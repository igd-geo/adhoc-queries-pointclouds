// Progressive index:
// Takes a query and a dataset and:
//   1) Executes the query using whatever index is available
//   2) Refines the existing index while answering the query

// Progressive index thus knows datasets and keeps track of one or more indices per dataset. This could be e.g. one `BlockIndex` per file

// Executing a query goes like this:
//   1) Check if dataset is known. If not, make sure all files have the same format
//   2) Compile the query into target format
//   3) Generate a stream of blocks that have to be investigated closer
//      3.1) A block refers to a contiguous range of points within a specific file
//      3.2) If an index exists, the index will take the query and spit out all relevant blocks (potentially with a classification of either 'all match' or 'some match')
//      3.3) If no index exists, all points in all files are converted to blocks and each block has to be processed
//      3.4) Process all blocks in parallel, e.g. using `rayon`
//      3.5) Implement some refinement procedure that generates or improves the index for each block (this is TBD)

use anyhow::{anyhow, bail, Context, Result};
use human_repr::HumanCount;
use itertools::Itertools;
use log::info;
use pasture_core::{
    layout::{
        attributes::{GPS_TIME, NUMBER_OF_RETURNS, RETURN_NUMBER},
        PointLayout,
    },
    math::AABB,
    meta::Metadata,
};
use pasture_io::las::{ATTRIBUTE_BASIC_FLAGS, ATTRIBUTE_EXTENDED_FLAGS};
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    ffi::OsStr,
    fmt::Display,
    path::{Path, PathBuf},
    sync::atomic::{AtomicUsize, Ordering},
    time::Instant,
};

use super::{
    Block, BlockIndex, IndexResult, PointRange, PositionIndex, Query, RefinementStrategy, ValueType,
};
use crate::{
    io::{FileHandle, InputLayer, PointOutput},
    search::{compile_query, WhichIndicesToLoopOver},
    stats::{BlockQueryRuntimeTracker, QueryStats},
};

/// Result of a rough query: A set of blocks in each file and a set of blocks that might be
/// good candidates for refinement
#[derive(Debug, Default)]
pub struct RoughQueryResult {
    /// Collection of matching blocks (sorted ascending) per file in a dataset
    matching_blocks: FxHashMap<usize, Vec<(PointRange, IndexResult)>>,
    /// Set of blocks for each ValueType that are candidates for index refinement
    blocks_for_refinement: FxHashMap<ValueType, FxHashSet<PointRange>>,
}

#[derive(Debug, Clone)]
pub struct DatasetStats {
    num_files: usize,
    num_points: usize,
    bounds: Option<AABB<f64>>,
    point_layout: PointLayout,
}

impl DatasetStats {
    pub fn num_files(&self) -> usize {
        self.num_files
    }

    pub fn num_points(&self) -> usize {
        self.num_points
    }

    pub fn bounds(&self) -> Option<&AABB<f64>> {
        self.bounds.as_ref()
    }

    pub fn point_layout(&self) -> &PointLayout {
        &self.point_layout
    }
}

impl Display for DatasetStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Number of files:   {}", self.num_files)?;
        writeln!(f, "Number of points:  {}", self.num_points)?;
        if let Some(bounds) = self.bounds.as_ref() {
            writeln!(
                f,
                "Bounds:            ({}, {}, {}) ({}, {}, {})",
                bounds.min().x,
                bounds.min().y,
                bounds.min().z,
                bounds.max().x,
                bounds.max().y,
                bounds.max().z
            )?;
        } else {
            writeln!(f, "Bounds:            Unknown")?;
        }
        Ok(())
    }
}

#[derive(Serialize, Deserialize)]
pub struct KnownDataset {
    files: Vec<PathBuf>,
    indices: FxHashMap<ValueType, BlockIndex>,
    common_file_extension: String,
}

impl KnownDataset {
    /// Evaluate the rough query, i.e. the query that returns a set of `PointRange`s that contain points that might
    /// match the query. The rough query also returns a set of blocks per index that are candidates for index refinement
    pub fn rough_query(&self, query: &Query) -> RoughQueryResult {
        let _span = tracy_client::span!("rough_query");
        // Which indices does the query need?
        // If only one index, loop over the blocks of that index and filter them using the query
        // If multiple indices, get an iterator over all blocks per index and start with the first block in each index. Advance the iterator
        // for the smallest block(s) and always evaluate using all indices for the current blocks

        let query_result = query.eval(&self.indices);
        RoughQueryResult {
            matching_blocks: query_result.matching_blocks_by_file(),
            blocks_for_refinement: query_result.potential_blocks_for_refinement,
        }
    }

    /// Access to the indices of this dataset
    pub fn indices(&self) -> &FxHashMap<ValueType, BlockIndex> {
        &self.indices
    }

    pub fn file_paths(&self) -> &[PathBuf] {
        &self.files
    }

    /// Apply the given range of IndexRefinements to the indices of this dataset
    fn apply_refinements<I: Iterator<Item = (ValueType, FxHashSet<PointRange>)>>(
        &mut self,
        refinements: I,
        refinement_strategy: &dyn RefinementStrategy,
        input_layer: &InputLayer,
        dataset_id: DatasetID,
    ) -> Result<()> {
        // Group refinements by their index ValueType, then apply each set of refinements to each of the indices
        for (value_type, refinements) in refinements {
            if let Some(index) = self.indices.get_mut(&value_type) {
                let actual_candidates = refinement_strategy.select_best_candidates(
                    refinements,
                    value_type,
                    dataset_id,
                    input_layer,
                );

                let total_points_to_refine = actual_candidates
                    .iter()
                    .map(|block| block.points_in_file.len())
                    .sum::<usize>();
                info!(
                    "Refining {} blocks with a total of {} points",
                    actual_candidates.len(),
                    total_points_to_refine.human_count_bare()
                );

                index
                    .apply_refinements(actual_candidates, value_type, input_layer, dataset_id)
                    .context(format!(
                        "Failed to refine index for ValueType {}",
                        value_type
                    ))?;
            }
        }

        Ok(())
    }
}

pub type DatasetID = usize;

pub struct ProgressiveIndex {
    datasets: HashMap<DatasetID, KnownDataset>,
    input_layer: InputLayer,
}

impl ProgressiveIndex {
    pub fn new() -> Self {
        Self {
            datasets: HashMap::new(),
            input_layer: Default::default(),
        }
    }

    pub fn with_datasets(datasets: HashMap<DatasetID, KnownDataset>) -> Self {
        Self {
            datasets,
            input_layer: Default::default(),
        }
    }

    /// Adds a new dataset to the ProgressiveIndex. A dataset is modeled as a set of files, for which the ProgressiveIndex
    /// will build an initial, very rough block index
    pub fn add_dataset<'a, P: AsRef<Path>>(&mut self, files: &'a [P]) -> Result<DatasetID> {
        info!("Adding new dataset");
        if files.is_empty() {
            bail!("No files in dataset");
        }

        let id = self.datasets.len();
        self.input_layer
            .add_files(files, id)
            .context("Failed to add files to input layer")?;

        let (indices, common_file_extension) =
            build_initial_block_indices(files, id, &self.input_layer)
                .context("Failed do build initial index for dataset")?;
        self.datasets.insert(
            id,
            KnownDataset {
                files: files
                    .into_iter()
                    .map(|file| file.as_ref().to_owned())
                    .collect(),
                indices,
                common_file_extension,
            },
        );
        Ok(id)
    }

    /// Information about the dataset, i.e. how many points, bounds etc.
    pub fn dataset_stats(&self, dataset_id: DatasetID) -> DatasetStats {
        let mut num_points: usize = 0;
        let mut global_bounds: Option<AABB<f64>> = None;
        let mut point_layout: Option<PointLayout> = None;

        let num_files = self
            .datasets
            .get(&dataset_id)
            .expect("Unknown dataset")
            .file_paths()
            .len();
        for idx in 0..num_files {
            let las_metadata = self
                .input_layer
                .get_las_metadata(FileHandle(dataset_id, idx))
                .expect("Stats are only supported for LAS-like files");
            let bounds = las_metadata.bounds().expect("No bounds found");
            num_points += las_metadata.point_count();
            if let Some(current_bounds) = global_bounds {
                global_bounds = Some(AABB::union(&current_bounds, &bounds));
            } else {
                global_bounds = Some(bounds);
            }

            let file_point_layout = self
                .input_layer
                .get_default_point_layout_of_file(FileHandle(dataset_id, idx))
                .expect("Could not get default PointLayout of file");
            if let Some(point_layout) = point_layout.as_ref() {
                if *point_layout != file_point_layout {
                    panic!("All files must have the same PointLayout!");
                }
            } else {
                point_layout = Some(file_point_layout);
            }
        }

        DatasetStats {
            num_files,
            num_points,
            bounds: global_bounds,
            point_layout: point_layout.expect("No files found in dataset"),
        }
    }

    /// Run a query on the ProgressiveIndex for the given dataset. The resulting points of the query are collected by
    /// the given `result_collector`
    pub fn query(
        &mut self,
        dataset_id: DatasetID,
        query: Query,
        refinement_strategy: &dyn RefinementStrategy,
        data_output: &impl PointOutput,
    ) -> Result<QueryStats> {
        let _span = tracy_client::span!("query");

        info!("Querying dataset {}", dataset_id);
        let timer = Instant::now();
        let runtime_tracker = BlockQueryRuntimeTracker::default();

        let dataset = self.datasets.get_mut(&dataset_id).expect("Unknown dataset");
        let compiled_query =
            compile_query(&query, &dataset.common_file_extension).context(format!(
                "Can't compile query for file format {}",
                dataset.common_file_extension
            ))?;
        // let data_extractor = match dataset.common_file_extension.as_str() {
        //     "las" => LASExtractor {},
        //     _ => bail!(
        //         "Unsupported file extension {}",
        //         &dataset.common_file_extension
        //     ),
        // };

        // Query with index to get matching blocks, group the blocks by their file
        let rough_query_result = dataset.rough_query(&query);
        // info!("Matching blocks:");
        // for (file_index, ranges) in rough_query_result.matching_blocks.iter() {
        //     for range in ranges {
        //         info!(
        //             "File {} - Points {:#?} - {:#?}",
        //             dataset.files[*file_index].display(),
        //             range.0.points_in_file,
        //             range.1
        //         );
        //     }
        // }

        let largest_block = dataset
            .indices
            .iter()
            .map(|(_, index)| index.largest_block().expect("No blocks").len())
            .max()
            .expect("No indices");
        let all_matching_indices = vec![true; largest_block];

        let partial_match_blocks = AtomicUsize::new(0);
        let full_match_blocks = AtomicUsize::new(0);
        let total_points_queried = AtomicUsize::new(0);
        let matching_points = AtomicUsize::new(0);

        rough_query_result
            .matching_blocks
            .into_par_iter()
            .map(|(_, mut blocks)| -> Result<()> {
                // Sort blocks so that blocks with many matches are handled first
                blocks.sort_by(|a,b| a.1.cmp(&b.1));
                blocks
                    .into_par_iter()
                    .map_with(
                        all_matching_indices.clone(),
                        |all_matching_indices, (point_range, index_result)| -> Result<()> {
                            let _span = tracy_client::span!("fine_query_block");

                            let block_length = point_range.points_in_file.len();
                            // let mut cursor = Cursor::new(file_data);

                            let matching_indices_within_this_block =
                                &mut all_matching_indices[..block_length];

                            // IndexResult will be either partial match or full match
                            match index_result {
                                super::IndexResult::MatchAll => {
                                    data_output.output(
                                        &self.input_layer,
                                        dataset_id,
                                        point_range,
                                        matching_indices_within_this_block,
                                    )?;

                                    full_match_blocks.fetch_add(1, Ordering::SeqCst);
                                    total_points_queried.fetch_add(block_length, Ordering::SeqCst);
                                    matching_points.fetch_add(block_length, Ordering::SeqCst);
                                }
                                super::IndexResult::MatchSome(_how_many) => {
                                    // We have to run the actual query on this block where only some indices match
                                    // eval() will correctly set the valid indices in 'all_matching_indices', so we don't have to reset this array in every loop iteration
                                    let num_matches = compiled_query.eval(
                                        &self.input_layer,
                                        point_range.clone(),
                                        dataset_id,
                                        matching_indices_within_this_block,
                                        block_length,
                                        WhichIndicesToLoopOver::All,
                                        &runtime_tracker,
                                    )?;

                                    if num_matches > 0 {
                                        data_output.output(
                                            &self.input_layer,
                                            dataset_id,
                                            point_range,
                                            matching_indices_within_this_block,
                                        )?;
                                    }

                                    partial_match_blocks.fetch_add(1, Ordering::SeqCst);
                                    total_points_queried.fetch_add(block_length, Ordering::SeqCst);
                                    matching_points.fetch_add(num_matches, Ordering::SeqCst);
                                }
                                _ => panic!(
                                    "get_matching_blocks should not return a block with IndexResult::NoMatch"
                                ),
                            }
                            Ok(())
                        },
                    )
                    .collect::<Result<Vec<_>, _>>()?;

                Ok(())
            })
            .collect::<Result<Vec<_>, _>>()?;

        let refinement_time = {
            let timer = Instant::now();
            dataset
                .apply_refinements(
                    rough_query_result.blocks_for_refinement.into_iter(),
                    refinement_strategy,
                    &self.input_layer,
                    dataset_id,
                )
                .context("Failed to refine index")?;
            timer.elapsed()
        };

        // info!(
        //     "Blocks queried: {} full, {} partial ({} total in dataset)",
        //     full_match_blocks,
        //     partial_match_blocks,
        //     dataset.index.blocks_count()
        // );

        // runtime_tracker.to_csv(BufWriter::new(File::create("query.csv")?))?;

        Ok(QueryStats {
            total_blocks_queried: 0, //TODO
            full_match_blocks: full_match_blocks.into_inner(),
            partial_match_blocks: partial_match_blocks.into_inner(),
            total_points_queried: total_points_queried.into_inner(),
            matching_points: matching_points.into_inner(),
            runtime: timer.elapsed(),
            refinement_time,
        })
    }

    /// Returns all known datasets for this ProgressiveIndex
    pub fn datasets(&self) -> &HashMap<DatasetID, KnownDataset> {
        &self.datasets
    }
}

// const INITIAL_BLOCK_SIZE: usize = 50_000; // Same size as the default LAZ block size

/// Maximum size of a Block in the BlockIndex. We set an upper bound for two reasons:
/// 1) To enable parallelization even if we only have one (large) file
/// 2) To prevent excessive memory usage for large files that otherwise would have just one block
const MAX_BLOCK_SIZE: usize = 1_000_000;

fn build_initial_block_indices<P: AsRef<Path>>(
    files: &'_ [P],
    dataset_id: DatasetID,
    input_layer: &InputLayer,
) -> Result<(FxHashMap<ValueType, BlockIndex>, String)> {
    let _span = tracy_client::span!("build_initial_block_indices");

    let common_extension =
        common_file_extension(files).context("Can't get common file extension")?;

    let mut blocks_per_value_type: HashMap<ValueType, Vec<Block>> = Default::default();

    for (file_id, file) in files.iter().enumerate() {
        let file_handle = FileHandle(dataset_id, file_id);
        let las_metadata = input_layer.get_las_metadata(file_handle).ok_or_else(|| {
            anyhow!(
                "File {} is not a LAS-derivate, could not get metadata information",
                file.as_ref().display()
            )
        })?;

        let point_count = las_metadata
            .number_of_points()
            .ok_or(anyhow!("Can't determine number of points"))?;
        let num_blocks = (point_count + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE;
        let block_point_ranges = (0..num_blocks)
            .map(|block_index| {
                let point_start = block_index * MAX_BLOCK_SIZE;
                let point_end = ((block_index + 1) * MAX_BLOCK_SIZE).min(point_count);
                point_start..point_end
            })
            .collect_vec();

        let bounds = las_metadata.bounds().ok_or(anyhow!("Can't get bounds"))?;

        // Create initial blocks for PositionIndex. For larger files, we have multiple blocks that all share the
        // same bounds, which is pessimistic but still allows parallelization
        let mut position_blocks = block_point_ranges
            .iter()
            .map(|point_range| {
                let mut block = Block::new(point_range.clone(), file_id);
                let position_index = PositionIndex::new(bounds);
                block.set_index(Box::new(position_index));
                block
            })
            .collect_vec();

        if let Some(existing_blocks) = blocks_per_value_type.get_mut(&ValueType::Position3D) {
            existing_blocks.append(&mut position_blocks);
        } else {
            blocks_per_value_type.insert(ValueType::Position3D, position_blocks);
        }

        // Can't create an initial index for classifications because there is no histogram within the header. So we use a block without
        // an index
        let mut classification_blocks = block_point_ranges
            .iter()
            .map(|point_range| Block::new(point_range.clone(), file_id))
            .collect_vec();
        if let Some(existing_blocks) = blocks_per_value_type.get_mut(&ValueType::Classification) {
            existing_blocks.append(&mut classification_blocks);
        } else {
            blocks_per_value_type.insert(ValueType::Classification, classification_blocks);
        }

        // Add indices for other attributes
        let point_layout = input_layer
            .get_default_point_layout_of_file(file_handle)
            .with_context(|| {
                format!(
                    "Could not determine default PointLayout for file {}",
                    file.as_ref().display()
                )
            })?;
        // Return numbers and number of returns might not be stored directly but as packed LAS flags
        if point_layout.has_attribute(&RETURN_NUMBER)
            || point_layout.has_attribute(&ATTRIBUTE_BASIC_FLAGS)
            || point_layout.has_attribute(&ATTRIBUTE_EXTENDED_FLAGS)
        {
            let mut return_number_blocks = block_point_ranges
                .iter()
                .map(|point_range| Block::new(point_range.clone(), file_id))
                .collect_vec();
            if let Some(blocks) = blocks_per_value_type.get_mut(&ValueType::ReturnNumber) {
                blocks.append(&mut return_number_blocks);
            } else {
                blocks_per_value_type.insert(ValueType::ReturnNumber, return_number_blocks);
            }
        }

        if point_layout.has_attribute(&NUMBER_OF_RETURNS)
            || point_layout.has_attribute(&ATTRIBUTE_BASIC_FLAGS)
            || point_layout.has_attribute(&ATTRIBUTE_EXTENDED_FLAGS)
        {
            let mut number_of_returns_blocks = block_point_ranges
                .iter()
                .map(|point_range| Block::new(point_range.clone(), file_id))
                .collect_vec();
            if let Some(blocks) = blocks_per_value_type.get_mut(&ValueType::NumberOfReturns) {
                blocks.append(&mut number_of_returns_blocks);
            } else {
                blocks_per_value_type.insert(ValueType::NumberOfReturns, number_of_returns_blocks);
            }
        }

        if point_layout.has_attribute(&GPS_TIME) {
            let mut gps_time_blocks = block_point_ranges
                .iter()
                .map(|point_range| Block::new(point_range.clone(), file_id))
                .collect_vec();
            if let Some(blocks) = blocks_per_value_type.get_mut(&ValueType::GpsTime) {
                blocks.append(&mut gps_time_blocks);
            } else {
                blocks_per_value_type.insert(ValueType::GpsTime, gps_time_blocks);
            }
        }
    }

    let mut indices: FxHashMap<_, _> = Default::default();
    for (value_type, blocks) in blocks_per_value_type {
        indices.insert(value_type, BlockIndex::new(blocks));
    }

    Ok((indices, common_extension.to_string_lossy().to_lowercase()))
}

fn common_file_extension<P: AsRef<Path>>(files: &[P]) -> Result<&OsStr> {
    let mut common_extension = None;
    for (file, maybe_extension) in files.iter().map(|f| (f, f.as_ref().extension())) {
        match maybe_extension {
            None => bail!(
                "Can't determine file extension of file {}",
                file.as_ref().display()
            ),
            Some(ex) => match common_extension {
                None => common_extension = Some(ex),
                Some(prev_extension) => {
                    if prev_extension != ex {
                        bail!(
                            "Files have different file extensions ({} and {})",
                            prev_extension.to_string_lossy(),
                            ex.to_string_lossy(),
                        );
                    }
                }
            },
        }
    }
    common_extension.ok_or(anyhow!("No files found"))
}
