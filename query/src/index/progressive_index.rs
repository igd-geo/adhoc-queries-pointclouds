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
use log::info;
use pasture_io::base::{GenericPointReader, PointReader};
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    ffi::OsStr,
    fs::File,
    io::BufWriter,
    path::{Path, PathBuf},
    sync::atomic::{AtomicUsize, Ordering},
    time::Instant,
};

use super::{
    Block, BlockIndex, IndexResult, PointRange, PositionIndex, Query, RefinementStrategy, ValueType,
};
use crate::{
    io::{InputLayer, PointOutput},
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

    /// Apply the given range of IndexRefinements to the indices of this dataset
    fn apply_refinements<I: Iterator<Item = (ValueType, FxHashSet<PointRange>)>>(
        &mut self,
        refinements: I,
        refinement_strategy: &dyn RefinementStrategy,
    ) -> Result<()> {
        // Group refinements by their index ValueType, then apply each set of refinements to each of the indices
        for (value_type, refinements) in refinements {
            if let Some(index) = self.indices.get_mut(&value_type) {
                let actual_candidates = refinement_strategy.select_best_candidates(refinements);
                index
                    .apply_refinements(actual_candidates.into_iter(), value_type, &self.files)
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
        let id = self.datasets.len();
        let (indices, common_file_extension) = build_initial_block_indices(files)
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
        self.input_layer
            .add_files(files, id)
            .context("Failed to add files to input layer")?;
        Ok(id)
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
            .map(|(_, blocks)| -> Result<()> {
                blocks
                    .into_par_iter()
                    .map_with(
                        all_matching_indices.clone(),
                        |all_matching_indices, (point_range, index_result)| -> Result<()> {
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
                                super::IndexResult::MatchSome => {
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
                                    data_output.output(
                                        &self.input_layer,
                                        dataset_id,
                                        point_range,
                                        matching_indices_within_this_block,
                                    )?;

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

        dataset
            .apply_refinements(
                rough_query_result.blocks_for_refinement.into_iter(),
                refinement_strategy,
            )
            .context("Failed to refine index")?;

        // info!(
        //     "Blocks queried: {} full, {} partial ({} total in dataset)",
        //     full_match_blocks,
        //     partial_match_blocks,
        //     dataset.index.blocks_count()
        // );

        runtime_tracker.to_csv(BufWriter::new(File::create("query.csv")?))?;

        Ok(QueryStats {
            total_blocks_queried: 0, //TODO
            full_match_blocks: full_match_blocks.into_inner(),
            partial_match_blocks: partial_match_blocks.into_inner(),
            total_points_queried: total_points_queried.into_inner(),
            matching_points: matching_points.into_inner(),
            runtime: timer.elapsed(),
        })
    }

    /// Returns all known datasets for this ProgressiveIndex
    pub fn datasets(&self) -> &HashMap<DatasetID, KnownDataset> {
        &self.datasets
    }
}

// const INITIAL_BLOCK_SIZE: usize = 50_000; // Same size as the default LAZ block size

fn build_initial_block_indices<P: AsRef<Path>>(
    files: &'_ [P],
) -> Result<(FxHashMap<ValueType, BlockIndex>, String)> {
    let common_extension =
        common_file_extension(files).context("Can't get common file extension")?;
    let mut position_blocks = vec![];
    let mut classification_blocks = vec![];
    for (file_id, file) in files.iter().enumerate() {
        let reader = GenericPointReader::open_file(file.as_ref()).context(format!(
            "Can't open file reader for file {}",
            file.as_ref().display()
        ))?;
        let point_count = reader
            .get_metadata()
            .number_of_points()
            .ok_or(anyhow!("Can't determine number of points"))?;
        let bounds = reader
            .get_metadata()
            .bounds()
            .ok_or(anyhow!("Can't get bounds"))?;
        // let num_blocks = (point_count + INITIAL_BLOCK_SIZE - 1) / INITIAL_BLOCK_SIZE;
        // for block_idx in 0..num_blocks {
        //     let block_start = block_idx * INITIAL_BLOCK_SIZE;
        //     let block_end = (block_start + INITIAL_BLOCK_SIZE).min(point_count);
        //     blocks.push(Block::new(block_start..block_end, file_id));
        // }

        // Create one block per file and add a position index to it (we can do that because we have the bounds in the LAS header)
        let mut block = Block::new(0..point_count, file_id);
        let position_index = PositionIndex::new(bounds);
        block.set_index(Box::new(position_index));
        position_blocks.push(block);
        // Can't create an initial index for classifications because there is no histogram within the header. So we use a block without
        // an index
        classification_blocks.push(Block::new(0..point_count, file_id));
    }

    let mut indices: FxHashMap<_, _> = Default::default();
    indices.insert(ValueType::Position3D, BlockIndex::new(position_blocks));
    indices.insert(
        ValueType::Classification,
        BlockIndex::new(classification_blocks),
    );

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
