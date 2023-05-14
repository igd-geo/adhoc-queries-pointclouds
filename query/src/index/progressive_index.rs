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
use itertools::Itertools;
use log::info;
use memmap::Mmap;
use pasture_io::{base::IOFactory, las_rs::raw};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::{
    collections::HashMap,
    ffi::OsStr,
    fs::File,
    io::{BufWriter, Cursor},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
    time::Instant,
};

use super::{Block, BlockIndex, IndexResult, PointRange, PositionIndex, Query, ValueType};
use crate::{
    collect_points::ResultCollector,
    index::IndexRefinement,
    search::{compile_query, Extractor, LASExtractor, WhichIndicesToLoopOver},
    stats::{BlockQueryRuntimeTracker, QueryStats},
};

struct KnownDataset {
    files: Vec<PathBuf>,
    indices: FxHashMap<ValueType, BlockIndex>,
    common_file_extension: String,
}

impl KnownDataset {
    pub(crate) fn get_matching_blocks(
        &self,
        query: &Query,
    ) -> FxHashMap<usize, Vec<(PointRange, IndexResult)>> {
        // Which indices does the query need?
        // If only one index, loop over the blocks of that index and filter them using the query
        // If multiple indices, get an iterator over all blocks per index and start with the first block in each index. Advance the iterator
        // for the smallest block(s) and always evaluate using all indices for the current blocks

        let mut result: FxHashMap<usize, Vec<(PointRange, IndexResult)>> = Default::default();

        for (point_range, query_result) in query.eval(&self.indices) {
            match query_result {
                IndexResult::MatchAll | IndexResult::MatchSome => {
                    if let Some(blocks_of_file) = result.get_mut(&point_range.file_index) {
                        blocks_of_file.push((point_range, query_result));
                    } else {
                        result.insert(point_range.file_index, vec![(point_range, query_result)]);
                    }
                }
                _ => (),
            }
        }

        result
    }

    /// Apply the given range of IndexRefinements to the indices of this dataset
    fn apply_refinements<I: Iterator<Item = IndexRefinement>>(&mut self, refinements: I) {
        // Group refinements by their index ValueType, then apply each set of refinements to each of the indices
        for (value_type, refinements) in &refinements.group_by(|refinement| refinement.value_type) {
            if let Some(index) = self.indices.get_mut(&value_type) {
                index.apply_refinements(&refinements.collect::<Vec<_>>());
            }
        }
    }
}

pub type DatasetID = usize;

pub struct ProgressiveIndex {
    datasets: HashMap<DatasetID, KnownDataset>,
}

impl ProgressiveIndex {
    pub fn new() -> Self {
        Self {
            datasets: HashMap::new(),
        }
    }

    pub fn add_dataset(&mut self, files: Vec<PathBuf>) -> Result<DatasetID> {
        info!("Adding new dataset");
        let id = self.datasets.len();
        let (indices, common_file_extension) = build_initial_block_indices(&files)
            .context("Failed do build initial index for dataset")?;
        self.datasets.insert(
            id,
            KnownDataset {
                files,
                indices,
                common_file_extension,
            },
        );
        Ok(id)
    }

    pub fn query(
        &mut self,
        dataset_id: DatasetID,
        query: Query,
        result_collector: Arc<Mutex<dyn ResultCollector>>,
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
        let data_extractor = match dataset.common_file_extension.as_str() {
            "las" => LASExtractor {},
            _ => bail!(
                "Unsupported file extension {}",
                &dataset.common_file_extension
            ),
        };

        // Query with index to get matching blocks, group the blocks by their file
        let blocks_per_file = dataset.get_matching_blocks(&query);

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

        let refined_indices = blocks_per_file
            .into_par_iter()
            .map(|(file_id, blocks)| -> Result<Vec<IndexRefinement>> {
                let path = dataset.files[file_id].as_path();

                // TODO This is LAS-specific, extract this
                let file = File::open(path).context("Can't open file")?;
                let file_mmap = unsafe { Mmap::map(&file).context("Can't mmap file")? };
                let file_data: &[u8] = &file_mmap;

                let raw_header = raw::Header::read_from(&mut Cursor::new(file_data))
                    .context("Can't read LAS header")?;

                let refined_indices = blocks
                    .into_par_iter()
                    .map_with(
                        all_matching_indices.clone(),
                        |all_matching_indices,
                         (point_range, index_result)|
                         -> Result<Vec<IndexRefinement>> {
                            let block_length = point_range.points_in_file.len();
                            let mut cursor = Cursor::new(file_data);

                            // Simpler way to refine would be to refine as a separate step AFTER evaluating the query
                            let mut refined_indices = Some(vec![]);

                            // IndexResult will be either partial match or full match
                            match index_result {
                                super::IndexResult::MatchAll => {
                                    let data = data_extractor
                                        .extract_data(
                                            &mut cursor,
                                            &raw_header,
                                            point_range,
                                            all_matching_indices,
                                            block_length,
                                            &runtime_tracker,
                                        )
                                        .context("Failed to extract data for block")?;

                                    result_collector.lock().unwrap().collect(data);

                                    full_match_blocks.fetch_add(1, Ordering::SeqCst);
                                    total_points_queried.fetch_add(block_length, Ordering::SeqCst);
                                    matching_points.fetch_add(block_length, Ordering::SeqCst);
                                }
                                super::IndexResult::MatchSome => {
                                    // We have to run the actual query on this block where only some indices match
                                    // eval() will correctly set the valid indices in 'all_matching_indices', so we don't have to reset this array in every loop iteration
                                    let num_matches = compiled_query.eval(
                                        &mut cursor,
                                        &raw_header,
                                        point_range.clone(),
                                        all_matching_indices,
                                        block_length,
                                        WhichIndicesToLoopOver::All,
                                        &mut refined_indices,
                                        &runtime_tracker,
                                    )?;
                                    let data = data_extractor
                                        .extract_data(
                                            &mut cursor,
                                            &raw_header,
                                            point_range.clone(),
                                            &mut all_matching_indices[..block_length],
                                            num_matches,
                                            &runtime_tracker,
                                        )
                                        .context("Failed to extract data for block")?;

                                    result_collector.lock().unwrap().collect(data);

                                    partial_match_blocks.fetch_add(1, Ordering::SeqCst);
                                    total_points_queried.fetch_add(block_length, Ordering::SeqCst);
                                    matching_points.fetch_add(num_matches, Ordering::SeqCst);
                                }
                                _ => panic!(
                        "get_matching_blocks should not return a block with IndexResult::NoMatch"
                    ),
                            }
                            Ok(refined_indices.unwrap())
                        },
                    )
                    .collect::<Result<Vec<_>, _>>()?;

                Ok(refined_indices.into_iter().flatten().collect())
            })
            .collect::<Result<Vec<_>, _>>()?;

        dataset.apply_refinements(refined_indices.into_iter().flatten());

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
}

// const INITIAL_BLOCK_SIZE: usize = 50_000; // Same size as the default LAZ block size

fn build_initial_block_indices<P: AsRef<Path>>(
    files: &'_ [P],
) -> Result<(FxHashMap<ValueType, BlockIndex>, String)> {
    let common_extension =
        common_file_extension(files).context("Can't get common file extension")?;
    let io_factory = IOFactory::default();
    let mut position_blocks = vec![];
    let mut classification_blocks = vec![];
    for (file_id, file) in files.iter().enumerate() {
        let reader = io_factory.make_reader(file.as_ref()).context(format!(
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
