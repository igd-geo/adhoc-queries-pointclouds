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
use memmap::Mmap;
use pasture_io::{base::IOFactory, las_rs::raw};
use rayon::prelude::*;
use std::{
    collections::HashMap,
    ffi::OsStr,
    fs::File,
    io::Cursor,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
    time::Instant,
};

use super::{Block, BlockIndex, PositionIndex, Query, ValueType};
use crate::{
    collect_points::ResultCollector,
    search::{compile_query, Extractor, LASExtractor, WhichIndicesToLoopOver},
    stats::QueryStats,
};

struct KnownDataset {
    files: Vec<PathBuf>,
    index: BlockIndex,
    common_file_extension: String,
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
        let (index, common_file_extension) = build_initial_block_index(&files)
            .context("Failed do build initial index for dataset")?;
        self.datasets.insert(
            id,
            KnownDataset {
                files,
                index,
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

        let dataset = self.datasets.get(&dataset_id).expect("Unknown dataset");
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
        let blocks = dataset.index.get_matching_blocks(&query);
        let mut blocks_per_file: HashMap<usize, Vec<_>> = HashMap::new();
        for (block, query_result) in blocks {
            if let Some(blocks_of_file) = blocks_per_file.get_mut(&block.file_id()) {
                blocks_of_file.push((block.len(), block.point_range(), query_result));
            } else {
                blocks_per_file.insert(
                    block.file_id(),
                    vec![(block.len(), block.point_range(), query_result)],
                );
            }
        }

        let largest_block = dataset
            .index
            .largest_block()
            .expect("There are no blocks in the index")
            .len();
        let all_matching_indices = vec![true; largest_block];

        let partial_match_blocks = AtomicUsize::new(0);
        let full_match_blocks = AtomicUsize::new(0);
        let total_points_queried = AtomicUsize::new(0);
        let matching_points = AtomicUsize::new(0);

        blocks_per_file
            .into_par_iter()
            .map(|(file_id, blocks)| -> Result<()> {
                let path = dataset.files[file_id].as_path();

                let file = File::open(path).context("Can't open file")?;
                let file_mmap = unsafe { Mmap::map(&file).context("Can't mmap file")? };
                let file_data: &[u8] = &file_mmap;

                let raw_header = raw::Header::read_from(&mut Cursor::new(file_data))
                    .context("Can't read LAS header")?;

                blocks
                    .into_par_iter()
                    .map_with(
                        all_matching_indices.clone(),
                        |all_matching_indices,
                         (block_length, block_point_range, index_result)|
                         -> Result<()> {
                            let mut cursor = Cursor::new(file_data);
                            // IndexResult will be either partial match or full match
                            match index_result {
                                super::IndexResult::MatchAll => {
                                    let data = data_extractor
                                        .extract_data(
                                            &mut cursor,
                                            &raw_header,
                                            block_point_range,
                                            all_matching_indices,
                                            block_length,
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
                                        block_point_range.clone(),
                                        all_matching_indices,
                                        block_length,
                                        WhichIndicesToLoopOver::All,
                                    )?;
                                    let data = data_extractor
                                        .extract_data(
                                            &mut cursor,
                                            &raw_header,
                                            block_point_range.clone(),
                                            &mut all_matching_indices[..block_length],
                                            num_matches,
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
                            Ok(())
                        },
                    )
                    .collect::<Result<_>>()?;

                Ok(())
            })
            .collect::<Result<_>>()?;

        // info!(
        //     "Blocks queried: {} full, {} partial ({} total in dataset)",
        //     full_match_blocks,
        //     partial_match_blocks,
        //     dataset.index.blocks_count()
        // );

        Ok(QueryStats {
            total_blocks_queried: dataset.index.blocks_count(),
            full_match_blocks: full_match_blocks.into_inner(),
            partial_match_blocks: partial_match_blocks.into_inner(),
            total_points_queried: total_points_queried.into_inner(),
            matching_points: matching_points.into_inner(),
            runtime: timer.elapsed(),
        })
    }
}

// const INITIAL_BLOCK_SIZE: usize = 50_000; // Same size as the default LAZ block size

fn build_initial_block_index<P: AsRef<Path>>(files: &'_ [P]) -> Result<(BlockIndex, String)> {
    let common_extension =
        common_file_extension(files).context("Can't get common file extension")?;
    let io_factory = IOFactory::default();
    let mut blocks = vec![];
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
        block
            .indices_mut()
            .insert(ValueType::Position3D, Box::new(position_index));
        blocks.push(block);
    }

    Ok((
        BlockIndex::new(blocks),
        common_extension.to_string_lossy().to_lowercase(),
    ))
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
