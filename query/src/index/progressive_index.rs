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
use memmap::Mmap;
use pasture_io::{base::IOFactory, las::LASReader};
use std::{
    collections::HashMap,
    ffi::OsStr,
    fs::File,
    io::Cursor,
    path::{Path, PathBuf},
};

use super::{Block, BlockIndex, Query};
use crate::{
    collect_points::ResultCollector,
    search::{compile_query, Extractor, LASExtractor, WhichIndicesToLoopOver},
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
        result_collector: &mut dyn ResultCollector,
    ) -> Result<()> {
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
        let blocks_per_file = dataset
            .index
            .get_matching_blocks(&query)
            .group_by(|(block, _)| block.file_id());

        let mut all_matching_indices = vec![true; INITIAL_BLOCK_SIZE];

        for (file_id, blocks) in &blocks_per_file {
            let path = dataset.files[file_id].as_path();
            let file_header = LASReader::from_path(path)
                .context("Can't open LAS file")?
                .header()
                .to_owned();
            let raw_header = file_header.into_raw().context("Can't get raw LAS header")?;

            let file = File::open(path).context("Can't open file")?;
            let file_mmap = unsafe { Mmap::map(&file).context("Can't mmap file")? };
            let mut cursor = Cursor::new(file_mmap);

            for (block, index_result) in blocks {
                // IndexResult will be either partial match or full match
                match index_result {
                    super::IndexResult::MatchAll => {
                        data_extractor
                            .extract_data(
                                &mut cursor,
                                &raw_header,
                                block.point_range(),
                                &mut all_matching_indices,
                                block.len(),
                                result_collector,
                            )
                            .context("Failed to extract data for block")?;
                    }
                    super::IndexResult::MatchSome => {
                        // We have to run the actual query on this block where only some indices match
                        // eval() will correctly set the valid indices in 'all_matching_indices', so we don't have to reset this array in every loop iteration
                        let num_matches = compiled_query.eval(
                            &mut cursor,
                            &raw_header,
                            block.point_range(),
                            &mut all_matching_indices,
                            block.len(),
                            WhichIndicesToLoopOver::All,
                        )?;
                        data_extractor
                            .extract_data(
                                &mut cursor,
                                &raw_header,
                                block.point_range(),
                                &mut all_matching_indices,
                                num_matches,
                                result_collector,
                            )
                            .context("Failed to extract data for block")?;
                    }
                    _ => panic!(
                        "get_matching_blocks should not return a block with IndexResult::NoMatch"
                    ),
                }
            }
        }

        Ok(())
    }
}

const INITIAL_BLOCK_SIZE: usize = 50_000; // Same size as the default LAZ block size

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
        let num_blocks = (point_count + INITIAL_BLOCK_SIZE - 1) / INITIAL_BLOCK_SIZE;
        for block_idx in 0..num_blocks {
            let block_start = block_idx * INITIAL_BLOCK_SIZE;
            let block_end = (block_start + INITIAL_BLOCK_SIZE).min(point_count);
            blocks.push(Block::new(block_start..block_end, file_id));
        }
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