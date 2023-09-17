use std::{
    ffi::OsStr,
    fs::{File, OpenOptions},
    io::{BufReader, Cursor},
    path::{Path, PathBuf},
};

use anyhow::{bail, Context, Result};
use clap::Parser;
use io::last::las_to_last;
use log::info;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    input: PathBuf,
    output: Option<PathBuf>,
}

fn is_las_file(path: &Path) -> bool {
    path.extension()
        .and_then(OsStr::to_str)
        .map(|ex| ex == "las" || ex == "LAS")
        .unwrap_or_default()
}

fn get_all_input_output_file_pairings(args: &Args) -> Result<Vec<(PathBuf, PathBuf)>> {
    if !args.input.exists() {
        bail!("Input path {} does not exist", args.input.display());
    }

    if args.input.is_file() {
        let output = args
            .output
            .clone()
            .unwrap_or(args.input.with_extension("last"));
        Ok(vec![(args.input.clone(), output)])
    } else if args.input.is_dir() {
        let all_las_files_in_dir =
            walkdir::WalkDir::new(&args.input)
                .into_iter()
                .filter_map(|entry| {
                    entry.ok().and_then(|entry| {
                        if is_las_file(entry.path()) {
                            Some(entry.path().to_owned())
                        } else {
                            None
                        }
                    })
                });

        if let Some(out_dir) = &args.output {
            if !out_dir.is_dir() {
                bail!("If input path is a directory, output path must also point to a directory!");
            }
            if !out_dir.exists() {
                std::fs::create_dir_all(&out_dir).with_context(|| {
                    format!("Could not create output directory {}", out_dir.display())
                })?;
            }

            // Get file stem for all input files, put them into out dir as `out_dir/stem.last`
            all_las_files_in_dir
                .map(|in_path| -> Result<(PathBuf, PathBuf)> {
                    let stem = in_path.file_stem().with_context(|| {
                        format!("Can't determine file stem of file {}", in_path.display())
                    })?;
                    let mut out_file = out_dir.join(stem);
                    out_file.set_extension("last");
                    Ok((in_path, out_file))
                })
                .collect()
        } else {
            // Replace extension of each input file with `.last` and return (in_path, last_in_path)
            all_las_files_in_dir
                .map(|in_path| -> Result<(PathBuf, PathBuf)> {
                    let out_path = in_path.with_extension("last");
                    Ok((in_path, out_path))
                })
                .collect()
        }
    } else {
        bail!("Unsupported input path {}", args.input.display())
    }
}

fn main() -> Result<()> {
    pretty_env_logger::init();

    let args = Args::parse();
    let input_output_file_pairs = get_all_input_output_file_pairings(&args)?;
    info!("Converting {} files", input_output_file_pairs.len());

    for (in_file, out_file) in input_output_file_pairs {
        if !is_las_file(&in_file) {
            bail!("Found non-LAS file {}", in_file.display());
        }

        let las_file = File::open(&in_file)
            .with_context(|| format!("Could not open file {}", in_file.display()))?;
        let num_bytes = las_file.metadata()?.len();
        let last_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&out_file)
            .with_context(|| {
                format!(
                    "Could not open output file {} for writing",
                    out_file.display()
                )
            })?;
        last_file
            .set_len(num_bytes)
            .with_context(|| format!("Could not resize output file {}", out_file.display()))?;

        let mut last_file_mmap = unsafe {
            memmap::MmapMut::map_mut(&last_file)
                .with_context(|| format!("Could not mmap output file {}", out_file.display()))?
        };

        las_to_last(
            BufReader::new(las_file),
            Cursor::new(&mut last_file_mmap[..]),
        )
        .with_context(|| format!("Conversion to LAST failed for file {}", in_file.display()))?;
    }
    Ok(())
}
