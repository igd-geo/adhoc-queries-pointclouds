use std::{
    ffi::OsStr,
    fs::File,
    io::{BufWriter, Cursor},
    path::{Path, PathBuf},
    sync::atomic::{AtomicUsize, Ordering},
};

use anyhow::{bail, Context, Result};
use clap::Parser;
use io::lazer::LazerWriter;
use log::info;
use lz4::EncoderBuilder;
use pasture_core::containers::ExternalMemoryBuffer;
use pasture_io::{
    base::PointWriter,
    las::point_layout_from_las_point_format,
    las_rs::{point::Format, raw, Builder},
};
use rayon::prelude::*;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to input dataset. Can be a single LAS file or a directory containing zero or more LAS files. The directory
    /// is not parsed recursively, and all files without either `.las` or `.LAS` at the end are ignored
    input: PathBuf,
    /// Optional output path. Must be a valid file path if `input` refers to a single LAS file, or a valid directory
    /// path if `input` refers to a directory. In the directory case, output files will have the same name as input
    /// files, but with `.lazer` as their extension instead of `.las`
    output: Option<PathBuf>,
    /// The LZ4 compression level. Must be between 0 and 12, default is 9
    #[arg(short, long, default_value_t = 9)]
    compression_level: u32,
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
            .unwrap_or(args.input.with_extension("lazer"));
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

            // Get file stem for all input files, put them into out dir as `out_dir/stem.lazer`
            all_las_files_in_dir
                .map(|in_path| -> Result<(PathBuf, PathBuf)> {
                    let stem = in_path.file_stem().with_context(|| {
                        format!("Can't determine file stem of file {}", in_path.display())
                    })?;
                    let mut out_file = out_dir.join(stem);
                    out_file.set_extension("lazer");
                    Ok((in_path, out_file))
                })
                .collect()
        } else {
            // Replace extension of each input file with `.lazer` and return (in_path, lazer_in_path)
            all_las_files_in_dir
                .map(|in_path| -> Result<(PathBuf, PathBuf)> {
                    let out_path = in_path.with_extension("lazer");
                    Ok((in_path, out_path))
                })
                .collect()
        }
    } else {
        bail!("Unsupported input path {}", args.input.display())
    }
}

fn las_to_lazer(las_path: &Path, lazer_path: &Path, encoder_builder: EncoderBuilder) -> Result<()> {
    let las_file_bytes = std::fs::read(las_path)?;
    let (las_points, raw_header) = {
        let header = raw::Header::read_from(Cursor::new(&las_file_bytes))?;
        let offset_to_point_data = header.offset_to_point_data as usize;
        let point_layout = point_layout_from_las_point_format(
            &Format::new(header.point_data_record_format)?,
            true,
        )?;
        (
            ExternalMemoryBuffer::new(&las_file_bytes[offset_to_point_data..], point_layout),
            header,
        )
    };
    let las_header = Builder::new(raw_header)?.into_header()?;

    let lazer_file = BufWriter::new(
        File::create(lazer_path)
            .with_context(|| format!("Could not create LAZER file {}", lazer_path.display()))?,
    );
    let mut lazer_writer = LazerWriter::new(lazer_file, las_header, encoder_builder)
        .context("Could not open LAZER writer")?;
    lazer_writer
        .write(&las_points)
        .context("Failed to write points to LAZER file")?;
    lazer_writer.flush().context("Failed to flush LAZER file")?;

    Ok(())
}

fn main() -> Result<()> {
    pretty_env_logger::init();

    let args = Args::parse();
    let input_output_file_pairs = get_all_input_output_file_pairings(&args)?;
    info!("Converting {} files", input_output_file_pairs.len());

    let mut encoder_builder = EncoderBuilder::new();
    encoder_builder.level(args.compression_level);
    info!("LZ4 parameters: {:#?}", encoder_builder);

    let num_files_processed = AtomicUsize::default();
    input_output_file_pairs
        .par_iter()
        .map(|(in_file, out_file)| {
            if !is_las_file(&in_file) {
                bail!("Found non-LAS file {}", in_file.display());
            }

            let global_file_number = num_files_processed.fetch_add(1, Ordering::SeqCst);
            info!(
                "{global_file_number:4} / {:4}",
                input_output_file_pairs.len()
            );

            las_to_lazer(&in_file, &out_file, encoder_builder.clone()).with_context(|| {
                format!(
                    "Could not convert file {} to LAZER format",
                    in_file.display()
                )
            })?;

            Ok(())
        })
        .collect::<Result<()>>()?;

    Ok(())
}
