use std::{
    ffi::OsStr,
    fs::File,
    io::BufWriter,
    path::{Path, PathBuf},
};

use anyhow::{bail, Context, Result};
use clap::Parser;
use log::info;
use pasture_core::containers::{
    BorrowedBuffer, InterleavedBuffer, OwningBuffer, SliceBuffer, VectorBuffer,
};
use pasture_io::{
    base::{PointReader, PointWriter},
    las::{LASReader, LASWriter},
    las_rs::point::Format,
};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    input: PathBuf,
    output: PathBuf,
    #[arg(
        short,
        long,
        help = "Downsampling factor, between 0 and 1. Describes ratio of points to keep in merged files to total point count"
    )]
    downsampling: Option<f64>,
}

fn is_las_file(path: &Path) -> bool {
    path.extension()
        .and_then(OsStr::to_str)
        .map(|ex| ex == "las" || ex == "LAS")
        .unwrap_or_default()
}

fn get_all_input_files(args: &Args) -> Result<Vec<PathBuf>> {
    if !args.input.exists() {
        bail!("Input path {} does not exist", args.input.display());
    }

    if args.input.is_file() {
        Ok(vec![args.input.clone()])
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
        Ok(all_las_files_in_dir.collect())
    } else {
        bail!("Input path must point to a valid file or directory");
    }
}

fn merge_files(input: Vec<PathBuf>, args: &Args) -> Result<()> {
    info!("Merging {} files...", input.len());

    let mut read_buffer: Option<VectorBuffer> = None;
    let mut point_format: Option<Format> = None;
    let mut writer: Option<LASWriter<BufWriter<File>>> = None;
    for (file_number, input_file) in input.iter().enumerate() {
        let mut las_reader = LASReader::from_path(&input_file, false).with_context(|| {
            format!(
                "Could not open LAS reader for file {}",
                input_file.display()
            )
        })?;

        if let Some(format) = point_format.as_ref() {
            let this_file_format = las_reader.las_metadata().point_format();
            if *format != this_file_format {
                bail!("Point formats of all files must match. Expected format {} but found format {} in file {}", format, this_file_format, input_file.display());
            }
        } else {
            point_format = Some(las_reader.las_metadata().point_format());
            let las_writer = LASWriter::from_path_and_point_layout(
                args.output.as_path(),
                las_reader.get_default_point_layout(),
            )
            .with_context(|| {
                format!(
                    "Could not open LAS writer for output file {}",
                    args.output.display()
                )
            })?;
            writer = Some(las_writer);
        }

        let read_buffer = if let Some(buf) = read_buffer.as_mut() {
            buf
        } else {
            let mut buffer =
                VectorBuffer::with_capacity(50_000, las_reader.get_default_point_layout().clone());
            buffer.resize(50_000);
            read_buffer = Some(buffer);
            read_buffer.as_mut().unwrap()
        };

        let writer = writer.as_mut().unwrap();

        while las_reader.remaining_points() > 0 {
            let num_read = las_reader.read_into(read_buffer, read_buffer.len())?;
            if let Some(downsampling_factor) = args.downsampling {
                let step_size = (1.0 / downsampling_factor) as usize;
                let steps = num_read / step_size;
                let mut gather_buffer =
                    VectorBuffer::with_capacity(steps, read_buffer.point_layout().clone());
                for idx in (0..num_read).step_by(step_size) {
                    // Safe because both buffers have the same PointLayout
                    unsafe {
                        gather_buffer.push_points(read_buffer.get_point_ref(idx));
                    }
                }

                writer
                    .write(&gather_buffer)
                    .context("Failed to write points")?;
            } else {
                writer
                    .write(&read_buffer.slice(0..num_read))
                    .context("Failed to write points")?;
            }
        }

        info!("{:3} / {:3}", file_number, input.len());
    }

    if let Some(mut writer) = writer {
        writer.flush().context("Failed to flush writer")?;
    }

    Ok(())
}

fn main() -> Result<()> {
    pretty_env_logger::init();

    let args = Args::parse();
    let input_files = get_all_input_files(&args).context("Failed to get input files")?;

    if input_files.is_empty() {
        bail!("No input files found");
    }

    if let Err(e) = merge_files(input_files, &args) {
        std::fs::remove_file(args.output.as_path())?;
        return Err(e);
    }

    Ok(())
}
