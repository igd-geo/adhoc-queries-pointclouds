use std::{
    ffi::OsStr,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc::channel,
        Arc,
    },
    time::Duration,
};

use anyhow::{bail, Context, Result};
use clap::Parser;
use human_repr::HumanCount;
use log::info;
use pasture_core::{
    containers::{BorrowedBuffer, InterleavedBuffer, OwningBuffer, SliceBuffer, VectorBuffer},
    layout::PointLayout,
    math::AABB,
    meta::Metadata,
};
use pasture_io::{
    base::{PointReader, PointWriter},
    las::{las_point_format_from_point_layout, LASReader, LASWriter},
    las_rs::{Builder, Transform},
};
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};

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
        .map(|ex| ex == "las" || ex == "LAS" || ex == "laz" || ex == "LAZ")
        .unwrap_or_default()
}

fn get_all_input_files(args: &Args) -> Result<Vec<PathBuf>> {
    if !args.input.exists() {
        bail!("Input path {} does not exist", args.input.display());
    }

    if args.input.is_file() {
        Ok(vec![args.input.clone()])
    } else if args.input.is_dir() {
        let all_las_files_in_dir = walkdir::WalkDir::new(&args.input).into_iter().map(
            |entry| -> Result<Option<PathBuf>> {
                let entry = entry?;
                if is_las_file(entry.path()) {
                    Ok(Some(entry.path().to_owned()))
                } else {
                    Ok(None)
                }
            },
        );
        let with_empty_entries = all_las_files_in_dir.collect::<Result<Vec<_>>>()?;
        Ok(with_empty_entries.into_iter().filter_map(|e| e).collect())
    } else {
        bail!("Input path must point to a valid file or directory");
    }
}

fn merge_files(input: Vec<PathBuf>, args: &Args) -> Result<()> {
    info!("Merging {} files...", input.len());

    // Gather metadata of all files
    let mut metadata: Option<(AABB<f64>, PointLayout)> = None;
    let mut total_point_count: usize = 0;
    for input_file in input.iter() {
        let las_reader = LASReader::from_path(&input_file, false).with_context(|| {
            format!(
                "Could not open LAS reader for file {}",
                input_file.display()
            )
        })?;
        total_point_count += las_reader.remaining_points();

        if let Some(metadata) = metadata.as_mut() {
            let this_point_layout = las_reader.get_default_point_layout();
            if metadata.1 != *this_point_layout {
                bail!("Point layouts of all files must match. Expected layout {} but found layout {} in file {}", metadata.1, this_point_layout, input_file.display());
            }

            let bounds = las_reader
                .las_metadata()
                .bounds()
                .expect("No bounds in LAS file");
            metadata.0 = AABB::union(&metadata.0, &bounds);
        } else {
            let point_layout = las_reader.get_default_point_layout().clone();
            let bounds = las_reader
                .las_metadata()
                .bounds()
                .expect("No bounds in LAS file");
            metadata = Some((bounds, point_layout));
        }
    }

    let (bounds, point_layout) = metadata.expect("No files found");
    let point_layout_copy = point_layout.clone();

    let (filled_buffer_sender, filled_buffer_receiver) = channel::<(usize, VectorBuffer)>();

    let read_count = Arc::new(AtomicUsize::default());
    let read_count_clone = read_count.clone();
    let write_count = Arc::new(AtomicUsize::default());
    let write_count_clone = write_count.clone();
    let done = Arc::new(AtomicBool::default());
    let done_clone = done.clone();

    let log_thread_handle = std::thread::spawn(move || {
        while !done.load(Ordering::SeqCst) {
            eprintln!(
                "Read: {:5.3}/{total_point_count}  Write: {:5.3}",
                read_count.load(Ordering::SeqCst).human_count_bare(),
                write_count.load(Ordering::SeqCst).human_count_bare()
            );
            std::thread::sleep(Duration::from_secs(5));
        }
    });

    let output_file = args.output.clone();
    let write_thread_handle = std::thread::spawn(move || -> Result<()> {
        let mut header_builder = Builder::from((1, 4));
        header_builder.transforms.x = Transform {
            offset: bounds.center().x,
            scale: 0.001,
        };
        header_builder.transforms.y = Transform {
            offset: bounds.center().y,
            scale: 0.001,
        };
        header_builder.transforms.z = Transform {
            offset: bounds.center().z,
            scale: 0.001,
        };
        header_builder.point_format = las_point_format_from_point_layout(&point_layout_copy);
        let mut las_writer =
            LASWriter::from_path_and_header(output_file.as_path(), header_builder.into_header()?)
                .with_context(|| {
                format!(
                    "Could not open LAS writer for output file {}",
                    output_file.display()
                )
            })?;

        while let Ok((count, filled_buffer)) = filled_buffer_receiver.recv() {
            las_writer
                .write(&filled_buffer.slice(0..count))
                .context("Failed to write points")?;

            write_count_clone.fetch_add(count, Ordering::SeqCst);
        }

        las_writer.flush()?;
        done_clone.store(true, Ordering::SeqCst);
        Ok(())
    });

    input
        .par_iter()
        .map_with(filled_buffer_sender, |buffer_sender, file| -> Result<()> {
            const READ_BUFFER_CAPACITY: usize = 1_000_000;

            let mut las_reader = LASReader::from_path(&file, false)?;
            while las_reader.remaining_points() > 0 {
                let mut read_buffer =
                    VectorBuffer::with_capacity(READ_BUFFER_CAPACITY, point_layout.clone());
                read_buffer.resize(READ_BUFFER_CAPACITY);
                let num_read = las_reader.read_into(&mut read_buffer, READ_BUFFER_CAPACITY)?;
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

                    buffer_sender.send((gather_buffer.len(), gather_buffer))?;
                } else {
                    buffer_sender.send((num_read, read_buffer))?;
                }

                read_count_clone.fetch_add(num_read, Ordering::SeqCst);
            }

            Ok(())
        })
        .collect::<Result<()>>()?;

    write_thread_handle
        .join()
        .expect("Error on write thread")
        .context("Writing failed")?;
    log_thread_handle.join().expect("Error on log thread");

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
        if args.output.exists() {
            std::fs::remove_file(args.output.as_path())?;
        }
        return Err(e);
    }

    Ok(())
}
