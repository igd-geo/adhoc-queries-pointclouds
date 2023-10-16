use std::{
    ffi::OsStr,
    fs::{File, OpenOptions},
    io::{BufReader, Cursor, Write},
    path::{Path, PathBuf},
    process::Command,
    time::{Duration, Instant},
};

use anyhow::{bail, Context, Result};
use clap::Parser;
use exar::{
    experiment::{ExperimentInstance, ExperimentVersion},
    variable::GenericValue,
};
use io::{last::LASTReader, lazer::LazerReader};
use pasture_core::containers::{BorrowedBuffer, HashMapBuffer, MakeBufferFromLayout, OwningBuffer};
use pasture_io::{
    base::{read_all, PointReader},
    las::{point_layout_from_las_metadata, LASReader},
    las_rs::{Point, Read},
};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    input_file: PathBuf,
    #[arg(long, short, default_value_t = false)]
    purge_cache: bool,
}

#[derive(Clone)]
struct Stats {
    runtime: Duration,
    number_of_points: usize,
    points_per_second: f64,
    bytes_per_second: f64,
}

fn flush_disk_cache() -> Result<()> {
    let sync_output = Command::new("sync")
        .output()
        .context("Could not execute sync command")?;
    if !sync_output.status.success() {
        bail!("Sync command failed with exit code {}", sync_output.status);
    }

    if std::env::consts::OS == "macos" {
        let purge_output = Command::new("purge")
            .output()
            .context("Could not execute purge command")?;
        if !purge_output.status.success() {
            bail!(
                "Purge command failed with exit code {}",
                purge_output.status
            );
        }
    } else if std::env::consts::OS == "linux" {
        let mut drop_caches = OpenOptions::new()
            .write(true)
            .open("/proc/sys/vm/drop_caches")?;
        drop_caches.write_all("3".as_bytes())?;
    }

    Ok(())
}

fn benchmark_las_pasture(file: &Path, mmapped: bool) -> Result<Stats> {
    let file_size = file
        .metadata()
        .context("Could not get file metadata")?
        .len();
    let is_compressed =
        file.extension().map(|ex| ex.to_ascii_lowercase()) == Some("laz".to_string().into());

    let timer = Instant::now();
    let points = if mmapped {
        let _span = tracy_client::span!("pasture_read_mmap");
        let file = File::open(file).context("failed to open file")?;
        let mmap = unsafe { memmap::Mmap::map(&file)? };
        let mut reader = LASReader::from_read(Cursor::new(&mmap[..]), is_compressed, false)
            .context("failed to open LAS reader")?;
        reader
            .read::<HashMapBuffer>(reader.remaining_points())
            .context("failed to read points")?
    } else {
        let _span = tracy_client::span!("pasture_read_file");
        read_all::<HashMapBuffer, _>(file).context("failed to read points")?
    };
    let runtime = timer.elapsed();

    let bytes_per_second = file_size as f64 / runtime.as_secs_f64();
    let points_per_second = points.len() as f64 / runtime.as_secs_f64();

    Ok(Stats {
        bytes_per_second,
        number_of_points: points.len(),
        points_per_second,
        runtime,
    })
}

fn benchmark_las_lasrs(file: &Path, mmapped: bool) -> Result<Stats> {
    let file_size = file
        .metadata()
        .context("Could not get file metadata")?
        .len();
    let timer = Instant::now();

    let points = if mmapped {
        let file = File::open(file).context("failed to open file")?;
        let mmap = unsafe { memmap::Mmap::map(&file)? };
        let mut reader = pasture_io::las_rs::Reader::new(Cursor::new(&mmap[..]))
            .context("failed to open LAS reader")?;
        reader
            .points()
            .map(|p| -> Result<Point> { Ok(p?) })
            .collect::<Result<Vec<_>>>()
    } else {
        let mut reader =
            pasture_io::las_rs::Reader::from_path(file).context("failed to open LAS reader")?;
        reader
            .points()
            .map(|p| -> Result<Point> { Ok(p?) })
            .collect::<Result<Vec<_>>>()
    }?;

    let runtime = timer.elapsed();

    let bytes_per_second = file_size as f64 / runtime.as_secs_f64();
    let points_per_second = points.len() as f64 / runtime.as_secs_f64();

    Ok(Stats {
        bytes_per_second,
        number_of_points: points.len(),
        points_per_second,
        runtime,
    })
}

fn benchmark_las_pdal(file: &Path) -> Result<Stats> {
    let file_size = file
        .metadata()
        .context("Could not get file metadata")?
        .len();
    let number_of_points = LASReader::from_path(file, false)?.remaining_points();

    let pipeline = format!(
        "[{{\"type\":\"readers.las\", \"filename\":\"{}\"}},{{\"type\":\"writers.null\"}}]",
        file.display()
    );
    std::fs::write("pipeline.json", pipeline)?;

    let timer = Instant::now();
    let mut pdal = Command::new("pdal")
        .arg("pipeline")
        .arg("-i")
        .arg("./pipeline.json")
        .spawn()?;

    let pdal_status = pdal.wait()?;
    let mut runtime = timer.elapsed();

    if !pdal_status.success() {
        let mut pdal_stderr = pdal.stderr.expect("could not get stderr stream for PDAL");
        std::io::copy(&mut pdal_stderr, &mut std::io::stderr())?;
        bail!("PDAL failed");
    }

    // Try to estimate the overhead of running the PDAL executable and subtract that from the total runtime
    // A more fair test would use a custom executable that links with the PDAL library and profile time within
    // that executable, but there are no working PDAL bindings for Rust, so we use this way...

    let pdal_executable_baseline_duration = {
        let timer = Instant::now();
        Command::new("pdal").arg("pipeline").output()?;
        timer.elapsed()
    };

    runtime = runtime.saturating_sub(pdal_executable_baseline_duration);

    let bytes_per_second = file_size as f64 / runtime.as_secs_f64();
    let points_per_second = number_of_points as f64 / runtime.as_secs_f64();

    std::fs::remove_file("pipeline.json")?;

    Ok(Stats {
        bytes_per_second,
        number_of_points,
        points_per_second,
        runtime,
    })
}

fn benchmark_last_pasture(file: &Path, mmapped: bool) -> Result<Stats> {
    let file_size = file
        .metadata()
        .context("Could not get file metadata")?
        .len();

    let timer = Instant::now();

    let points = if mmapped {
        let file = File::open(file).context("failed to open file")?;
        let mmap = unsafe { memmap::Mmap::map(&file)? };
        let mut last_reader =
            LASTReader::from_read(Cursor::new(&mmap[..])).context("failed to open LAST reader")?;

        // Read the point data in a layout that has world-space positions and extracted flags attributes to be comparable
        // to the LAS/LAZ reading. Otherwise it would be unfair, because the LAST and LAZER readers by default read the
        // points in a PointLayout that matches the binary layout of the file, whereas LAS/LAZ readers parse positions and
        // flag attributes

        let point_layout_with_high_level_attributes =
            point_layout_from_las_metadata(last_reader.las_metadata(), false)
                .context("Could not get matching PointLayout")?;
        let mut buffer = HashMapBuffer::new_from_layout(point_layout_with_high_level_attributes);
        buffer.resize(last_reader.remaining_points());
        last_reader
            .read_into(&mut buffer, last_reader.remaining_points())
            .context("failed to read points")?;
        buffer
    } else {
        let file = File::open(file).context("failed to open file")?;
        let mut last_reader =
            LASTReader::from_read(BufReader::new(file)).context("failed to open LAST reader")?;
        let point_layout_with_high_level_attributes =
            point_layout_from_las_metadata(last_reader.las_metadata(), false)
                .context("Could not get matching PointLayout")?;
        let mut buffer = HashMapBuffer::new_from_layout(point_layout_with_high_level_attributes);
        buffer.resize(last_reader.remaining_points());
        last_reader
            .read_into(&mut buffer, last_reader.remaining_points())
            .context("failed to read points")?;
        buffer
    };

    let runtime = timer.elapsed();

    let bytes_per_second = file_size as f64 / runtime.as_secs_f64();
    let points_per_second = points.len() as f64 / runtime.as_secs_f64();

    Ok(Stats {
        bytes_per_second,
        number_of_points: points.len(),
        points_per_second,
        runtime,
    })
}

fn benchmark_lazer_pasture(file: &Path, mmapped: bool) -> Result<Stats> {
    let file_size = file
        .metadata()
        .context("Could not get file metadata")?
        .len();

    let timer = Instant::now();

    let points = if mmapped {
        let file = File::open(file).context("failed to open file")?;
        let mmap = unsafe { memmap::Mmap::map(&file)? };
        let mut lazer_reader =
            LazerReader::new(Cursor::new(&mmap[..])).context("failed to open LAZER reader")?;
        let point_layout_with_high_level_attributes =
            point_layout_from_las_metadata(lazer_reader.las_metadata(), false)
                .context("Could not get matching PointLayout")?;
        let mut buffer = HashMapBuffer::new_from_layout(point_layout_with_high_level_attributes);
        buffer.resize(lazer_reader.remaining_points());
        lazer_reader
            .read_into(&mut buffer, lazer_reader.remaining_points())
            .context("failed to read points")?;
        buffer
    } else {
        let file = File::open(file).context("failed to open file")?;
        let mut lazer_reader =
            LazerReader::new(BufReader::new(file)).context("failed to open LAZER reader")?;
        let point_layout_with_high_level_attributes =
            point_layout_from_las_metadata(lazer_reader.las_metadata(), false)
                .context("Could not get matching PointLayout")?;
        let mut buffer = HashMapBuffer::new_from_layout(point_layout_with_high_level_attributes);
        buffer.resize(lazer_reader.remaining_points());
        lazer_reader
            .read_into(&mut buffer, lazer_reader.remaining_points())
            .context("failed to read points")?;
        buffer
    };

    let runtime = timer.elapsed();

    let bytes_per_second = file_size as f64 / runtime.as_secs_f64();
    let points_per_second = points.len() as f64 / runtime.as_secs_f64();

    Ok(Stats {
        bytes_per_second,
        number_of_points: points.len(),
        points_per_second,
        runtime,
    })
}

fn run_experiment(
    file: &Path,
    runner: impl Fn(&Path) -> Result<Stats>,
    experiment: ExperimentInstance<'_>,
    purge_cache: bool,
) -> Result<()> {
    const RUNS: usize = 10;

    for _ in 0..RUNS {
        experiment
            .run(|context| -> Result<()> {
                if purge_cache {
                    flush_disk_cache().context("Failed to reset page cache")?;
                }
                let run_stats = runner(file).context("Failed to run experiment")?;

                context.add_measurement(
                    "Runtime",
                    GenericValue::Numeric(run_stats.runtime.as_secs_f64()),
                );
                context.add_measurement(
                    "Point throughput",
                    GenericValue::Numeric(run_stats.points_per_second),
                );
                context.add_measurement(
                    "Memory throughput",
                    GenericValue::Numeric(run_stats.bytes_per_second),
                );
                context.add_measurement(
                    "Point count",
                    GenericValue::Numeric(run_stats.number_of_points as f64),
                );
                Ok(())
            })
            .context("Experiment run failed")?;
    }

    Ok(())
}

fn main() -> Result<()> {
    dotenv::dotenv().context("Failed to initialize with .env file")?;
    pretty_env_logger::init();
    let _client = tracy_client::Client::start();
    let args = Args::parse();

    let extension = args
        .input_file
        .extension()
        .map(OsStr::to_string_lossy)
        .context("Could not get file extension")?;

    let machine = std::env::var("MACHINE").context("To run experiments, please set the 'MACHINE' environment variable to the name of the machine that you are running this experiment on. This is required so that experiment data can be mapped to the actual machine that ran the experiment. This will typically be the name or system configuration of the computer that runs the experiment.")?;

    let experiment_description = include_str!("yaml/reader_performance.yaml");
    let experiment = ExperimentVersion::from_yaml_str(experiment_description)
        .context("Could not get experiment version")?;

    let dataset = args.input_file.display().to_string();

    match extension.as_ref() {
        "las" | "laz" => {
            run_experiment(
                &args.input_file,
                |path| benchmark_las_pasture(path, false),
                experiment.make_instance([
                    ("Dataset", GenericValue::String(dataset.clone())),
                    ("Machine", GenericValue::String(machine.clone())),
                    ("Tool", GenericValue::String("pasture (read)".to_string())),
                    ("Purge cache", GenericValue::Bool(args.purge_cache)),
                ])?,
                args.purge_cache,
            )
            .context("Benchmarking pasture failed")?;
            run_experiment(
                &args.input_file,
                |path| benchmark_las_pasture(path, true),
                experiment.make_instance([
                    ("Dataset", GenericValue::String(dataset.clone())),
                    ("Machine", GenericValue::String(machine.clone())),
                    ("Tool", GenericValue::String("pasture (mmap)".to_string())),
                    ("Purge cache", GenericValue::Bool(args.purge_cache)),
                ])?,
                args.purge_cache,
            )
            .context("Benchmarking pasture failed")?;
            run_experiment(
                &args.input_file,
                |path| benchmark_las_lasrs(path, false),
                experiment.make_instance([
                    ("Dataset", GenericValue::String(dataset.clone())),
                    ("Machine", GenericValue::String(machine.clone())),
                    ("Tool", GenericValue::String("las-rs (read)".to_string())),
                    ("Purge cache", GenericValue::Bool(args.purge_cache)),
                ])?,
                args.purge_cache,
            )
            .context("Benchmarking las-rs failed")?;
            run_experiment(
                &args.input_file,
                |path| benchmark_las_lasrs(path, true),
                experiment.make_instance([
                    ("Dataset", GenericValue::String(dataset.clone())),
                    ("Machine", GenericValue::String(machine.clone())),
                    ("Tool", GenericValue::String("las-rs (mmap)".to_string())),
                    ("Purge cache", GenericValue::Bool(args.purge_cache)),
                ])?,
                args.purge_cache,
            )
            .context("Benchmarking las-rs failed")?;
            run_experiment(
                &args.input_file,
                benchmark_las_pdal,
                experiment.make_instance([
                    ("Dataset", GenericValue::String(dataset.clone())),
                    ("Machine", GenericValue::String(machine.clone())),
                    ("Tool", GenericValue::String("PDAL".to_string())),
                    ("Purge cache", GenericValue::Bool(args.purge_cache)),
                ])?,
                args.purge_cache,
            )
            .context("Benchmarking PDAL failed")?;
        }
        "last" => {
            run_experiment(
                &args.input_file,
                |path| benchmark_last_pasture(path, false),
                experiment.make_instance([
                    ("Dataset", GenericValue::String(dataset.clone())),
                    ("Machine", GenericValue::String(machine.clone())),
                    ("Tool", GenericValue::String("pasture (read)".to_string())),
                    ("Purge cache", GenericValue::Bool(args.purge_cache)),
                ])?,
                args.purge_cache,
            )
            .context("Benchmarking pasture failed")?;
            run_experiment(
                &args.input_file,
                |path| benchmark_last_pasture(path, true),
                experiment.make_instance([
                    ("Dataset", GenericValue::String(dataset.clone())),
                    ("Machine", GenericValue::String(machine.clone())),
                    ("Tool", GenericValue::String("pasture (mmap)".to_string())),
                    ("Purge cache", GenericValue::Bool(args.purge_cache)),
                ])?,
                args.purge_cache,
            )
            .context("Benchmarking pasture failed")?;
        }
        "lazer" => {
            run_experiment(
                &args.input_file,
                |path| benchmark_lazer_pasture(path, false),
                experiment.make_instance([
                    ("Dataset", GenericValue::String(dataset.clone())),
                    ("Machine", GenericValue::String(machine.clone())),
                    ("Tool", GenericValue::String("pasture (read)".to_string())),
                    ("Purge cache", GenericValue::Bool(args.purge_cache)),
                ])?,
                args.purge_cache,
            )
            .context("Benchmarking pasture failed")?;
            run_experiment(
                &args.input_file,
                |path| benchmark_lazer_pasture(path, true),
                experiment.make_instance([
                    ("Dataset", GenericValue::String(dataset.clone())),
                    ("Machine", GenericValue::String(machine.clone())),
                    ("Tool", GenericValue::String("pasture (mmap)".to_string())),
                    ("Purge cache", GenericValue::Bool(args.purge_cache)),
                ])?,
                args.purge_cache,
            )
            .context("Benchmarking pasture failed")?;
        }
        _ => bail!("Invalid file extension {extension}"),
    }

    Ok(())
}
