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
use io::last::LASTReader;
use pasture_core::{
    containers::{BorrowedBuffer, HashMapBuffer, OwningBuffer, VectorBuffer},
    layout::{attributes::CLASSIFICATION, PointLayout},
};
use pasture_io::{
    base::PointReader,
    las::{point_layout_from_las_metadata, LASReader, ATTRIBUTE_LOCAL_LAS_POSITION},
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

fn reset_page_cache() -> Result<()> {
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

/// Read data in the default PointLayout from a LAS file. Default means positions as f64 in world-space and unpacked
/// bit attributes!
fn benchmark_las_default_layout(file: &Path, mmapped: bool) -> Result<Stats> {
    let file_size = file
        .metadata()
        .context("Could not get file metadata")?
        .len();
    let is_compressed =
        file.extension().map(|ex| ex.to_ascii_lowercase()) == Some("laz".to_string().into());

    let timer = Instant::now();
    let points = if mmapped {
        let file = File::open(file).context("failed to open file")?;
        let mmap = unsafe { memmap::Mmap::map(&file)? };
        let mut reader = LASReader::from_read(Cursor::new(&mmap[..]), is_compressed, false)
            .context("failed to open LAS reader")?;
        // Read into HashMapBuffer instead of VectorBuffer because this is faster with layout conversion, and it is the
        // same thing we do in the `reader_performance` experiment!
        reader
            .read::<HashMapBuffer>(reader.remaining_points())
            .context("Failed to read points")?
    } else {
        let file = File::open(file).context("failed to open file")?;
        let mut reader = LASReader::from_read(BufReader::new(file), is_compressed, false)
            .context("failed to open LAS reader")?;
        reader
            .read::<HashMapBuffer>(reader.remaining_points())
            .context("Failed to read points")?
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

fn benchmark_las_pasture_custom_layout(
    file: &Path,
    mmapped: bool,
    custom_layout: &PointLayout,
) -> Result<Stats> {
    let file_size = file
        .metadata()
        .context("Could not get file metadata")?
        .len();
    let is_compressed =
        file.extension().map(|ex| ex.to_ascii_lowercase()) == Some("laz".to_string().into());

    let timer = Instant::now();
    let points = if mmapped {
        let file = File::open(file).context("failed to open file")?;
        let mmap = unsafe { memmap::Mmap::map(&file)? };
        let mut reader = LASReader::from_read(Cursor::new(&mmap[..]), is_compressed, true)
            .context("failed to open LAS reader")?;
        let num_points = reader.remaining_points();
        // Here we read into a VectorBuffer instead of a HashMapBuffer because this will be faster if we don't need
        // layout conversions, which (contrary to the name of the method) is what is going on here: We read in the default
        // memory layout, or a subset of it. Technically, for the subsets a HashMapBuffer _might_ be faster, but we want
        // to stay consistent. Also, the current custom layouts only include 1 attribute, so it makes no difference
        let mut points = VectorBuffer::with_capacity(num_points, custom_layout.clone());
        points.resize(num_points);
        reader
            .read_into(&mut points, num_points)
            .context("Failed to read points")?;
        points
    } else {
        let file = File::open(file).context("failed to open file")?;
        let mut reader = LASReader::from_read(BufReader::new(file), is_compressed, true)
            .context("failed to open LAS reader")?;
        let num_points = reader.remaining_points();
        let mut points = VectorBuffer::with_capacity(num_points, custom_layout.clone());
        points.resize(num_points);
        reader
            .read_into(&mut points, num_points)
            .context("Failed to read points")?;
        points
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

fn benchmark_last_pasture_default_layout(file: &Path, mmapped: bool) -> Result<Stats> {
    let file_size = file
        .metadata()
        .context("Could not get file metadata")?
        .len();

    let timer = Instant::now();
    let points = if mmapped {
        let file = File::open(file).context("failed to open file")?;
        let mmap = unsafe { memmap::Mmap::map(&file)? };
        let mut reader =
            LASTReader::from_read(Cursor::new(&mmap[..])).context("failed to open LAST reader")?;
        let layout = point_layout_from_las_metadata(reader.las_metadata(), false)?;
        let num_points = reader.remaining_points();
        let mut points = HashMapBuffer::with_capacity(num_points, layout);
        points.resize(num_points);
        reader
            .read_into(&mut points, num_points)
            .context("Failed to read points")?;
        points
    } else {
        let file = File::open(file).context("failed to open file")?;
        let mut reader =
            LASTReader::from_read(BufReader::new(file)).context("failed to open LAST reader")?;
        let layout = point_layout_from_las_metadata(reader.las_metadata(), false)?;
        let num_points = reader.remaining_points();
        let mut points = HashMapBuffer::with_capacity(num_points, layout);
        points.resize(num_points);
        reader
            .read_into(&mut points, num_points)
            .context("Failed to read points")?;
        points
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

fn benchmark_last_pasture_custom_layout(
    file: &Path,
    mmapped: bool,
    custom_layout: &PointLayout,
) -> Result<Stats> {
    let timer = Instant::now();
    let points = if mmapped {
        let file = File::open(file).context("failed to open file")?;
        let mmap = unsafe { memmap::Mmap::map(&file)? };
        let mut reader =
            LASTReader::from_read(Cursor::new(&mmap[..])).context("failed to open LAST reader")?;
        let num_points = reader.remaining_points();
        let mut points = HashMapBuffer::with_capacity(num_points, custom_layout.clone());
        points.resize(num_points);
        reader
            .read_into(&mut points, num_points)
            .context("Failed to read points")?;
        points
    } else {
        let file = File::open(file).context("failed to open file")?;
        let mut reader =
            LASTReader::from_read(BufReader::new(file)).context("failed to open LAST reader")?;
        let num_points = reader.remaining_points();
        let mut points = HashMapBuffer::with_capacity(num_points, custom_layout.clone());
        points.resize(num_points);
        reader
            .read_into(&mut points, num_points)
            .context("Failed to read points")?;
        points
    };
    let runtime = timer.elapsed();

    // In contrast to LAS, the throughput for LAST can be measured from the size of the PointLayout
    // (since all tests within this experiment work without any format conversion, i.e. they only
    // read a subset of the LAST file)
    // Calculating throughput from file size would be wrong if we for example only read classifications
    let bytes_per_second = (points.len() as f64
        * points.point_layout().size_of_point_entry() as f64)
        / runtime.as_secs_f64();
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
        experiment.run(|context| -> Result<()> {
            if purge_cache {
                reset_page_cache().context("Failed to reset page cache")?;
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
        })?;
    }

    Ok(())
}

fn main() -> Result<()> {
    dotenv::dotenv().context("Failed to initialize with .env file")?;
    pretty_env_logger::init();
    let args = Args::parse();

    let extension = args
        .input_file
        .extension()
        .map(OsStr::to_string_lossy)
        .context("Could not get file extension")?;

    let machine = std::env::var("MACHINE").context("To run experiments, please set the 'MACHINE' environment variable to the name of the machine that you are running this experiment on. This is required so that experiment data can be mapped to the actual machine that ran the experiment. This will typically be the name or system configuration of the computer that runs the experiment.")?;

    let experiment_description = include_str!("yaml/optimized_reader_performance.yaml");
    let experiment = ExperimentVersion::from_yaml_str(experiment_description)
        .context("Could not get experiment version")?;

    let mmapped = [false, true];

    let matching_memory_layout = {
        let reader = LASReader::from_path(&args.input_file, true)?;
        reader.get_default_point_layout().clone()
    };
    let only_positions_layout = [ATTRIBUTE_LOCAL_LAS_POSITION]
        .into_iter()
        .collect::<PointLayout>();
    let only_classifications_layout = [CLASSIFICATION].into_iter().collect::<PointLayout>();

    let custom_layouts = [
        ("All (native)", matching_memory_layout),
        ("Positions", only_positions_layout),
        ("Classifications", only_classifications_layout),
    ];

    let dataset = args.input_file.display().to_string();

    match extension.as_ref() {
        "las" => {
            for mmap in &mmapped {
                run_experiment(
                    &args.input_file,
                    |path| benchmark_las_default_layout(path, *mmap),
                    experiment.make_instance([
                        ("Dataset", GenericValue::String(dataset.clone())),
                        ("Machine", GenericValue::String(machine.clone())),
                        ("Purge cache", GenericValue::Bool(args.purge_cache)),
                        ("mmap", GenericValue::Bool(*mmap)),
                        (
                            "Point layout",
                            GenericValue::String("All (default)".to_string()),
                        ),
                    ])?,
                    args.purge_cache,
                )
                .context("Benchmarking pasture failed")?;

                for (label, layout) in &custom_layouts {
                    run_experiment(
                        &args.input_file,
                        |path| benchmark_las_pasture_custom_layout(path, *mmap, layout),
                        experiment.make_instance([
                            ("Dataset", GenericValue::String(dataset.clone())),
                            ("Machine", GenericValue::String(machine.clone())),
                            ("Purge cache", GenericValue::Bool(args.purge_cache)),
                            ("mmap", GenericValue::Bool(*mmap)),
                            ("Point layout", GenericValue::String(label.to_string())),
                        ])?,
                        args.purge_cache,
                    )
                    .context("Benchmarking pasture failed")?;
                }
            }
        }
        "last" => {
            for mmap in &mmapped {
                run_experiment(
                    &args.input_file,
                    |path| benchmark_last_pasture_default_layout(path, *mmap),
                    experiment.make_instance([
                        ("Dataset", GenericValue::String(dataset.clone())),
                        ("Machine", GenericValue::String(machine.clone())),
                        ("Purge cache", GenericValue::Bool(args.purge_cache)),
                        ("mmap", GenericValue::Bool(*mmap)),
                        (
                            "Point layout",
                            GenericValue::String("All (default)".to_string()),
                        ),
                    ])?,
                    args.purge_cache,
                )
                .context("Benchmarking pasture failed")?;

                for (label, layout) in &custom_layouts {
                    run_experiment(
                        &args.input_file,
                        |path| benchmark_last_pasture_custom_layout(path, *mmap, layout),
                        experiment.make_instance([
                            ("Dataset", GenericValue::String(dataset.clone())),
                            ("Machine", GenericValue::String(machine.clone())),
                            ("Purge cache", GenericValue::Bool(args.purge_cache)),
                            ("mmap", GenericValue::Bool(*mmap)),
                            ("Point layout", GenericValue::String(label.to_string())),
                        ])?,
                        args.purge_cache,
                    )
                    .context("Benchmarking pasture failed")?;
                }
            }
        }
        _ => bail!("Invalid file extension {extension}"),
    }

    Ok(())
}
