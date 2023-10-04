use std::{
    borrow::Cow,
    collections::HashSet,
    ffi::OsStr,
    fs::File,
    io::{BufReader, Cursor},
    path::{Path, PathBuf},
    process::Command,
    time::{Duration, Instant},
};

use anyhow::{bail, Context, Result};
use clap::Parser;
use experiment_archiver::{Experiment, VariableTemplate};
use io::{last::LASTReader, lazer::LazerReader};
use pasture_core::containers::{
    BorrowedBuffer, HashMapBuffer, MakeBufferFromLayout, OwningBuffer, VectorBuffer,
};
use pasture_io::{
    base::{read_all, PointReader},
    las::{point_layout_from_las_metadata, LASReader},
    las_rs::{Point, Read},
};
use statrs::statistics::{Data, Distribution};

const VARIABLE_DATASET: VariableTemplate = VariableTemplate::new(
    Cow::Borrowed("Dataset"),
    Cow::Borrowed("The dataset used in the experiment"),
    Cow::Borrowed("none"),
);
const VARIABLE_TOOL: VariableTemplate = VariableTemplate::new(
    Cow::Borrowed("Tool / crate"),
    Cow::Borrowed("For which tool/crate is this measurement?"),
    Cow::Borrowed("text"),
);
const VARIABLE_RUNTIME: VariableTemplate = VariableTemplate::new(
    Cow::Borrowed("Runtime"),
    Cow::Borrowed("The runtime of the tool/crate"),
    Cow::Borrowed("ms"),
);
const VARIABLE_RUNTIME_ERR: VariableTemplate = VariableTemplate::new(
    Cow::Borrowed("Runtime error (1 sigma)"),
    Cow::Borrowed("The standard deviation of the measured runtime"),
    Cow::Borrowed("ms"),
);
const VARIABLE_NR_POINTS: VariableTemplate = VariableTemplate::new(
    Cow::Borrowed("Number of points"),
    Cow::Borrowed("The number of points read from the dataset"),
    Cow::Borrowed("number"),
);
const VARIABLE_POINT_THROUGHPUT: VariableTemplate = VariableTemplate::new(
    Cow::Borrowed("Throughput (points)"),
    Cow::Borrowed("The number of points read per second"),
    Cow::Borrowed("points/s"),
);
const VARIABLE_MEMORY_THROUGHPUT: VariableTemplate = VariableTemplate::new(
    Cow::Borrowed("Throughput (memory)"),
    Cow::Borrowed("The memory throughput, i.e. how many bytes were read per second"),
    Cow::Borrowed("bytes/s"),
);
const VARIABLE_PURGE_CACHE: VariableTemplate = VariableTemplate::new(
    Cow::Borrowed("Purge cache before run?"),
    Cow::Borrowed("Is the disk cache being purged before each run of the experiment?"),
    Cow::Borrowed("bool"),
);
const VARIABLE_MACHINE: VariableTemplate = VariableTemplate::new(
    Cow::Borrowed("Machine"),
    Cow::Borrowed("The machine that the experiment is run on"),
    Cow::Borrowed("text"),
);

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
    tool: &'static str,
    dataset: String,
}

fn reset_page_cache() -> Result<()> {
    let sync_output = Command::new("sync")
        .output()
        .context("Could not execute sync command")?;
    if !sync_output.status.success() {
        bail!("Sync command failed with exit code {}", sync_output.status);
    }

    let purge_output = Command::new("purge")
        .output()
        .context("Could not execute purge command")?;
    if !purge_output.status.success() {
        bail!(
            "Purge command failed with exit code {}",
            purge_output.status
        );
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
            .read::<VectorBuffer>(reader.remaining_points())
            .context("failed to read points")?
    } else {
        let _span = tracy_client::span!("pasture_read_file");
        read_all::<VectorBuffer, _>(file).context("failed to read points")?
    };
    let runtime = timer.elapsed();

    let bytes_per_second = file_size as f64 / runtime.as_secs_f64();
    let points_per_second = points.len() as f64 / runtime.as_secs_f64();

    let tool = if mmapped {
        "pasture (mmap)"
    } else {
        "pasture (file)"
    };

    Ok(Stats {
        bytes_per_second,
        number_of_points: points.len(),
        points_per_second,
        runtime,
        tool: tool,
        dataset: file.display().to_string(),
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

    let tool = if mmapped {
        "las-rs (mmap)"
    } else {
        "las-rs (file)"
    };

    Ok(Stats {
        bytes_per_second,
        number_of_points: points.len(),
        points_per_second,
        runtime,
        tool,
        dataset: file.display().to_string(),
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
        tool: "PDAL",
        dataset: file.display().to_string(),
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

    let tool = if mmapped {
        "pasture (mmap)"
    } else {
        "pasture (file)"
    };

    Ok(Stats {
        bytes_per_second,
        number_of_points: points.len(),
        points_per_second,
        runtime,
        tool: tool,
        dataset: file.display().to_string(),
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

    let tool = if mmapped {
        "pasture (mmap)"
    } else {
        "pasture (file)"
    };

    Ok(Stats {
        bytes_per_second,
        number_of_points: points.len(),
        points_per_second,
        runtime,
        tool: tool,
        dataset: file.display().to_string(),
    })
}

fn run_experiment(
    file: &Path,
    runner: impl Fn(&Path) -> Result<Stats>,
    experiment: &mut Experiment,
    machine: &str,
    purge_cache: bool,
) -> Result<()> {
    const RUNS: usize = 10;

    let stats_per_run = (0..RUNS)
        .map(|_| -> Result<Stats> {
            if purge_cache {
                reset_page_cache().context("Failed to reset page cache")?;
            }
            runner(file).context("Failed to run experiment")
        })
        .collect::<Result<Vec<_>>>()?;

    let runtimes_secs = Data::new(
        stats_per_run
            .iter()
            .map(|stat| stat.runtime.as_secs_f64())
            .collect::<Vec<_>>(),
    );
    let mean_runtime_ms = runtimes_secs
        .mean()
        .expect("Could not calculate mean runtime")
        * 1000.0;
    let stddev_runtime_ms = runtimes_secs
        .std_dev()
        .expect("Could not calculate runtime standard deviation")
        * 1000.0;

    let stats = stats_per_run[0].clone();

    experiment.run(|context| {
        context.add_value_by_name(VARIABLE_DATASET.name(), stats.dataset);
        context.add_value_by_name(VARIABLE_MACHINE.name(), machine);
        context.add_value_by_name(VARIABLE_MEMORY_THROUGHPUT.name(), stats.bytes_per_second);
        context.add_value_by_name(VARIABLE_NR_POINTS.name(), stats.number_of_points);
        context.add_value_by_name(VARIABLE_POINT_THROUGHPUT.name(), stats.points_per_second);
        context.add_value_by_name(VARIABLE_RUNTIME.name(), format!("{mean_runtime_ms:2}"));
        context.add_value_by_name(
            VARIABLE_RUNTIME_ERR.name(),
            format!("{stddev_runtime_ms:2}"),
        );
        context.add_value_by_name(VARIABLE_PURGE_CACHE.name(), purge_cache);
        context.add_value_by_name(VARIABLE_TOOL.name(), stats.tool);

        Ok(())
    })?;
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

    let variables = [
        VARIABLE_DATASET.clone(),
        VARIABLE_TOOL.clone(),
        VARIABLE_RUNTIME.clone(),
        VARIABLE_RUNTIME_ERR.clone(),
        VARIABLE_NR_POINTS.clone(),
        VARIABLE_POINT_THROUGHPUT.clone(),
        VARIABLE_MEMORY_THROUGHPUT.clone(),
        VARIABLE_PURGE_CACHE.clone(),
        VARIABLE_MACHINE.clone(),
    ]
    .into_iter()
    .collect::<HashSet<VariableTemplate>>();

    let mut experiment = Experiment::new(
        "Pointcloud read performance".into(),
        "Measures the performance of reading point clouds using various libraries and tools".into(),
        "Pascal Bormann".into(),
        variables,
    )
    .context("Failed to setup experiment")?;
    experiment.set_autolog_runs(true);

    match extension.as_ref() {
        "las" | "laz" => {
            run_experiment(
                &args.input_file,
                |path| benchmark_las_pasture(path, false),
                &mut experiment,
                &machine,
                args.purge_cache,
            )
            .context("Benchmarking pasture failed")?;
            run_experiment(
                &args.input_file,
                |path| benchmark_las_pasture(path, true),
                &mut experiment,
                &machine,
                args.purge_cache,
            )
            .context("Benchmarking pasture failed")?;
            run_experiment(
                &args.input_file,
                |path| benchmark_las_lasrs(path, false),
                &mut experiment,
                &machine,
                args.purge_cache,
            )
            .context("Benchmarking las-rs failed")?;
            run_experiment(
                &args.input_file,
                |path| benchmark_las_lasrs(path, true),
                &mut experiment,
                &machine,
                args.purge_cache,
            )
            .context("Benchmarking las-rs failed")?;
            run_experiment(
                &args.input_file,
                benchmark_las_pdal,
                &mut experiment,
                &machine,
                args.purge_cache,
            )
            .context("Benchmarking PDAL failed")?;
        }
        "last" => {
            run_experiment(
                &args.input_file,
                |path| benchmark_last_pasture(path, false),
                &mut experiment,
                &machine,
                args.purge_cache,
            )
            .context("Benchmarking pasture failed")?;
            run_experiment(
                &args.input_file,
                |path| benchmark_last_pasture(path, true),
                &mut experiment,
                &machine,
                args.purge_cache,
            )
            .context("Benchmarking pasture failed")?;
        }
        "lazer" => {
            run_experiment(
                &args.input_file,
                |path| benchmark_lazer_pasture(path, false),
                &mut experiment,
                &machine,
                args.purge_cache,
            )
            .context("Benchmarking pasture failed")?;
            run_experiment(
                &args.input_file,
                |path| benchmark_lazer_pasture(path, true),
                &mut experiment,
                &machine,
                args.purge_cache,
            )
            .context("Benchmarking pasture failed")?;
        }
        _ => bail!("Invalid file extension {extension}"),
    }

    Ok(())
}
