use std::{
    borrow::Cow,
    collections::HashSet,
    ffi::OsStr,
    fs::{File, OpenOptions},
    io::{BufReader, Cursor, Write},
    path::{Path, PathBuf},
    process::Command,
    time::{Duration, Instant},
};

use anyhow::{bail, Context, Result};
use clap::Parser;
use experiment_archiver::{Experiment, VariableTemplate};

use io::last::LASTReader;
use itertools::Itertools;
use pasture_core::{
    containers::{BorrowedBuffer, HashMapBuffer, OwningBuffer, VectorBuffer},
    layout::{attributes::CLASSIFICATION, PointLayout},
};
use pasture_io::{
    base::PointReader,
    las::{LASReader, ATTRIBUTE_LOCAL_LAS_POSITION},
};
use statrs::statistics::{Data, Distribution};

const VARIABLE_DATASET: VariableTemplate = VariableTemplate::new(
    Cow::Borrowed("Dataset"),
    Cow::Borrowed("The dataset used in the experiment"),
    Cow::Borrowed("none"),
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
const VARIABLE_POINT_THROUGHPUT_ERR: VariableTemplate = VariableTemplate::new(
    Cow::Borrowed("Throughput (points) error (1 sigma)"),
    Cow::Borrowed("Standard error for point throughput"),
    Cow::Borrowed("points/s"),
);
const VARIABLE_MEMORY_THROUGHPUT: VariableTemplate = VariableTemplate::new(
    Cow::Borrowed("Throughput (memory)"),
    Cow::Borrowed("The memory throughput, i.e. how many bytes were read per second"),
    Cow::Borrowed("bytes/s"),
);
const VARIABLE_MEMORY_THROUGHPUT_ERR: VariableTemplate = VariableTemplate::new(
    Cow::Borrowed("Throughput (memory) error (1 sigma)"),
    Cow::Borrowed("Standard error for memory throughput"),
    Cow::Borrowed("bytes/s"),
);
const VARIABLE_LOADED_ATTRIBUTES: VariableTemplate = VariableTemplate::new(
    Cow::Borrowed("Attributes"),
    Cow::Borrowed("The actual point attributes that were loaded from the dataset"),
    Cow::Borrowed("text"),
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
    dataset: String,
    point_layout: PointLayout,
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
        let mut reader = LASReader::from_read(Cursor::new(&mmap[..]), is_compressed, true)
            .context("failed to open LAS reader")?;
        reader
            .read::<VectorBuffer>(reader.remaining_points())
            .context("Failed to read points")?
    } else {
        let file = File::open(file).context("failed to open file")?;
        let mut reader = LASReader::from_read(BufReader::new(file), is_compressed, true)
            .context("failed to open LAS reader")?;
        reader
            .read::<VectorBuffer>(reader.remaining_points())
            .context("Failed to read points")?
    };
    let runtime = timer.elapsed();

    let bytes_per_second = file_size as f64 / runtime.as_secs_f64();
    let points_per_second = points.len() as f64 / runtime.as_secs_f64();

    let dataset = if mmapped {
        format!("{} (mmap)", file.display())
    } else {
        format!("{} (file)", file.display())
    };

    Ok(Stats {
        bytes_per_second,
        number_of_points: points.len(),
        points_per_second,
        runtime,
        dataset,
        point_layout: points.point_layout().clone(),
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

    let dataset = if mmapped {
        format!("{} (mmap)", file.display())
    } else {
        format!("{} (file)", file.display())
    };

    Ok(Stats {
        bytes_per_second,
        number_of_points: points.len(),
        points_per_second,
        runtime,
        dataset,
        point_layout: points.point_layout().clone(),
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
        reader
            .read::<HashMapBuffer>(reader.remaining_points())
            .context("Failed to read points")?
    } else {
        let file = File::open(file).context("failed to open file")?;
        let mut reader =
            LASTReader::from_read(BufReader::new(file)).context("failed to open LAST reader")?;
        reader
            .read::<HashMapBuffer>(reader.remaining_points())
            .context("Failed to read points")?
    };
    let runtime = timer.elapsed();

    let bytes_per_second = file_size as f64 / runtime.as_secs_f64();
    let points_per_second = points.len() as f64 / runtime.as_secs_f64();

    let dataset = if mmapped {
        format!("{} (mmap)", file.display())
    } else {
        format!("{} (file)", file.display())
    };

    Ok(Stats {
        bytes_per_second,
        number_of_points: points.len(),
        points_per_second,
        runtime,
        dataset,
        point_layout: points.point_layout().clone(),
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

    let dataset = if mmapped {
        format!("{} (mmap)", file.display())
    } else {
        format!("{} (file)", file.display())
    };

    Ok(Stats {
        bytes_per_second,
        number_of_points: points.len(),
        points_per_second,
        runtime,
        dataset,
        point_layout: points.point_layout().clone(),
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

    let point_throughputs = Data::new(
        stats_per_run
            .iter()
            .map(|stat| stat.points_per_second)
            .collect_vec(),
    );
    let mean_point_throughput = point_throughputs
        .mean()
        .expect("Could not calculate mean point throughput")
        as usize;
    let stddev_point_throughput = point_throughputs.std_dev().unwrap() as usize;

    let memory_throghputs = Data::new(
        stats_per_run
            .iter()
            .map(|stat| stat.bytes_per_second)
            .collect_vec(),
    );
    let mean_memory_throughput = memory_throghputs
        .mean()
        .expect("Could not calculate mean memory throughput")
        as usize;
    let stddev_memory_throughput = memory_throghputs.std_dev().unwrap() as usize;

    let stats = stats_per_run[0].clone();

    experiment.run(|context| {
        context.add_value_by_name(VARIABLE_DATASET.name(), stats.dataset);
        context.add_value_by_name(VARIABLE_MACHINE.name(), machine);
        context.add_value_by_name(
            VARIABLE_MEMORY_THROUGHPUT.name(),
            format!("{mean_memory_throughput}"),
        );
        context.add_value_by_name(
            VARIABLE_MEMORY_THROUGHPUT_ERR.name(),
            format!("{stddev_memory_throughput}"),
        );
        context.add_value_by_name(VARIABLE_NR_POINTS.name(), stats.number_of_points);
        context.add_value_by_name(
            VARIABLE_POINT_THROUGHPUT.name(),
            format!("{mean_point_throughput}"),
        );
        context.add_value_by_name(
            VARIABLE_POINT_THROUGHPUT_ERR.name(),
            format!("{stddev_point_throughput}"),
        );
        context.add_value_by_name(VARIABLE_RUNTIME.name(), format!("{mean_runtime_ms:.3}"));
        context.add_value_by_name(
            VARIABLE_RUNTIME_ERR.name(),
            format!("{stddev_runtime_ms:.3}"),
        );

        let attributes_str = stats
            .point_layout
            .attributes()
            .map(|a| a.attribute_definition().to_string())
            .join(";");
        context.add_value_by_name(VARIABLE_LOADED_ATTRIBUTES.name(), attributes_str);
        context.add_value_by_name(VARIABLE_PURGE_CACHE.name(), purge_cache);

        Ok(())
    })?;
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

    let variables = [
        VARIABLE_DATASET.clone(),
        VARIABLE_RUNTIME.clone(),
        VARIABLE_RUNTIME_ERR.clone(),
        VARIABLE_NR_POINTS.clone(),
        VARIABLE_POINT_THROUGHPUT.clone(),
        VARIABLE_POINT_THROUGHPUT_ERR.clone(),
        VARIABLE_MEMORY_THROUGHPUT.clone(),
        VARIABLE_MEMORY_THROUGHPUT_ERR.clone(),
        VARIABLE_PURGE_CACHE.clone(),
        VARIABLE_MACHINE.clone(),
        VARIABLE_LOADED_ATTRIBUTES.clone(),
    ]
    .into_iter()
    .collect::<HashSet<VariableTemplate>>();

    let mut experiment = Experiment::new(
        "Optimized LAS/LAST reading performance".into(),
        "Measures the performance of reading LAS and LAST point clouds with optimized code (e.g. reading in matching LAS binary layout, using mmap and ExternalMemoryBuffer etc.)".into(),
        "Pascal Bormann".into(),
        variables,
    )
    .context("Failed to setup experiment")?;
    experiment.set_autolog_runs(true);

    let mmapped = [false, true];

    let only_positions_layout = [ATTRIBUTE_LOCAL_LAS_POSITION]
        .into_iter()
        .collect::<PointLayout>();
    let only_classifications_layout = [CLASSIFICATION].into_iter().collect::<PointLayout>();

    let custom_layouts = [only_positions_layout, only_classifications_layout];

    match extension.as_ref() {
        "las" => {
            for mmap in &mmapped {
                run_experiment(
                    &args.input_file,
                    |path| benchmark_las_default_layout(path, *mmap),
                    &mut experiment,
                    &machine,
                    args.purge_cache,
                )
                .context("Benchmarking pasture failed")?;

                for layout in &custom_layouts {
                    run_experiment(
                        &args.input_file,
                        |path| benchmark_las_pasture_custom_layout(path, *mmap, layout),
                        &mut experiment,
                        &machine,
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
                    &mut experiment,
                    &machine,
                    args.purge_cache,
                )
                .context("Benchmarking pasture failed")?;

                for layout in &custom_layouts {
                    run_experiment(
                        &args.input_file,
                        |path| benchmark_last_pasture_custom_layout(path, *mmap, layout),
                        &mut experiment,
                        &machine,
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
