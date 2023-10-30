use std::{
    ffi::OsStr,
    fs::OpenOptions,
    io::Write,
    path::{Path, PathBuf},
    process::Command,
    time::Duration,
};

use anyhow::{bail, Context, Result};
use exar::{experiment::ExperimentVersion, variable::GenericValue};
use itertools::Itertools;
use pasture_core::{
    layout::{attributes::POSITION_3D, PointLayout},
    math::AABB,
    nalgebra::Point3,
};
use pasture_io::las::{point_layout_from_las_metadata, LASReader};
use query::{
    index::{
        AlwaysRefinementStrategy, AtomicExpression, Classification, CompareExpression,
        NoRefinementStrategy, Position, ProgressiveIndex, QueryExpression, RefinementStrategy,
        TimeBudgetRefinementStrategy, Value,
    },
    io::StdoutOutput,
};

struct QueryParams<'a> {
    query_series: Vec<QueryExpression>,
    query_label: String,
    files: Vec<PathBuf>,
    dataset_label: String,
    file_format: &'static str,
    target_layout: PointLayout,
    refinement_strategy: &'a dyn RefinementStrategy,
    refinement_strategy_label: String,
}

fn get_files_with_extension<P: AsRef<Path>>(extension: &str, path: P) -> Vec<PathBuf> {
    walkdir::WalkDir::new(path)
        .into_iter()
        .filter_map(|entry| {
            entry.ok().and_then(|entry| {
                let extension_lower = entry
                    .path()
                    .extension()
                    .and_then(OsStr::to_str)
                    .map(|s| s.to_ascii_lowercase());
                if extension_lower.as_ref().map(|s| s.as_str()) == Some(extension) {
                    Some(entry.path().to_owned())
                } else {
                    None
                }
            })
        })
        .collect()
}

fn query_from_bounds(bounds: &AABB<f64>) -> QueryExpression {
    QueryExpression::Atomic(AtomicExpression::Within(
        Value::Position(Position(bounds.min().coords))
            ..Value::Position(Position(bounds.max().coords)),
    ))
}

fn lerp_bounds(from: &AABB<f64>, to: &AABB<f64>, f: f64) -> AABB<f64> {
    let new_min = (from.min() * (1.0 - f)).coords + (to.min() * f).coords;
    let new_max = (from.max() * (1.0 - f)).coords + (to.max() * f).coords;
    AABB::from_min_max(new_min.into(), new_max.into())
}

fn get_test_queries_ahn4s() -> Vec<(String, Vec<QueryExpression>)> {
    const REPEATS: usize = 8;
    let large_bounds = AABB::from_min_max(
        Point3::new(122000.0, 481250.0, 0.0),
        Point3::new(124000.0, 487500.0, 200.0),
    );
    let small_bounds = AABB::from_min_max(
        Point3::new(122000.0, 481250.0, 0.0),
        Point3::new(122500.0, 482500.0, 100.0),
    );

    let small_bounds_query = query_from_bounds(&small_bounds);
    let large_bounds_query = query_from_bounds(&large_bounds);

    let large_to_small_queries = (0..REPEATS)
        .map(|idx| {
            let f = idx as f64 / (REPEATS - 1) as f64;
            let bounds = lerp_bounds(&large_bounds, &small_bounds, f);
            query_from_bounds(&bounds)
        })
        .collect_vec();

    let buildings_query = QueryExpression::Atomic(AtomicExpression::Compare((
        CompareExpression::Equals,
        Value::Classification(Classification(6)),
    )));

    // Ideas:
    // 1) N times the small query
    // 2) N times the large query
    // 3) From the large query, interpolated to the small query, in N steps
    // 4) N times a buildings query

    vec![
        (
            "Small repeated".to_string(),
            vec![small_bounds_query; REPEATS],
        ),
        (
            "Large repeated".to_string(),
            vec![large_bounds_query; REPEATS],
        ),
        ("Large to small".to_string(), large_to_small_queries),
        (
            "Buildings repeated".to_string(),
            vec![buildings_query; REPEATS],
        ),
    ]
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

fn run_adaptive_query(
    query_params: QueryParams<'_>,
    experiment_version: &ExperimentVersion,
    machine: &str,
) -> Result<()> {
    flush_disk_cache().context("Failed to flush disk cache")?;

    let mut index = ProgressiveIndex::new();
    let dataset_id = index
        .add_dataset(query_params.files.as_slice())
        .context("Failed to add new dataset")?;

    for (query_number, query) in query_params.query_series.into_iter().enumerate() {
        let positions_in_world_space = query_params.target_layout.has_attribute(&POSITION_3D);
        let stdout_output = StdoutOutput::new(
            query_params.target_layout.clone(),
            positions_in_world_space,
            true,
        );
        let query_stats = index
            .query(
                dataset_id,
                query,
                query_params.refinement_strategy,
                &stdout_output,
            )
            .context("Query failed")?;

        let bytes_over_time = stdout_output.bytes_over_time().unwrap();
        let t_start = bytes_over_time[0].0;
        let bytes_since_start = bytes_over_time
            .into_iter()
            .map(|(time, bytes)| {
                let duration = time - t_start;
                (duration.as_secs_f64(), bytes)
            })
            .collect_vec();
        // Instead of having tons of values for all timestamps where something changed, convert to an even distribution
        // over the total runtime. This gives nice tables like 'After 5% of the runtime, we had X% of the data' etc.
        const TIME_STEPS: usize = 20;
        let total_time = bytes_since_start.last().unwrap().0;
        let bytes_per_timestep = (0..=TIME_STEPS)
            .map(|step| {
                let step_percentage = step as f64 / TIME_STEPS as f64;
                let current_timestep = step_percentage * total_time;
                let index_of_closest_value =
                    bytes_since_start.partition_point(|(time, _)| *time <= current_timestep) - 1;
                let bytes_at_timestep = bytes_since_start[index_of_closest_value].1;
                (current_timestep, bytes_at_timestep)
            })
            .collect_vec();

        let instance = experiment_version
            .make_instance([
                (
                    "Dataset",
                    GenericValue::String(query_params.dataset_label.clone()),
                ),
                ("Machine", GenericValue::String(machine.to_string())),
                (
                    "Query",
                    GenericValue::String(query_params.query_label.to_string()),
                ),
                (
                    "Query iteration",
                    GenericValue::Numeric(query_number as f64),
                ),
                (
                    "Refinement strategy",
                    GenericValue::String(query_params.refinement_strategy_label.to_string()),
                ),
                (
                    "File format",
                    GenericValue::String(query_params.file_format.to_string()),
                ),
            ])
            .context("Failed to create ExperimentInstance")?;

        instance
            .run(|context| {
                let only_query_runtime = query_stats.runtime - query_stats.refinement_time;
                context.add_measurement(
                    "Query runtime",
                    GenericValue::Numeric(only_query_runtime.as_secs_f64()),
                );
                context.add_measurement(
                    "Refinement time",
                    GenericValue::Numeric(query_stats.refinement_time.as_secs_f64()),
                );
                context.add_measurement(
                    "Queried points count",
                    GenericValue::Numeric(query_stats.total_points_queried as f64),
                );

                let bytes_since_start_as_csvlike = bytes_per_timestep
                    .iter()
                    .map(|(time, bytes)| format!("({time},{bytes})"))
                    .join(",");
                context.add_measurement(
                    "Bytes over time",
                    GenericValue::String(bytes_since_start_as_csvlike),
                );

                Ok(())
            })
            .context("Failed to log data for experiment run")?;

        eprintln!("Query {query_number} stats: {query_stats}");
        eprintln!("Bytes since start: {bytes_per_timestep:?}");
    }

    Ok(())
}

fn main() -> Result<()> {
    pretty_env_logger::init();
    let _client = tracy_client::Client::start();

    // If desired, keep 2 logical cores free so we can do other stuff while the experiment runs
    // Doesn't work too well though...
    // let max_hardware_threads = std::thread::available_parallelism()?;
    // rayon::ThreadPoolBuilder::new()
    //     .num_threads(max_hardware_threads.get() - 2)
    //     .build_global()?;

    let machine = std::env::var("MACHINE").context("To run experiments, please set the 'MACHINE' environment variable to the name of the machine that you are running this experiment on. This is required so that experiment data can be mapped to the actual machine that ran the experiment. This will typically be the name or system configuration of the computer that runs the experiment.")?;

    let experiment_description = include_str!("yaml/adaptive_indexing.yaml");
    let experiment = ExperimentVersion::from_yaml_str(experiment_description)
        .context("Could not get current version of experiment")?;

    let refinement_strategies: Vec<(&'static str, Box<dyn RefinementStrategy>)> = vec![
        ("No adaptive indexing", Box::new(NoRefinementStrategy)),
        ("Refine always", Box::new(AlwaysRefinementStrategy)),
        (
            "Refine timed (5s)",
            Box::new(TimeBudgetRefinementStrategy::new(Duration::from_secs(5))),
        ),
        (
            "Refine timed (10s)",
            Box::new(TimeBudgetRefinementStrategy::new(Duration::from_secs(10))),
        ),
    ];

    let file_extensions = [
        //"las",
        // "last",
        "laz",
        // "lazer",
    ];
    // let file_extensions = &file_extensions[1..2];

    for extension in file_extensions {
        let ahn4s_files = get_files_with_extension(
            extension,
            format!("/Users/pbormann/data/projects/progressive_indexing/experiment_data/ahn4s/{extension}"),
        );
        let ahn4s_metadata = LASReader::from_path(&ahn4s_files[0], false)
            .context("Failed to open first file from AHN4-S dataset")?
            .las_metadata()
            .clone();
        let ahn4s_default_point_layout = point_layout_from_las_metadata(&ahn4s_metadata, false)?;
        let ahn4s_queries = get_test_queries_ahn4s();

        for (query_label, queries) in ahn4s_queries {
            for (refinement_strategy_label, refinement_strategy) in &refinement_strategies {
                let config = QueryParams {
                    file_format: extension,
                    files: ahn4s_files.clone(),
                    dataset_label: "AHN4-S".to_string(),
                    query_label: query_label.clone(),
                    query_series: queries.clone(),
                    target_layout: ahn4s_default_point_layout.clone(),
                    refinement_strategy: refinement_strategy.as_ref(),
                    refinement_strategy_label: refinement_strategy_label.to_string(),
                };

                run_adaptive_query(config, &experiment, &machine)
                    .with_context(|| format!("Query {query_label} failed"))?;
            }
        }
    }

    Ok(())
}
