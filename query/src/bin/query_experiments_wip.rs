use std::{
    ffi::OsStr,
    path::{Path, PathBuf}, process::Command, fs::OpenOptions, io::Write, borrow::Cow,
};

use anyhow::{anyhow, bail, Context, Result};
use clap::{Parser, ValueEnum};
use exar::{experiment::ExperimentVersion, variable::GenericValue};
use geo::MultiPolygon;
use log::info;
use pasture_core::{
    layout::{
        attributes::{CLASSIFICATION, INTENSITY, POSITION_3D},
        PointLayout,
    },
    nalgebra::Vector3,
};
use pasture_io::las::{LASReader, point_layout_from_las_metadata};
use query::{
    index::{
        AtomicExpression, Classification, CompareExpression, Geometry,
        NoRefinementStrategy, Position, ProgressiveIndex, QueryExpression, ReturnNumber, Value, NumberOfReturns, DiscreteLod,
    },
    io::StdoutOutput,
    stats::QueryStats,
};
use shapefile::{Shape, dbase::FieldValue};

#[derive(ValueEnum, Copy, Clone, Debug)]
enum Dataset {
    Doc,
    CA13,
    AHN4S,
}

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    dataset: Dataset,
    data_path: PathBuf,
    shapefiles_path: PathBuf,
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

fn parse_shapefile<P: AsRef<Path>>(path: P) -> Result<Vec<(Cow<'static, str>, QueryExpression)>> {
    let mut reader = shapefile::Reader::from_path(path)?;
    reader.iter_shapes_and_records().map(|shape_record| -> Result<(Cow<'static, str>, QueryExpression)> {
        let (shape, record) = shape_record?;
        let shape_name = match record.get("name").ok_or_else(|| anyhow!("Missing field 'name' in shape"))? {
            FieldValue::Character(text) => text.clone().unwrap_or_default(),
            other => bail!("Unexpected attribute value for field 'name' ({other:?}"),
        };
        let query = match shape {
            Shape::Polygon(poly) => {
                let geo_polygon: MultiPolygon = poly.clone().into();
                if geo_polygon.0.len() > 1 {
                    bail!("Shape {shape_name} has more than one Polygons, but only one Polygon per shape is supported currently");
                }
            let first_polygon = geo_polygon.0[0].clone();
            eprintln!("Shape: {first_polygon:?}");
            QueryExpression::Atomic(AtomicExpression::Intersects(
                Geometry::Polygon(first_polygon),
            ))
            }
            other => bail!("Invalid Shape, expected Polygon but got {other}"),
        };
        Ok((Cow::Owned(shape_name), query))
    }).collect()
}

/// Configuration for a specific dataset (file paths, queries, output layouts)
struct DatasetConfig {
    dataset_name: &'static str,
    queries: Vec<(Cow<'static, str>, QueryExpression)>,
    datasets: Vec<(&'static str, Vec<PathBuf>)>,
    output_point_layouts: Vec<(&'static str, PointLayout)>,
}

impl DatasetConfig {
    pub fn total_runs(&self) -> usize {
        self.queries.len() * self.datasets.len() * self.output_point_layouts.len()
    }
}

struct QueryParams {
    query: QueryExpression,
    query_label: String,
    files: Vec<PathBuf>,
    file_format: &'static str,
    target_layout: PointLayout,
    layout_label: String,
    flush_disk_cache: bool,
}

struct OutputStats {
    bytes_written: usize,
}

fn run_query(params: QueryParams, run_number: usize, total_runs: usize,) -> Result<(QueryStats, OutputStats)> {
    info!("Run {run_number:3} / {total_runs:3}: file format: {} - query: {} - output attributes: {}", params.file_format, params.query_label, params.layout_label);

    if params.flush_disk_cache {
        flush_disk_cache().context("Failed to flush disk cache")?;
    }

    let mut index = ProgressiveIndex::new();
    let dataset_id = index
        .add_dataset(params.files.as_slice())
        .context("Failed to add new dataset")?;

    let positions_in_world_space = params.target_layout.has_attribute(&POSITION_3D);
    let stdout_output = StdoutOutput::new(params.target_layout, positions_in_world_space);
    let query_stats = index
        .query(
            dataset_id,
            params.query,
            &NoRefinementStrategy,
            &stdout_output,
        )
        .context("Query failed")?;

    let output_stats = OutputStats {
        bytes_written: stdout_output.bytes_output(),
    };

    Ok((query_stats, output_stats))
}

/// Returns a collection of queries for the experiment with the district of columbia dataset
fn experiment_queries_doc(args: &Args) -> Result<Vec<(Cow<'static, str>, QueryExpression)>> {
    let shapefile_queries = parse_shapefile(&args.shapefiles_path).with_context(|| format!("Failed to parse shapefile {}", args.shapefiles_path.display()))?;

    let aabb_large = QueryExpression::Atomic(AtomicExpression::Within(
        Value::Position(Position(Vector3::new(390000.0, 130000.0, 0.0)))
            ..Value::Position(Position(Vector3::new(400000.0, 140000.0, 200.0))),
    ));
    let aabb_small = QueryExpression::Atomic(AtomicExpression::Within(
        Value::Position(Position(Vector3::new(390000.0, 130000.0, 0.0)))
            ..Value::Position(Position(Vector3::new(390500.0, 140000.0, 200.0))),
    ));
    let aabb_no_matches = QueryExpression::Atomic(AtomicExpression::Within(
        Value::Position(Position(Vector3::new(39000.0, 130000.0, 0.0)))
            ..Value::Position(Position(Vector3::new(40000.0, 140000.0, 200.0))),
    ));
    let aabb_all = QueryExpression::Atomic(AtomicExpression::Within(
        Value::Position(Position(Vector3::new(0.0, 0.0, -1000.0)))
            ..Value::Position(Position(Vector3::new(
                1_000_000.0,
                1_000_000.0,
                1_000_000.0,
            ))),
    ));

    let class_buildings = QueryExpression::Atomic(AtomicExpression::Compare((
        CompareExpression::Equals,
        Value::Classification(Classification(6)),
    )));

    let small_polygon_query = shapefile_queries.iter().find(|(name, _)| name == "Polygon small").ok_or_else(|| anyhow!("Missing shape 'Polygon small"))?;
    let buildings_in_small_polygon = QueryExpression::And(Box::new(small_polygon_query.1.clone()), Box::new(class_buildings.clone()));

    let vegetation_classes = QueryExpression::Atomic(AtomicExpression::Within(
        Value::Classification(Classification(3))..Value::Classification(Classification(6))
    ));

    let first_returns = QueryExpression::Atomic(AtomicExpression::Compare((
        CompareExpression::Equals,
        Value::ReturnNumber(ReturnNumber(1)),
    )));

    let canopies_estimate = QueryExpression::And(
        Box::new(QueryExpression::Atomic(AtomicExpression::Compare(
            (CompareExpression::GreaterThan,
            Value::NumberOfReturns(NumberOfReturns(1))),
        ))),
        Box::new(first_returns.clone()),
    );

    let lod0 = QueryExpression::Atomic(
        AtomicExpression::Compare(
            (CompareExpression::Equals, Value::LOD(DiscreteLod(0))),
        )
    );

    let lod3 = QueryExpression::Atomic(
        AtomicExpression::Compare(
            (CompareExpression::Equals, Value::LOD(DiscreteLod(3))),
        )
    );

    Ok(shapefile_queries.into_iter().chain([
        (Cow::Borrowed("AABB (small)"), aabb_small),
        (Cow::Borrowed("AABB (large)"), aabb_large),
        (Cow::Borrowed("AABB (full)"), aabb_all),
        (Cow::Borrowed("AABB (none)"), aabb_no_matches),
        (Cow::Borrowed("Buildings"), class_buildings),
        (Cow::Borrowed("Buildings in small polygon"), buildings_in_small_polygon),
        (Cow::Borrowed("Vegetation"), vegetation_classes),
        (Cow::Borrowed("First returns"), first_returns),
        (Cow::Borrowed("Canopies estimate"), canopies_estimate),
        (Cow::Borrowed("LOD0"), lod0),
        (Cow::Borrowed("LOD3"), lod3),
    ].into_iter()).collect())
}

fn experiment_queries_ahn4s(args: &Args) -> Result<Vec<(Cow<'static, str>, QueryExpression)>> {
    let bounds_full = QueryExpression::Atomic(
        AtomicExpression::Within(
            Value::Position(Position(Vector3::new(120000.0, 481250.0, -8.0)))..Value::Position(
                Position(Vector3::new(125000.0, 487500.0, 200.0))
            )
        )
    );
    let bounds_large = QueryExpression::Atomic(
        AtomicExpression::Within(
            Value::Position(Position(Vector3::new(122000.0, 481250.0, 0.0)))..Value::Position(
                Position(Vector3::new(124000.0, 487500.0, 200.0))
            )
        )
    );
    let bounds_small = QueryExpression::Atomic(
        AtomicExpression::Within(
            Value::Position(Position(Vector3::new(122000.0, 481250.0, 0.0)))..Value::Position(
                Position(Vector3::new(122500.0, 482500.0, 100.0))
            )
        )
    );
    let bounds_none = QueryExpression::Atomic(
        AtomicExpression::Within(
            Value::Position(Position(Vector3::new(100000.0, 481250.0, 0.0)))..Value::Position(
                Position(Vector3::new(110000.0, 487500.0, 200.0))
            )
        )
    );

    let shapefile_queries = parse_shapefile(&args.shapefiles_path).with_context(|| format!("Failed to parse shapefile {}", args.shapefiles_path.display()))?;

    let class_buildings = QueryExpression::Atomic(AtomicExpression::Compare((
        CompareExpression::Equals,
        Value::Classification(Classification(6)),
    )));

    let small_polygon_query = shapefile_queries.iter().find(|(name, _)| name == "Polygon small").ok_or_else(|| anyhow!("Missing shape 'Polygon small"))?;
    let buildings_in_small_polygon = QueryExpression::And(Box::new(small_polygon_query.1.clone()), Box::new(class_buildings.clone()));

    let vegetation_classes = QueryExpression::Atomic(AtomicExpression::Within(
        Value::Classification(Classification(3))..Value::Classification(Classification(6))
    ));

    let first_returns = QueryExpression::Atomic(AtomicExpression::Compare((
        CompareExpression::Equals,
        Value::ReturnNumber(ReturnNumber(1)),
    )));

    let canopies_estimate = QueryExpression::And(
        Box::new(QueryExpression::Atomic(AtomicExpression::Compare(
            (CompareExpression::GreaterThan,
            Value::NumberOfReturns(NumberOfReturns(1))),
        ))),
        Box::new(first_returns.clone()),
    );

    let lod0 = QueryExpression::Atomic(
        AtomicExpression::Compare(
            (CompareExpression::Equals, Value::LOD(DiscreteLod(0))),
        )
    );

    let lod3 = QueryExpression::Atomic(
        AtomicExpression::Compare(
            (CompareExpression::Equals, Value::LOD(DiscreteLod(3))),
        )
    );

    Ok(shapefile_queries.into_iter().chain([
        (Cow::Borrowed("AABB (full)"), bounds_full),
        (Cow::Borrowed("AABB (large)"), bounds_large),
        (Cow::Borrowed("AABB (small)"), bounds_small),
        (Cow::Borrowed("AABB (none)"), bounds_none),
        (Cow::Borrowed("Buildings"), class_buildings),
        (Cow::Borrowed("Buildings in small polygon"), buildings_in_small_polygon),
        (Cow::Borrowed("Vegetation"), vegetation_classes),
        (Cow::Borrowed("First returns"), first_returns),
        (Cow::Borrowed("Canopies estimate"), canopies_estimate),
        (Cow::Borrowed("LOD0"), lod0),
        (Cow::Borrowed("LOD3"), lod3),
    ].into_iter()).collect())
}

fn get_files_with_extension<P: AsRef<Path>>(extension: &str, path: P) -> Vec<PathBuf> {
    walkdir::WalkDir::new(path)
        .into_iter()
        .filter_map(|entry| {
            entry.ok().and_then(|entry| {
                if entry.path().extension().and_then(OsStr::to_str) == Some(extension) {
                    Some(entry.path().to_owned())
                } else {
                    None
                }
            })
        })
        .collect()
}

fn get_doc_config(args: &Args) -> Result<DatasetConfig> {
    let doc_queries =
    experiment_queries_doc(&args).context("Failed to build queries for doc dataset")?;

    let las_files = get_files_with_extension("las", &args.data_path.join("doc/las"));
    let last_files = get_files_with_extension("last", &args.data_path.join("doc/last"));
    let laz_files = get_files_with_extension("laz", &args.data_path.join("doc/laz"));
    let lazer_files = get_files_with_extension("lazer", &args.data_path.join("doc/lazer"));

    let doc_datasets = vec![
        ("LAS", las_files),
        ("LAST", last_files),
        ("LAZ", laz_files),
        ("LAZER", lazer_files),
    ];

    // Instead of hardcoding the point formats, we get the default and native PointLayouts from the dataset
    let doc_metadata = LASReader::from_path(&doc_datasets[0].1[0], false).context("Failed to open first file from DoC dataset")?.las_metadata().clone();

    let doc_output_point_layouts = vec![
        ("All (default)", point_layout_from_las_metadata(&doc_metadata, false)?),
        ("All (native)", point_layout_from_las_metadata(&doc_metadata, true)?),
        ("Positions", [POSITION_3D].into_iter().collect::<PointLayout>()),
        ("Positions, classifications, intensities", [POSITION_3D, CLASSIFICATION, INTENSITY]
            .into_iter()
            .collect::<PointLayout>()),
    ];

    Ok(DatasetConfig { dataset_name: "doc", queries: doc_queries, datasets: doc_datasets, output_point_layouts: doc_output_point_layouts })
}

fn get_ahn4s_config(args: &Args) -> Result<DatasetConfig> {
    let queries =
    experiment_queries_ahn4s(&args).context("Failed to build queries for AHN4-S dataset")?;

    let las_files = get_files_with_extension("las", &args.data_path.join("ahn4s/las"));
    let last_files = get_files_with_extension("last", &args.data_path.join("ahn4s/last"));
    let laz_files = get_files_with_extension("laz", &args.data_path.join("ahn4s/laz"));
    let lazer_files = get_files_with_extension("lazer", &args.data_path.join("ahn4s/lazer"));

    let datasets = vec![
        ("LAS", las_files),
        ("LAST", last_files),
        ("LAZ", laz_files),
        ("LAZER", lazer_files),
    ];

    // Instead of hardcoding the point formats, we get the default and native PointLayouts from the dataset
    let metadata = LASReader::from_path(&datasets[0].1[0], false).context("Failed to open first file from AHN4-S dataset")?.las_metadata().clone();

    let output_point_layouts = vec![
        ("All (default)", point_layout_from_las_metadata(&metadata, false)?),
        ("All (native)", point_layout_from_las_metadata(&metadata, true)?),
        ("Positions", [POSITION_3D].into_iter().collect::<PointLayout>()),
        ("Positions, classifications, intensities", [POSITION_3D, CLASSIFICATION, INTENSITY]
            .into_iter()
            .collect::<PointLayout>()),
    ];

    Ok(DatasetConfig { dataset_name: "AHN4-S", queries, datasets, output_point_layouts, })
}

fn main() -> Result<()> {
    // dotenv::dotenv().context("Failed to initialize with .env file")?;
    pretty_env_logger::init();

    let args = Args::parse();

    info!("Ad-hoc query experiment - {:?}", args.dataset);

    let machine = std::env::var("MACHINE").context("To run experiments, please set the 'MACHINE' environment variable to the name of the machine that you are running this experiment on. This is required so that experiment data can be mapped to the actual machine that ran the experiment. This will typically be the name or system configuration of the computer that runs the experiment.")?;

    let experiment_description = include_str!("yaml/ad_hoc_queries.yaml");
    let experiment = ExperimentVersion::from_yaml_str(experiment_description).context("Could not get current version of experiment")?;

    let dataset_config = match args.dataset {
        Dataset::Doc => get_doc_config(&args)?,
        Dataset::AHN4S => get_ahn4s_config(&args)?,
        _ => unimplemented!(),
    };

    let total_runs = dataset_config.total_runs() * 2; //x2 for flushing disk cache
    let mut current_run = 0; 

    // Dataset is in outermost loop so that - if we don't flush the disk cache - we immediately get results for
    // cached data. Which we might want
    for (file_format, files) in &dataset_config.datasets {
        for flush_disk_cache in [true, false] {
            for (query_label, query) in &dataset_config.queries {
                for (layout_label, output_layout) in &dataset_config.output_point_layouts {
                    let params = QueryParams {
                        files: files.clone(),
                        flush_disk_cache,
                        query: query.clone(),
                        query_label: query_label.to_string(),
                        target_layout: output_layout.clone(),
                        layout_label: layout_label.to_string(),
                        file_format,
                    };
                    let experiment_instance = experiment.make_instance([
                        ("Dataset", GenericValue::String(format!("{} ({file_format})", dataset_config.dataset_name))),
                        ("Machine", GenericValue::String(machine.clone())),
                        ("System", GenericValue::String("Ad-hoc query engine".to_string())),
                        ("Query", GenericValue::String(query_label.to_string())),
                        ("Output attributes", GenericValue::String(layout_label.to_string())),
                        ("Purge cache", GenericValue::Bool(flush_disk_cache)),
                    ]).context("Could not create experiment instance")?;

                    current_run += 1;
                    experiment_instance.run(|run_context| {
                        let (query_stats, output_stats) = run_query(params, current_run, total_runs).context("Executing query failed")?;
                        
                        run_context.add_measurement("Runtime", GenericValue::Numeric(query_stats.runtime.as_secs_f64()));
                        run_context.add_measurement("Bytes written", GenericValue::Numeric(output_stats.bytes_written as f64));
                        run_context.add_measurement("Match count", GenericValue::Numeric(query_stats.matching_points as f64));
                        run_context.add_measurement("Queried points count", GenericValue::Numeric(query_stats.total_points_queried as f64));

                        Ok(())
                    }).with_context(|| format!("Experiment run failed (query {query}, file format {file_format}, output layout {output_layout})"))?;
                }
            }
        }
    }

    Ok(())
}
