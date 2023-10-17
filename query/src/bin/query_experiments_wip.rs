use std::{
    ffi::OsStr,
    path::{Path, PathBuf}, process::Command, fs::OpenOptions, io::Write,
};

use anyhow::{bail, Context, Result};
use clap::Parser;
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
use pasture_io::{las::point_layout_from_las_point_format, las_rs::point::Format};
use query::{
    index::{
        AtomicExpression, Classification, CompareExpression, Geometry,
        NoRefinementStrategy, Position, ProgressiveIndex, QueryExpression, ReturnNumber, Value, NumberOfReturns, DiscreteLod,
    },
    io::StdoutOutput,
    stats::QueryStats,
};
use shapefile::{Shape, ShapeReader};

const DOC_SHAPEFILE_SMALL_POLY_WITH_HOLES_PATH: &str =
    "doc_polygon_small_with_holes_1.shp";
const DOC_SHAPEFILE_SMALL_RECT: &str =
    "doc_polygon_small.shp";
const DOC_SHAPEFILE_SMALL_POLY: &str =
    "doc_polygon_small_2.shp";
const DOC_SHAPEFILE_LARGE_RECT: &str =
    "doc_polygon_large_1.shp";
    const DOC_SHAPEFILE_LARGE_POLY: &str =
    "doc_polygon_large_1.shp";

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
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

fn crude_shapefile_parsing(path: impl AsRef<Path>) -> Result<QueryExpression> {
    let shape_reader = ShapeReader::from_path(path).context("Can't read shapefile")?;
    let shapes = shape_reader
        .read()
        .context("Failed to read shapes from shapefile")?;
    if shapes.is_empty() {
        bail!("No shapes found in shapefile");
    }
    let first_shape = &shapes[0];
    match first_shape {
        Shape::Polygon(poly) => {
            let geo_polygon: MultiPolygon = poly.clone().into();
            let first_polygon = geo_polygon.0[0].clone();
            Ok(QueryExpression::Atomic(AtomicExpression::Intersects(
                Geometry::Polygon(first_polygon),
            )))
        }
        _ => bail!("Unsupported shape type"),
    }
}

struct QueryParams {
    query: QueryExpression,
    files: Vec<PathBuf>,
    file_format: &'static str,
    target_layout: PointLayout,
    flush_disk_cache: bool,
}

struct OutputStats {
    bytes_written: usize,
}

fn run_query(params: QueryParams, run_number: usize, total_runs: usize,) -> Result<(QueryStats, OutputStats)> {
    info!("Run {run_number:3} / {total_runs:3}: file format: {} - query: {} - output attributes: {}", params.file_format, params.query, params.target_layout);

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
fn experiment_queries_doc(args: &Args) -> Result<Vec<(&'static str, QueryExpression)>> {
    let small_rect_query = crude_shapefile_parsing(args.shapefiles_path.join(DOC_SHAPEFILE_SMALL_RECT))?;
    let small_poly_query = crude_shapefile_parsing(args.shapefiles_path.join(DOC_SHAPEFILE_SMALL_POLY))?;
    let small_poly_with_holes_query =
        crude_shapefile_parsing(args.shapefiles_path.join(DOC_SHAPEFILE_SMALL_POLY_WITH_HOLES_PATH))?;
    let large_rect_query_1 = crude_shapefile_parsing(args.shapefiles_path.join(DOC_SHAPEFILE_LARGE_RECT))?;
    let large_poly_query = crude_shapefile_parsing(args.shapefiles_path.join(DOC_SHAPEFILE_LARGE_POLY))?;

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

    let buildings_in_small_polygon = QueryExpression::And(Box::new(small_poly_query.clone()), Box::new(class_buildings.clone()));

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

    Ok(vec![
        ("AABB (small)", aabb_small),
        ("AABB (large)", aabb_large),
        ("AABB (full)", aabb_all),
        ("AABB (none)", aabb_no_matches),
        ("Rect (small)", small_rect_query),
        ("Polygon (small)", small_poly_query),
        ("Rect (large)", large_rect_query_1),
        ("Polygon (large)", large_poly_query),
        ("Polygon (holes)", small_poly_with_holes_query),
        ("Buildings", class_buildings),
        ("Buildings in small polygon", buildings_in_small_polygon),
        ("Vegetation", vegetation_classes),
        ("First returns", first_returns),
        ("Canopies estimate", canopies_estimate),
        ("LOD0", lod0),
        ("LOD3", lod3),
    ])
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

fn main() -> Result<()> {
    // dotenv::dotenv().context("Failed to initialize with .env file")?;
    pretty_env_logger::init();

    let args = Args::parse();

    info!("Ad-hoc query experiment - doc dataset");

    let machine = std::env::var("MACHINE").context("To run experiments, please set the 'MACHINE' environment variable to the name of the machine that you are running this experiment on. This is required so that experiment data can be mapped to the actual machine that ran the experiment. This will typically be the name or system configuration of the computer that runs the experiment.")?;

    let doc_queries =
        experiment_queries_doc(&args).context("Failed to build queries for doc dataset")?;

    let las_files = get_files_with_extension("las", &args.data_path.join("doc/las"));
    let last_files = get_files_with_extension("last", &args.data_path.join("doc/last"));
    let laz_files = get_files_with_extension("laz", &args.data_path.join("doc/laz"));
    let lazer_files = get_files_with_extension("lazer", &args.data_path.join("doc/lazer"));

    let doc_datasets = [
        ("LAS", las_files),
        ("LAST", last_files),
        ("LAZ", laz_files),
        ("LAZER", lazer_files),
    ];

    let output_point_layouts = [
        ("All (default)", point_layout_from_las_point_format(&Format::new(6)?, false)?),
        ("All (native)", point_layout_from_las_point_format(&Format::new(6)?, true)?),
        ("Positions", [POSITION_3D].into_iter().collect::<PointLayout>()),
        ("Positions, classifications, intensities", [POSITION_3D, CLASSIFICATION, INTENSITY]
            .into_iter()
            .collect::<PointLayout>()),
    ];

    let experiment_description = include_str!("yaml/ad_hoc_queries.yaml");
    let experiment = ExperimentVersion::from_yaml_str(experiment_description).context("Could not get current version of experiment")?;

    let total_runs = doc_queries.len() * output_point_layouts.len() * doc_datasets.len();
    let mut current_run = 0; 

    // Dataset is in outermost loop so that - if we don't flush the disk cache - we immediately get results for
    // cached data. Which we might want
    for (file_format, files) in &doc_datasets {
        for flush_disk_cache in [false, true] {
            for (query_label, query) in &doc_queries {
                for (layout_label, output_layout) in &output_point_layouts {
                    let params = QueryParams {
                        files: files.clone(),
                        flush_disk_cache,
                        query: query.clone(),
                        target_layout: output_layout.clone(),
                        file_format,
                    };
                    let experiment_instance = experiment.make_instance([
                        ("Dataset", GenericValue::String(format!("DoC ({file_format})"))),
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
