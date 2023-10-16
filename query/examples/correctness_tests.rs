use std::{
    collections::HashSet,
    ffi::OsStr,
    ops::Range,
    path::{Path, PathBuf},
};

use anyhow::{bail, Context, Result};
use clap::Parser;
use colored::Colorize;
use geo::{Contains, MultiPolygon};
use pasture_core::{
    containers::{BorrowedBuffer, VectorBuffer},
    layout::attributes::{CLASSIFICATION, GPS_TIME, NUMBER_OF_RETURNS, POSITION_3D, RETURN_NUMBER},
    math::AABB,
    nalgebra::Vector3,
};
use pasture_io::{base::PointReader, las::LASReader};
use query::{
    index::{
        AtomicExpression, Classification, CompareExpression, Geometry, GpsTime,
        NoRefinementStrategy, Position, ProgressiveIndex, QueryExpression, ReturnNumber, Value,
    },
    io::CountOutput,
};
use rayon::prelude::*;
use shapefile::{Shape, ShapeReader};

// const DATASET_DOC_LAS_PATH: &str =
//     "/Users/pbormann/data/projects/progressive_indexing/experiment_data/doc/las";
// const DATASET_DOC_LAST_PATH: &str =
//     "/Users/pbormann/data/projects/progressive_indexing/experiment_data/doc/last";
// const DATASET_DOC_LAZ_PATH: &str =
//     "/Users/pbormann/data/projects/progressive_indexing/experiment_data/doc/laz";
// const DATASET_DOC_LAZER_PATH: &str =
//     "/Users/pbormann/data/projects/progressive_indexing/experiment_data/doc/lazer";

// const DOC_SHAPEFILE_SMALL_WITH_HOLES_PATH: &str =
//     "/Users/pbormann/data/projects/progressive_indexing/queries/doc_polygon_small_with_holes_1.shp";

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    data_path: PathBuf,
    shapefile_path: PathBuf,
}

fn get_compare_func<T: PartialEq + PartialOrd + Copy>(
    expr: &CompareExpression,
) -> Box<dyn Fn(T, T) -> bool> {
    match expr {
        CompareExpression::Equals => Box::new(|l, r| l == r),
        CompareExpression::NotEquals => Box::new(|l, r| l != r),
        CompareExpression::LessThan => Box::new(|l, r| l < r),
        CompareExpression::LessThanOrEquals => Box::new(|l, r| l <= r),
        CompareExpression::GreaterThan => Box::new(|l, r| l > r),
        CompareExpression::GreaterThanOrEquals => Box::new(|l, r| l >= r),
    }
}

fn count_matches<T: PartialEq + PartialOrd + Copy>(
    data: impl Iterator<Item = T>,
    cmp: &dyn Fn(T, T) -> bool,
    ref_value: T,
) -> HashSet<usize> {
    data.enumerate()
        .filter_map(|(idx, value)| {
            if cmp(value, ref_value) {
                Some(idx)
            } else {
                None
            }
        })
        .collect()
}

fn eval_compare_query<'a>(
    expr: &CompareExpression,
    value: &Value,
    points: &'a impl BorrowedBuffer<'a>,
) -> HashSet<usize> {
    match value {
        Value::Classification(class) => {
            let data = points.view_attribute::<u8>(&CLASSIFICATION);
            let cmp = get_compare_func::<u8>(expr);
            count_matches(data.into_iter(), cmp.as_ref(), class.0)
        }
        Value::Position(position) => {
            let data = points.view_attribute::<Vector3<f64>>(&POSITION_3D);
            let cmp = get_compare_func::<Vector3<f64>>(expr);
            count_matches(data.into_iter(), cmp.as_ref(), position.0)
        }
        Value::ReturnNumber(return_number) => {
            let data = points.view_attribute::<u8>(&RETURN_NUMBER);
            let cmp = get_compare_func::<u8>(expr);
            count_matches(data.into_iter(), cmp.as_ref(), return_number.0)
        }
        Value::NumberOfReturns(nr_of_returns) => {
            let data = points.view_attribute::<u8>(&NUMBER_OF_RETURNS);
            let cmp = get_compare_func::<u8>(expr);
            count_matches(data.into_iter(), cmp.as_ref(), nr_of_returns.0)
        }
        Value::GpsTime(gps_time) => {
            let data = points.view_attribute::<f64>(&GPS_TIME);
            let cmp = get_compare_func::<f64>(expr);
            count_matches(data.into_iter(), cmp.as_ref(), gps_time.0)
        }
        other => panic!("Unsupported value {other} in compare query"),
    }
}

fn eval_intersect_query<'a>(
    geometry: &Geometry,
    points: &'a impl BorrowedBuffer<'a>,
) -> HashSet<usize> {
    match geometry {
        Geometry::Polygon(polygon) => {
            let positions = points.view_attribute::<Vector3<f64>>(&POSITION_3D);
            positions
                .into_iter()
                .enumerate()
                .filter_map(|(idx, position)| {
                    let geo_pos = geo::point! { x: position.x, y: position.y };
                    if polygon.contains(&geo_pos) {
                        Some(idx)
                    } else {
                        None
                    }
                })
                .collect()
        }
    }
}

fn eval_within_query<'a>(
    range: &Range<Value>,
    points: &'a impl BorrowedBuffer<'a>,
) -> HashSet<usize> {
    match (range.start, range.end) {
        (Value::Classification(start), Value::Classification(end)) => {
            let classifications = points.view_attribute::<u8>(&CLASSIFICATION);
            classifications
                .into_iter()
                .enumerate()
                .filter_map(|(idx, class)| {
                    if class < start.0 || class > end.0 {
                        None
                    } else {
                        Some(idx)
                    }
                })
                .collect()
        }
        (Value::Position(min), Value::Position(max)) => {
            let bounds = AABB::from_min_max(min.0.into(), max.0.into());
            let positions = points.view_attribute::<Vector3<f64>>(&POSITION_3D);
            positions
                .into_iter()
                .enumerate()
                .filter_map(|(idx, pos)| {
                    if bounds.contains(&pos.into()) {
                        Some(idx)
                    } else {
                        None
                    }
                })
                .collect()
        }
        (Value::ReturnNumber(min), Value::ReturnNumber(max)) => {
            let data = points.view_attribute::<u8>(&RETURN_NUMBER);
            data.into_iter()
                .enumerate()
                .filter_map(|(idx, value)| {
                    if value < min.0 || value > max.0 {
                        None
                    } else {
                        Some(idx)
                    }
                })
                .collect()
        }
        (Value::NumberOfReturns(min), Value::NumberOfReturns(max)) => {
            let data = points.view_attribute::<u8>(&NUMBER_OF_RETURNS);
            data.into_iter()
                .enumerate()
                .filter_map(|(idx, value)| {
                    if value < min.0 || value > max.0 {
                        None
                    } else {
                        Some(idx)
                    }
                })
                .collect()
        }
        (Value::GpsTime(min), Value::GpsTime(max)) => {
            let data = points.view_attribute::<f64>(&GPS_TIME);
            data.into_iter()
                .enumerate()
                .filter_map(|(idx, value)| {
                    if value < min.0 || value > max.0 {
                        None
                    } else {
                        Some(idx)
                    }
                })
                .collect()
        }
        (other_low, other_high) => panic!(
            "Invalid parameters for WITHIN query ({} and {})",
            other_low, other_high
        ),
    }
}

/// Counts the number of matching points within `points` for the given `query`
fn get_matching_indices<'a>(
    query: &QueryExpression,
    points: &'a impl BorrowedBuffer<'a>,
) -> HashSet<usize> {
    match query {
        QueryExpression::Atomic(atom) => match atom {
            AtomicExpression::Compare((expr, value)) => eval_compare_query(expr, value, points),
            AtomicExpression::Intersects(geometry) => eval_intersect_query(geometry, points),
            AtomicExpression::Within(range) => eval_within_query(range, points),
        },
        QueryExpression::And(l, r) => {
            let l_matches = get_matching_indices(l.as_ref(), points);
            let r_matches = get_matching_indices(r.as_ref(), points);
            l_matches.intersection(&r_matches).copied().collect()
        }
        QueryExpression::Or(l, r) => {
            let l_matches = get_matching_indices(l.as_ref(), points);
            let r_matches = get_matching_indices(r.as_ref(), points);
            l_matches.union(&r_matches).copied().collect()
        }
    }
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

/// Counts the number of matching points for the `query` in the dataset pointed to by `path`. This should be a directory
/// containing LAS files
fn apply_query_to_las_dataset(query: &QueryExpression, files: &[PathBuf]) -> Result<usize> {
    files
        .into_par_iter()
        .map(|file| -> Result<usize> {
            let mut las_reader = LASReader::from_path(&file, false)?;
            let points = las_reader.read::<VectorBuffer>(las_reader.remaining_points())?;
            let matching_indices = get_matching_indices(query, &points);
            Ok(matching_indices.len())
        })
        .sum::<Result<usize>>()
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

fn test_queries_doc(shapefile_path: &Path) -> Vec<QueryExpression> {
    let aabb_large = QueryExpression::Atomic(AtomicExpression::Within(
        Value::Position(Position(Vector3::new(390000.0, 130000.0, 0.0)))
            ..Value::Position(Position(Vector3::new(400000.0, 140000.0, 200.0))),
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

    let everything_not_unclassified = QueryExpression::Atomic(AtomicExpression::Compare((
        CompareExpression::GreaterThan,
        Value::Classification(Classification(0)),
    )));

    let first_returns = QueryExpression::Atomic(AtomicExpression::Compare((
        CompareExpression::Equals,
        Value::ReturnNumber(ReturnNumber(1)),
    )));

    let first_returns_with_less_than = QueryExpression::Atomic(AtomicExpression::Compare((
        CompareExpression::LessThan,
        Value::ReturnNumber(ReturnNumber(2)),
    )));

    let gps_time_range = QueryExpression::Atomic(AtomicExpression::Within(
        Value::GpsTime(GpsTime(207011500.0))..Value::GpsTime(GpsTime(207012000.0)),
    ));

    let polygon_small = crude_shapefile_parsing(shapefile_path).expect("Could not parse shapefile");

    let buildings_and_first_returns = QueryExpression::And(
        Box::new(class_buildings.clone()),
        Box::new(first_returns.clone()),
    );
    let time_range_or_bounds = QueryExpression::Or(
        Box::new(gps_time_range.clone()),
        Box::new(aabb_large.clone()),
    );

    vec![
        aabb_large,
        aabb_all,
        aabb_no_matches,
        class_buildings,
        everything_not_unclassified,
        first_returns,
        first_returns_with_less_than,
        polygon_small,
        gps_time_range,
        buildings_and_first_returns,
        time_range_or_bounds,
    ]
}

fn assert_correctness_with_doc_dataset() -> Result<()> {
    let args = Args::parse();

    let queries = test_queries_doc(&args.shapefile_path);

    let las_files = get_files_with_extension("las", &args.data_path.join("doc/las"));
    let last_files = get_files_with_extension("last", &args.data_path.join("doc/last"));
    let laz_files = get_files_with_extension("laz", &args.data_path.join("doc/laz"));
    let lazer_files = get_files_with_extension("lazer", &args.data_path.join("doc/lazer"));

    struct ReportEntry {
        file_format: &'static str,
        expected_matches: usize,
        actual_matches: usize,
    }
    let mut reports_by_query: Vec<(QueryExpression, Vec<ReportEntry>)> = vec![];

    for query in queries {
        let query_clone = query.clone();
        let expected_num_matches = apply_query_to_las_dataset(&query, &las_files)?;
        let report_entries = [
            (las_files.as_slice(), "las"),
            (last_files.as_slice(), "last"),
            (laz_files.as_slice(), "laz"),
            (lazer_files.as_slice(), "lazer"),
        ]
        .into_par_iter()
        .map(move |(files, extension)| -> Result<ReportEntry> {
            let mut progressive_index = ProgressiveIndex::new();
            let dataset_id = progressive_index.add_dataset(&files)?;

            let output = CountOutput::default();
            progressive_index.query(dataset_id, query.clone(), &NoRefinementStrategy, &output)?;

            Ok(ReportEntry {
                file_format: extension,
                expected_matches: expected_num_matches,
                actual_matches: output.count(),
            })
        })
        .collect::<Result<Vec<_>>>()?;

        reports_by_query.push((query_clone, report_entries));
    }

    eprintln!("Report - District of Columbia dataset");
    for (query, entries) in reports_by_query {
        eprintln!("Query: {query}");
        for entry in entries {
            if entry.actual_matches == entry.expected_matches {
                eprintln!(
                    "{} {} ({} matches)",
                    entry.file_format,
                    "correct".green(),
                    entry.actual_matches
                );
            } else {
                eprintln!(
                    "{} {}! Expected {} matches but found {} matches",
                    entry.file_format,
                    "incorrect".red(),
                    entry.expected_matches,
                    entry.actual_matches
                );
            }
        }
    }

    Ok(())
}

fn main() -> Result<()> {
    let _client = tracy_client::Client::start();
    assert_correctness_with_doc_dataset()
}
