use std::sync::{Arc, Mutex};
use std::{cmp::min, ops::Range, path::Path};

use anyhow::Context;
use anyhow::Result;
use pasture_core::containers::PointBufferExt;
use pasture_core::layout::attributes::{CLASSIFICATION, GPS_TIME, POSITION_3D};
use pasture_core::{
    containers::{InterleavedPointBufferExt, InterleavedVecPointStorage, PointBuffer},
    layout::PointType,
    math::AABB,
    nalgebra::{Point3, Vector3},
};
use pasture_derive::PointType;
use pasture_io::base::PointReader;
use pasture_io::las::LASReader;
use pasture_io::las_rs::{Builder, GpsTimeType};
use pasture_io::{base::PointWriter, las::LASWriter};
use query::collect_points::BufferCollector;
use query::index::Value;
use query::index::{Classification, Position};
use query::index::{DatasetID, Query};
use query::{self, index::ProgressiveIndex};
use rand::{thread_rng, Rng};
use scopeguard::defer;

// Tests could be as 'simple' as a set of input/output LAS files, together with a query. The output LAS files
// could be generated with a baseline tool, e.g. LAStools
// To compare the files, we should use point IDs and sort the output files by point IDs

#[derive(PointType, Debug, Copy, Clone)]
#[repr(C, packed)]
struct Point {
    #[pasture(BUILTIN_POSITION_3D)]
    pub position: Vector3<f64>,
    #[pasture(BUILTIN_CLASSIFICATION)]
    pub classification: u8,
    // Hijack GPS time for point IDs due to its high precision
    #[pasture(BUILTIN_GPS_TIME)]
    pub id: f64,
}

const CLASSIFICATION_RANGE: Range<u8> = 0..4;
const POSITION_RANGE: Range<i32> = 0..1024;

/// Generate `count` random points
fn gen_random_points(count: usize) -> InterleavedVecPointStorage {
    let mut buffer = InterleavedVecPointStorage::with_capacity(count, Point::layout());
    let mut rng = thread_rng();

    for id in 0..count {
        buffer.push_point(Point {
            classification: rng.gen_range(CLASSIFICATION_RANGE),
            id: id as f64,
            position: Vector3::new(
                // Using i32 values, but cast to f64, together with a LAS scale of 1.0 to prevent rounding problems
                rng.gen_range(POSITION_RANGE) as f64,
                rng.gen_range(POSITION_RANGE) as f64,
                rng.gen_range(POSITION_RANGE) as f64,
            ),
        });
    }

    buffer
}

/// Splits the given points into chunks
fn split_into_chunks(
    points: InterleavedVecPointStorage,
    num_chunks: usize,
) -> Vec<InterleavedVecPointStorage> {
    let mut ret = vec![];

    let points_per_chunk = (points.len() + num_chunks - 1) / num_chunks;
    for chunk in 0..num_chunks {
        let chunk_start = chunk * points_per_chunk;
        let chunk_end = min((chunk + 1) * points_per_chunk, points.len());

        let mut chunk_buffer = InterleavedVecPointStorage::with_capacity(
            chunk_end - chunk_start,
            points.point_layout().clone(),
        );
        chunk_buffer.push_points(points.get_points_ref::<Point>(chunk_start..chunk_end));
        ret.push(chunk_buffer);
    }

    ret
}

/// Writes the given point data into a LAS file and returns the data from that LAS file after reading. This is done
/// to make sure that the PointLayout of the query result matches the PointLayout of the reference data
fn write_as_las(
    points: &InterleavedVecPointStorage,
    path: &Path,
) -> Result<InterleavedVecPointStorage> {
    let mut las_header_builder = Builder::from((1, 4));
    las_header_builder.point_format.has_gps_time = true;
    las_header_builder.gps_time_type = GpsTimeType::Standard;
    las_header_builder.transforms.x.scale = 1.0;
    las_header_builder.transforms.y.scale = 1.0;
    las_header_builder.transforms.z.scale = 1.0;

    {
        let mut writer = LASWriter::from_path_and_header(
            path,
            las_header_builder
                .into_header()
                .expect("Failed to build LAS header"),
        )
        .context("Could not create LAS writer")?;
        writer.write(points).context("Failed to write points")?;
        writer.flush().context("Failed to flush LAS writer")?;
    }

    let mut reader = LASReader::from_path(path).context("Can't open LAS file")?;
    let mut ret =
        InterleavedVecPointStorage::with_capacity(points.len(), points.point_layout().clone());
    reader
        .read_into(&mut ret, points.len())
        .context("Error while reading points from LAS file")?;
    Ok(ret)
}

/// Reference implementation for a point query. This is used to compare the progressive indexer query implementation
/// to a correct baseline
fn get_matching_points_reference(
    points: &InterleavedVecPointStorage,
    bounds: Option<AABB<f64>>,
    classification_range: Option<Range<u8>>,
) -> InterleavedVecPointStorage {
    let mut ret = InterleavedVecPointStorage::new(points.point_layout().clone());
    for point in points.iter_point_ref::<Point>() {
        if let Some(ref bounds) = bounds {
            let pos = point.position;
            if !bounds.contains(&Point3::new(pos[0], pos[1], pos[2])) {
                continue;
            }
        }

        if let Some(ref classification_range) = classification_range {
            if !classification_range.contains(&point.classification) {
                continue;
            }
        }

        ret.push_point(*point);
    }

    ret
}

/// Converts the data in `points` into the `Point` type
fn convert_to_point_type(points: &InterleavedVecPointStorage) -> InterleavedVecPointStorage {
    let mut ret = InterleavedVecPointStorage::with_capacity(points.len(), Point::layout());
    for idx in 0..points.len() {
        let position = points.get_attribute::<Vector3<f64>>(&POSITION_3D, idx);
        let classification = points.get_attribute::<u8>(&CLASSIFICATION, idx);
        let id = points.get_attribute::<f64>(&GPS_TIME, idx);
        ret.push_point(Point {
            classification,
            id,
            position,
        });
    }

    ret
}

/// Asserts that the given set of points match. Assumes that both buffers are sorted in ascending order based
/// on the .id field
fn assert_points_match(expected: &InterleavedVecPointStorage, actual: &InterleavedVecPointStorage) {
    // assert_eq!(expected.len(), actual.len());
    for (expected_point, actual_point) in expected
        .iter_point_ref::<Point>()
        .zip(actual.iter_point_ref::<Point>())
    {
        let expected_id = expected_point.id;
        let actual_id = actual_point.id;
        assert_eq!(
            expected_id, actual_id,
            "Expected point {:#?} but found point {:#?}",
            expected_point, actual_point
        );
    }

    if expected.len() != actual.len() {
        if expected.len() < actual.len() {
            let unexpected_point = actual.get_point::<Point>(expected.len());
            assert!(
                false,
                "Query result has unexpected point {:#?} at index {}",
                unexpected_point,
                expected.len()
            );
        } else {
            let missing_point = expected.get_point::<Point>(actual.len());
            assert!(
                false,
                "Query result has missing point {:#?} at index {}",
                missing_point,
                actual.len()
            );
        };
    }
}

/// Shorthand for creating a Query object AND returning the expected points based on this query
fn setup_query(
    points: &InterleavedVecPointStorage,
    bounds: Option<AABB<f64>>,
    classification_range: Option<Range<u8>>,
) -> (InterleavedVecPointStorage, Query) {
    let expected_result =
        get_matching_points_reference(points, bounds, classification_range.clone());

    let bounds_query = bounds.map(|bounds| {
        Query::Within(
            Value::Position(Position(bounds.min().coords))
                ..Value::Position(Position(bounds.max().coords)),
        )
    });

    let classification_query = classification_range.map(|classification_range| {
        Query::Within(
            Value::Classification(Classification(classification_range.start))
                ..Value::Classification(Classification(classification_range.end)),
        )
    });

    let query = match (bounds_query, classification_query) {
        (Some(a), Some(b)) => Query::And(Box::new(a), Box::new(b)),
        (None, Some(b)) => b,
        (Some(a), _) => a,
        _ => panic!("No empty query allowed!"),
    };

    (expected_result, query)
}

/// Helper to run a query on a ProgressiveIndex and return the resulting points sorted in ascending order by point ID
fn run_query_and_sort_result(
    query: Query,
    indexer: &mut ProgressiveIndex,
    dataset_id: DatasetID,
) -> Result<InterleavedVecPointStorage> {
    let collector = Arc::new(Mutex::new(BufferCollector::new()));
    indexer
        .query(dataset_id, query, collector.clone())
        .context("Query failed")?;

    let collector = Arc::try_unwrap(collector)
        .ok()
        .expect("Can't take ownership of result collector")
        .into_inner()
        .expect("Mutex was poisoned");
    let combined_data = collector
        .as_single_buffer()
        .context("Query returned no data")?;

    let mut combined_data = convert_to_point_type(&combined_data);

    // Sort ascending by point ID
    combined_data.sort_by(|a: &Point, b: &Point| {
        let a_id = a.id;
        let b_id = b.id;
        a_id.total_cmp(&b_id)
    });

    Ok(combined_data)
}

fn _assert_is_sorted(points: &InterleavedVecPointStorage) {
    for (p1, p2) in points
        .iter_point_ref::<Point>()
        .zip(points.iter_point_ref::<Point>().skip(1))
    {
        assert!(p1.id < p2.id);
    }
}

#[test]
fn test_basic_query() -> Result<()> {
    const COUNT: usize = 1 << 12;
    let test_data = gen_random_points(COUNT);

    let file_path = Path::new("input.las");
    write_as_las(&test_data, file_path).context("Failed to write test data")?;
    defer! {
        std::fs::remove_file(file_path).expect("Failed to cleanup temporary files");
    }

    let mut indexer = ProgressiveIndex::new();
    let dataset_id = indexer
        .add_dataset(&[file_path])
        .context("Failed to add dataset to ProgressiveIndex")?;

    let bounds = AABB::from_min_max(Point3::new(1.0, 2.0, 3.0), Point3::new(2.0, 3.0, 4.0));
    let classification_range: Range<u8> = 2..3;

    let (expected_result, query) =
        setup_query(&test_data, Some(bounds), Some(classification_range));

    let collector = Arc::new(Mutex::new(BufferCollector::new()));
    indexer
        .query(dataset_id, query, collector.clone())
        .context("Query failed")?;

    let collector = Arc::try_unwrap(collector)
        .ok()
        .expect("Can't take ownership of result collector")
        .into_inner()
        .expect("Mutex was poisoned");
    let combined_data = collector
        .as_single_buffer()
        .context("Query returned no data")?;

    let mut combined_data = convert_to_point_type(&combined_data);

    // Sort ascending by point ID
    combined_data.sort_by(|a: &Point, b: &Point| {
        let a_id = a.id;
        let b_id = b.id;
        a_id.total_cmp(&b_id)
    });

    assert_points_match(&expected_result, &combined_data);

    Ok(())
}

#[test]
fn test_multifile_query() -> Result<()> {
    const COUNT: usize = 1 << 16;
    let test_data = gen_random_points(COUNT);

    let (expected_points, query) = setup_query(
        &test_data,
        Some(AABB::from_min_max(
            Point3::new(1.0, 2.0, 3.0),
            Point3::new(2.0, 3.0, 4.0),
        )),
        Some(2..4),
    );

    let test_data_splitted = split_into_chunks(test_data, 8);
    let file_names = test_data_splitted
        .iter()
        .enumerate()
        .map(|(idx, _)| format!("input_{}.las", idx))
        .collect::<Vec<_>>();

    for (points, file_name) in test_data_splitted.iter().zip(file_names.iter()) {
        write_as_las(points, Path::new(file_name.as_str()))
            .context("Could not write temporary file")?;
    }

    defer! {
        for file_name in &file_names {
            std::fs::remove_file(file_name).expect("Could not remove temporary file");
        }
    }

    let mut indexer = ProgressiveIndex::new();
    let dataset_id = indexer
        .add_dataset(file_names.as_slice())
        .context("Failed to add dataset to ProgressiveIndex")?;

    let actual_query_result =
        run_query_and_sort_result(query, &mut indexer, dataset_id).context("Query failed")?;

    assert_points_match(&expected_points, &actual_query_result);

    Ok(())
}
