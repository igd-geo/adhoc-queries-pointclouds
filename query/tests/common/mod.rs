use std::{cmp::min, ops::Range, path::Path};

use anyhow::Context;
use anyhow::Result;
use pasture_core::containers::{
    BorrowedBuffer, BorrowedMutBuffer, InterleavedBuffer, MakeBufferFromLayout, OwningBuffer,
    VectorBuffer,
};
use pasture_core::layout::attributes::{CLASSIFICATION, GPS_TIME, POSITION_3D};
use pasture_core::{
    layout::PointType,
    math::AABB,
    nalgebra::{Point3, Vector3},
};
use pasture_derive::PointType;
use pasture_io::base::PointReader;
use pasture_io::las::LASReader;
use pasture_io::las_rs::{Builder, GpsTimeType};
use pasture_io::{base::PointWriter, las::LASWriter};
use query::index::{Classification, Position};
use query::index::{DatasetID, Query};
use query::index::{NoRefinementStrategy, Value};
use query::io::InMemoryOutput;
use query::{self, index::ProgressiveIndex};
use rand::{thread_rng, Rng};

#[derive(PointType, Debug, Copy, Clone, bytemuck::AnyBitPattern, bytemuck::NoUninit)]
#[repr(C, packed)]
pub struct Point {
    #[pasture(BUILTIN_POSITION_3D)]
    pub position: Vector3<f64>,
    #[pasture(BUILTIN_CLASSIFICATION)]
    pub classification: u8,
    // Hijack GPS time for point IDs due to its high precision
    #[pasture(BUILTIN_GPS_TIME)]
    pub id: f64,
}

pub const CLASSIFICATION_RANGE: Range<u8> = 0..4;
pub const POSITION_RANGE: Range<i32> = 0..1024;

/// Generate `count` random points
pub fn gen_random_points(count: usize) -> VectorBuffer {
    let mut buffer = VectorBuffer::with_capacity(count, Point::layout());
    let mut rng = thread_rng();

    for id in 0..count {
        buffer.view_mut().push_point(Point {
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
pub fn split_into_chunks(points: VectorBuffer, num_chunks: usize) -> Vec<VectorBuffer> {
    let mut ret = vec![];

    let points_per_chunk = (points.len() + num_chunks - 1) / num_chunks;
    for chunk in 0..num_chunks {
        let chunk_start = chunk * points_per_chunk;
        let chunk_end = min((chunk + 1) * points_per_chunk, points.len());

        let mut chunk_buffer =
            VectorBuffer::with_capacity(chunk_end - chunk_start, points.point_layout().clone());
        // Safe because buffers share the same PointLayout
        unsafe {
            chunk_buffer.push_points(points.get_point_range_ref(chunk_start..chunk_end));
        }
        ret.push(chunk_buffer);
    }

    ret
}

/// Writes the given point data into a LAS file and returns the data from that LAS file after reading. This is done
/// to make sure that the PointLayout of the query result matches the PointLayout of the reference data
pub fn write_as_las(points: &VectorBuffer, path: &Path) -> Result<VectorBuffer> {
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
    let mut ret = VectorBuffer::with_capacity(points.len(), points.point_layout().clone());
    reader
        .read_into(&mut ret, points.len())
        .context("Error while reading points from LAS file")?;
    Ok(ret)
}

/// Reference implementation for a point query. This is used to compare the progressive indexer query implementation
/// to a correct baseline
pub fn get_matching_points_reference(
    points: &VectorBuffer,
    bounds: Option<AABB<f64>>,
    classification_range: Option<Range<u8>>,
) -> VectorBuffer {
    let mut ret = VectorBuffer::new_from_layout(points.point_layout().clone());
    for point in points.view::<Point>().iter() {
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

        ret.view_mut().push_point(*point);
    }

    ret
}

/// Converts the data in `points` into the `Point` type
pub fn convert_to_point_type(points: &VectorBuffer) -> VectorBuffer {
    let mut ret = VectorBuffer::with_capacity(points.len(), Point::layout());
    for idx in 0..points.len() {
        let position = points.view_attribute::<Vector3<f64>>(&POSITION_3D).at(idx);
        let classification = points.view_attribute::<u8>(&CLASSIFICATION).at(idx);
        let id = points.view_attribute::<f64>(&GPS_TIME).at(idx);
        ret.view_mut().push_point(Point {
            classification,
            id,
            position,
        });
    }

    ret
}

/// Extract the points from the given collector, sort them by their PointID and return them as a buffer
pub fn get_sorted_points_from_collector(collector: InMemoryOutput) -> VectorBuffer {
    let combined_data = collector
        .into_single_buffer()
        .expect("Query returned no data");

    let mut combined_data = convert_to_point_type(&combined_data);

    // Sort ascending by point ID
    combined_data.view_mut().sort_by(|a: &Point, b: &Point| {
        let a_id = a.id;
        let b_id = b.id;
        a_id.total_cmp(&b_id)
    });

    combined_data
}

/// Asserts that the given set of points match. Assumes that both buffers are sorted in ascending order based
/// on the .id field
pub fn assert_points_match(expected: &VectorBuffer, actual: &VectorBuffer) {
    // assert_eq!(expected.len(), actual.len());
    for (expected_point, actual_point) in expected
        .view::<Point>()
        .into_iter()
        .zip(actual.view::<Point>().into_iter())
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
            let unexpected_point = actual.view::<Point>().at(expected.len());
            assert!(
                false,
                "Query result has unexpected point {:#?} at index {}",
                unexpected_point,
                expected.len()
            );
        } else {
            let missing_point = expected.view::<Point>().at(actual.len());
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
pub fn setup_query(
    points: &VectorBuffer,
    bounds: Option<AABB<f64>>,
    classification_range: Option<Range<u8>>,
) -> (VectorBuffer, Query) {
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
pub fn run_query_and_sort_result(
    query: Query,
    indexer: &mut ProgressiveIndex,
    dataset_id: DatasetID,
) -> Result<VectorBuffer> {
    let output = InMemoryOutput::default();
    indexer
        .query(dataset_id, query, &NoRefinementStrategy, &output)
        .context("Query failed")?;

    let combined_data = output
        .into_single_buffer()
        .context("Query returned no data")?;

    let mut combined_data = convert_to_point_type(&combined_data);

    // Sort ascending by point ID
    combined_data.view_mut().sort_by(|a: &Point, b: &Point| {
        let a_id = a.id;
        let b_id = b.id;
        a_id.total_cmp(&b_id)
    });

    Ok(combined_data)
}
