use std::{ops::Range, path::Path};

use anyhow::Context;
use anyhow::Result;
use pasture_core::{math::AABB, nalgebra::Point3};

use query::index::NoRefinementStrategy;
use query::io::InMemoryOutput;
use query::{self, index::ProgressiveIndex};
use scopeguard::defer;

use crate::common::{
    assert_points_match, gen_random_points, get_sorted_points_from_collector,
    run_query_and_sort_result, setup_query, split_into_chunks, write_as_las,
};

mod common;

// Tests could be as 'simple' as a set of input/output LAS files, together with a query. The output LAS files
// could be generated with a baseline tool, e.g. LAStools
// To compare the files, we should use point IDs and sort the output files by point IDs

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

    let output = InMemoryOutput::default();
    indexer
        .query(dataset_id, query, &NoRefinementStrategy, &output)
        .context("Query failed")?;

    let combined_data = get_sorted_points_from_collector(output);

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
