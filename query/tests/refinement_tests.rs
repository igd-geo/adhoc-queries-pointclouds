use std::{ops::Range, path::Path};

use anyhow::{Context, Result};
use pasture_core::{math::AABB, nalgebra::Point3};
use query::{
    index::{ProgressiveIndex, RefinementStrategy, ValueType},
    io::InMemoryOutput,
};
use scopeguard::defer;

use crate::common::{
    assert_points_match, gen_random_points, get_sorted_points_from_collector, setup_query,
    write_as_las,
};

mod common;

/// Refinement strategy that always refines all candidate blocks
struct RefinementStrategyAlways;

impl RefinementStrategy for RefinementStrategyAlways {
    fn select_best_candidates(
        &self,
        potential_refinements: rustc_hash::FxHashSet<query::index::PointRange>,
        _value_type: ValueType,
        _dataset_id: query::index::DatasetID,
        _input_layer: &query::io::InputLayer,
    ) -> Vec<query::index::PointRange> {
        potential_refinements.into_iter().collect()
    }
}

#[test]
fn test_refinement_position_index() -> Result<()> {
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

    let (_expected_result, query) = setup_query(&test_data, Some(bounds), None);

    // This query will find one matching range of positions, because the ProgressiveIndex always starts out with a basic
    // PositionIndex covering the whole file. This range of points is large enough to be refined into 4 sub-blocks, so this
    // is what we expect after we run the query: 4 blocks instead of 1

    let output = InMemoryOutput::default();
    indexer
        .query(dataset_id, query, &RefinementStrategyAlways, &output)
        .context("Error while querying")?;

    let dataset = indexer.datasets().get(&dataset_id).unwrap();
    let position_index = dataset
        .indices()
        .get(&ValueType::Position3D)
        .expect("Expected dataset to have an index over positions");

    // It's probably not great to check the EXACT number of new blocks, as that might change depending on the
    // refinement implementation. But we can check some reasonable invariants:
    // - there should be more blocks than before
    // - the blocks must not overlap
    // - each new block must be smaller than the original block (`COUNT` in our case)
    // - each block should have an index
    assert!(position_index.blocks_count() > 1);
    for block in position_index.blocks() {
        assert!(block.point_range().points_in_file.len() < COUNT);
        assert!(block.index().is_some());
    }

    Ok(())
}

#[test]
fn test_running_query_multiple_times_gives_identical_results() -> Result<()> {
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

    let output_first_query = InMemoryOutput::default();
    indexer
        .query(
            dataset_id,
            query.clone(),
            &RefinementStrategyAlways,
            &output_first_query,
        )
        .context("Error while querying")?;

    let output_second_query = InMemoryOutput::default();
    indexer
        .query(
            dataset_id,
            query.clone(),
            &RefinementStrategyAlways,
            &output_second_query,
        )
        .context("Error while querying")?;

    let first_query_points = get_sorted_points_from_collector(output_first_query);
    let second_query_points = get_sorted_points_from_collector(output_second_query);

    assert_points_match(&expected_result, &first_query_points);
    assert_points_match(&expected_result, &second_query_points);

    Ok(())
}
