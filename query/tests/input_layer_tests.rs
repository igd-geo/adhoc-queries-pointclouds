use anyhow::{anyhow, Result};
use pasture_core::containers::BorrowedBuffer;
use query::{
    index::PointRange,
    io::{FileHandle, InputLayer, PointDataMemoryLayout},
};

const TEST_LAS_FILE: &str =
    "/Users/pbormann/data/geodata/pointclouds/datasets/district_of_columbia/1120_1.las";
const TEST_LAST_FILE: &str =
    "/Users/pbormann/data/projects/progressive_indexing/experiment_data/doc/last/1120_1.last";
const TEST_LAZ_FILE: &str =
    "/Users/pbormann/data/projects/progressive_indexing/experiment_data/doc/laz/1120_1.laz";

#[test]
fn test_input_layer_format_equivalence() -> Result<()> {
    let mut input_layer = InputLayer::default();
    let las_dataset_id = 1;
    let laz_dataset_id = 2;
    let last_dataset_id = 3;
    input_layer.add_files(&[TEST_LAS_FILE], las_dataset_id)?;
    input_layer.add_files(&[TEST_LAZ_FILE], laz_dataset_id)?;
    input_layer.add_files(&[TEST_LAST_FILE], last_dataset_id)?;

    let file_meta = input_layer
        .get_las_metadata(FileHandle(las_dataset_id, 0))
        .ok_or(anyhow!("LAS metadata not found"))?;
    let num_points = file_meta.point_count();

    // Assert the equivalence of the data for LAS, LAZ, and LAST file formats when using the InputLayer
    let las_data = input_layer.get_point_data(
        las_dataset_id,
        PointRange::new(0, 0..num_points),
        PointDataMemoryLayout::Interleaved,
    )?;
    let laz_data = input_layer.get_point_data(
        laz_dataset_id,
        PointRange::new(0, 0..num_points),
        PointDataMemoryLayout::Interleaved,
    )?;
    let last_data = input_layer.get_point_data(
        last_dataset_id,
        PointRange::new(0, 0..num_points),
        PointDataMemoryLayout::Columnar,
    )?;

    assert_eq!(las_data.point_layout(), laz_data.point_layout());
    assert_eq!(las_data.point_layout(), last_data.point_layout());

    let mut raw_point_buffer1 = vec![0; las_data.point_layout().size_of_point_entry() as usize];
    let mut raw_point_buffer2 = raw_point_buffer1.clone();

    for point_idx in 0..num_points {
        las_data.get_point(point_idx, &mut raw_point_buffer1);
        laz_data.get_point(point_idx, &mut raw_point_buffer2);

        assert_eq!(
            raw_point_buffer1, raw_point_buffer2,
            "LAZ point at index {point_idx} is different from expected LAS point"
        );

        last_data.get_point(point_idx, &mut raw_point_buffer2);
        assert_eq!(
            raw_point_buffer1, raw_point_buffer2,
            "LAST point at index {point_idx} is different from expected LAS point"
        );
    }

    Ok(())
}
