use std::{fs::File, io::BufReader};

use anyhow::Result;
use io::{last::LASTReader, lazer::LazerReader};
use pasture_core::containers::{
    BorrowedBuffer, HashMapBuffer, MakeBufferFromLayout, OwningBuffer, VectorBuffer,
};
use pasture_io::base::{read_all, PointReader};

const LAS_PATH: &str =
    "/Users/pbormann/data/geodata/pointclouds/datasets/district_of_columbia/1321_1.las";
const LAST_PATH: &str =
    "/Users/pbormann/data/projects/progressive_indexing/experiment_data/doc/last/1321_1.last";
const LAZER_PATH: &str =
    "/Users/pbormann/data/projects/progressive_indexing/experiment_data/doc/lazer/1321_1.lazer";

fn main() -> Result<()> {
    let las_points = read_all::<VectorBuffer, _>(LAS_PATH)?;

    let last_points = {
        let mut last_reader = LASTReader::from_read(BufReader::new(File::open(LAST_PATH)?))?;
        let mut buffer = HashMapBuffer::new_from_layout(las_points.point_layout().clone());
        buffer.resize(last_reader.remaining_points());
        last_reader.read_into(&mut buffer, last_reader.remaining_points())?;
        buffer
    };

    let lazer_points = {
        let mut lazer_reader = LazerReader::new(BufReader::new(File::open(LAZER_PATH)?))?;
        let mut buffer = HashMapBuffer::new_from_layout(las_points.point_layout().clone());
        buffer.resize(lazer_reader.remaining_points());
        lazer_reader.read_into(&mut buffer, lazer_reader.remaining_points())?;
        buffer
    };

    assert_eq!(las_points.len(), last_points.len());
    assert_eq!(las_points.len(), lazer_points.len());

    let mut expected_point = vec![0; las_points.point_layout().size_of_point_entry() as usize];
    let mut actual_point = expected_point.clone();
    for point_index in 0..las_points.len() {
        las_points.get_point(point_index, &mut expected_point);
        last_points.get_point(point_index, &mut actual_point);
        assert_eq!(
            expected_point, actual_point,
            "LAST point at index {point_index} is different"
        );

        lazer_points.get_point(point_index, &mut expected_point);
        assert_eq!(
            expected_point, actual_point,
            "LASZER point at index {point_index} is different"
        );
    }

    Ok(())
}
