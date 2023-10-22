use std::{fs::File, io::BufReader};

use anyhow::{bail, Result};
use human_repr::HumanCount;
use io::{last::LASTReader, lazer::LazerReader};
use log::info;
use pasture_core::containers::{BorrowedBuffer, HashMapBuffer, OwningBuffer, VectorBuffer};
use pasture_io::{base::PointReader, las::LASReader};

const LAS_PATH: &str =
    "/Users/pbormann/data/projects/progressive_indexing/experiment_data/ahn4s/las/C_25GN1.las";
const LAST_PATH: &str =
    "/Users/pbormann/data/projects/progressive_indexing/experiment_data/ahn4s/last/C_25GN1.last";
const LAZER_PATH: &str =
    "/Users/pbormann/data/projects/progressive_indexing/experiment_data/ahn4s/lazer/C_25GN1.lazer";

fn main() -> Result<()> {
    pretty_env_logger::init();

    let mut las_reader = LASReader::from_path(LAS_PATH, true)?;
    let mut last_reader = LASTReader::from_read(BufReader::new(File::open(LAST_PATH)?))?;
    let mut lazer_reader = LazerReader::new(BufReader::new(File::open(LAZER_PATH)?))?;

    const CAPACITY: usize = 1_000_000;
    let mut las_buffer =
        VectorBuffer::with_capacity(CAPACITY, las_reader.get_default_point_layout().clone());
    las_buffer.resize(CAPACITY);
    let mut last_buffer =
        HashMapBuffer::with_capacity(CAPACITY, last_reader.get_default_point_layout().clone());
    last_buffer.resize(CAPACITY);
    let mut lazer_buffer =
        HashMapBuffer::with_capacity(CAPACITY, lazer_reader.get_default_point_layout().clone());
    lazer_buffer.resize(CAPACITY);

    if last_reader.remaining_points() != las_reader.remaining_points() {
        bail!(
            "Different point counts in LAST file ({}) as in LAS file ({})",
            last_reader.remaining_points(),
            las_reader.remaining_points()
        );
    }
    if lazer_reader.remaining_points() != las_reader.remaining_points() {
        bail!(
            "Different point counts in LAZER file ({}) as in LAS file ({})",
            lazer_reader.remaining_points(),
            las_reader.remaining_points()
        );
    }

    let point_count = las_reader.remaining_points();
    let chunks = (point_count + CAPACITY - 1) / CAPACITY;
    for chunk_index in 0..chunks {
        info!(
            "{:5} / {:5}",
            ((chunk_index + 1) * CAPACITY).human_count_bare(),
            point_count.human_count_bare()
        );

        let num_points = las_reader.read_into(&mut las_buffer, CAPACITY)?;
        last_reader.read_into(&mut last_buffer, CAPACITY)?;
        lazer_reader.read_into(&mut lazer_buffer, CAPACITY)?;

        let mut expected_point = vec![0; las_buffer.point_layout().size_of_point_entry() as usize];
        let mut actual_point = expected_point.clone();
        for point_index in 0..num_points {
            las_buffer.get_point(point_index, &mut expected_point);
            last_buffer.get_point(point_index, &mut actual_point);
            assert_eq!(
                expected_point, actual_point,
                "LAST point at index {point_index} is different"
            );

            lazer_buffer.get_point(point_index, &mut expected_point);
            assert_eq!(
                expected_point, actual_point,
                "LASZER point at index {point_index} is different"
            );
        }
    }

    Ok(())
}
