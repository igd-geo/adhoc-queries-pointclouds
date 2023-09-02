use crate::{
    index::{Classification, CompareExpression, DatasetID, PointRange, Position},
    io::{FileHandle, InputLayer},
    stats::{BlockQueryRuntimeTracker, BlockQueryRuntimeType},
};
use anyhow::{bail, Context, Result};
use byteorder::{LittleEndian, ReadBytesExt};

use pasture_core::{
    containers::BorrowedBuffer,
    layout::{
        attributes::{CLASSIFICATION, POSITION_3D},
        PointAttributeDataType,
    },
    math::AABB,
    nalgebra::{clamp, Point3, Vector3},
};
use pasture_io::las_rs::{raw, Transform};
use std::io::{Cursor, Seek};
use std::ops::Range;
use std::{io::SeekFrom, time::Instant};

use super::CompiledQueryAtom;

/// Convert a world-space position into the local integer-value space of the given LAS file using the offset and scale parameters in the header.
/// If the position is outside the representable range, i32::MIN/i32::MAX values are returned instead
pub(crate) fn to_local_integer_position(
    position_world: &Vector3<f64>,
    las_transforms: &pasture_io::las_rs::Vector<Transform>,
) -> Vector3<i32> {
    let local_x = (position_world.x / las_transforms.x.scale) - las_transforms.x.offset;
    let local_y = (position_world.y / las_transforms.y.scale) - las_transforms.y.offset;
    let local_z = (position_world.z / las_transforms.z.scale) - las_transforms.z.offset;
    Vector3::new(
        clamp(local_x, i32::MIN as f64, i32::MAX as f64) as i32,
        clamp(local_y, i32::MIN as f64, i32::MAX as f64) as i32,
        clamp(local_z, i32::MIN as f64, i32::MAX as f64) as i32,
    )
}

/// Convert from a position in local space to a world space using the given LAS file header
pub(crate) fn to_world_space_position(
    position_local: &Vector3<i32>,
    las_header: &raw::Header,
) -> Vector3<f64> {
    Vector3::new(
        (position_local.x as f64 * las_header.x_scale_factor) + las_header.x_offset,
        (position_local.y as f64 * las_header.y_scale_factor) + las_header.y_offset,
        (position_local.z as f64 * las_header.z_scale_factor) + las_header.z_offset,
    )
}

/// Calculates the world-space bounding box for the given range of points in the given file
fn _get_bounds_of_point_range(
    file: &mut Cursor<&[u8]>,
    file_header: &raw::Header,
    point_range: Range<usize>,
) -> Result<AABB<f64>> {
    let mut local_min = Vector3::<i32>::new(i32::MAX, i32::MAX, i32::MAX);
    let mut local_max = Vector3::<i32>::new(i32::MIN, i32::MIN, i32::MIN);
    for point_index in point_range {
        let point_offset: u64 = point_index as u64 * file_header.point_data_record_length as u64
            + file_header.offset_to_point_data as u64;
        file.seek(SeekFrom::Start(point_offset))?;

        let x = file.read_i32::<LittleEndian>()?;
        let y = file.read_i32::<LittleEndian>()?;
        let z = file.read_i32::<LittleEndian>()?;

        local_min.x = local_min.x.min(x);
        local_min.y = local_min.y.min(y);
        local_min.z = local_min.z.min(z);

        local_max.x = local_max.x.max(x);
        local_max.y = local_max.y.max(y);
        local_max.z = local_max.z.max(z);
    }

    Ok(AABB::from_min_max(
        Point3::new(
            (local_min.x as f64 * file_header.x_scale_factor) + file_header.x_offset,
            (local_min.y as f64 * file_header.y_scale_factor) + file_header.y_offset,
            (local_min.z as f64 * file_header.z_scale_factor) + file_header.z_offset,
        ),
        Point3::new(
            (local_max.x as f64 * file_header.x_scale_factor) + file_header.x_offset,
            (local_max.y as f64 * file_header.y_scale_factor) + file_header.y_offset,
            (local_max.z as f64 * file_header.z_scale_factor) + file_header.z_offset,
        ),
    ))
}

// pub struct LASExtractor;

// impl Extractor for LASExtractor {
//     fn extract_data(
//         &self,
//         file: &mut Cursor<&[u8]>,
//         file_header: &raw::Header,
//         block: PointRange,
//         matching_indices: &[bool],
//         num_matches: usize,
//         runtime_tracker: &BlockQueryRuntimeTracker,
//     ) -> Result<Box<dyn PointBufferSend>> {
//         let t_start = Instant::now();
//         defer! {
//             runtime_tracker.log_runtime(block.clone(), BlockQueryRuntimeType::Extraction, t_start.elapsed());
//         }

//         {
//             let mut num_disjoint_ranges: usize = 0;
//             for (current_included, next_included) in
//                 matching_indices.iter().zip(matching_indices.iter().skip(1))
//             {
//                 if current_included != next_included {
//                     num_disjoint_ranges += 1;
//                 }
//             }
//             info!("Num disjoint ranges @ {block}: {num_disjoint_ranges}");
//         }

//         let point_format =
//             Format::new(file_header.point_data_record_format).expect("Invalid point format");
//         let mut buffer = VectorBuffer::with_capacity(
//             num_matches,
//             point_layout_from_las_point_format(&point_format, false)?,
//         );
//         buffer.resize(num_matches);

//         // let num_matches_counted: usize = matching_indices.iter().filter(|b| **b).count();
//         // assert_eq!(num_matches, num_matches_counted);

//         let point_raw_data = buffer.get_raw_points_mut(0..num_matches);
//         let mut point_data_writer = Cursor::new(point_raw_data);

//         for (relative_index, _) in matching_indices
//             .iter()
//             .enumerate()
//             .filter(|(_, is_match)| **is_match)
//         {
//             let point_index = relative_index + block.points_in_file.start;
//             let point_offset: u64 = point_index as u64
//                 * file_header.point_data_record_length as u64
//                 + file_header.offset_to_point_data as u64;
//             file.seek(SeekFrom::Start(point_offset))?;

//             // XYZ
//             let local_x = file.read_i32::<LittleEndian>()?;
//             let local_y = file.read_i32::<LittleEndian>()?;
//             let local_z = file.read_i32::<LittleEndian>()?;
//             let global_x = (local_x as f64 * file_header.x_scale_factor) + file_header.x_offset;
//             let global_y = (local_y as f64 * file_header.y_scale_factor) + file_header.y_offset;
//             let global_z = (local_z as f64 * file_header.z_scale_factor) + file_header.z_offset;
//             point_data_writer.write_f64::<NativeEndian>(global_x)?;
//             point_data_writer.write_f64::<NativeEndian>(global_y)?;
//             point_data_writer.write_f64::<NativeEndian>(global_z)?;

//             // Intensity
//             point_data_writer.write_i16::<NativeEndian>(file.read_i16::<LittleEndian>()?)?;

//             // Bit attributes
//             if point_format.is_extended {
//                 let bit_attributes_first_byte = file.read_u8()?;
//                 let bit_attributes_second_byte = file.read_u8()?;

//                 let return_number = bit_attributes_first_byte & 0b1111;
//                 let number_of_returns = (bit_attributes_first_byte >> 4) & 0b1111;
//                 let classification_flags = bit_attributes_second_byte & 0b1111;
//                 let scanner_channel = (bit_attributes_second_byte >> 4) & 0b11;
//                 let scan_direction_flag = (bit_attributes_second_byte >> 6) & 0b1;
//                 let edge_of_flight_line = (bit_attributes_second_byte >> 7) & 0b1;

//                 point_data_writer.write_u8(return_number)?;
//                 point_data_writer.write_u8(number_of_returns)?;
//                 point_data_writer.write_u8(classification_flags)?;
//                 point_data_writer.write_u8(scanner_channel)?;
//                 point_data_writer.write_u8(scan_direction_flag)?;
//                 point_data_writer.write_u8(edge_of_flight_line)?;
//             } else {
//                 let bit_attributes = file.read_u8()?;
//                 let return_number = bit_attributes & 0b111;
//                 let number_of_returns = (bit_attributes >> 3) & 0b111;
//                 let scan_direction_flag = (bit_attributes >> 6) & 0b1;
//                 let edge_of_flight_line = (bit_attributes >> 7) & 0b1;

//                 point_data_writer.write_u8(return_number)?;
//                 point_data_writer.write_u8(number_of_returns)?;
//                 point_data_writer.write_u8(scan_direction_flag)?;
//                 point_data_writer.write_u8(edge_of_flight_line)?;
//             }

//             // Classification
//             point_data_writer.write_u8(file.read_u8()?)?;

//             // User data in format > 5, scan angle rank in format <= 5
//             point_data_writer.write_u8(file.read_u8()?)?;

//             if !point_format.is_extended {
//                 // User data
//                 point_data_writer.write_u8(file.read_u8()?)?;
//             } else {
//                 // Scan angle
//                 point_data_writer.write_i16::<NativeEndian>(file.read_i16::<LittleEndian>()?)?;
//             }

//             // Point source ID
//             point_data_writer.write_u16::<NativeEndian>(file.read_u16::<LittleEndian>()?)?;

//             // Format 0 is done here, the other formats are handled now

//             if point_format.has_gps_time {
//                 point_data_writer.write_f64::<NativeEndian>(file.read_f64::<LittleEndian>()?)?;
//             }

//             if point_format.has_color {
//                 point_data_writer.write_u16::<NativeEndian>(file.read_u16::<LittleEndian>()?)?;
//                 point_data_writer.write_u16::<NativeEndian>(file.read_u16::<LittleEndian>()?)?;
//                 point_data_writer.write_u16::<NativeEndian>(file.read_u16::<LittleEndian>()?)?;
//             }

//             if point_format.has_nir {
//                 point_data_writer.write_u16::<NativeEndian>(file.read_u16::<LittleEndian>()?)?;
//             }

//             if point_format.has_waveform {
//                 point_data_writer.write_u8(file.read_u8()?)?;
//                 point_data_writer.write_u64::<NativeEndian>(file.read_u64::<LittleEndian>()?)?;
//                 point_data_writer.write_u32::<NativeEndian>(file.read_u32::<LittleEndian>()?)?;
//                 point_data_writer.write_f32::<NativeEndian>(file.read_f32::<LittleEndian>()?)?;
//                 point_data_writer.write_f32::<NativeEndian>(file.read_f32::<LittleEndian>()?)?;
//                 point_data_writer.write_f32::<NativeEndian>(file.read_f32::<LittleEndian>()?)?;
//                 point_data_writer.write_f32::<NativeEndian>(file.read_f32::<LittleEndian>()?)?;
//             }
//         }

//         Ok(Box::new(buffer))
//     }
// }

/// Implementation of the 'Within' query for LAS files
pub(crate) struct LasQueryAtomWithin<T> {
    min: T,
    max: T,
}

impl<T> LasQueryAtomWithin<T> {
    pub(crate) fn new(min: T, max: T) -> Self {
        Self { min, max }
    }
}

/// Implementation of the 'Compare' query for LAS files
pub(crate) struct LasQueryAtomCompare<T> {
    value: T,
    compare_expression: CompareExpression,
}

impl<T> LasQueryAtomCompare<T> {
    pub(crate) fn new(value: T, compare_expression: CompareExpression) -> Self {
        Self {
            value,
            compare_expression,
        }
    }
}

fn eval_impl<F: FnMut(usize) -> Result<bool>>(
    block: PointRange,
    matching_indices: &'_ mut [bool],
    which_indices_to_loop_over: super::WhichIndicesToLoopOver,
    mut test_point: F,
    runtime_tracker: &BlockQueryRuntimeTracker,
) -> Result<usize> {
    let timer = Instant::now();

    let mut num_matches = 0;
    match which_indices_to_loop_over {
        super::WhichIndicesToLoopOver::All => {
            assert!(block.points_in_file.len() <= matching_indices.len());
            for point_index in block.points_in_file.clone() {
                let local_index = point_index - block.points_in_file.start;
                matching_indices[local_index] = test_point(point_index)?;
                if matching_indices[local_index] {
                    num_matches += 1;
                }
            }
        }
        super::WhichIndicesToLoopOver::Matching => {
            for (local_index, is_match) in matching_indices
                .iter_mut()
                .enumerate()
                .filter(|(_, is_match)| **is_match)
            {
                let point_index = local_index + block.points_in_file.start;
                *is_match = test_point(point_index)?;
                if *is_match {
                    num_matches += 1;
                }
            }
        }
        super::WhichIndicesToLoopOver::NotMatching => {
            for (local_index, is_match) in matching_indices
                .iter_mut()
                .enumerate()
                .filter(|(_, is_match)| !**is_match)
            {
                let point_index = local_index + block.points_in_file.start;
                *is_match = test_point(point_index)?;
                if *is_match {
                    num_matches += 1;
                }
            }
        }
    }

    runtime_tracker.log_runtime(block, BlockQueryRuntimeType::Eval, timer.elapsed());

    Ok(num_matches)
}

impl CompiledQueryAtom for LasQueryAtomWithin<Position> {
    fn eval(
        &self,
        input_layer: &InputLayer,
        block: PointRange,
        dataset_id: DatasetID,
        matching_indices: &'_ mut [bool],
        which_indices_to_loop_over: super::WhichIndicesToLoopOver,
        runtime_tracker: &BlockQueryRuntimeTracker,
    ) -> Result<usize> {
        let las_metadata = input_layer
            .get_las_metadata(FileHandle(dataset_id, block.file_index))
            .context("Could not get LAS metadata for file")?;

        let las_transforms = las_metadata.raw_las_header().unwrap().transforms();

        let local_min = to_local_integer_position(&self.min.0, las_transforms);
        let local_max = to_local_integer_position(&self.max.0, las_transforms);
        let local_bounds = AABB::from_min_max(local_min.into(), local_max.into());

        // We can access the raw position data using a pasture `AttributeView`
        let point_data = input_layer
            .get_point_data(dataset_id, block.clone())
            .context("Could not access point data")?;
        let positions = point_data.view_attribute::<Vector3<i32>>(
            &POSITION_3D.with_custom_datatype(PointAttributeDataType::Vec3i32),
        );

        let test_point = |point_index: usize| -> Result<bool> {
            let position = positions.at(point_index);
            Ok(local_bounds.contains(&position.into()))
        };

        eval_impl(
            block,
            matching_indices,
            which_indices_to_loop_over,
            test_point,
            runtime_tracker,
        )
    }
}

impl CompiledQueryAtom for LasQueryAtomWithin<Classification> {
    fn eval(
        &self,
        input_layer: &InputLayer,
        block: PointRange,
        dataset_id: DatasetID,
        matching_indices: &'_ mut [bool],
        which_indices_to_loop_over: super::WhichIndicesToLoopOver,
        runtime_tracker: &BlockQueryRuntimeTracker,
    ) -> Result<usize> {
        let point_data = input_layer
            .get_point_data(dataset_id, block.clone())
            .context("Could not access point data")?;
        let classifications = point_data.view_attribute::<u8>(&CLASSIFICATION);

        let test_point = |point_index: usize| -> Result<bool> {
            let classification = classifications.at(point_index);
            Ok(classification >= self.min.0 && classification < self.max.0)
        };

        eval_impl(
            block,
            matching_indices,
            which_indices_to_loop_over,
            test_point,
            runtime_tracker,
        )
    }
}

impl CompiledQueryAtom for LasQueryAtomCompare<Position> {
    fn eval(
        &self,
        input_layer: &InputLayer,
        block: PointRange,
        dataset_id: DatasetID,
        matching_indices: &'_ mut [bool],
        which_indices_to_loop_over: super::WhichIndicesToLoopOver,
        runtime_tracker: &BlockQueryRuntimeTracker,
    ) -> Result<usize> {
        let las_metadata = input_layer
            .get_las_metadata(FileHandle(dataset_id, block.file_index))
            .context("Could not get LAS metadata for file")?;

        let las_transforms = las_metadata.raw_las_header().unwrap().transforms();
        let local_position = to_local_integer_position(&self.value.0, las_transforms);

        // We can access the raw position data using a pasture `AttributeView`
        let point_data = input_layer
            .get_point_data(dataset_id, block.clone())
            .context("Could not access point data")?;
        let positions = point_data.view_attribute::<Vector3<i32>>(
            &POSITION_3D.with_custom_datatype(PointAttributeDataType::Vec3i32),
        );

        match self.compare_expression {
            CompareExpression::Equals => {
                let test_point = |point_index: usize| -> Result<bool> {
                    let position = positions.at(point_index);
                    Ok(position == local_position)
                };

                eval_impl(
                    block,
                    matching_indices,
                    which_indices_to_loop_over,
                    test_point,
                    runtime_tracker,
                )
            }
            CompareExpression::NotEquals => {
                let test_point = |point_index: usize| -> Result<bool> {
                    let position = positions.at(point_index);
                    Ok(position != local_position)
                };

                eval_impl(
                    block,
                    matching_indices,
                    which_indices_to_loop_over,
                    test_point,
                    runtime_tracker,
                )
            }
            other => bail!("Unsupported compare expression {other:#?} for Position attribute"),
        }
    }
}

impl CompiledQueryAtom for LasQueryAtomCompare<Classification> {
    fn eval(
        &self,
        input_layer: &InputLayer,
        block: PointRange,
        dataset_id: DatasetID,
        matching_indices: &'_ mut [bool],
        which_indices_to_loop_over: super::WhichIndicesToLoopOver,
        runtime_tracker: &BlockQueryRuntimeTracker,
    ) -> Result<usize> {
        let point_data = input_layer
            .get_point_data(dataset_id, block.clone())
            .context("Could not access point data")?;
        let classifications = point_data.view_attribute::<u8>(&CLASSIFICATION);

        match self.compare_expression {
            CompareExpression::Equals => {
                let test_point = |point_index: usize| -> Result<bool> {
                    let classification = classifications.at(point_index);
                    Ok(classification == self.value.0)
                };

                eval_impl(
                    block,
                    matching_indices,
                    which_indices_to_loop_over,
                    test_point,
                    runtime_tracker,
                )
            }
            CompareExpression::NotEquals => {
                let test_point = |point_index: usize| -> Result<bool> {
                    let classification = classifications.at(point_index);
                    Ok(classification != self.value.0)
                };

                eval_impl(
                    block,
                    matching_indices,
                    which_indices_to_loop_over,
                    test_point,
                    runtime_tracker,
                )
            }
            CompareExpression::LessThan => {
                let test_point = |point_index: usize| -> Result<bool> {
                    let classification = classifications.at(point_index);
                    Ok(classification < self.value.0)
                };

                eval_impl(
                    block,
                    matching_indices,
                    which_indices_to_loop_over,
                    test_point,
                    runtime_tracker,
                )
            }
            CompareExpression::LessThanOrEquals => {
                let test_point = |point_index: usize| -> Result<bool> {
                    let classification = classifications.at(point_index);
                    Ok(classification <= self.value.0)
                };

                eval_impl(
                    block,
                    matching_indices,
                    which_indices_to_loop_over,
                    test_point,
                    runtime_tracker,
                )
            }
            CompareExpression::GreaterThan => {
                let test_point = |point_index: usize| -> Result<bool> {
                    let classification = classifications.at(point_index);
                    Ok(classification > self.value.0)
                };

                eval_impl(
                    block,
                    matching_indices,
                    which_indices_to_loop_over,
                    test_point,
                    runtime_tracker,
                )
            }
            CompareExpression::GreaterThanOrEquals => {
                let test_point = |point_index: usize| -> Result<bool> {
                    let classification = classifications.at(point_index);
                    Ok(classification >= self.value.0)
                };

                eval_impl(
                    block,
                    matching_indices,
                    which_indices_to_loop_over,
                    test_point,
                    runtime_tracker,
                )
            }
        }
    }
}
