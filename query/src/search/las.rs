use crate::{
    collect_points::ResultCollector,
    index::{Classification, Position},
};
use anyhow::Result;
use byteorder::{LittleEndian, NativeEndian, ReadBytesExt, WriteBytesExt};
use memmap::Mmap;
use pasture_core::{
    containers::{InterleavedPointBufferMut, InterleavedVecPointStorage, PointBufferWriteable},
    nalgebra::{clamp, Vector3},
};
use pasture_io::{
    las::point_layout_from_las_point_format,
    las_rs::{point::Format, raw},
};
use std::io::SeekFrom;
use std::io::{Cursor, Seek};
use std::ops::Range;

use super::{CompiledQueryAtom, Extractor};

/// Convert a world-space position into the local integer-value space of the given LAS file using the offset and scale parameters in the header.
/// If the position is outside the representable range, i32::MIN/i32::MAX values are returned instead
pub(crate) fn to_local_integer_position(
    position_world: &Vector3<f64>,
    las_header: &raw::Header,
) -> Vector3<i32> {
    let local_x = (position_world.x / las_header.x_scale_factor) - las_header.x_offset;
    let local_y = (position_world.y / las_header.y_scale_factor) - las_header.y_offset;
    let local_z = (position_world.z / las_header.z_scale_factor) - las_header.z_offset;
    Vector3::new(
        clamp(local_x, i32::MIN as f64, i32::MAX as f64) as i32,
        clamp(local_y, i32::MIN as f64, i32::MAX as f64) as i32,
        clamp(local_z, i32::MIN as f64, i32::MAX as f64) as i32,
    )
}

pub struct LASExtractor;

impl Extractor for LASExtractor {
    fn extract_data(
        &self,
        file: &mut Cursor<Mmap>,
        file_header: &raw::Header,
        block: Range<usize>,
        matching_indices: &mut [bool],
        num_matches: usize,
        result_collector: &mut dyn ResultCollector,
    ) -> Result<()> {
        let point_format =
            Format::new(file_header.point_data_record_format).expect("Invalid point format");
        let mut buffer = InterleavedVecPointStorage::with_capacity(
            num_matches,
            point_layout_from_las_point_format(&point_format)?,
        );
        buffer.resize(num_matches);

        let mut current_point = 0;
        for (relative_index, _) in matching_indices
            .iter()
            .enumerate()
            .filter(|(_, is_match)| **is_match)
        {
            let point_index = relative_index + block.start;
            let point_offset: u64 = point_index as u64
                * file_header.point_data_record_length as u64
                + file_header.offset_to_point_data as u64;
            file.seek(SeekFrom::Start(point_offset))?;

            let point_raw_data = buffer.get_raw_point_mut(current_point);
            current_point += 1;

            let mut point_data_writer = Cursor::new(point_raw_data);

            // XYZ
            let local_x = file.read_i32::<LittleEndian>()?;
            let local_y = file.read_i32::<LittleEndian>()?;
            let local_z = file.read_i32::<LittleEndian>()?;
            let global_x = (local_x as f64 * file_header.x_scale_factor) + file_header.x_offset;
            let global_y = (local_y as f64 * file_header.y_scale_factor) + file_header.y_offset;
            let global_z = (local_z as f64 * file_header.z_scale_factor) + file_header.z_offset;
            point_data_writer.write_f64::<NativeEndian>(global_x)?;
            point_data_writer.write_f64::<NativeEndian>(global_y)?;
            point_data_writer.write_f64::<NativeEndian>(global_z)?;

            // Intensity
            point_data_writer.write_i16::<NativeEndian>(file.read_i16::<LittleEndian>()?)?;

            // Bit attributes
            if point_format.is_extended {
                let bit_attributes_first_byte = file.read_u8()?;
                let bit_attributes_second_byte = file.read_u8()?;

                let return_number = bit_attributes_first_byte & 0b1111;
                let number_of_returns = (bit_attributes_first_byte >> 4) & 0b1111;
                let classification_flags = bit_attributes_second_byte & 0b1111;
                let scanner_channel = (bit_attributes_second_byte >> 4) & 0b11;
                let scan_direction_flag = (bit_attributes_second_byte >> 6) & 0b1;
                let edge_of_flight_line = (bit_attributes_second_byte >> 7) & 0b1;

                point_data_writer.write_u8(return_number)?;
                point_data_writer.write_u8(number_of_returns)?;
                point_data_writer.write_u8(classification_flags)?;
                point_data_writer.write_u8(scanner_channel)?;
                point_data_writer.write_u8(scan_direction_flag)?;
                point_data_writer.write_u8(edge_of_flight_line)?;
            } else {
                let bit_attributes = file.read_u8()?;
                let return_number = bit_attributes & 0b111;
                let number_of_returns = (bit_attributes >> 3) & 0b111;
                let scan_direction_flag = (bit_attributes >> 6) & 0b1;
                let edge_of_flight_line = (bit_attributes >> 7) & 0b1;

                point_data_writer.write_u8(return_number)?;
                point_data_writer.write_u8(number_of_returns)?;
                point_data_writer.write_u8(scan_direction_flag)?;
                point_data_writer.write_u8(edge_of_flight_line)?;
            }

            // Classification
            point_data_writer.write_u8(file.read_u8()?)?;

            // User data in format > 5, scan angle rank in format <= 5
            point_data_writer.write_u8(file.read_u8()?)?;

            if !point_format.is_extended {
                // User data
                point_data_writer.write_u8(file.read_u8()?)?;
            } else {
                // Scan angle
                point_data_writer.write_i16::<NativeEndian>(file.read_i16::<LittleEndian>()?)?;
            }

            // Point source ID
            point_data_writer.write_u16::<NativeEndian>(file.read_u16::<LittleEndian>()?)?;

            // Format 0 is done here, the other formats are handled now

            if point_format.has_gps_time {
                point_data_writer.write_f64::<NativeEndian>(file.read_f64::<LittleEndian>()?)?;
            }

            if point_format.has_color {
                point_data_writer.write_u16::<NativeEndian>(file.read_u16::<LittleEndian>()?)?;
                point_data_writer.write_u16::<NativeEndian>(file.read_u16::<LittleEndian>()?)?;
                point_data_writer.write_u16::<NativeEndian>(file.read_u16::<LittleEndian>()?)?;
            }

            if point_format.has_nir {
                point_data_writer.write_u16::<NativeEndian>(file.read_u16::<LittleEndian>()?)?;
            }

            if point_format.has_waveform {
                point_data_writer.write_u8(file.read_u8()?)?;
                point_data_writer.write_u64::<NativeEndian>(file.read_u64::<LittleEndian>()?)?;
                point_data_writer.write_u32::<NativeEndian>(file.read_u32::<LittleEndian>()?)?;
                point_data_writer.write_f32::<NativeEndian>(file.read_f32::<LittleEndian>()?)?;
                point_data_writer.write_f32::<NativeEndian>(file.read_f32::<LittleEndian>()?)?;
                point_data_writer.write_f32::<NativeEndian>(file.read_f32::<LittleEndian>()?)?;
                point_data_writer.write_f32::<NativeEndian>(file.read_f32::<LittleEndian>()?)?;
            }
        }

        result_collector.collect(Box::new(buffer));

        Ok(())
    }
}

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

/// Implementation of the 'Equals' query for LAS files
pub(crate) struct LasQueryAtomEquals<T> {
    value: T,
}

impl<T> LasQueryAtomEquals<T> {
    pub(crate) fn new(value: T) -> Self {
        Self { value }
    }
}

fn eval_impl<F: FnMut(usize) -> Result<bool>>(
    block: Range<usize>,
    matching_indices: &'_ mut [bool],
    which_indices_to_loop_over: super::WhichIndicesToLoopOver,
    mut test_point: F,
) -> Result<usize> {
    let mut num_matches = 0;
    match which_indices_to_loop_over {
        super::WhichIndicesToLoopOver::All => {
            assert!(block.len() == matching_indices.len());
            for point_index in block.clone() {
                let local_index = point_index - block.start;
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
                let point_index = local_index + block.start;
                *is_match = test_point(point_index)?;
                if *is_match {
                    num_matches *= 1;
                }
            }
        }
        super::WhichIndicesToLoopOver::NotMatching => {
            for (local_index, is_match) in matching_indices
                .iter_mut()
                .enumerate()
                .filter(|(_, is_match)| !**is_match)
            {
                let point_index = local_index + block.start;
                *is_match = test_point(point_index)?;
                if *is_match {
                    num_matches *= 1;
                }
            }
        }
    }

    Ok(num_matches)
}

impl CompiledQueryAtom for LasQueryAtomWithin<Position> {
    fn eval(
        &self,
        file: &mut Cursor<Mmap>,
        file_header: &pasture_io::las_rs::raw::Header,
        block: Range<usize>,
        matching_indices: &'_ mut [bool],
        which_indices_to_loop_over: super::WhichIndicesToLoopOver,
    ) -> Result<usize> {
        let local_min = to_local_integer_position(&self.min.0, file_header);
        let local_max = to_local_integer_position(&self.max.0, file_header);

        let test_point = |point_index: usize| -> Result<bool> {
            // Seek to point start, read X, Y, and Z in LAS i32 format and check
            let point_offset: u64 = point_index as u64
                * file_header.point_data_record_length as u64
                + file_header.offset_to_point_data as u64;
            file.seek(SeekFrom::Start(point_offset))?;

            let local_x = file.read_i32::<LittleEndian>()?;
            if local_x < local_min.x || local_x > local_max.x {
                return Ok(false);
            }

            let local_y = file.read_i32::<LittleEndian>()?;
            if local_y < local_min.y || local_y > local_max.y {
                return Ok(false);
            }

            let local_z = file.read_i32::<LittleEndian>()?;
            if local_z < local_min.z || local_z > local_max.z {
                return Ok(false);
            }

            Ok(true)
        };

        eval_impl(
            block,
            matching_indices,
            which_indices_to_loop_over,
            test_point,
        )
    }
}

impl CompiledQueryAtom for LasQueryAtomWithin<Classification> {
    fn eval(
        &self,
        file: &mut Cursor<Mmap>,
        file_header: &pasture_io::las_rs::raw::Header,
        block: Range<usize>,
        matching_indices: &'_ mut [bool],
        which_indices_to_loop_over: super::WhichIndicesToLoopOver,
    ) -> Result<usize> {
        let offset_to_classification = if file_header.point_data_record_format > 5 {
            16
        } else {
            15
        };

        let test_point = |point_index: usize| -> Result<bool> {
            // Seek to point start, read X, Y, and Z in LAS i32 format and check
            let point_offset: u64 = point_index as u64
                * file_header.point_data_record_length as u64
                + file_header.offset_to_point_data as u64;
            file.seek(SeekFrom::Start(point_offset + offset_to_classification))?;

            let classification = file.read_u8()?;
            Ok(classification >= self.min.0 && classification < self.max.0)
        };

        eval_impl(
            block,
            matching_indices,
            which_indices_to_loop_over,
            test_point,
        )
    }
}

impl CompiledQueryAtom for LasQueryAtomEquals<Position> {
    fn eval(
        &self,
        file: &mut Cursor<Mmap>,
        file_header: &pasture_io::las_rs::raw::Header,
        block: Range<usize>,
        matching_indices: &'_ mut [bool],
        which_indices_to_loop_over: super::WhichIndicesToLoopOver,
    ) -> Result<usize> {
        let local_position = to_local_integer_position(&self.value.0, file_header);
        let test_point = |point_index: usize| -> Result<bool> {
            // Seek to point start, read X, Y, and Z in LAS i32 format and check
            let point_offset: u64 = point_index as u64
                * file_header.point_data_record_length as u64
                + file_header.offset_to_point_data as u64;
            file.seek(SeekFrom::Start(point_offset))?;

            let local_x = file.read_i32::<LittleEndian>()?;
            if local_x != local_position.x {
                return Ok(false);
            }

            let local_y = file.read_i32::<LittleEndian>()?;
            if local_y != local_position.y {
                return Ok(false);
            }

            let local_z = file.read_i32::<LittleEndian>()?;
            if local_z != local_position.z {
                return Ok(false);
            }

            Ok(true)
        };

        eval_impl(
            block,
            matching_indices,
            which_indices_to_loop_over,
            test_point,
        )
    }
}

impl CompiledQueryAtom for LasQueryAtomEquals<Classification> {
    fn eval(
        &self,
        file: &mut Cursor<Mmap>,
        file_header: &pasture_io::las_rs::raw::Header,
        block: Range<usize>,
        matching_indices: &'_ mut [bool],
        which_indices_to_loop_over: super::WhichIndicesToLoopOver,
    ) -> Result<usize> {
        let offset_to_classification = if file_header.point_data_record_format > 5 {
            16
        } else {
            15
        };

        let test_point = |point_index: usize| -> Result<bool> {
            // Seek to point start, read X, Y, and Z in LAS i32 format and check
            let point_offset: u64 = point_index as u64
                * file_header.point_data_record_length as u64
                + file_header.offset_to_point_data as u64;
            file.seek(SeekFrom::Start(point_offset + offset_to_classification))?;

            let classification = file.read_u8()?;
            Ok(classification == self.value.0)
        };

        eval_impl(
            block,
            matching_indices,
            which_indices_to_loop_over,
            test_point,
        )
    }
}
