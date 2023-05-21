use std::{
    fs::File,
    io::{Cursor, Seek, SeekFrom},
    ops::Range,
    path::Path,
};

use anyhow::{Context, Result};
use byteorder::{LittleEndian, ReadBytesExt};
use memmap::Mmap;
use pasture_core::nalgebra::Vector3;
use pasture_io::las_rs::raw::Header;

use crate::search::to_world_space_position;

use super::PointReader;

/// Reader for LAS files
pub(crate) struct LASReader {
    data: Mmap,
    raw_header: Header,
}

impl LASReader {
    /// Open a LASReader to the given LAS file
    pub(crate) fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path).context("Can't open file")?;
        let file_mmap = unsafe { Mmap::map(&file).context("Can't mmap file")? };
        let file_data: &[u8] = &file_mmap;

        let raw_header =
            Header::read_from(&mut Cursor::new(file_data)).context("Can't read LAS header")?;

        Ok(Self {
            data: file_mmap,
            raw_header,
        })
    }
}

impl PointReader for LASReader {
    fn read_positions(&mut self, point_indices: Range<usize>) -> Result<Vec<Vector3<f64>>> {
        let mut data = Cursor::new(&self.data);
        point_indices
            .map(|index: usize| {
                let point_offset: u64 = index as u64
                    * self.raw_header.point_data_record_length as u64
                    + self.raw_header.offset_to_point_data as u64;
                data.seek(SeekFrom::Start(point_offset))?;

                let x = data.read_i32::<LittleEndian>()?;
                let y = data.read_i32::<LittleEndian>()?;
                let z = data.read_i32::<LittleEndian>()?;

                Ok(to_world_space_position(
                    &Vector3::new(x, y, z),
                    &self.raw_header,
                ))
            })
            .collect::<Result<Vec<_>, _>>()
    }

    fn read_positions_in_local_space(
        &mut self,
        point_indices: Range<usize>,
    ) -> Option<Result<Vec<Vector3<i32>>>> {
        let mut data = Cursor::new(&self.data);
        Some(
            point_indices
                .map(|index: usize| {
                    let point_offset: u64 = index as u64
                        * self.raw_header.point_data_record_length as u64
                        + self.raw_header.offset_to_point_data as u64;
                    data.seek(SeekFrom::Start(point_offset))?;

                    let x = data.read_i32::<LittleEndian>()?;
                    let y = data.read_i32::<LittleEndian>()?;
                    let z = data.read_i32::<LittleEndian>()?;

                    Ok(Vector3::new(x, y, z))
                })
                .collect::<Result<Vec<_>, _>>(),
        )
    }

    fn read_classifications(&mut self, point_indices: Range<usize>) -> Result<Vec<u8>> {
        let mut data = Cursor::new(&self.data);
        let offset_to_classification = if self.raw_header.point_data_record_format > 5 {
            16
        } else {
            15
        };

        point_indices
            .map(|index: usize| {
                let point_offset: u64 = index as u64
                    * self.raw_header.point_data_record_length as u64
                    + self.raw_header.offset_to_point_data as u64;
                data.seek(SeekFrom::Start(point_offset + offset_to_classification))?;

                let classification = data.read_u8()?;

                Ok(classification)
            })
            .collect::<Result<Vec<_>, _>>()
    }
}
