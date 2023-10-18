use std::io::{Cursor, Read, Seek, SeekFrom, Write};

use anyhow::{bail, Context, Result};
use pasture_core::{
    containers::{BorrowedBuffer, BorrowedMutBuffer, HashMapBuffer, OwningBuffer},
    layout::{PointAttributeMember, PointLayout},
};
use pasture_io::{
    base::{PointReader, SeekToPoint},
    las::{
        point_layout_from_las_metadata, point_layout_from_las_point_format, LASMetadata, LASReader,
    },
    las_rs::{point::Format, raw::Header},
};

use crate::las_common::{get_default_las_converter, get_minimum_layout_for_las_conversion};

/// Convert a LAS file to a LAST file
pub fn las_to_last<R: Read + Seek + Send>(
    mut las_reader: R,
    mut last_writer: Cursor<&mut [u8]>,
) -> Result<()> {
    let header = Header::read_from(&mut las_reader).context("Failed to read LAS header")?;
    if header.evlr.is_some() {
        bail!("Files with EVLRs are not supported currently");
    }

    let start_of_point_records = header.offset_to_point_data as usize;

    // Copy all data before the point records into the LAST file
    {
        las_reader.seek(std::io::SeekFrom::Start(0))?;
        let mut buf = vec![0; start_of_point_records];
        las_reader
            .read_exact(&mut buf)
            .context("Could not read data prior to point records")?;
        last_writer
            .write_all(&buf)
            .context("Could not copy data prior to point records to LAST file")?;
    }

    let raw_las_point_layout =
        point_layout_from_las_point_format(&Format::new(header.point_data_record_format)?, true)?;

    const CHUNK_SIZE: usize = 1 << 20;
    let mut chunk_buffer =
        vec![0; CHUNK_SIZE * raw_las_point_layout.size_of_point_entry() as usize];
    let num_points = if let Some(large_file) = header.large_file.as_ref() {
        large_file.number_of_point_records as usize
    } else {
        header.number_of_point_records as usize
    };

    let num_chunks = (num_points + CHUNK_SIZE - 1) / CHUNK_SIZE;
    for chunk_id in 0..num_chunks {
        let points_in_chunk = std::cmp::min(CHUNK_SIZE, num_points - (chunk_id * CHUNK_SIZE));

        let points_chunk = &mut chunk_buffer
            [..(points_in_chunk * raw_las_point_layout.size_of_point_entry() as usize)];
        las_reader.read_exact(points_chunk)?;

        // TODO Would be easier if we had an ExternalMemoryColumnarBuffer
        // But oh boy does this open a can of worms... Turns out, it would have been better to have a single
        // `PointBuffer<T>` type, where `T: PointStorage` for a trait hierarchy of storage types that in essence
        // implement what the separate buffer traits currently implement. This would simplify code since for
        // example `PointBuffer<Vec<u8>>` and `PointBuffer<&[u8]>` share almost all functionality (expect resizing)

        // Copy all attribute data to the correct position in the LAST file
        for attribute in raw_las_point_layout.attributes() {
            let start_of_attribute_records =
                start_of_point_records + (num_points * attribute.offset() as usize);
            let offset_in_last_file =
                start_of_attribute_records + (chunk_id * CHUNK_SIZE * attribute.size() as usize);

            last_writer.seek(SeekFrom::Start(offset_in_last_file as u64))?;

            for local_idx in 0..points_in_chunk {
                let src_start_of_point =
                    local_idx * raw_las_point_layout.size_of_point_entry() as usize;
                let point_bytes = &points_chunk[src_start_of_point
                    ..(src_start_of_point + raw_las_point_layout.size_of_point_entry() as usize)];
                let src_bytes_of_attribute = &point_bytes[attribute.byte_range_within_point()];

                last_writer.write_all(src_bytes_of_attribute)?;
            }
        }
    }

    last_writer.flush()?;

    Ok(())
}

// TODO LAST to LAS, or maybe a pasture PointReader for LAST
// TODO Equivalence tests for the point records in a LAST file and its corresponding LAS file
pub struct LASTReader<R: Read + Seek> {
    read: R,
    las_metadata: LASMetadata,
    raw_point_layout: PointLayout,
    current_point_index: usize,
    offset_to_point_records: usize,
}

impl<R: Read + Seek + Send> LASTReader<R> {
    pub fn from_read(mut read: R) -> Result<Self> {
        let las_metadata = {
            let las_reader = LASReader::from_read(&mut read, false, false)?;
            las_reader.las_metadata().clone()
        };
        let raw_las_header = las_metadata
            .raw_las_header()
            .cloned()
            .expect("No LAS header found")
            .into_raw()?;
        let raw_point_layout = point_layout_from_las_metadata(&las_metadata, true)
            .context("No matching PointLayout found for this LAST file")?;

        Ok(Self {
            read,
            las_metadata,
            raw_point_layout,
            current_point_index: 0,
            offset_to_point_records: raw_las_header.offset_to_point_data as usize,
        })
    }

    pub fn remaining_points(&self) -> usize {
        self.las_metadata.point_count() - self.current_point_index
    }

    pub fn las_metadata(&self) -> &LASMetadata {
        &self.las_metadata
    }

    pub fn into_inner(self) -> R {
        self.read
    }

    fn byte_offset_of_attribute_data(&self, attribute: &PointAttributeMember) -> usize {
        self.offset_to_point_records
            + (attribute.offset() as usize * self.las_metadata.point_count())
    }

    /// Read data for all attributes that are in `buffer`, assuming that the attributes are a subset of
    /// all native point attributes of the current LAST file. If an attribute is present in `buffer`, its
    /// datatype must match the default datatype of the attribute in the LAST file. For example, if the
    /// buffer has a `POSITION_3D` attribute, it MUST be of datatype `Vec3i32`
    fn read_default_attributes<'a, 'b, B: BorrowedMutBuffer<'a>>(
        &mut self,
        buffer: &'b mut B,
        count: usize,
    ) -> Result<()> {
        let new_points_range = 0..buffer.len();
        let required_attributes = self
            .raw_point_layout
            .attributes()
            .filter(|a| {
                buffer
                    .point_layout()
                    .has_attribute(a.attribute_definition())
            })
            .collect::<Vec<_>>();

        for attribute in required_attributes {
            let offset_to_start_of_attribute = self.byte_offset_of_attribute_data(attribute);
            let offset_to_point = offset_to_start_of_attribute
                + (self.current_point_index * attribute.size() as usize);
            self.read.seek(SeekFrom::Start(offset_to_point as u64))?;

            if let Some(columnar_buffer) = buffer.as_columnar_mut() {
                let _span = tracy_client::span!("LastReader::read_attribute columnar");

                let attribute_data_range = columnar_buffer.get_attribute_range_mut(
                    attribute.attribute_definition(),
                    new_points_range.clone(),
                );
                self.read.read_exact(attribute_data_range)?;
            } else {
                let _span = tracy_client::span!("LastReader::read_attribute unknown layout");

                let mut buf = vec![0; count * attribute.size() as usize];
                self.read.read_exact(&mut buf)?;
                // Safe because we know the attribute exists within the buffer and the attribute size matches
                unsafe {
                    buffer.set_attribute_range(
                        attribute.attribute_definition(),
                        new_points_range.clone(),
                        &buf,
                    );
                }
            }
        }

        Ok(())
    }
}

impl<R: Read + Seek + Send> PointReader for LASTReader<R> {
    fn read_into<'a, 'b, B: BorrowedMutBuffer<'a>>(
        &mut self,
        point_buffer: &'b mut B,
        count: usize,
    ) -> Result<usize>
    where
        'a: 'b,
    {
        let remaining = self.las_metadata.point_count() - self.current_point_index;
        let num_points_to_read = std::cmp::min(remaining, count);
        if num_points_to_read == 0 {
            return Ok(0);
        }

        let minimum_layout_to_parse = get_minimum_layout_for_las_conversion(
            &self.raw_point_layout,
            point_buffer.point_layout(),
        )
        .context(
            "Could not determine appropriate PointLayout for data conversion from LAST file",
        )?;

        if *point_buffer.point_layout() == minimum_layout_to_parse {
            let _span = tracy_client::span!("LastReader::read_into default layout");
            let _plot = tracy_client::plot!("points", count as f64);

            self.read_default_attributes(point_buffer, num_points_to_read)?;
        } else {
            let _span = tracy_client::span!("LastReader::read_into with conversion");
            let _plot = tracy_client::plot!("points", count as f64);

            let mut tmp_buffer =
                HashMapBuffer::with_capacity(num_points_to_read, minimum_layout_to_parse);
            tmp_buffer.resize(num_points_to_read);
            self.read_default_attributes(&mut tmp_buffer, num_points_to_read)?;

            let target_layout = point_buffer.point_layout().clone();

            let raw_las_header = self
                .las_metadata
                .raw_las_header()
                .cloned()
                .expect("No LAS header found")
                .into_raw()?;
            let converter = get_default_las_converter(
                tmp_buffer.point_layout(),
                &target_layout,
                raw_las_header,
            )
            .context("Could not get a buffer layout converter for the target buffer")?;
            {
                let _span = tracy_client::span!("convert_into");
                converter.convert_into(&tmp_buffer, point_buffer);
            }
        }

        self.current_point_index += num_points_to_read;
        Ok(num_points_to_read)
    }

    fn get_metadata(&self) -> &dyn pasture_core::meta::Metadata {
        &self.las_metadata
    }

    fn get_default_point_layout(&self) -> &pasture_core::layout::PointLayout {
        &self.raw_point_layout
    }
}

impl<R: Read + Seek + Send> SeekToPoint for LASTReader<R> {
    fn seek_point(&mut self, position: SeekFrom) -> Result<usize> {
        match position {
            SeekFrom::Start(offset) => {
                let clamped_offset =
                    std::cmp::min(self.las_metadata.point_count(), offset as usize);
                self.current_point_index = clamped_offset;
                Ok(clamped_offset)
            }
            SeekFrom::End(offset) => {
                let count = self.las_metadata.point_count() as i64;
                let clamped_offset_from_start = count.min(count + offset).max(0) as usize;
                self.current_point_index = clamped_offset_from_start;
                Ok(clamped_offset_from_start)
            }
            SeekFrom::Current(offset) => {
                let count = self.las_metadata.point_count() as i64;
                let clamped_offset_from_start =
                    count.min(self.current_point_index as i64 + offset).max(0) as usize;
                self.current_point_index = clamped_offset_from_start;
                Ok(clamped_offset_from_start)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use pasture_core::containers::{ExternalMemoryBuffer, InterleavedBuffer, VectorBuffer};
    use pasture_io::{las::point_layout_from_las_point_format, las_rs::point::Format};

    use super::*;

    #[test]
    fn test_las_last_roundtrip() -> Result<()> {
        let in_file_path =
            "/Users/pbormann/data/geodata/pointclouds/datasets/district_of_columbia/1318.las";

        let las_file_bytes = std::fs::read(in_file_path)?;
        let (las_points, header) = {
            let header = Header::read_from(Cursor::new(&las_file_bytes))?;
            let offset_to_point_data = header.offset_to_point_data as usize;
            let point_layout = point_layout_from_las_point_format(
                &Format::new(header.point_data_record_format)?,
                true,
            )?;
            (
                ExternalMemoryBuffer::new(&las_file_bytes[offset_to_point_data..], point_layout),
                header,
            )
        };
        let mut last_bytes = vec![0; las_file_bytes.len()];
        las_to_last(Cursor::new(&las_file_bytes), Cursor::new(&mut last_bytes))?;

        let mut last_reader = LASTReader::from_read(Cursor::new(&last_bytes))?;
        // let  last_points = VectorBuffer::with_capacity(
        //     last_reader.remaining_points(),
        //     las_points.point_layout().clone(),
        // );
        let last_points = last_reader.read::<VectorBuffer>(last_reader.remaining_points())?;

        assert_eq!(las_points.len(), last_points.len());
        assert_eq!(las_points.point_layout(), last_points.point_layout());

        let las_point_records = las_points.get_point_range_ref(0..las_points.len());
        let last_point_records = last_points.get_point_range_ref(0..las_points.len());
        for (idx, (las_point, last_point)) in las_point_records
            .chunks(header.point_data_record_length as usize)
            .zip(last_point_records.chunks(header.point_data_record_length as usize))
            .enumerate()
        {
            assert_eq!(
                las_point, last_point,
                "Point records at index {idx} are different"
            );
        }

        Ok(())
    }
}
