use std::{
    collections::HashMap,
    convert::TryInto,
    fs::File,
    io::{BufReader, Read, Seek, SeekFrom},
    ops::Range,
    path::Path,
};

use byteorder::{LittleEndian, ReadBytesExt};
use pasture_core::{
    containers::{
        PerAttributePointView, PerAttributeVecPointStorage, PointBuffer, PointBufferWriteable,
    },
    layout::{
        attributes::{CLASSIFICATION, COLOR_RGB, INTENSITY, POSITION_3D},
        conversion::{get_converter_for_attributes, AttributeConversionFn},
        FieldAlignment, PointAttributeMember, PointLayout, PrimitiveType,
    },
    nalgebra::Vector3,
    util::view_raw_bytes,
};
use pasture_io::{
    base::{PointReader, SeekToPoint},
    las::LASMetadata,
    las_rs::{
        raw::{self},
        Header,
    },
};

use anyhow::{Context, Result};

fn push_convert_attribute<T: PrimitiveType>(
    val: T,
    attribute: &PointAttributeMember,
    index: usize,
    target_slice: &mut [u8],
    maybe_converter: &Option<AttributeConversionFn>,
) {
    let target_start_idx = index * attribute.size() as usize;
    let target_end_idx = target_start_idx + attribute.size() as usize;
    let target_slice = &mut target_slice[target_start_idx..target_end_idx];
    if let Some(ref converter) = maybe_converter {
        unsafe {
            converter(view_raw_bytes(&val), target_slice);
        }
    } else {
        unsafe {
            target_slice.copy_from_slice(view_raw_bytes(&val));
        }
    }
}

trait SeekRead: Seek + Read {}
impl<T: Seek + Read> SeekRead for T {}

pub struct LASTReader {
    last_reader: Box<dyn SeekRead>,
    raw_las_header: raw::Header,
    metadata: LASMetadata,
    default_point_layout: PointLayout,
    current_point_index: usize,
    byte_offsets_to_attributes: HashMap<&'static str, usize>,
}

impl LASTReader {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let reader = BufReader::new(File::open(path)?);
        Self::from(reader)
    }

    pub fn from<R: 'static + std::io::Read + Seek>(mut reader: R) -> Result<Self> {
        let mut raw_las_header =
            Self::read_las_header(&mut reader).context("Error while trying to read LAS header")?;
        // LAZ files sometimes have the seventh bit (128) set inside the point_data_record_format to indicate that they are
        // compressed. Pretty sure this is invalid, in any case the las_rs library doesn't correctly ignore this, so we blank
        // out the higher bits here manually
        raw_las_header.point_data_record_format &= 0b1111;
        let las_header = Header::from_raw(raw_las_header.clone())
            .context("Error while trying to parse LAS header")?;

        let offset_to_point_data: usize = raw_las_header
            .offset_to_point_data
            .try_into()
            .expect("Offset to point data could not be converted to usize");

        // Gather byte offsets to Attribute Records
        let mut byte_offsets_to_attributes = HashMap::new();
        byte_offsets_to_attributes.insert(POSITION_3D.name(), offset_to_point_data);

        let size_of_positions: usize = (12 * las_header.number_of_points())
            .try_into()
            .expect("Size of positions could not be converted to usize");
        byte_offsets_to_attributes
            .insert(INTENSITY.name(), offset_to_point_data + size_of_positions);

        if las_header.point_format().is_extended {
            // Extended format uses 2 bytes for bit attributes instead of 1 byte
            let size_of_blocks_prior_to_classification: usize = (16
                * las_header.number_of_points())
            .try_into()
            .expect("Size of blocks prior to classification could not be converted to usize");
            byte_offsets_to_attributes.insert(
                CLASSIFICATION.name(),
                offset_to_point_data + size_of_blocks_prior_to_classification,
            );
        } else {
            let size_of_blocks_prior_to_classification: usize = (15
                * las_header.number_of_points())
            .try_into()
            .expect("Size of blocks prior to classification could not be converted to usize");
            byte_offsets_to_attributes.insert(
                CLASSIFICATION.name(),
                offset_to_point_data + size_of_blocks_prior_to_classification,
            );
        }

        let mut point_layout =
            PointLayout::from_attributes(&[POSITION_3D, INTENSITY, CLASSIFICATION]);

        if las_header.point_format().has_color {
            let offset_to_colors_in_single_point_record: usize = match las_header
                .point_format()
                .to_u8()
                .expect("Could not get LAS format number")
            {
                2 => 20,
                3 => 28,
                5 => 28,
                7 => 30,
                8 => 30,
                10 => 30,
                _ => 0,
            };
            let size_of_blocks_prior_to_colors_block: usize =
                offset_to_colors_in_single_point_record * (las_header.number_of_points() as usize);
            byte_offsets_to_attributes.insert(
                COLOR_RGB.name(),
                offset_to_point_data + size_of_blocks_prior_to_colors_block,
            );

            point_layout.add_attribute(COLOR_RGB, FieldAlignment::Packed(1));
        }

        Ok(Self {
            last_reader: Box::new(reader),
            raw_las_header: raw_las_header,
            metadata: LASMetadata::from(&las_header),
            default_point_layout: point_layout,
            current_point_index: 0,
            byte_offsets_to_attributes: byte_offsets_to_attributes,
        })
    }

    fn read_las_header<R: std::io::Read>(reader: R) -> Result<raw::Header> {
        let raw_header =
            raw::Header::read_from(reader).context("Error while trying to read LAS header")?;
        Ok(raw_header)
    }

    fn read_attribute(
        &mut self,
        range: Range<usize>,
        target_attribute: &PointAttributeMember,
        target_buffer: &mut [u8],
    ) -> Result<()> {
        // Read the corresponding attribute, if it exists in a LAST file (LAST has the same attributes as LAS)
        // TODO Implement reading of more LAS attributes. For now, these four will do
        match target_attribute.name() {
            "Position3D" => {
                let maybe_converter =
                    get_converter_for_attributes(&POSITION_3D, &target_attribute.into());
                let offset_of_first_position = *self
                    .byte_offsets_to_attributes
                    .get(POSITION_3D.name())
                    .unwrap();

                let size_of_single_position_in_file = 12;
                let offset_to_target_position =
                    offset_of_first_position + (size_of_single_position_in_file * range.start);
                self.last_reader
                    .seek(SeekFrom::Start(offset_to_target_position as u64))?;

                let num_to_read = range.len();
                for idx in 0..num_to_read {
                    let x = self.last_reader.read_i32::<LittleEndian>()?;
                    let y = self.last_reader.read_i32::<LittleEndian>()?;
                    let z = self.last_reader.read_i32::<LittleEndian>()?;

                    let world_space_pos = Vector3::new(
                        self.raw_las_header.x_offset
                            + self.raw_las_header.x_scale_factor * x as f64,
                        self.raw_las_header.y_offset
                            + self.raw_las_header.y_scale_factor * y as f64,
                        self.raw_las_header.z_offset
                            + self.raw_las_header.z_scale_factor * z as f64,
                    );

                    push_convert_attribute(
                        world_space_pos,
                        target_attribute,
                        idx,
                        target_buffer,
                        &maybe_converter,
                    );
                }
            }
            "Intensity" => {
                let maybe_converter =
                    get_converter_for_attributes(&INTENSITY, &target_attribute.into());
                let offset_to_first_intensity = *self
                    .byte_offsets_to_attributes
                    .get(INTENSITY.name())
                    .unwrap();

                let size_of_single_intensity_in_file = 2;
                let offset_to_target_intensity =
                    offset_to_first_intensity + (size_of_single_intensity_in_file * range.start);
                self.last_reader
                    .seek(SeekFrom::Start(offset_to_target_intensity as u64))?;

                let num_to_read = range.len();
                for idx in 0..num_to_read {
                    let intensity = self.last_reader.read_u16::<LittleEndian>()?;
                    push_convert_attribute(
                        intensity,
                        target_attribute,
                        idx,
                        target_buffer,
                        &maybe_converter,
                    );
                }
            }
            "ColorRGB" => {
                if let Some(offset_to_first_color) =
                    self.byte_offsets_to_attributes.get(COLOR_RGB.name())
                {
                    let maybe_converter =
                        get_converter_for_attributes(&COLOR_RGB, &target_attribute.into());

                    let size_of_single_color_in_file = 6;
                    let offset_to_target_color =
                        offset_to_first_color + (size_of_single_color_in_file * range.start);
                    self.last_reader
                        .seek(SeekFrom::Start(offset_to_target_color as u64))?;

                    let num_to_read = range.len();
                    for idx in 0..num_to_read {
                        let r = self.last_reader.read_u16::<LittleEndian>()?;
                        let g = self.last_reader.read_u16::<LittleEndian>()?;
                        let b = self.last_reader.read_u16::<LittleEndian>()?;

                        push_convert_attribute(
                            Vector3::new(r, g, b),
                            target_attribute,
                            idx,
                            target_buffer,
                            &maybe_converter,
                        );
                    }
                }
            }
            "Classification" => {
                let maybe_converter =
                    get_converter_for_attributes(&CLASSIFICATION, &target_attribute.into());
                let offset_to_first_classification = *self
                    .byte_offsets_to_attributes
                    .get(CLASSIFICATION.name())
                    .unwrap();

                let size_of_single_classification_in_file = 1;
                let offset_to_target_classification = offset_to_first_classification
                    + (size_of_single_classification_in_file * range.start);
                self.last_reader
                    .seek(SeekFrom::Start(offset_to_target_classification as u64))?;

                let num_to_read = range.len();
                for idx in 0..num_to_read {
                    let class = self.last_reader.read_u8()?;
                    push_convert_attribute(
                        class,
                        target_attribute,
                        idx,
                        target_buffer,
                        &maybe_converter,
                    );
                }
            }
            _ => (),
        }

        Ok(())
    }
}

impl PointReader for LASTReader {
    fn read_into(
        &mut self,
        point_buffer: &mut dyn PointBufferWriteable,
        count: usize,
    ) -> Result<usize> {
        if point_buffer.as_per_attribute().is_none() {
            panic!("LASTReader::read_into is only supported for PerAttributePointBuffer types!");
        }

        let num_points_to_read = usize::min(
            count,
            self.metadata.point_count() - self.current_point_index,
        );

        if num_points_to_read == 0 {
            return Ok(0);
        }

        let first_point_index = self.current_point_index;
        let last_point_index = first_point_index + num_points_to_read;

        let target_layout = point_buffer.point_layout().clone();

        let chunk_size = usize::min(num_points_to_read, 50_000 as usize);
        let mut temporary_buffers_per_attribute = target_layout
            .attributes()
            .map(|a| vec![0; a.size() as usize * chunk_size])
            .collect::<Vec<Vec<u8>>>();

        let first_chunk_idx = first_point_index / chunk_size;
        let last_chunk_idx = last_point_index / chunk_size;

        for _ in first_chunk_idx..=last_chunk_idx {
            let current_chunk = self.current_point_index / chunk_size;
            let next_chunk = current_chunk + 1;
            let last_point_in_current_chunk = usize::min(last_point_index, next_chunk * chunk_size);
            let to_read_from_current_chunk = last_point_in_current_chunk - self.current_point_index;

            // For each attribute in the TARGET layout:
            //   Read 'to_read_from_current_chunk' attribute values from the current chunk, starting at point self.current_point_index
            for (attribute_idx, target_attribute) in target_layout.attributes().enumerate() {
                let target_buffer = &mut temporary_buffers_per_attribute[attribute_idx];
                self.read_attribute(
                    self.current_point_index
                        ..(self.current_point_index + to_read_from_current_chunk),
                    target_attribute,
                    target_buffer,
                )?;
            }

            // Push from temporary buffers into point_buffer
            let current_slices = target_layout
                .attributes()
                .enumerate()
                .map(|(idx, attribute)| {
                    let size = to_read_from_current_chunk * attribute.size() as usize;
                    &temporary_buffers_per_attribute[idx][..size]
                })
                .collect::<Vec<_>>();
            point_buffer.push(&PerAttributePointView::from_slices(
                current_slices,
                target_layout.clone(),
            ));

            // Update current point index
            self.current_point_index += to_read_from_current_chunk;
        }

        Ok(num_points_to_read)
    }

    fn read(&mut self, count: usize) -> Result<Box<dyn PointBuffer>> {
        let num_points_to_read = usize::min(
            count,
            self.metadata.point_count() - self.current_point_index,
        );
        let mut buffer = PerAttributeVecPointStorage::with_capacity(
            num_points_to_read,
            self.get_default_point_layout().clone(),
        );
        self.read_into(&mut buffer, num_points_to_read)?;
        Ok(Box::new(buffer))
    }

    fn get_metadata(&self) -> &dyn pasture_core::meta::Metadata {
        &self.metadata
    }

    fn get_default_point_layout(&self) -> &pasture_core::layout::PointLayout {
        &self.default_point_layout
    }
}

impl SeekToPoint for LASTReader {
    fn seek_point(&mut self, _position: SeekFrom) -> Result<usize> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use pasture_core::containers::PointBufferExt;
    use pasture_io::las::LASReader;

    use super::*;

    #[test]
    fn test_lazer_reader() -> Result<()> {
        let mut las_source = LASReader::from_path("/Users/pbormann/data/geodata/pointclouds/datasets/navvis_m6_3rdFloor/navvis_m6_HQ3rdFloor.laz")?;
        let mut last_source = LASTReader::new("/Users/pbormann/data/projects/progressive_indexing/experiment_data/navvis_m6_HQ3rdFloor.last")?;

        let chunk_size = 60000;
        for chunk_idx in 0..10 {
            let las_points = las_source.read(chunk_size)?;
            let last_points = last_source.read(chunk_size)?;

            for idx in 0..chunk_size {
                let las_pos = las_points.get_attribute::<Vector3<f64>>(&POSITION_3D, idx);
                let lazer_pos = last_points.get_attribute::<Vector3<f64>>(&POSITION_3D, idx);
                assert_eq!(
                    las_pos,
                    lazer_pos,
                    "Position {} is different",
                    (chunk_idx * chunk_size) + idx
                );

                let las_col = las_points.get_attribute::<Vector3<u16>>(&COLOR_RGB, idx);
                let lazer_col = last_points.get_attribute::<Vector3<u16>>(&COLOR_RGB, idx);
                assert_eq!(
                    las_col,
                    lazer_col,
                    "Color {} is different",
                    (chunk_idx * chunk_size) + idx
                );
            }
        }

        Ok(())
    }
}
