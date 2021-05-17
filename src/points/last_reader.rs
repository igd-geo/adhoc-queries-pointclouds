use std::{collections::HashMap, convert::TryInto, fs::File, io::{BufReader, Read, Seek, SeekFrom}, path::Path};

use byteorder::{LittleEndian, ReadBytesExt};
use pasture_core::{containers::{PerAttributeVecPointStorage, PointBuffer, PointBufferWriteable}, layout::{PointAttributeDefinition, attributes::{CLASSIFICATION, COLOR_RGB, INTENSITY, POSITION_3D}}};
use pasture_io::{base::{PointReader, SeekToPoint}, las::LASMetadata, las_rs::{Header, raw}};

use anyhow::Result;

trait SeekRead: Seek + Read {}
impl<T: Seek + Read> SeekRead for T {}

pub struct LASTReader {
    last_reader: Box<dyn SeekRead>,
    raw_las_header: raw::Header,
    metadata: LASMetadata,
    current_point_index: usize,
    byte_offsets_to_attributes: HashMap<&'static str, usize>,
}

impl LASTReader {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let reader = BufReader::new(File::open(path)?);
        Self::from(reader)
    }

    pub fn from<R: 'static + std::io::Read + Seek>(mut reader: R) -> Result<Self> {
        let raw_las_header = Self::read_las_header(&mut reader)?;
        let las_header = Header::from_raw(raw_las_header.clone())?;

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
        byte_offsets_to_attributes.insert(
            INTENSITY.name(),
            offset_to_point_data + size_of_positions,
        );

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
        }

        Ok(Self {
            last_reader: Box::new(reader),
            raw_las_header: raw_las_header,
            metadata: LASMetadata::from(&las_header),
            current_point_index: 0,
            byte_offsets_to_attributes: byte_offsets_to_attributes,
        })
    }

    fn read_las_header<R: std::io::Read>(reader: R) -> Result<raw::Header> {
        let raw_header = raw::Header::read_from(reader)?;
        Ok(raw_header)
    }
}

impl PointReader for LASTReader {
    fn read_into(
        &mut self,
        point_buffer: &mut dyn PointBufferWriteable,
        count: usize,
    ) -> Result<usize> {
        let num_points_to_read = usize::min(
            count,
            self.metadata.point_count() - self.current_point_index,
        );

        if num_points_to_read == 0 {
            return Ok(0);
        }

        // Read positions
        let positions_offset = self
            .byte_offsets_to_attributes
            .get(&POSITION_3D.name())
            .expect("Offset to positions not found in byte offsets map");
        let offset_of_current_position: u64 = (positions_offset + (12 * self.current_point_index))
            .try_into()
            .unwrap();
        self.last_reader
            .seek(SeekFrom::Start(offset_of_current_position))?;

            todo!()

        // for idx in 0..num_points_to_read {
        //     let x = self.last_reader.read_i32::<LittleEndian>()?;
        //     let y = self.last_reader.read_i32::<LittleEndian>()?;
        //     let z = self.last_reader.read_i32::<LittleEndian>()?;

        //     point_buffer.positions_mut()[idx] = Vector3::new(
        //         self.raw_las_header.x_offset + (x as f64 * self.raw_las_header.x_scale_factor),
        //         self.raw_las_header.y_offset + (y as f64 * self.raw_las_header.y_scale_factor),
        //         self.raw_las_header.z_offset + (z as f64 * self.raw_las_header.z_scale_factor),
        //     );
        // }

        // // TODO Read other attributes
        // // Read intensities
        // if point_buffer.intensities_mut().is_some() {
        //     let intensities_offset = self
        //         .byte_offsets_to_attributes
        //         .get(&PointAttributes::Intensity)
        //         .expect("Offset to intensities not found in byte offsets map");
        //     let offset_of_current_intensity: u64 = (intensities_offset
        //         + (2 * self.current_point_index))
        //         .try_into()
        //         .unwrap();
        //     self.last_reader
        //         .seek(SeekFrom::Start(offset_of_current_intensity))?;

        //     for idx in 0..num_points_to_read {
        //         let intensity = self.last_reader.read_u16::<LittleEndian>()?;

        //         point_buffer.intensities_mut().unwrap()[idx] = intensity;
        //     }
        // }

        // if point_buffer.classifications_mut().is_some() {
        //     let classifications_offset = self
        //         .byte_offsets_to_attributes
        //         .get(&PointAttributes::Classification)
        //         .expect("Offset to classifications not found in byte offsets map");
        //     let offset_of_current_classification: u64 = (classifications_offset
        //         + (1 * self.current_point_index))
        //         .try_into()
        //         .unwrap();
        //     self.last_reader
        //         .seek(SeekFrom::Start(offset_of_current_classification))?;

        //     for idx in 0..num_points_to_read {
        //         let classification = self.last_reader.read_u8()?;
        //         point_buffer.classifications_mut().unwrap()[idx] = classification;
        //     }
        // }

        // if point_buffer.colors_mut().is_some() {
        //     let colors_offset = self
        //         .byte_offsets_to_attributes
        //         .get(&PointAttributes::Color)
        //         .expect("Offset to colors not found in byte offsets map");
        //     let offset_of_current_color: u64 = (colors_offset + (6 * self.current_point_index))
        //         .try_into()
        //         .unwrap();
        //     self.last_reader
        //         .seek(SeekFrom::Start(offset_of_current_color))?;

        //     for idx in 0..num_points_to_read {
        //         let r = self.last_reader.read_u16::<LittleEndian>()?;
        //         let g = self.last_reader.read_u16::<LittleEndian>()?;
        //         let b = self.last_reader.read_u16::<LittleEndian>()?;
        //         point_buffer.colors_mut().unwrap()[idx] =
        //             Vector3::new((r >> 8) as u8, (g >> 8) as u8, (b >> 8) as u8);
        //     }
        // }

        // self.current_point_index += num_points_to_read;

        // Ok(())
    }

    fn read(&mut self, count: usize) -> Result<Box<dyn PointBuffer>> {
        todo!()
    }

    fn get_metadata(&self) -> &dyn pasture_core::meta::Metadata {
        todo!()
    }

    fn get_default_point_layout(&self) -> &pasture_core::layout::PointLayout {
        todo!()
    }
}

impl SeekToPoint for LASTReader {
    fn seek_point(&mut self, position: SeekFrom) -> Result<usize> {
        todo!()
    }
}