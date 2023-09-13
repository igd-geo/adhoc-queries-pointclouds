use anyhow::Result;
use byteorder::{LittleEndian, ReadBytesExt};
use lz4::Decoder;
use pasture_core::{
    containers::{
        InterleavedPointBufferMut, PerAttributePointView, PerAttributeVecPointStorage,
        PointBufferWriteable,
    },
    layout::{
        attributes::{CLASSIFICATION, COLOR_RGB, INTENSITY, POSITION_3D},
        conversion::{get_converter_for_attributes, AttributeConversionFn},
        FieldAlignment, PointLayout,
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

use std::collections::HashMap;
use std::convert::TryInto;

use std::fs::File;
use std::io::{BufReader, Cursor, Read, Seek, SeekFrom};
use std::path::Path;

trait SeekRead: Seek + Read {}
impl<T: Seek + Read> SeekRead for T {}

pub struct LAZERSource {
    reader: Box<dyn SeekRead>,
    raw_las_header: raw::Header,
    metadata: LASMetadata,
    default_point_layout: PointLayout,
    has_colors: bool,
    current_point_index: usize,
    block_size: u64,
    number_of_attributes: usize,
    block_offsets: Vec<u64>,
    block_byte_sizes: Vec<u64>,
    current_block_cache: Box<[u8]>,
    decoders_per_attribute: HashMap<&'static str, Decoder<Cursor<&'static [u8]>>>,
}

impl LAZERSource {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let reader = BufReader::new(File::open(path)?);
        Self::from(reader)
    }

    pub fn from<R: 'static + std::io::Read + Seek>(mut reader: R) -> Result<Self> {
        let raw_las_header = Self::read_las_header(&mut reader)?;
        let las_header = Header::from_raw(raw_las_header.clone())?;

        let offset_to_point_data: u64 = raw_las_header.offset_to_point_data.into();
        reader.seek(SeekFrom::Start(offset_to_point_data))?;

        // Read the block size and the block offsets
        let block_size = reader.read_u64::<LittleEndian>()?;
        let num_blocks = (las_header.number_of_points() + (block_size - 1)) / block_size;

        let mut block_offsets = Vec::with_capacity(num_blocks as usize);
        for _ in 0..num_blocks {
            block_offsets.push(reader.read_u64::<LittleEndian>()?);
        }

        // Get size of file
        let cur_reader_pos = reader.seek(SeekFrom::Current(0))?;
        let file_size = reader.seek(SeekFrom::End(0))?;
        reader.seek(SeekFrom::Start(cur_reader_pos))?;

        let mut block_byte_sizes = Vec::with_capacity(num_blocks as usize);
        for block_idx in 0..num_blocks {
            let size_of_block = if block_idx == num_blocks - 1 {
                file_size - block_offsets[block_idx as usize]
            } else {
                block_offsets[block_idx as usize + 1] - block_offsets[block_idx as usize]
            };
            block_byte_sizes.push(size_of_block);
        }

        // TODO More attributes from LAS format are supported by LAZER, but for this showcase, we only use these 4
        let mut point_layout =
            PointLayout::from_attributes(&[POSITION_3D, INTENSITY, CLASSIFICATION]);

        let mut number_of_attributes = 8;
        if las_header.point_format().has_color {
            number_of_attributes += 1;
            point_layout.add_attribute(COLOR_RGB, FieldAlignment::Packed(1));
        }
        if las_header.point_format().has_gps_time {
            number_of_attributes += 1;
        }
        if las_header.point_format().has_waveform {
            number_of_attributes += 1;
        }
        if las_header.point_format().has_nir {
            number_of_attributes += 1;
        }

        let mut myself = Self {
            reader: Box::new(reader),
            raw_las_header: raw_las_header,
            metadata: LASMetadata::from(&las_header),
            default_point_layout: point_layout,
            has_colors: las_header.point_format().has_color,
            current_point_index: 0,
            number_of_attributes: number_of_attributes,
            block_size: block_size,
            block_offsets: block_offsets,
            block_byte_sizes: block_byte_sizes,
            current_block_cache: Vec::new().into_boxed_slice(),
            decoders_per_attribute: HashMap::new(),
        };

        myself.move_decoders_to_point_in_block(0, 0)?;

        Ok(myself)
    }

    pub fn block_size(&self) -> u64 {
        self.block_size
    }

    fn read_las_header<R: std::io::Read>(reader: R) -> Result<raw::Header> {
        let raw_header = raw::Header::read_from(reader)?;
        Ok(raw_header)
    }

    fn move_decoders_to_point_in_block(
        &mut self,
        block_index: usize,
        point_in_block: usize,
    ) -> Result<()> {
        // This only works by reading the full compressed block into a temporary memory buffer
        // and pointing each decoder to a specific non-overlapping section in this buffer
        // Block contains one u64 attribute offset per attribute, followed by the compressed attribute blobs

        let block_offset_in_file = self.block_offsets[block_index];
        self.reader.seek(SeekFrom::Start(block_offset_in_file))?;

        // read the offsets to the compressed blobs
        let offsets_to_compressed_blobs = (0..self.number_of_attributes)
            .map(|_| self.reader.read_u64::<LittleEndian>())
            .collect::<Result<Vec<_>, _>>()?;

        // The offsets in the block header are relative to the file, we want them relative to the
        // first byte of the first compressed attribute in the block because we read only the compressed
        // attributes into self.current_block_cache
        let offset_to_compressed_blobs_in_cache = offsets_to_compressed_blobs
            .iter()
            .map(|offset| offset - offsets_to_compressed_blobs[0])
            .collect::<Vec<_>>();

        // read all compressed blobs into a single buffer
        let block_size = self.block_byte_sizes[block_index];
        let compressed_attributes_size = block_size - (self.number_of_attributes as u64 * 8);
        if self.current_block_cache.len() < compressed_attributes_size as usize {
            self.current_block_cache =
                (vec![0; compressed_attributes_size as usize]).into_boxed_slice();
        }

        // Read exactly compressed_attributes_size bytes. self.current_block_cache may be larger
        // because a previous block was larger
        self.reader
            .read_exact(&mut self.current_block_cache[0..compressed_attributes_size as usize])?;

        // split the buffer into chunks based on the offsets that we read. Create Decoders for each chunk

        let offset_to_positions = offset_to_compressed_blobs_in_cache[0] as usize;
        let offset_to_blob_after_positions = offset_to_compressed_blobs_in_cache[1] as usize;

        // TODO The Cursor inside the Decoder references memory that belongs to self, so we need
        // an appropriate lifetime. The current method is called form a trait method however and
        // this trait method has no lifetime...
        let mut positions_decoder = unsafe {
            let ptr = self.current_block_cache.as_ptr();
            let start_ptr = ptr.add(offset_to_positions);
            let slice = std::slice::from_raw_parts(
                start_ptr,
                offset_to_blob_after_positions - offset_to_positions,
            );

            Decoder::new(Cursor::new(slice))?
        };
        // Move decoder to correct point index
        for _ in 0..point_in_block {
            // TODO Could maybe be a single read call because we know the size (point_in_block * 12)
            positions_decoder.read_i32::<LittleEndian>()?;
            positions_decoder.read_i32::<LittleEndian>()?;
            positions_decoder.read_i32::<LittleEndian>()?;
        }
        self.decoders_per_attribute
            .insert(POSITION_3D.name(), positions_decoder);

        let offset_to_intensities = offset_to_compressed_blobs_in_cache[1] as usize;
        let offset_to_blob_after_intensities = offset_to_compressed_blobs_in_cache[2] as usize;

        let mut intensities_decoder = unsafe {
            let ptr = self.current_block_cache.as_ptr();
            let start_ptr = ptr.add(offset_to_intensities);
            let slice = std::slice::from_raw_parts(
                start_ptr,
                offset_to_blob_after_intensities - offset_to_intensities,
            );

            Decoder::new(Cursor::new(slice))?
        };
        for _ in 0..point_in_block {
            intensities_decoder.read_u16::<LittleEndian>()?;
        }
        self.decoders_per_attribute
            .insert(INTENSITY.name(), intensities_decoder);

        let offset_to_classifications = offset_to_compressed_blobs_in_cache[3] as usize;
        let offset_to_blob_after_classifications = offset_to_compressed_blobs_in_cache[4] as usize;
        let mut classifications_decoder = unsafe {
            let ptr = self.current_block_cache.as_ptr();
            let start_ptr = ptr.add(offset_to_classifications);
            let slice = std::slice::from_raw_parts(
                start_ptr,
                offset_to_blob_after_classifications - offset_to_classifications,
            );

            Decoder::new(Cursor::new(slice))?
        };
        for _ in 0..point_in_block {
            classifications_decoder.read_u8()?;
        }
        self.decoders_per_attribute
            .insert(CLASSIFICATION.name(), classifications_decoder);

        if self.has_colors {
            let offset_to_colors = offset_to_compressed_blobs_in_cache[8] as usize;
            let offset_to_blob_after_colors = if offset_to_compressed_blobs_in_cache.len() > 9 {
                offset_to_compressed_blobs_in_cache[9] as usize
            } else {
                block_offset_in_file as usize + block_size as usize
            };
            let mut colors_decoder = unsafe {
                let ptr = self.current_block_cache.as_ptr();
                let start_ptr = ptr.add(offset_to_colors);
                let slice = std::slice::from_raw_parts(
                    start_ptr,
                    offset_to_blob_after_colors - offset_to_colors,
                );
                Decoder::new(Cursor::new(slice))?
            };
            for _ in 0..point_in_block {
                colors_decoder.read_u16::<LittleEndian>()?;
                colors_decoder.read_u16::<LittleEndian>()?;
                colors_decoder.read_u16::<LittleEndian>()?;
            }
            self.decoders_per_attribute
                .insert(COLOR_RGB.name(), colors_decoder);
        }

        Ok(())
    }

    fn is_last_block(&self, block_index: u64) -> bool {
        (block_index as usize) == (self.block_offsets.len() - 1)
    }

    fn _read_into_interleaved(
        &mut self,
        _point_buffer: &dyn InterleavedPointBufferMut,
        _count: usize,
    ) -> Result<usize> {
        todo!()
        // let num_points_to_read = usize::min(
        //     count,
        //     self.metadata.point_count() - self.current_point_index,
        // ) as u64;
        // if num_points_to_read == 0 {
        //     return Ok(0);
        // }

        // let first_block_index = self.current_point_index as u64 / self.block_size;
        // let last_block_index_inclusive =
        //     (self.current_point_index as u64 + num_points_to_read) / self.block_size;

        // let first_point_index = self.current_point_index as u64;
        // let last_point_index = first_point_index + num_points_to_read;

        // fn get_attribute_parser(
        //     name: &str,
        //     source_layout: &PointLayout,
        //     target_layout: &PointLayout,
        // ) -> Option<(usize, usize, Option<AttributeConversionFn>)> {
        //     target_layout
        //         .get_attribute_by_name(name)
        //         .map_or(None, |target_attribute| {
        //             let converter =
        //                 source_layout
        //                     .get_attribute_by_name(name)
        //                     .and_then(|source_attribute| {
        //                         get_converter_for_attributes(
        //                             &source_attribute.into(),
        //                             &target_attribute.into(),
        //                         )
        //                     });
        //             let offset_of_attribute = target_attribute.offset() as usize;
        //             let size_of_attribute = target_attribute.size() as usize;
        //             Some((offset_of_attribute, size_of_attribute, converter))
        //         })
        // }

        // fn run_parser<U>(
        //     decoder_fn: impl Fn(Option<&mut Decoder<Cursor<&[u8]>>>) -> Result<U>,
        //     maybe_parser: Option<(usize, usize, Option<AttributeConversionFn>)>,
        //     start_of_target_point_in_chunk: usize,
        //     size_of_attribute: Option<usize>,
        //     decoder: Option<&mut Decoder<Cursor<&[u8]>>>,
        //     chunk_buffer: &mut [u8],
        // ) -> Result<()> {
        //     if let Some((offset, size, maybe_converter)) = maybe_parser {
        //         let source_data = decoder_fn(decoder)?;
        //         let source_slice = unsafe { view_raw_bytes(&source_data) };

        //         let pos_start = start_of_target_point_in_chunk + offset;
        //         let pos_end = pos_start + size;
        //         let target_slice = &mut chunk_buffer[pos_start..pos_end];

        //         if let Some(converter) = maybe_converter {
        //             unsafe {
        //                 converter(source_slice, target_slice);
        //             }
        //         } else {
        //             target_slice.copy_from_slice(source_slice);
        //         }
        //     } else if let Some(bytes_to_skip) = size_of_attribute {
        //         if let Some(actual_decoder) = decoder {
        //             for _ in 0..bytes_to_skip {
        //                 actual_decoder.read_u8()?;
        //             }
        //         }
        //     }

        //     Ok(())
        // }

        // let source_layout = self.get_default_point_layout();
        // let target_layout = point_buffer.point_layout().clone();
        // let target_point_size = target_layout.size_of_point_entry();

        // // This format currently only supports positions, intensities, classififcations and colors
        // let position_parser = get_attribute_parser(POSITION_3D.name(), source_layout, &target_layout);
        // let intensity_parser = get_attribute_parser(INTENSITY.name(), source_layout, &target_layout);
        // let classification_parser = get_attribute_parser(CLASSIFICATION.name(), source_layout, &target_layout);
        // let color_parser = get_attribute_parser(COLOR_RGB.name(), source_layout, &target_layout);

        // let mut block_buffer = vec![0u8; (target_point_size * self.block_size).try_into().unwrap()];

        // for block_idx in first_block_index..last_block_index_inclusive + 1 {
        //     let block_start_point = block_idx * self.block_size;
        //     let point_in_block_start = if first_point_index < block_start_point {
        //         0
        //     } else {
        //         first_point_index - block_start_point
        //     };
        //     let point_in_block_end =
        //         u64::min(last_point_index - block_start_point, self.block_size);

        //     let num_points_to_read_cur_block = point_in_block_end - point_in_block_start;

        //     // Read data from block

        //     {
        //         let positions_decoder = self
        //             .decoders_per_attribute
        //             .get_mut(&POSITION_3D.name())
        //             .expect("Positions Decoder was None");

        //         let offset_x = self.raw_las_header.x_offset;
        //         let offset_y = self.raw_las_header.y_offset;
        //         let offset_z = self.raw_las_header.z_offset;
        //         let scale_x = self.raw_las_header.x_scale_factor;
        //         let scale_y = self.raw_las_header.y_scale_factor;
        //         let scale_z = self.raw_las_header.z_scale_factor;

        //         for idx in 0..num_points_to_read_cur_block {
        //             let start_of_target_point_in_chunk = (idx * target_point_size) as usize;

        //             run_parser(|maybe_decoder| {
        //                 let decoder = maybe_decoder.unwrap();
        //                 let x = decoder.read_i32::<LittleEndian>()?;
        //                 let y = decoder.read_i32::<LittleEndian>()?;
        //                 let z = decoder.read_i32::<LittleEndian>()?;

        //                 Ok(Vector3::new(
        //                     offset_x + (x as f64 * scale_x),
        //                     offset_y + (y as f64 * scale_y),
        //                     offset_z + (z as f64 * scale_z),
        //                 ))
        //             },
        //                 position_parser, start_of_target_point_in_chunk, Some(12), Some(positions_decoder), &mut block_buffer)?;
        //         }
        //     }

        //     {
        //         let intensities_decoder = self
        //         .decoders_per_attribute
        //         .get_mut(&INTENSITY.name())
        //         .expect("Intensity Decoder was None");

        //         for idx in 0..num_points_to_read_cur_block {
        //             let start_of_target_point_in_chunk = (idx * target_point_size) as usize;

        //             run_parser(|maybe_decoder| {
        //                 let decoder = maybe_decoder.unwrap();
        //                 let intensity = decoder.read_u16::<LittleEndian>()?;
        //                 Ok(intensity)
        //             },
        //                 intensity_parser, start_of_target_point_in_chunk, Some(2), Some(intensities_decoder), &mut block_buffer)?;
        //         }
        //     }

        //     {
        //         let classifications_decoder = self
        //             .decoders_per_attribute
        //             .get_mut(&CLASSIFICATION.name())
        //             .expect("Classification Decoder was None");

        //             for idx in 0..num_points_to_read_cur_block {
        //                 let start_of_target_point_in_chunk = (idx * target_point_size) as usize;

        //                 run_parser(|maybe_decoder| {
        //                     let decoder = maybe_decoder.unwrap();
        //                     let classification = decoder.read_u8()?;
        //                     Ok(classification)
        //                 },
        //                     classification_parser, start_of_target_point_in_chunk, Some(1), Some(classifications_decoder), &mut block_buffer)?;
        //             }
        //         }

        //         {
        //             let colors_decoder = self
        //             .decoders_per_attribute
        //             .get_mut(&COLOR_RGB.name());

        //             match colors_decoder {
        //                 Some(dec) => {
        //                     for idx in 0..num_points_to_read_cur_block {
        //                         let start_of_target_point_in_chunk = (idx * target_point_size) as usize;

        //                         run_parser(|decoder| {
        //                             match decoder {
        //                                 Some(dec) => {
        //                                     let r = dec.read_u16::<LittleEndian>()?;
        //                                     let g = dec.read_u16::<LittleEndian>()?;
        //                                     let b = dec.read_u16::<LittleEndian>()?;
        //                                     Ok(Vector3::new(r,g,b))
        //                                 },
        //                                 None => {
        //                                     Ok(Vector3::default())
        //                                 }
        //                             }
        //                         },
        //                             color_parser, start_of_target_point_in_chunk, Some(6), Some(dec), &mut block_buffer)?;
        //                     }
        //                 },
        //                 None => {
        //                     for idx in 0..num_points_to_read_cur_block {
        //                         let start_of_target_point_in_chunk = (idx * target_point_size) as usize;

        //                         run_parser(|decoder| {
        //                             match decoder {
        //                                 Some(dec) => {
        //                                     let r = dec.read_u16::<LittleEndian>()?;
        //                                     let g = dec.read_u16::<LittleEndian>()?;
        //                                     let b = dec.read_u16::<LittleEndian>()?;
        //                                     Ok(Vector3::new(r,g,b))
        //                                 },
        //                                 None => {
        //                                     Ok(Vector3::default())
        //                                 }
        //                             }
        //                         },
        //                             color_parser, start_of_target_point_in_chunk, Some(6), None, &mut block_buffer)?;
        //                     }
        //                 }
        //             }

        //         }

        //         // Push data into point_buffer
        //         // TODO The parsers assume that the 'block_buffer' stores interleaved data. This might be inefficient in case that the
        //         // 'point_buffer' stores per-attribute data, because LAZER also stores per-attribute data...
        //         point_buffer.push(&InterleavedPointView::from_raw_slice(block_buffer.as_slice(), target_layout.clone()));

        //     // If we finished reading this block, then we have to move to the next block
        //     // This is because read_into assumes that the decoders are already at the correct
        //     // position within the file. If we skip moving to the next block, then a future
        //     // call to read_into will read garbage!
        //     if !self.is_last_block(block_idx) && point_in_block_end == self.block_size {
        //         self.move_decoders_to_point_in_block(block_idx as usize + 1, 0)?;
        //     }

        //     self.current_point_index += num_points_to_read_cur_block as usize;
        // }

        // Ok(num_points_to_read as usize)
    }
}

impl PointReader for LAZERSource {
    fn read_into(
        &mut self,
        point_buffer: &mut dyn PointBufferWriteable,
        count: usize,
    ) -> Result<usize> {
        // Since moving around in the compressed file is slow (Decoder has no random access), we move only in the seek
        // function. read_into assumes that we are at the correct position!
        // TODO read_into for now only works on per-attribute buffers
        point_buffer
            .as_per_attribute()
            .expect("LAZERSource currently only supports reading into PerAttribute buffers");

        let num_points_to_read = usize::min(
            count,
            self.metadata.point_count() - self.current_point_index,
        ) as u64;
        if num_points_to_read == 0 {
            return Ok(0);
        }

        let first_block_index = self.current_point_index as u64 / self.block_size;
        let last_block_index_inclusive =
            (self.current_point_index as u64 + num_points_to_read) / self.block_size;

        let first_point_index = self.current_point_index as u64;
        let last_point_index = first_point_index + num_points_to_read;

        let source_layout = self.get_default_point_layout();
        let target_layout = point_buffer.point_layout().clone();

        let mut temporary_buffers_per_attribute = target_layout
            .attributes()
            .map(|a| vec![0; (a.size() * self.block_size).try_into().unwrap()])
            .collect::<Vec<Vec<u8>>>();

        fn get_attribute_parser(
            name: &str,
            source_layout: &PointLayout,
            target_layout: &PointLayout,
        ) -> Option<AttributeConversionFn> {
            target_layout
                .get_attribute_by_name(name)
                .map_or(None, |target_attribute| {
                    source_layout
                        .get_attribute_by_name(name)
                        .and_then(|source_attribute| {
                            get_converter_for_attributes(
                                &source_attribute.into(),
                                &target_attribute.into(),
                            )
                        })
                })
        }

        let positions_converter =
            get_attribute_parser(POSITION_3D.name(), source_layout, &target_layout);
        let intensity_converter =
            get_attribute_parser(INTENSITY.name(), source_layout, &target_layout);
        let class_converter =
            get_attribute_parser(CLASSIFICATION.name(), source_layout, &target_layout);
        let color_converter = get_attribute_parser(COLOR_RGB.name(), source_layout, &target_layout);

        for block_idx in first_block_index..last_block_index_inclusive + 1 {
            let block_start_point = block_idx * self.block_size;
            let point_in_block_start = if first_point_index < block_start_point {
                0
            } else {
                first_point_index - block_start_point
            };
            let point_in_block_end =
                u64::min(last_point_index - block_start_point, self.block_size);

            let num_points_to_read_cur_block = point_in_block_end - point_in_block_start;

            // Read data from block
            if let Some(pos_buffer) = target_layout
                .index_of(&POSITION_3D)
                .map(|idx| &mut temporary_buffers_per_attribute[idx])
            {
                let pos_decoder = self
                    .decoders_per_attribute
                    .get_mut(POSITION_3D.name())
                    .expect("No positions decoder found");
                let pos_size_in_target = target_layout
                    .get_attribute_by_name(POSITION_3D.name())
                    .unwrap()
                    .size();
                for idx in 0..num_points_to_read_cur_block {
                    let x = pos_decoder.read_i32::<LittleEndian>()?;
                    let y = pos_decoder.read_i32::<LittleEndian>()?;
                    let z = pos_decoder.read_i32::<LittleEndian>()?;

                    let world_space_pos = Vector3::new(
                        self.raw_las_header.x_offset
                            + self.raw_las_header.x_scale_factor * x as f64,
                        self.raw_las_header.y_offset
                            + self.raw_las_header.y_scale_factor * y as f64,
                        self.raw_las_header.z_offset
                            + self.raw_las_header.z_scale_factor * z as f64,
                    );

                    let pos_start_idx = (idx * pos_size_in_target) as usize;
                    let pos_end_idx = pos_start_idx + pos_size_in_target as usize;
                    let target_slice = &mut pos_buffer[pos_start_idx..pos_end_idx];
                    if let Some(ref converter) = positions_converter {
                        unsafe {
                            converter(view_raw_bytes(&world_space_pos), target_slice);
                        }
                    } else {
                        unsafe {
                            target_slice.copy_from_slice(view_raw_bytes(&world_space_pos));
                        }
                    }
                }
            }

            if let Some(intensity_buffer) = target_layout
                .index_of(&INTENSITY)
                .map(|idx| &mut temporary_buffers_per_attribute[idx])
            {
                let intensity_decoder = self
                    .decoders_per_attribute
                    .get_mut(INTENSITY.name())
                    .expect("No intensity decoder found");
                let intensity_size_in_target = target_layout
                    .get_attribute_by_name(INTENSITY.name())
                    .unwrap()
                    .size();
                for idx in 0..num_points_to_read_cur_block {
                    let intensity = intensity_decoder.read_i16::<LittleEndian>()?;

                    let intensity_start_idx = (idx * intensity_size_in_target) as usize;
                    let intensity_end_idx = intensity_start_idx + intensity_size_in_target as usize;
                    let target_slice =
                        &mut intensity_buffer[intensity_start_idx..intensity_end_idx];
                    if let Some(ref converter) = intensity_converter {
                        unsafe {
                            converter(view_raw_bytes(&intensity), target_slice);
                        }
                    } else {
                        unsafe {
                            target_slice.copy_from_slice(view_raw_bytes(&intensity));
                        }
                    }
                }
            }

            if let Some(class_buffer) = target_layout
                .index_of(&CLASSIFICATION)
                .map(|idx| &mut temporary_buffers_per_attribute[idx])
            {
                let class_decoder = self
                    .decoders_per_attribute
                    .get_mut(CLASSIFICATION.name())
                    .expect("No classification decoder found");
                let class_size_in_target = target_layout
                    .get_attribute_by_name(CLASSIFICATION.name())
                    .unwrap()
                    .size();
                for idx in 0..num_points_to_read_cur_block {
                    let class = class_decoder.read_u8()?;

                    let class_start_idx = (idx * class_size_in_target) as usize;
                    let class_end_idx = class_start_idx + class_size_in_target as usize;
                    let target_slice = &mut class_buffer[class_start_idx..class_end_idx];
                    if let Some(ref converter) = class_converter {
                        unsafe {
                            converter(view_raw_bytes(&class), target_slice);
                        }
                    } else {
                        unsafe {
                            target_slice.copy_from_slice(view_raw_bytes(&class));
                        }
                    }
                }
            }

            if let Some(color_buffer) = target_layout
                .index_of(&COLOR_RGB)
                .map(|idx| &mut temporary_buffers_per_attribute[idx])
            {
                if let Some(color_decoder) = self.decoders_per_attribute.get_mut(COLOR_RGB.name()) {
                    let color_size_in_target = target_layout
                        .get_attribute_by_name(COLOR_RGB.name())
                        .unwrap()
                        .size();
                    for idx in 0..num_points_to_read_cur_block {
                        let r = color_decoder.read_u16::<LittleEndian>()?;
                        let g = color_decoder.read_u16::<LittleEndian>()?;
                        let b = color_decoder.read_u16::<LittleEndian>()?;

                        let color = Vector3::new(r, g, b);

                        let color_start_idx = (idx * color_size_in_target) as usize;
                        let color_end_idx = color_start_idx + color_size_in_target as usize;
                        let target_slice = &mut color_buffer[color_start_idx..color_end_idx];
                        if let Some(ref converter) = color_converter {
                            unsafe {
                                converter(view_raw_bytes(&color), target_slice);
                            }
                        } else {
                            unsafe {
                                target_slice.copy_from_slice(view_raw_bytes(&color));
                            }
                        }
                    }
                }
            }

            let current_slices = target_layout
                .attributes()
                .enumerate()
                .map(|(idx, attribute)| {
                    let size = (num_points_to_read_cur_block * attribute.size()) as usize;
                    &temporary_buffers_per_attribute[idx][..size]
                })
                .collect::<Vec<_>>();
            point_buffer.push(&PerAttributePointView::from_slices(
                current_slices,
                target_layout.clone(),
            ));

            // If we finished reading this block, then we have to move to the next block
            // This is because read_into assumes that the decoders are already at the correct
            // position within the file. If we skip moving to the next block, then a future
            // call to read_into will read garbage!
            if !self.is_last_block(block_idx) && point_in_block_end == self.block_size {
                self.move_decoders_to_point_in_block(block_idx as usize + 1, 0)?;
            }

            self.current_point_index += num_points_to_read_cur_block as usize;
        }

        Ok(num_points_to_read as usize)
    }

    // fn seek(&mut self, index: usize) {
    //     if index == self.current_point_index {
    //         return;
    //     }
    //     let block_index_of_point = index / self.block_size as usize;
    //     let block_start_index = block_index_of_point * self.block_size as usize;
    //     let index_within_block = index - block_start_index;

    //     self.move_decoders_to_point_in_block(block_index_of_point, index_within_block)
    //         .expect("Seek failed (TODO make seek return Result)");

    //     self.current_point_index = index;
    // }

    // fn current_index(&self) -> usize {
    //     self.current_point_index
    // }

    fn read(&mut self, count: usize) -> Result<Box<dyn pasture_core::containers::PointBuffer>> {
        let mut buffer = PerAttributeVecPointStorage::new(self.default_point_layout.clone());
        self.read_into(&mut buffer, count)?;
        Ok(Box::new(buffer))
    }

    fn get_metadata(&self) -> &dyn pasture_core::meta::Metadata {
        &self.metadata
    }

    fn get_default_point_layout(&self) -> &pasture_core::layout::PointLayout {
        &self.default_point_layout
    }
}

impl SeekToPoint for LAZERSource {
    fn seek_point(&mut self, _position: SeekFrom) -> Result<usize> {
        todo!()
        // if index == self.current_point_index {
        //     return;
        // }
        // let block_index_of_point = index / self.block_size as usize;
        // let block_start_index = block_index_of_point * self.block_size as usize;
        // let index_within_block = index - block_start_index;

        // self.move_decoders_to_point_in_block(block_index_of_point, index_within_block)
        //     .expect("Seek failed (TODO make seek return Result)");

        // self.current_point_index = index;
    }
}

#[cfg(test)]
mod tests {
    use pasture_core::containers::PointBufferExt;
    use pasture_io::las::LASReader;

    use super::*;

    // #[test]
    fn _test_lazer_reader() -> Result<()> {
        let mut las_source = LASReader::from_path("/Users/pbormann/data/geodata/pointclouds/datasets/navvis_m6_3rdFloor/navvis_m6_HQ3rdFloor.laz")?;
        let mut lazer_source = LAZERSource::new("/Users/pbormann/data/projects/progressive_indexing/experiment_data/navvis_m6_HQ3rdFloor.lazer")?;

        let count = lazer_source.block_size() as usize;
        for _ in 0..10 {
            let las_points = las_source.read(count)?;
            let lazer_points = lazer_source.read(count)?;

            for idx in 0..count {
                let las_pos = las_points.get_attribute::<Vector3<f64>>(&POSITION_3D, idx);
                let lazer_pos = lazer_points.get_attribute::<Vector3<f64>>(&POSITION_3D, idx);
                assert_eq!(las_pos, lazer_pos);

                let las_col = las_points.get_attribute::<Vector3<u16>>(&COLOR_RGB, idx);
                let lazer_col = lazer_points.get_attribute::<Vector3<u16>>(&COLOR_RGB, idx);
                assert_eq!(las_col, lazer_col);
            }
        }

        Ok(())
    }
}
