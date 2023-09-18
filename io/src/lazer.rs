use std::{
    io::{Cursor, Read, Seek, SeekFrom, Write},
    ops::Range,
};

use anyhow::{anyhow, bail, Context, Result};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use log::debug;
use lz4::{Decoder, Encoder, EncoderBuilder};
use pasture_core::{
    containers::{BorrowedBuffer, HashMapBuffer, OwningBuffer},
    layout::PointLayout,
    meta::Metadata,
};
use pasture_io::{
    base::{PointReader, PointWriter, SeekToPoint},
    las::{point_layout_from_las_point_format, LASMetadata},
    las_rs::{
        point::Format,
        raw::{self, vlr::RecordLength},
        Builder, Header,
    },
};

use crate::las_common::{get_default_las_converter, get_minimum_layout_for_las_conversion};

/// VLR that stores the offsets to the compressed blocks in a LAZER file
#[derive(Default, Debug)]
struct BlocksVlr {
    block_offsets_from_start_of_file: Vec<u64>,
}

impl BlocksVlr {
    pub const USER_ID: &[u8; 16] = b"LAZER format\0\0\0\0";
    pub const RECORD_ID: u16 = 2;

    pub fn add_block_offset(&mut self, offset: u64) {
        self.block_offsets_from_start_of_file.push(offset);
    }

    pub fn number_of_blocks(&self) -> usize {
        self.block_offsets_from_start_of_file.len()
    }

    pub fn to_raw_vlr(&self) -> Result<raw::Vlr> {
        let mut raw_vlr = raw::Vlr::default();
        raw_vlr.record_id = Self::RECORD_ID;
        raw_vlr.user_id = *Self::USER_ID;

        let mut data_writer: Cursor<Vec<u8>> = Cursor::new(Vec::default());
        data_writer.write_u64::<LittleEndian>(self.block_offsets_from_start_of_file.len() as u64)?;
        for offset in &self.block_offsets_from_start_of_file {
            data_writer.write_u64::<LittleEndian>(*offset)?;
        }
        raw_vlr.data = data_writer.into_inner();
        raw_vlr.record_length_after_header = RecordLength::Evlr(raw_vlr.data.len() as u64);
        {
            let mut write_helper = Cursor::new(raw_vlr.description.as_mut_slice());
            write_helper.write_all(b"LAZER block offsets")?;
        }
        Ok(raw_vlr)
    }

    pub fn from_raw_vlr(raw_vlr: &raw::Vlr) -> Result<Self> {
        if raw_vlr.user_id != *Self::USER_ID {
            bail!(
                "Invalid user ID, expected {:?} but got {:?}",
                Self::USER_ID,
                raw_vlr.user_id
            );
        }
        if raw_vlr.record_id != Self::RECORD_ID {
            bail!(
                "Invalid record ID, expected {} but got {}",
                Self::RECORD_ID,
                raw_vlr.record_id
            );
        }

        let mut data_reader = Cursor::new(&raw_vlr.data);
        let num_blocks = data_reader.read_u64::<LittleEndian>()?;
        let block_offsets = (0..num_blocks)
            .map(|_| {
                let offset = data_reader.read_u64::<LittleEndian>()?;
                Ok(offset)
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            block_offsets_from_start_of_file: block_offsets,
        })
    }
}

/// VLR that points to the end of the file where the `BlocksVlr` is stored. This LAZER-specific VLR is required
/// because without it, there would be no way to figure out at which byte offset the `BlocksVlr` is stored. We
/// can't store the `BlocksVlr` as a regular VLR, because we don't know its size until all points are compressed,
/// so we have to store it at the end and memorize the byte offset to it in this `LazerVlr` (which is fixed-size,
/// so we can store it as a regular VLR)
#[derive(Default)]
struct LazerVlr {
    offset_to_blocks_evlr: u64,
}

impl LazerVlr {
    pub const USER_ID: &[u8; 16] = b"LAZER format\0\0\0\0";
    pub const RECORD_ID: u16 = 1;

    pub fn to_raw_vlr(&self) -> Result<raw::Vlr> {
        let mut raw_vlr = raw::Vlr::default();
        raw_vlr.record_id = Self::RECORD_ID;
        raw_vlr.user_id = *Self::USER_ID;

        let mut data_writer: Cursor<Vec<u8>> = Cursor::new(Vec::default());
        data_writer.write_u64::<LittleEndian>(self.offset_to_blocks_evlr)?;
        raw_vlr.data = data_writer.into_inner();
        raw_vlr.record_length_after_header = RecordLength::Vlr(raw_vlr.data.len() as u16);
        {
            let mut write_helper = Cursor::new(raw_vlr.description.as_mut_slice());
            write_helper.write_all(b"Offset to blocks EVLR")?;
        }
        Ok(raw_vlr)
    }

    pub fn from_raw_vlr(raw_vlr: &raw::Vlr) -> Result<Self> {
        if raw_vlr.user_id != *Self::USER_ID {
            bail!(
                "Invalid user ID, expected {:?} but got {:?}",
                Self::USER_ID,
                raw_vlr.user_id
            );
        }
        if raw_vlr.record_id != Self::RECORD_ID {
            bail!(
                "Invalid record ID, expected {} but got {}",
                Self::RECORD_ID,
                raw_vlr.record_id
            );
        }

        let mut data_reader = Cursor::new(&raw_vlr.data);
        let offset_to_blocks_evlr = data_reader.read_u64::<LittleEndian>()?;

        Ok(Self {
            offset_to_blocks_evlr,
        })
    }
}

/// Header for a single compressed LAZER block. Contains the number of points and the byte offsets
/// to the compressed columnar attribute arrays from the start of the block (first byte after the header)
#[derive(Clone, Debug)]
struct BlockHeader {
    num_points_in_block: u32,
    attribute_byte_offsets: Vec<u32>,
}

impl BlockHeader {
    pub fn from_compressed_attributes(
        compressed_attributes: &[Vec<u8>],
        num_points: usize,
    ) -> Self {
        Self {
            num_points_in_block: num_points as u32,
            attribute_byte_offsets: compressed_attributes
                .iter()
                .map(|bytes| bytes.len() as u32)
                .scan(0u32, |state, cur| {
                    let old_state = *state;
                    *state += cur;
                    Some(old_state)
                })
                .collect(),
        }
    }

    /// Returns the size of this header within the LAZER file
    pub fn size(&self) -> usize {
        (1 + self.attribute_byte_offsets.len()) * std::mem::size_of::<u32>()
    }

    pub fn write_to<W: Write>(&self, mut write: W) -> Result<()> {
        write.write_u32::<LittleEndian>(self.num_points_in_block)?;
        // Write the offsets without first writing the NUMBER of offsets, since this is a fixed, known parameter in LAZER
        // (it is equal to the number of point attributes in the point format)
        for offset in &self.attribute_byte_offsets {
            write.write_u32::<LittleEndian>(*offset)?;
        }
        Ok(())
    }

    pub fn read_from<R: Read>(mut read: R, point_layout: &PointLayout) -> Result<Self> {
        let num_points = read.read_u32::<LittleEndian>()?;
        let offsets = (0..point_layout.attributes().count())
            .map(|_| {
                let offset = read.read_u32::<LittleEndian>()?;
                Ok(offset)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            num_points_in_block: num_points,
            attribute_byte_offsets: offsets,
        })
    }
}

type InMemoryEncoder = Encoder<Cursor<Vec<u8>>>;
type InMemoryDecoder = Decoder<Cursor<Vec<u8>>>;
// type MmapDecoder<'a> = Decoder<Cursor<&'a [u8]>>;

#[derive(Default)]
struct LazerBlockInfo {
    num_points: usize,
    block_index: usize,
    current_position_within_block: usize,
    decoders: Vec<InMemoryDecoder>,
}

impl LazerBlockInfo {
    pub fn remaining_points(&self) -> usize {
        self.num_points - self.current_position_within_block
    }
}

pub struct LazerReader<R: Read + Seek> {
    reader: R,
    raw_las_header: raw::Header,
    default_point_layout: PointLayout,
    las_metadata: LASMetadata,
    blocks_evlr: BlocksVlr,
    lazer_vlr: LazerVlr,
    // Store the index of the first point in that block together with the header of each block. This makes
    // seek operations easier. The `BlockHeader` structure itself does not need to store this information
    // so we use a tuple here
    block_headers: Vec<(usize, BlockHeader)>,
    current_block: Option<LazerBlockInfo>,
    current_point_index: usize,
}

impl<R: Read + Seek> LazerReader<R> {
    pub fn new(mut reader: R) -> Result<Self> {
        let raw_las_header =
            raw::Header::read_from(&mut reader).context("Failed to read LAS header")?;
        let format = Format::new(raw_las_header.point_data_record_format)
            .context("Invalid point record format")?;
        let default_point_layout = point_layout_from_las_point_format(&format, true)
            .context("Could not determine default point layout")?;

        let vlrs = (0..raw_las_header.number_of_variable_length_records)
            .map(|_| {
                let raw_vlr =
                    raw::Vlr::read_from(&mut reader, false).context("Failed to read VLR")?;
                Ok(raw_vlr)
            })
            .collect::<Result<Vec<_>>>()?;

        // We really only care for the LazerVlr and discard other VLRs. In a future implementation, we should
        // add support for other VLRs, e.g. extra bytes
        let raw_lazer_vlr = vlrs
            .iter()
            .find(|raw_vlr| {
                raw_vlr.user_id == *LazerVlr::USER_ID && raw_vlr.record_id == LazerVlr::RECORD_ID
            })
            .ok_or(anyhow!("No LAZER VLR found"))?;
        let lazer_vlr =
            LazerVlr::from_raw_vlr(raw_lazer_vlr).context("Failed to parse LAZER VLR")?;

        // With the offset in the LAZER VLR we can read the Blocks EVLR
        reader.seek(SeekFrom::Start(lazer_vlr.offset_to_blocks_evlr))?;
        let raw_blocks_evlr =
            raw::Vlr::read_from(&mut reader, true).context("Failed to read blocks EVLR")?;
        let blocks_evlr = BlocksVlr::from_raw_vlr(&raw_blocks_evlr)
            .context("Failed to deserialize blocks EVLR")?;

        let las_header = {
            let header_builder = Builder::new(raw_las_header.clone())
                .context("Can't create LAS header builder from raw LAS header")?;
            // header_builder.vlrs = vlrs; // TODO Support VLRs
            header_builder
                .into_header()
                .context("Failed to build LAS header")?
        };
        let las_metadata = las_header
            .try_into()
            .context("Failed to get LAS metadata information from header")?;

        // Read all block headers so that we can seek to blocks based on point indices!
        let block_headers = blocks_evlr
            .block_offsets_from_start_of_file
            .iter()
            .map(|block_offset| {
                reader.seek(SeekFrom::Start(*block_offset))?;
                BlockHeader::read_from(&mut reader, &default_point_layout)
            })
            .collect::<Result<Vec<_>>>()
            .context("Failed to read block headers")?;

        let total_number_of_points_in_blocks = block_headers
            .iter()
            .map(|block| block.num_points_in_block as usize)
            .sum::<usize>();
        if total_number_of_points_in_blocks != raw_las_header.number_of_point_records as usize {
            bail!("LAS header states that there are {} points in this LAZER file, but the total number of points in all blocks is different ({total_number_of_points_in_blocks}). These two numbers must match!", raw_las_header.number_of_point_records);
        }

        let block_headers_with_cumulative_point_counts = block_headers
            .into_iter()
            .scan::<usize, (usize, BlockHeader), _>(
                0,
                |cumulative_count, block_header| -> Option<(usize, BlockHeader)> {
                    let current_count = *cumulative_count;
                    *cumulative_count += block_header.num_points_in_block as usize;
                    Some((current_count, block_header))
                },
            )
            .collect::<Vec<_>>();

        Ok(Self {
            blocks_evlr,
            block_headers: block_headers_with_cumulative_point_counts,
            default_point_layout,
            raw_las_header,
            las_metadata,
            lazer_vlr,
            reader,
            current_block: None,
            current_point_index: 0,
        })
    }

    /// Begin to decode the block with the given index. This creates the LZ4 decoders for each attribute in
    /// the default PointLayout and sets up the `current_block` structure. No data is decoded, but the decoders
    /// are ready after this function is called!
    fn begin_decode_block(&mut self, block_index: usize) -> Result<()> {
        let (_, block_header) = self.block_headers[block_index].clone();

        // TODO Maybe store the size of the BlockHeader within `attribute_byte_offsets[0]`? i.e. the BlockHeader
        // stores offets from the start of the block INCLUDING itself?
        let block_point_data_start_byte = self.blocks_evlr.block_offsets_from_start_of_file
            [block_index]
            + block_header.size() as u64
            + block_header.attribute_byte_offsets[0] as u64;
        let block_point_data_end_byte = if block_index < self.blocks_evlr.number_of_blocks() - 1 {
            self.blocks_evlr.block_offsets_from_start_of_file[block_index + 1]
        } else {
            self.lazer_vlr.offset_to_blocks_evlr
        };
        let size_of_lazer_block = block_point_data_end_byte - block_point_data_start_byte;

        self.reader
            .seek(SeekFrom::Start(block_point_data_start_byte))?;

        let size_of_compressed_attribute_ranges = block_header
            .attribute_byte_offsets
            .iter()
            .copied()
            .zip(
                block_header
                    .attribute_byte_offsets
                    .iter()
                    .copied()
                    .skip(1)
                    .chain(std::iter::once(size_of_lazer_block as u32)),
            )
            .map(|(start, end)| end - start);

        let decoders = size_of_compressed_attribute_ranges
            .map(|size| {
                let mut buffer = vec![0; size as usize];
                self.reader.read_exact(&mut buffer)?;
                let decoder =
                    Decoder::new(Cursor::new(buffer)).context("failed to create lz4 decoder")?;
                Ok(decoder)
            })
            .collect::<Result<Vec<_>>>()?;

        self.current_block = Some(LazerBlockInfo {
            num_points: block_header.num_points_in_block as usize,
            block_index,
            current_position_within_block: 0,
            decoders,
        });

        Ok(())
    }

    /// Skip over the next `num_points` in the current block
    fn advance_current_block(&mut self, num_points: usize) -> Result<()> {
        if num_points == 0 {
            return Ok(());
        }

        let current_block = self
            .current_block
            .as_mut()
            .ok_or(anyhow!("No current block exists"))?;
        let remaining = current_block.num_points - current_block.current_position_within_block;
        if num_points > remaining {
            bail!("num_points exceeds remaining point count in current block");
        }

        for (attribute, decoder) in self
            .default_point_layout
            .attributes()
            .zip(current_block.decoders.iter_mut())
        {
            let bytes_to_skip = num_points * attribute.size() as usize;
            std::io::copy(
                &mut decoder.take(bytes_to_skip as u64),
                &mut std::io::sink(),
            )
            .context("Failed to skip bytes in LZ4 decoder")?;
        }

        current_block.current_position_within_block += num_points;
        Ok(())
    }

    pub fn is_at_end(&self) -> bool {
        self.remaining_points() == 0
    }

    pub fn remaining_points(&self) -> usize {
        self.raw_las_header.number_of_point_records as usize - self.current_point_index
    }

    /// Returns the index of the LAZER block that contains the point at the given index
    fn block_index_for_point(&self, point_index: usize) -> usize {
        assert!(point_index < self.raw_las_header.number_of_point_records as usize);
        self.block_headers
            .partition_point(|(cumulative_count, _)| *cumulative_count < point_index)
    }

    fn read_default_layout<'a, 'b, B: OwningBuffer<'a>>(
        &mut self,
        point_buffer: &'b mut B,
        count: usize,
    ) -> Result<()> {
        let point_buffer_offset = point_buffer.len();
        point_buffer.resize(point_buffer_offset + count);

        let mut points_to_read = count;
        let mut current_offset_in_points = point_buffer_offset;
        while points_to_read > 0 {
            let mut remaining_points_in_current_block = self
                .current_block
                .as_ref()
                .map(LazerBlockInfo::remaining_points)
                .unwrap_or_default();
            if remaining_points_in_current_block == 0 {
                self.begin_decode_block(
                    self.current_block
                        .as_ref()
                        .map(|block| block.block_index + 1)
                        .unwrap_or_default(),
                )?;
                remaining_points_in_current_block =
                    self.current_block.as_ref().unwrap().remaining_points();
            }

            // Encode the current chunk, which might fill the current LAZER block
            let current_chunk_size = remaining_points_in_current_block.min(points_to_read);
            self.decode_chunk_default_layout(
                current_offset_in_points..(current_offset_in_points + current_chunk_size),
                point_buffer,
            )?;

            current_offset_in_points += current_chunk_size;
            points_to_read -= current_chunk_size;
        }

        Ok(())
    }

    fn decode_chunk_default_layout<'a, 'b, B: OwningBuffer<'a>>(
        &mut self,
        point_range_in_buffer: Range<usize>,
        buffer: &'b mut B,
    ) -> Result<()> {
        let num_points_in_chunk = point_range_in_buffer.len();
        let current_block = self.current_block.as_mut().unwrap();

        // Annoying that we have to clone the layout here, but we need it in the filter operation which
        // runs at a time where we need mutable access to the buffer...
        let buffer_layout = buffer.point_layout().clone();

        if let Some(columnar_buffer) = buffer.as_columnar_mut() {
            for (attribute, decoder) in self
                .default_point_layout
                .attributes()
                .zip(current_block.decoders.iter_mut())
                // We filter for the actual attributes in the buffer, since a higher-level call might decide that
                // we don't actually have to decode every attribute
                .filter(|(attribute, _)| {
                    buffer_layout.has_attribute(attribute.attribute_definition())
                })
            {
                let attribute_data = columnar_buffer.get_attribute_range_mut(
                    attribute.attribute_definition(),
                    point_range_in_buffer.clone(),
                );
                decoder
                    .read_exact(attribute_data)
                    .with_context(|| format!("Failed to decode data for attribute {attribute}"))?;
            }
        } else {
            let largest_attribute = self
                .default_point_layout
                .attributes()
                .map(|a| a.size())
                .max()
                .expect("PointLayout must contain at least one attribute");
            let mut attribute_buffer = vec![0; largest_attribute as usize * num_points_in_chunk];
            for (attribute, decoder) in self
                .default_point_layout
                .attributes()
                .zip(current_block.decoders.iter_mut())
                .filter(|(attribute, _)| {
                    buffer_layout.has_attribute(attribute.attribute_definition())
                })
            {
                let attribute_data =
                    &mut attribute_buffer[..(attribute.size() as usize * num_points_in_chunk)];
                decoder
                    .read_exact(attribute_data)
                    .with_context(|| format!("Failed to decode data for attribute {attribute}"))?;
                // Is safe because we know the point layout of the buffer is the default point layout, so the
                // data for this attribute will be correct
                unsafe {
                    buffer.set_attribute_range(
                        attribute.attribute_definition(),
                        point_range_in_buffer.clone(),
                        attribute_data,
                    );
                }
            }
        }

        current_block.current_position_within_block += num_points_in_chunk;

        Ok(())
    }
}

impl<R: Read + Seek> PointReader for LazerReader<R> {
    fn read_into<'a, 'b, B: OwningBuffer<'a>>(
        &mut self,
        point_buffer: &'b mut B,
        count: usize,
    ) -> Result<usize>
    where
        'a: 'b,
    {
        if count == 0 || self.is_at_end() {
            return Ok(0);
        }

        let num_to_read = count.min(self.remaining_points());

        if *point_buffer.point_layout() == self.default_point_layout {
            self.read_default_layout(point_buffer, count)?;
            Ok(num_to_read)
        } else {
            // Reading into a custom layout has optimization potential by only decoding the attributes
            // that the target layout actually requires! The main problem I see with this approach is that
            // it makes reading the same file with different PointLayouts very difficult:
            //   Suppose we have a file with 1 block of 50k points and we only read the positions of the first
            //   25k points, but then want ALL attributes of the remaining 25k points. We would have to memorize
            //   that only the position decoder was advanced and then advance all other decoders upon the next
            //   read operation that requires them
            // Seems quite complicated. As a compromise, we could employ this strategy only if we read the whole
            // file. This should be sufficient for my experiments, at least until I implement the block index,
            // in which case I would have to upgrade to block-granularity for this optimization

            let target_layout = point_buffer.point_layout().clone();

            let minimum_source_layout = if num_to_read == self.remaining_points() {
                get_minimum_layout_for_las_conversion(&self.default_point_layout, &target_layout).context("Could not determine minimum PointLayout for parsing LAZER data into requested buffer layout")?
            } else {
                self.default_point_layout.clone()
            };

            let converter = get_default_las_converter(
                &minimum_source_layout,
                &target_layout,
                self.raw_las_header.clone(),
            )
            .context("Could not get buffer layout converter for target buffer")?;

            // Read into default layout buffer and convert this buffer into the target buffer
            let mut default_layout_buffer =
                HashMapBuffer::with_capacity(num_to_read, minimum_source_layout.clone());
            self.read_default_layout(&mut default_layout_buffer, num_to_read)?;

            point_buffer.resize(default_layout_buffer.len());
            converter.convert_into(&default_layout_buffer, point_buffer);

            Ok(num_to_read)
        }
    }

    fn get_metadata(&self) -> &dyn Metadata {
        &self.las_metadata
    }

    fn get_default_point_layout(&self) -> &PointLayout {
        &self.default_point_layout
    }
}

impl<R: Read + Seek> SeekToPoint for LazerReader<R> {
    fn seek_point(&mut self, position: SeekFrom) -> Result<usize> {
        let position_from_start = match position {
            SeekFrom::Start(offset) => offset as usize,
            SeekFrom::End(offset) => {
                (self.raw_las_header.number_of_point_records as i64 + offset).max(0) as usize
            }
            SeekFrom::Current(offset) => (self.current_point_index as i64 + offset).max(0) as usize,
        };

        if position_from_start >= self.raw_las_header.number_of_point_records as usize {
            self.current_point_index = self.raw_las_header.number_of_point_records as usize;
            return Ok(self.current_point_index);
        }

        let block_index = self.block_index_for_point(position_from_start);
        if let Some(current_block_info) = &mut self.current_block {
            let index_of_first_point_in_block = self.block_headers[block_index].0;
            let relative_offset_within_block = position_from_start - index_of_first_point_in_block;
            current_block_info.current_position_within_block = relative_offset_within_block;
        } else {
            self.begin_decode_block(block_index)
                .context("Failed to decode LAZER block")?;
            let index_of_first_point_in_block = self.block_headers[block_index].0;
            let relative_offset_within_block = position_from_start - index_of_first_point_in_block;

            self.advance_current_block(relative_offset_within_block)
                .context("Failed to seek within compressed LAZER block")?;

            let current_block_info = self
                .current_block
                .as_mut()
                .expect("current block is None after call to self.decode_block");
            current_block_info.current_position_within_block = relative_offset_within_block;
        }

        Ok(position_from_start)
    }
}

pub struct LazerWriter<W: Write + Seek> {
    writer: W,
    raw_las_header: raw::Header,
    default_point_layout: PointLayout,
    lazer_vlr: LazerVlr,
    lazer_vlr_byte_offset: usize,
    blocks_evlr: BlocksVlr,
    encoder_builder: EncoderBuilder,
    encoders: Vec<InMemoryEncoder>,
    num_points_in_current_block: usize,
    block_size: usize,
    byte_offset_of_current_block: usize,
    // Buffer that is used to copy attribute data from an opaque point buffer into the encoders
    copy_attribute_buffer: Vec<u8>,
}

impl<W: Write + Seek> LazerWriter<W> {
    const DEFAULT_BLOCK_SIZE: usize = 50_000;

    pub fn new(mut writer: W, las_header: Header, encoder_builder: EncoderBuilder) -> Result<Self> {
        if !las_header.evlrs().is_empty() {
            bail!("Writing LAZER file with extended VLRs is currently unsupported");
        }

        let vlrs = las_header.vlrs().clone();
        let mut raw_header = las_header
            .into_raw()
            .context("Could not convert LAS header into raw header")?;
        // Reset point counts in header
        raw_header.number_of_point_records = 0;
        raw_header.number_of_points_by_return = Default::default();

        // Add one VLR (the LazerVlr) to the count of VLRs
        raw_header.number_of_variable_length_records += 1;

        // Write raw header and VLRs, then memorize current position as offset ot point records (also update this
        // in the header, if the header has padding bytes)
        raw_header
            .write_to(&mut writer)
            .context("Failed to write LAS header")?;
        for vlr in vlrs {
            let raw_vlr = vlr
                .into_raw(false)
                .context("Can't convert VLR into raw VLR")?;
            raw_vlr
                .write_to(&mut writer)
                .context("Failed to write VLR")?;
        }

        // Also write the LAZER VLR, which will contain invalid data (a byte offset of 0), but we only reserve
        // space for this VLR and will update it accordingly in `flush()`
        let lazer_vlr_byte_offset = writer.stream_position()? as usize;
        let lazer_vlr = LazerVlr::default();
        let raw_lazer_vlr = lazer_vlr
            .to_raw_vlr()
            .context("Error while serializing LAZER VLR")?;
        raw_lazer_vlr
            .write_to(&mut writer)
            .context("Failed to write LAZER VLR")?;

        let offset_to_point_records = writer.stream_position()?;
        if offset_to_point_records > u32::MAX as u64 {
            bail!("Too many VLRs, size exceeds u32::MAX");
        }
        raw_header.offset_to_point_data = offset_to_point_records as u32;

        let point_format = Format::new(raw_header.point_data_record_format).with_context(|| {
            anyhow!(
                "Invalid LAS point record format {}",
                raw_header.point_data_record_format
            )
        })?;
        let default_point_layout = point_layout_from_las_point_format(&point_format, true)
            .context("Could not determine default PointLayout for LAS point record format")?;
        let encoders =
            Self::create_encoders_for_point_layout(&default_point_layout, &encoder_builder)
                .context("Failed to create LZ4 encoders")?;

        let max_attribute_size = default_point_layout
            .attributes()
            .max_by(|l, r| l.size().cmp(&r.size()))
            .map(|a| a.size())
            .unwrap_or_default();
        let copy_attribute_buffer = vec![0; max_attribute_size as usize * Self::DEFAULT_BLOCK_SIZE];

        Ok(Self {
            block_size: Self::DEFAULT_BLOCK_SIZE,
            blocks_evlr: Default::default(),
            byte_offset_of_current_block: offset_to_point_records as usize,
            copy_attribute_buffer,
            default_point_layout,
            encoders,
            encoder_builder,
            lazer_vlr,
            lazer_vlr_byte_offset,
            num_points_in_current_block: 0,
            raw_las_header: raw_header,
            writer,
        })
    }

    pub fn into_inner(mut self) -> Result<W> {
        self.flush().context("Failed to flush")?;
        Ok(self.writer)
    }

    fn write_lazer_vlr_and_block_evlr(&mut self) -> Result<()> {
        self.lazer_vlr.offset_to_blocks_evlr = self.byte_offset_of_current_block as u64;
        let raw_lazer_vlr = self
            .lazer_vlr
            .to_raw_vlr()
            .context("Can't serialize LAZER VLR")?;
        self.writer
            .seek(SeekFrom::Start(self.lazer_vlr_byte_offset as u64))?;
        raw_lazer_vlr
            .write_to(&mut self.writer)
            .context("Failed to write LAZER VLR")?;

        self.writer
            .seek(SeekFrom::Start(self.byte_offset_of_current_block as u64))?;
        let vlr = self
            .blocks_evlr
            .to_raw_vlr()
            .context("Can't serialize blocks EVLR")?;
        vlr.write_to(&mut self.writer)
            .context("Failed to write blocks EVLR")?;
        Ok(())
    }

    fn write_header(&mut self) -> Result<()> {
        self.writer.seek(SeekFrom::Start(0))?;
        self.raw_las_header
            .write_to(&mut self.writer)
            .context("Failed to write LAS header")?;
        Ok(())
    }

    fn finish_current_block(&mut self) -> Result<()> {
        if self.num_points_in_current_block == 0 {
            return Ok(());
        }

        let compressed_attributes = self
            .encoders
            .drain(..)
            .map(|encoder| -> Result<Vec<u8>> {
                let (write, err) = encoder.finish();
                err?;
                Ok(write.into_inner())
            })
            .collect::<Result<Vec<_>>>()
            .context("Error while flushing attribute encoders")?;

        let block_header = BlockHeader::from_compressed_attributes(
            &compressed_attributes,
            self.num_points_in_current_block,
        );
        block_header
            .write_to(&mut self.writer)
            .context("Failed to write block header")?;
        for compressed_attribute in compressed_attributes {
            self.writer
                .write_all(&compressed_attribute)
                .context("Failed to write compressed attribute")?;
        }

        // Housekeeping for the current block, and starting of a new block
        self.num_points_in_current_block = 0;
        self.blocks_evlr
            .add_block_offset(self.byte_offset_of_current_block as u64);
        self.byte_offset_of_current_block = self.writer.stream_position()? as usize;
        self.encoders = Self::create_encoders_for_point_layout(
            &self.default_point_layout,
            &self.encoder_builder,
        )?;

        Ok(())
    }

    fn create_encoders_for_point_layout(
        point_layout: &PointLayout,
        encoder_builder: &EncoderBuilder,
    ) -> Result<Vec<InMemoryEncoder>> {
        // Create one encoder per attribute within the point layout. The encoder writes to an in-memory buffer
        point_layout
            .attributes()
            .map(|_| -> Result<InMemoryEncoder> {
                let encoder = encoder_builder.build(Cursor::new(Vec::default()))?;
                Ok(encoder)
            })
            .collect::<Result<Vec<_>>>()
    }

    fn encode_default_layout<'a, B: BorrowedBuffer<'a>>(&mut self, points: &'a B) -> Result<()> {
        // Write as many blocks as necessary given the length of `points`
        let mut points_to_write = points.len();
        let mut current_offset_in_points = 0;
        while points_to_write > 0 {
            let remaining_points_in_current_block =
                self.block_size - self.num_points_in_current_block;

            // Encode the current chunk, which might fill the current LAZER block
            let current_chunk_size = remaining_points_in_current_block.min(points_to_write);
            self.encode_chunk_default_layout(
                current_offset_in_points..(current_offset_in_points + current_chunk_size),
                points,
            )?;

            current_offset_in_points += current_chunk_size;
            points_to_write -= current_chunk_size;

            // If we filled the current block, finish it (write block header and block memory to writer)
            if current_chunk_size == remaining_points_in_current_block {
                self.finish_current_block()?;
            }
        }

        Ok(())
    }

    fn encode_custom_layout<'a, B: BorrowedBuffer<'a>>(&mut self, points: &'a B) -> Result<()> {
        if self.raw_las_header.point_data_record_format >= 6 {
            // Not supported because BufferLayoutConverter does not support transform functions that run
            // on the source data, and extracting bit attributes for extended formats requires this
            bail!("LAZER writing not supported for extended LAS point record formats (>= 6)");
        }
        let target_layout = points.point_layout();
        let converter = get_default_las_converter(
            &self.default_point_layout,
            target_layout,
            self.raw_las_header.clone(),
        )
        .context("Could not get buffer layout converter for target buffer")?;
        let points_in_default_layout = converter.convert::<HashMapBuffer, _>(points);

        let mut points_to_write = points.len();
        let mut current_offset_in_points = 0;
        while points_to_write > 0 {
            let remaining_points_in_current_block =
                self.block_size - self.num_points_in_current_block;

            // Encode the current chunk, which might fill the current LAZER block
            let current_chunk_size = remaining_points_in_current_block.min(points_to_write);
            self.encode_chunk_default_layout(
                current_offset_in_points..(current_offset_in_points + current_chunk_size),
                &points_in_default_layout,
            )?;

            current_offset_in_points += current_chunk_size;
            points_to_write -= current_chunk_size;

            // If we filled the current block, finish it (write block header and block memory to writer)
            if current_chunk_size == remaining_points_in_current_block {
                self.finish_current_block()?;
            }
        }

        Ok(())
    }

    fn encode_chunk_default_layout<'a, B: BorrowedBuffer<'a>>(
        &mut self,
        chunk_range: Range<usize>,
        points: &'a B,
    ) -> Result<()> {
        for (attribute, encoder) in self
            .default_point_layout
            .attributes()
            .zip(self.encoders.iter_mut())
        {
            let attribute_bytes = if let Some(columnar_buffer) = points.as_columnar() {
                columnar_buffer
                    .get_attribute_range_ref(attribute.attribute_definition(), chunk_range.clone())
            } else {
                let num_bytes_of_attributes = chunk_range.len() * attribute.size() as usize;
                let buffer = &mut self.copy_attribute_buffer[..num_bytes_of_attributes];
                points.get_attribute_range(
                    attribute.attribute_definition(),
                    chunk_range.clone(),
                    buffer,
                );
                buffer
            };

            encoder
                .write_all(attribute_bytes)
                .context("LZ4 encoding of attribute failed")?;
        }

        self.num_points_in_current_block += chunk_range.len();
        self.raw_las_header.number_of_point_records += chunk_range.len() as u32;

        Ok(())
    }
}

impl<W: Write + Seek> PointWriter for LazerWriter<W> {
    fn write<'a, B: BorrowedBuffer<'a>>(&mut self, points: &'a B) -> Result<()> {
        if points.is_empty() {
            return Ok(());
        }

        if *points.point_layout() == self.default_point_layout {
            self.encode_default_layout(points)
        } else {
            self.encode_custom_layout(points)
        }
    }

    fn flush(&mut self) -> Result<()> {
        self.finish_current_block()
            .context("Failed to write LAZER block")?;
        self.write_header().context("Failed to write LAS header")?;
        self.write_lazer_vlr_and_block_evlr()
            .context("Failed to write blocks EVLR")?;
        Ok(())
    }

    fn get_default_point_layout(&self) -> &PointLayout {
        &self.default_point_layout
    }
}

#[cfg(test)]
mod tests {
    use pasture_core::{
        containers::{ExternalMemoryBuffer, InterleavedBuffer, VectorBuffer},
        layout::{attributes::POSITION_3D, PointAttributeDataType},
        nalgebra::Vector3,
    };

    use super::*;

    #[test]
    fn test_las_lazer_roundtrip() -> Result<()> {
        let in_file_path =
            "/Users/pbormann/data/geodata/pointclouds/datasets/district_of_columbia/1318_1.las";

        let las_file_bytes = std::fs::read(in_file_path)?;
        let (las_points, raw_header) = {
            let header = raw::Header::read_from(Cursor::new(&las_file_bytes))?;
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
        let las_header = Builder::new(raw_header)?.into_header()?;

        let mut lazer_writer = LazerWriter::new(
            Cursor::new(Vec::<u8>::default()),
            las_header.clone(),
            EncoderBuilder::new(),
        )?;
        lazer_writer.write(&las_points)?;

        let lazer_bytes = lazer_writer.into_inner()?.into_inner();

        let read_points = {
            let mut lazer_reader = LazerReader::new(Cursor::new(lazer_bytes))?;
            lazer_reader.read::<VectorBuffer>(lazer_reader.remaining_points())?
        };

        assert_eq!(las_points.point_layout(), read_points.point_layout());
        assert_eq!(las_points.len(), read_points.len());

        for idx in 0..las_points.len() {
            let expected_point_memory = las_points.get_point_ref(idx);
            let actual_point_memory = read_points.get_point_ref(idx);
            assert_eq!(
                expected_point_memory, actual_point_memory,
                "Point {idx} does not match"
            );
        }

        Ok(())
    }

    #[test]
    fn test_las_lazer_roundtrip_custom_layout() -> Result<()> {
        let in_file_path =
            "/Users/pbormann/data/geodata/pointclouds/datasets/district_of_columbia/1318_1.las";

        let las_file_bytes = std::fs::read(in_file_path)?;
        let (las_points, raw_header) = {
            let header = raw::Header::read_from(Cursor::new(&las_file_bytes))?;
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
        let las_header = Builder::new(raw_header)?.into_header()?;

        let mut lazer_writer = LazerWriter::new(
            Cursor::new(Vec::<u8>::default()),
            las_header.clone(),
            EncoderBuilder::new(),
        )?;
        lazer_writer.write(&las_points)?;

        let lazer_bytes = lazer_writer.into_inner()?.into_inner();

        let local_position_attribute =
            POSITION_3D.with_custom_datatype(PointAttributeDataType::Vec3i32);
        let custom_layout = [local_position_attribute.clone()]
            .into_iter()
            .collect::<PointLayout>();

        let read_points = {
            let mut lazer_reader = LazerReader::new(Cursor::new(lazer_bytes))?;
            let mut buf =
                HashMapBuffer::with_capacity(las_header.number_of_points() as usize, custom_layout);
            lazer_reader.read_into(&mut buf, lazer_reader.remaining_points())?;
            buf
        };

        assert_eq!(read_points.len(), las_points.len());
        for (expected_position, actual_position) in las_points
            .view_attribute::<Vector3<i32>>(&local_position_attribute)
            .into_iter()
            .zip(
                read_points
                    .view_attribute::<Vector3<i32>>(&local_position_attribute)
                    .into_iter(),
            )
        {
            assert_eq!(expected_position, actual_position)
        }

        Ok(())
    }
}
