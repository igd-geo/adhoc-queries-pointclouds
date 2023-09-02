use anyhow::{anyhow, bail, Context, Result};
use memmap::Mmap;
use ouroboros::self_referencing;
use pasture_core::{
    containers::{
        BorrowedBuffer, BufferSlice, ExternalMemoryBuffer, InterleavedBuffer, SliceBuffer,
        VectorBuffer,
    },
    layout::PointLayout,
};
use pasture_io::las::{point_layout_from_las_metadata, LASMetadata, LASReader};
use std::{
    collections::HashMap,
    fmt::Display,
    fs::File,
    io::Cursor,
    ops::Range,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use crate::index::{DatasetID, PointRange};

/// Handle to a file in the input layer. The file is uniquely identified by the ID of the dataset it belongs to,
/// and by its index within that dataset
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct FileHandle(pub DatasetID, pub usize);

impl Display for FileHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "(Dataset: {}, File index: {})", self.0, self.1)
    }
}

#[self_referencing]
pub struct BorrowedPointData {
    mapped_file: Arc<MappedLASFile>,
    #[covariant]
    #[borrows(mapped_file)]
    buffer_slice: BufferSlice<'this, ExternalMemoryBuffer<&'this [u8]>>,
}

pub enum PointData {
    Mmapped(BorrowedPointData),
    Owned(VectorBuffer),
}

impl<'a> BorrowedBuffer<'a> for PointData {
    fn len(&self) -> usize {
        match self {
            PointData::Mmapped(borrowed) => borrowed.borrow_buffer_slice().len(),
            PointData::Owned(buffer) => buffer.len(),
        }
    }

    fn point_layout(&self) -> &PointLayout {
        match self {
            PointData::Mmapped(borrowed) => borrowed.borrow_buffer_slice().point_layout(),
            PointData::Owned(buffer) => buffer.point_layout(),
        }
    }

    fn get_point(&self, index: usize, data: &mut [u8]) {
        match self {
            PointData::Mmapped(borrowed) => borrowed.borrow_buffer_slice().get_point(index, data),
            PointData::Owned(buffer) => buffer.get_point(index, data),
        }
    }

    fn get_point_range(&self, range: Range<usize>, data: &mut [u8]) {
        match self {
            PointData::Mmapped(borrowed) => {
                borrowed.borrow_buffer_slice().get_point_range(range, data)
            }
            PointData::Owned(buffer) => buffer.get_point_range(range, data),
        }
    }

    unsafe fn get_attribute_unchecked(
        &self,
        attribute_member: &pasture_core::layout::PointAttributeMember,
        index: usize,
        data: &mut [u8],
    ) {
        match self {
            PointData::Mmapped(borrowed) => borrowed.borrow_buffer_slice().get_attribute_unchecked(
                attribute_member,
                index,
                data,
            ),
            PointData::Owned(buffer) => {
                buffer.get_attribute_unchecked(attribute_member, index, data)
            }
        }
    }
}

impl<'a> InterleavedBuffer<'a> for PointData {
    fn get_point_ref<'b>(&'b self, index: usize) -> &'b [u8]
    where
        'a: 'b,
    {
        match self {
            PointData::Mmapped(borrowed) => borrowed.borrow_buffer_slice().get_point_ref(index),
            PointData::Owned(buffer) => buffer.get_point_ref(index),
        }
    }

    fn get_point_range_ref<'b>(&'b self, range: Range<usize>) -> &'b [u8]
    where
        'a: 'b,
    {
        match self {
            PointData::Mmapped(borrowed) => {
                borrowed.borrow_buffer_slice().get_point_range_ref(range)
            }
            PointData::Owned(buffer) => buffer.get_point_range_ref(range),
        }
    }
}

#[self_referencing]
pub struct MappedLASFile {
    metadata: LASMetadata,
    mmap: memmap::Mmap,
    #[covariant]
    #[borrows(mmap)]
    las_points_buffer: ExternalMemoryBuffer<&'this [u8]>,
}

impl MappedLASFile {
    /// Map the file at `path` into memory
    pub(crate) fn map_path(path: &Path) -> Result<Self> {
        let file =
            File::open(path).context(format!("Failed to open LAS file {}", path.display()))?;
        let mmap = unsafe { memmap::Mmap::map(&file).context("Failed to mmap LAS file")? };

        let las_metadata = LASReader::from_read(Cursor::new(&mmap), false)
            .context("Failed to read LAS file")?
            .las_metadata()
            .clone();

        let point_layout = point_layout_from_las_metadata(&las_metadata, true).context(format!(
            "No matching PointLayout found for LAS file {}",
            path.display()
        ))?;

        let las_header = las_metadata
            .raw_las_header()
            .ok_or(anyhow!("No LAS header found"))?;
        let raw_las_header = las_header
            .clone()
            .into_raw()
            .context("Can't convert LAS header into raw format")?;
        let point_record_byte_range = (raw_las_header.offset_to_point_data as usize)
            ..(raw_las_header.offset_to_end_of_points() as usize);

        Ok(MappedLASFileBuilder {
            metadata: las_metadata,
            mmap,
            las_points_buffer_builder: |mmap: &Mmap| {
                ExternalMemoryBuffer::new(&mmap[point_record_byte_range], point_layout)
            },
        }
        .build())
    }

    pub(crate) fn get_buffer_for_points(
        &self,
        point_range: Range<usize>,
    ) -> BufferSlice<'_, ExternalMemoryBuffer<&[u8]>> {
        self.borrow_las_points_buffer().slice(point_range)
    }
}

/// Handles low-level data access and provides access to point data for the query layer
#[derive(Default)]
pub struct InputLayer {
    known_files: HashMap<FileHandle, PathBuf>,
    mapped_las_files: Mutex<HashMap<FileHandle, Arc<MappedLASFile>>>,
    las_files_metadata: HashMap<FileHandle, LASMetadata>,
}

impl InputLayer {
    /// Add the given `files` for the given dataset to the input layer
    pub fn add_files<'a, P: AsRef<Path>>(
        &mut self,
        files: &'a [P],
        dataset_id: DatasetID,
    ) -> Result<Vec<FileHandle>> {
        let file_handles = files
            .iter()
            .enumerate()
            .map(|(file_number, _)| FileHandle(dataset_id, file_number))
            .collect();
        for (file_number, path) in files.iter().enumerate() {
            if !path.as_ref().exists() {
                bail!("File {} does not exist", path.as_ref().display());
            }
            self.known_files.insert(
                FileHandle(dataset_id, file_number),
                path.as_ref().to_owned(),
            );

            let las_reader = LASReader::from_path(path)?;
            self.las_files_metadata.insert(
                FileHandle(dataset_id, file_number),
                las_reader.las_metadata().clone(),
            );
        }
        Ok(file_handles)
    }

    /// Returns the data for the given `PointRange` in the given dataset
    pub fn get_point_data(
        &self,
        dataset_id: DatasetID,
        point_range: PointRange,
    ) -> Result<PointData> {
        // TODO Support more than just LAS files
        let file_handle = FileHandle(dataset_id, point_range.file_index);
        self.get_or_map_las_file(file_handle)
            .map(|mapped_las_file| {
                PointData::Mmapped(
                    BorrowedPointDataBuilder {
                        mapped_file: mapped_las_file,
                        buffer_slice_builder: |borrowed_file: &Arc<MappedLASFile>| {
                            borrowed_file.get_buffer_for_points(point_range.points_in_file)
                        },
                    }
                    .build(),
                )
            })
    }

    /// Returns the metadata of the LAS file for the given `file_handle`
    pub fn get_las_metadata(&self, file_handle: FileHandle) -> Option<&LASMetadata> {
        self.las_files_metadata.get(&file_handle)
    }

    fn get_or_map_las_file(&self, file_handle: FileHandle) -> Result<Arc<MappedLASFile>> {
        let mut mapped_las_files = self.mapped_las_files.lock().expect("Mutex was poisoned");

        if let Some(mapped_file) = mapped_las_files.get(&file_handle) {
            Ok(mapped_file.clone())
        } else {
            let file_path = self
                .known_files
                .get(&file_handle)
                .ok_or(anyhow!("Unknown file {}", file_handle))?;
            let mapped_file = MappedLASFile::map_path(file_path.as_path()).context(format!(
                "Failed to map file {} into memory",
                file_path.display()
            ))?;
            mapped_las_files.insert(file_handle, Arc::new(mapped_file));
            let mapped_file = mapped_las_files.get(&file_handle).unwrap();
            Ok(mapped_file.clone())
        }
    }
}
