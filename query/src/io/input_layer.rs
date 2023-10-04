use anyhow::{anyhow, bail, Context, Result};
use pasture_core::{
    containers::{BorrowedBuffer, HashMapBuffer, InterleavedBuffer, VectorBuffer},
    layout::PointLayout,
};
use pasture_io::las::{point_layout_from_las_metadata, LASMetadata, LASReader};
use std::{
    collections::HashMap,
    ffi::OsStr,
    fmt::Display,
    ops::Range,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use crate::index::{DatasetID, PointRange};

use super::{
    BorrowedLasPointData, LASPointDataReader, LASTPointDataReader, LAZERPointDataLoader,
    LAZPointDataReader,
};

/// Handle to a file in the input layer. The file is uniquely identified by the ID of the dataset it belongs to,
/// and by its index within that dataset
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct FileHandle(pub DatasetID, pub usize);

impl Display for FileHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "(Dataset: {}, File index: {})", self.0, self.1)
    }
}

pub enum PointData {
    MmappedLas(BorrowedLasPointData),
    OwnedInterleaved(VectorBuffer),
    OwnedColumnar(HashMapBuffer),
}

impl<'a> BorrowedBuffer<'a> for PointData {
    fn len(&self) -> usize {
        match self {
            PointData::MmappedLas(borrowed) => borrowed.borrow_buffer_slice().len(),
            PointData::OwnedInterleaved(buffer) => buffer.len(),
            PointData::OwnedColumnar(buffer) => buffer.len(),
        }
    }

    fn point_layout(&self) -> &PointLayout {
        match self {
            PointData::MmappedLas(borrowed) => borrowed.borrow_buffer_slice().point_layout(),
            PointData::OwnedInterleaved(buffer) => buffer.point_layout(),
            PointData::OwnedColumnar(buffer) => buffer.point_layout(),
        }
    }

    fn get_point(&self, index: usize, data: &mut [u8]) {
        match self {
            PointData::MmappedLas(borrowed) => {
                borrowed.borrow_buffer_slice().get_point(index, data)
            }
            PointData::OwnedInterleaved(buffer) => buffer.get_point(index, data),
            PointData::OwnedColumnar(buffer) => buffer.get_point(index, data),
        }
    }

    fn get_point_range(&self, range: Range<usize>, data: &mut [u8]) {
        match self {
            PointData::MmappedLas(borrowed) => {
                borrowed.borrow_buffer_slice().get_point_range(range, data)
            }
            PointData::OwnedInterleaved(buffer) => buffer.get_point_range(range, data),
            PointData::OwnedColumnar(buffer) => buffer.get_point_range(range, data),
        }
    }

    unsafe fn get_attribute_unchecked(
        &self,
        attribute_member: &pasture_core::layout::PointAttributeMember,
        index: usize,
        data: &mut [u8],
    ) {
        match self {
            PointData::MmappedLas(borrowed) => borrowed
                .borrow_buffer_slice()
                .get_attribute_unchecked(attribute_member, index, data),
            PointData::OwnedInterleaved(buffer) => {
                buffer.get_attribute_unchecked(attribute_member, index, data)
            }
            PointData::OwnedColumnar(buffer) => {
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
            PointData::MmappedLas(borrowed) => borrowed.borrow_buffer_slice().get_point_ref(index),
            PointData::OwnedInterleaved(buffer) => buffer.get_point_ref(index),
            PointData::OwnedColumnar(_) => panic!("Unsupported operation"),
        }
    }

    fn get_point_range_ref<'b>(&'b self, range: Range<usize>) -> &'b [u8]
    where
        'a: 'b,
    {
        match self {
            PointData::MmappedLas(borrowed) => {
                borrowed.borrow_buffer_slice().get_point_range_ref(range)
            }
            PointData::OwnedInterleaved(buffer) => buffer.get_point_range_ref(range),
            PointData::OwnedColumnar(_) => panic!("Unsupported operation"),
        }
    }
}

/// Trait for point data loaders that specific format loaders can implement. This makes the input layer code
/// a bit simpler
pub(crate) trait PointDataLoader: Send + Sync {
    /// Returns data for the given points in the given `PointLayout`
    fn get_point_data(
        &self,
        point_range: Range<usize>,
        target_layout: &PointLayout,
        positions_in_world_space: bool,
    ) -> Result<PointData>;
    /// Return the number of bytes that this data loader currently uses. This is necessary to
    /// prevent memory overflows for data loaders that use `mmap`
    fn mem_size(&self) -> usize;
    /// Returns the default `PointLayout` for this data loader
    fn default_point_layout(&self) -> &PointLayout;
    /// Returns `true` if the underlying file format stores point positions in world space
    fn has_positions_in_world_space(&self) -> bool;
    /// Does this `PointDataLoader` support returning borrowed data (e.g. using `mmap`)?
    fn supports_borrowed_data(&self) -> bool;
}

/// Handles low-level data access and provides access to point data for the query layer
pub struct InputLayer {
    known_files: HashMap<FileHandle, PathBuf>,
    active_point_loaders: Mutex<HashMap<FileHandle, Arc<dyn PointDataLoader>>>,
    las_files_metadata: HashMap<FileHandle, LASMetadata>,
    max_ram_consumption: usize,
}

impl InputLayer {
    const DEFAULT_MAX_MEMORY_CONSUMPTION: usize = 1 << 30; // 1 GiB

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

            // TODO Does the LAS reader also work for LAZ? For LAST, it will work. But in any case, might be better
            // to replace this with a `LASMetadata::from_path` function
            // And of course also handle non-LAS files here
            let las_reader = LASReader::from_path(path, false)?;
            self.las_files_metadata.insert(
                FileHandle(dataset_id, file_number),
                las_reader.las_metadata().clone(),
            );
        }
        Ok(file_handles)
    }

    /// Does the input layer support returning borrowed point data for the given `point_range` within the given dataset?
    /// If so, this enables optimizations in the query layer as we can safely view attributes of points without any copying
    /// or parsing
    pub fn can_get_borrowed_point_data(
        &self,
        dataset_id: DatasetID,
        point_range: PointRange,
    ) -> Result<bool> {
        let file_handle = FileHandle(dataset_id, point_range.file_index);
        self.get_or_create_loader(file_handle)
            .map(|loader| loader.supports_borrowed_data())
    }

    /// Returns the data for the given `PointRange` in the given dataset
    pub fn get_point_data(
        &self,
        dataset_id: DatasetID,
        point_range: PointRange,
    ) -> Result<PointData> {
        let _span = tracy_client::span!("InputLayer::get_point_data");
        let file_handle = FileHandle(dataset_id, point_range.file_index);
        self.get_or_create_loader(file_handle).and_then(|loader| {
            loader.get_point_data(
                point_range.points_in_file,
                loader.default_point_layout(),
                loader.has_positions_in_world_space(),
            )
        })
    }

    pub fn get_point_data_in_layout(
        &self,
        dataset_id: DatasetID,
        point_range: PointRange,
        point_layout: &PointLayout,
        positions_in_worldspace: bool,
    ) -> Result<PointData> {
        let _span = tracy_client::span!("InputLayer::get_point_data_in_layout");
        let file_handle = FileHandle(dataset_id, point_range.file_index);
        self.get_or_create_loader(file_handle).and_then(|loader| {
            loader.get_point_data(
                point_range.points_in_file,
                point_layout,
                positions_in_worldspace,
            )
        })
    }

    /// Returns the metadata of the LAS file for the given `file_handle`
    pub fn get_las_metadata(&self, file_handle: FileHandle) -> Option<&LASMetadata> {
        self.las_files_metadata.get(&file_handle)
    }

    /// Returns the default `PointLayout` for the given file. This is the `PointLayout` that exactly represents
    /// the binary memory layout of the point records in the file
    pub fn get_default_point_layout_of_file(&self, file_handle: FileHandle) -> Result<PointLayout> {
        let metadata = self
            .get_las_metadata(file_handle)
            .ok_or(anyhow!("File metadata for file {file_handle} not found"))?;
        point_layout_from_las_metadata(metadata, true)
    }

    pub fn set_max_ram_consumption(&mut self, max_bytes: usize) {
        self.max_ram_consumption = max_bytes;
    }

    fn get_or_create_loader(&self, file_handle: FileHandle) -> Result<Arc<dyn PointDataLoader>> {
        let mut active_loaders = self
            .active_point_loaders
            .lock()
            .expect("Mutex was poisoned");

        if let Some(loader) = active_loaders.get(&file_handle) {
            Ok(loader.clone())
        } else {
            // Before we map a new file, make sure we don't run out of memory. If so, first evict some old files from
            // memory!
            self.evict_loaders_until_below_memory_threshold(&mut active_loaders);

            let file_path = self
                .known_files
                .get(&file_handle)
                .ok_or(anyhow!("Unknown file {}", file_handle))?;
            let loader = Self::make_loader_from_path(&file_path)
                .context("Could not create PointDataLoader for file")?;
            active_loaders.insert(file_handle, loader.clone());
            Ok(loader)
        }
    }

    fn make_loader_from_path(path: &Path) -> Result<Arc<dyn PointDataLoader>> {
        let file_extension = path.extension().and_then(OsStr::to_str).ok_or(anyhow!(
            "Could not determine file extension of file {}",
            path.display()
        ))?;

        match file_extension {
            "las" | "LAS" => LASPointDataReader::new(path)
                .map(|reader| -> Arc<dyn PointDataLoader> { Arc::new(reader) }),
            "last" | "LAST" => LASTPointDataReader::new(path)
                .map(|reader| -> Arc<dyn PointDataLoader> { Arc::new(reader) }),
            "laz" | "LAZ" => LAZPointDataReader::new(path)
                .map(|reader| -> Arc<dyn PointDataLoader> { Arc::new(reader) }),
            "lazer" | "LAZER" => LAZERPointDataLoader::new(path)
                .map(|reader| -> Arc<dyn PointDataLoader> { Arc::new(reader) }),
            other => Err(anyhow!("Unsupported file extension {other}")),
        }
    }

    fn evict_loaders_until_below_memory_threshold(
        &self,
        loaders: &mut HashMap<FileHandle, Arc<dyn PointDataLoader>>,
    ) {
        let _span = tracy_client::span!("evict_loaders");

        let current_memory = loaders
            .values()
            .map(|loader| loader.mem_size())
            .sum::<usize>();
        if current_memory <= self.max_ram_consumption {
            return;
        }

        let mut memdiff = (current_memory - self.max_ram_consumption) as i64;
        // Collect files to evict by their size (could use other metrics, like LRU cache)
        // Evicting just means removing their Arc from the input layer, we can't be sure that someone might still
        // be using the file, so this might not be a *true* eviction, but should at least prevent running out of memory
        while memdiff > 0 {
            let (largest_file_handle, file_memsize) = loaders
                .iter()
                .max_by(|(_, lfile), (_, rfile)| lfile.mem_size().cmp(&rfile.mem_size()))
                .map(|(handle, file)| (*handle, file.mem_size()))
                .expect("No file found, but we are above memory threshold. This shouldn't be");
            loaders.remove_entry(&largest_file_handle);
            // info!("Memory consumption exceeded by {memdiff} bytes. Evicted file {largest_file_handle} to free {file_memsize} bytes");
            memdiff -= file_memsize as i64;
        }
    }
}

impl Default for InputLayer {
    fn default() -> Self {
        Self {
            known_files: Default::default(),
            active_point_loaders: Default::default(),
            las_files_metadata: Default::default(),
            max_ram_consumption: Self::DEFAULT_MAX_MEMORY_CONSUMPTION,
        }
    }
}
