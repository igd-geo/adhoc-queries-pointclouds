use std::{
    fs::File,
    io::{BufWriter, Write},
    path::Path,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Mutex,
    },
    time::Instant,
};

use anyhow::{Context, Result};
use pasture_core::{
    containers::{
        BorrowedBuffer, BorrowedMutBuffer, InterleavedBuffer, MakeBufferFromLayout, OwningBuffer,
        VectorBuffer,
    },
    layout::PointLayout,
};
use pasture_io::{base::PointWriter, las::LASWriter};

use crate::{
    index::{DatasetID, PointRange},
    io::{FileHandle, PointData, PointDataMemoryLayout},
};

use super::InputLayer;

/// Trait for the output component responsible for outputting the point data from the query engine
pub trait PointOutput: Send + Sync {
    fn output(
        &self,
        input_layer: &InputLayer,
        dataset_id: DatasetID,
        point_range: PointRange,
        matching_indices: &[bool],
        match_count: usize,
    ) -> Result<()>;
}

/// Output points to `stdout` in the given `PointLayout`
pub struct StdoutOutput {
    output_layout: PointLayout,
    positions_in_world_space: bool,
    bytes_output: AtomicUsize,
    bytes_over_time: Option<Mutex<Vec<(Instant, usize)>>>,
    // TODO Support interleaved and columnar output formats
}

impl StdoutOutput {
    pub fn new(
        output_layout: PointLayout,
        positions_in_world_space: bool,
        track_data_over_time: bool,
    ) -> Self {
        Self {
            output_layout,
            positions_in_world_space,
            bytes_output: AtomicUsize::default(),
            bytes_over_time: if track_data_over_time {
                let initial_bytes = (Instant::now(), 0);
                Some(Mutex::new(vec![initial_bytes]))
            } else {
                None
            },
        }
    }

    /// Returns the total number of bytes that were written to `stdout`
    pub fn bytes_output(&self) -> usize {
        self.bytes_output.load(Ordering::SeqCst)
    }

    /// Returns a collection of values for how many bytes were written at a given point in time. Returns `None` if
    /// `track_data_over_time` was passed to `StdoutOutput::new`
    pub fn bytes_over_time(&self) -> Option<Vec<(Instant, usize)>> {
        self.bytes_over_time
            .as_ref()
            .map(|b| b.lock().expect("Failed to lock").to_vec())
    }

    fn log_write(&self, added_bytes: usize) {
        if let Some(bytes_at_timestamp) = self.bytes_over_time.as_ref() {
            self.bytes_output.fetch_add(added_bytes, Ordering::SeqCst);
            let current_bytes = self.bytes_output.load(Ordering::SeqCst);
            let now = Instant::now();
            bytes_at_timestamp
                .lock()
                .expect("Failed to lock")
                .push((now, current_bytes));
        } else {
            self.bytes_output.fetch_add(added_bytes, Ordering::SeqCst);
        }
    }
}

impl PointOutput for StdoutOutput {
    fn output(
        &self,
        input_layer: &InputLayer,
        dataset_id: DatasetID,
        point_range: PointRange,
        matching_indices: &[bool],
        match_count: usize,
    ) -> Result<()> {
        let _span = tracy_client::span!("StdoutOutput::output");

        assert_eq!(point_range.points_in_file.len(), matching_indices.len());
        let file_point_layout = input_layer
            .get_default_point_layout_of_file(FileHandle(dataset_id, point_range.file_index))
            .context("Could not determine default PointLayout of file")?;

        let preferred_memory_layout = input_layer
            .get_preferred_memory_layout(FileHandle(dataset_id, point_range.file_index))?;
        let memory = if self.output_layout == file_point_layout {
            input_layer
                .get_point_data(dataset_id, point_range.clone(), preferred_memory_layout)
                .context(format!(
                    "Could not get point data for points {point_range} in dataset {dataset_id}"
                ))
        } else {
            input_layer
                .get_point_data_in_layout(
                    dataset_id,
                    point_range.clone(),
                    &self.output_layout,
                    self.positions_in_world_space,
                    preferred_memory_layout,
                )
                .context(format!(
                    "Could not get point data for points {point_range} in dataset {dataset_id}"
                ))
        }?;

        let size_of_point = self.output_layout.size_of_point_entry() as usize;

        // For now we always output data in interleaved format. We could support columnar data output as well,
        // but since most applications assume interleaved point data it seems to be a reasonable default
        match memory {
            PointData::OwnedColumnar(columnar_buffer) => {
                let filtered_data =
                    columnar_buffer.filter::<VectorBuffer, _>(|idx| matching_indices[idx]);
                let filtered_memory = filtered_data.get_point_range_ref(0..filtered_data.len());
                let mut stdout = std::io::stdout().lock();
                stdout.write_all(&filtered_memory)?;
                self.log_write(filtered_memory.len());
            }
            _ => {
                let points_range = memory.get_point_range_ref(0..memory.len());

                if match_count == memory.len() {
                    let mut stdout = std::io::stdout().lock();
                    stdout.write_all(points_range)?;
                    self.log_write(points_range.len());
                } else {
                    let filtered_memory = points_range
                        .chunks_exact(size_of_point)
                        .enumerate()
                        .filter_map(|(idx, data)| {
                            if matching_indices[idx] {
                                Some(data)
                            } else {
                                None
                            }
                        })
                        .flatten()
                        .copied()
                        .collect::<Vec<_>>();

                    let mut stdout = std::io::stdout().lock();
                    stdout.write_all(&filtered_memory)?;
                    self.log_write(filtered_memory.len());
                }
            }
        }

        Ok(())
    }
}

#[derive(Default)]
pub struct InMemoryOutput {
    buffers: Mutex<Vec<VectorBuffer>>,
}

impl InMemoryOutput {
    /// Get all cached buffers as a single point buffer. Returns `None` if no buffer was cached
    pub fn into_single_buffer(self) -> Option<VectorBuffer> {
        let buffers = self.buffers.into_inner().expect("Mutex was poisoned");
        buffers.into_iter().reduce(|mut acc, buffer| {
            acc.append_interleaved(&buffer);
            acc
        })
    }
}

impl PointOutput for InMemoryOutput {
    fn output(
        &self,
        input_layer: &InputLayer,
        dataset_id: DatasetID,
        point_range: PointRange,
        matching_indices: &[bool],
        _match_count: usize,
    ) -> Result<()> {
        assert_eq!(point_range.points_in_file.len(), matching_indices.len());
        let memory = input_layer
            .get_point_data(
                dataset_id,
                point_range.clone(),
                PointDataMemoryLayout::Interleaved,
            )
            .context(format!(
                "Could not get point data for points {point_range} in dataset {dataset_id}"
            ))?;

        let mut owned_buffer = VectorBuffer::new_from_layout(memory.point_layout().clone());

        for (point_index, _) in matching_indices
            .iter()
            .enumerate()
            .filter(|(_, is_match)| **is_match)
        {
            // Is safe because both buffers have the same point layout
            unsafe {
                owned_buffer.push_points(memory.get_point_ref(point_index));
            }
        }

        self.buffers.lock().unwrap().push(owned_buffer);

        Ok(())
    }
}

#[derive(Default)]
pub struct NullOutput;

impl PointOutput for NullOutput {
    fn output(
        &self,
        _input_layer: &InputLayer,
        _dataset_id: DatasetID,
        _point_range: PointRange,
        _matching_indices: &[bool],
        _match_count: usize,
    ) -> Result<()> {
        // intentionally does nothing
        Ok(())
    }
}

pub struct LASOutput {
    writer: Mutex<LASWriter<BufWriter<File>>>,
    output_layout: PointLayout,
}

impl LASOutput {
    pub fn new<P: AsRef<Path>>(path: P, point_layout: &PointLayout) -> Result<Self> {
        let writer = LASWriter::from_path_and_point_layout(path, point_layout)?;
        let output_layout = writer.get_default_point_layout().clone();
        Ok(Self {
            writer: Mutex::new(writer),
            output_layout,
        })
    }
}

impl Drop for LASOutput {
    fn drop(&mut self) {
        let mut writer = self.writer.lock().expect("Can't lock LAS writer");
        writer.flush().expect("flush() failed");
    }
}

impl PointOutput for LASOutput {
    fn output(
        &self,
        input_layer: &InputLayer,
        dataset_id: DatasetID,
        point_range: PointRange,
        matching_indices: &[bool],
        match_count: usize,
    ) -> Result<()> {
        let memory = input_layer
            .get_point_data_in_layout(
                dataset_id,
                point_range.clone(),
                &self.output_layout,
                true,
                PointDataMemoryLayout::Interleaved,
            )
            .context("Could not get point data")?;

        if match_count == point_range.points_in_file.len() {
            let mut writer = self.writer.lock().expect("Could not lock writer");
            writer.write(&memory)?;
        } else {
            // This is pretty inefficient, but I currently see no other way besides collecting the data into a new
            // buffer, since the `PointWriter::write` method expects a `BorrowedBuffer`, but we can't implement `SliceBuffer`
            // on `PointData`, since the type of the slice depends on the variant of `PointData` (either a `VectorBuffer` or
            // an `ExternalMemoryBuffer`)
            let mut tmp_buffer = VectorBuffer::new_from_layout(self.output_layout.clone());
            tmp_buffer.resize(match_count);
            let mut current_point: usize = 0;

            let mut single_point_buffer =
                vec![0; self.output_layout.size_of_point_entry() as usize];

            for (point_index, _) in matching_indices
                .iter()
                .enumerate()
                .filter(|(_, is_match)| **is_match)
            {
                memory.get_point(point_index, &mut single_point_buffer);
                unsafe {
                    tmp_buffer.set_point(current_point, &single_point_buffer);
                }
                current_point += 1;
            }

            let mut writer = self.writer.lock().expect("Could not lock writer");
            writer.write(&tmp_buffer)?;
        }

        Ok(())
    }
}

/// Point output that only counts the number of matches but outputs no actual data
#[derive(Default)]
pub struct CountOutput {
    count: AtomicUsize,
}

impl CountOutput {
    pub fn count(&self) -> usize {
        self.count.load(Ordering::SeqCst)
    }
}

impl PointOutput for CountOutput {
    fn output(
        &self,
        _input_layer: &InputLayer,
        _dataset_id: DatasetID,
        _point_range: PointRange,
        _matching_indices: &[bool],
        match_count: usize,
    ) -> Result<()> {
        self.count.fetch_add(match_count, Ordering::SeqCst);
        Ok(())
    }
}
