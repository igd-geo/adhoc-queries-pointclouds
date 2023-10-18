use std::{
    fs::File,
    io::{BufWriter, Write},
    path::Path,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Mutex,
    },
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
    io::{FileHandle, PointData},
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
    ) -> Result<()>;
}

/// Output points to `stdout` in the given `PointLayout`
pub struct StdoutOutput {
    output_layout: PointLayout,
    positions_in_world_space: bool,
    bytes_output: AtomicUsize,
    // TODO Support interleaved and columnar output formats
}

impl StdoutOutput {
    pub fn new(output_layout: PointLayout, positions_in_world_space: bool) -> Self {
        Self {
            output_layout,
            positions_in_world_space,
            bytes_output: AtomicUsize::default(),
        }
    }

    /// Returns the total number of bytes that were written to `stdout`
    pub fn bytes_output(&self) -> usize {
        self.bytes_output.load(Ordering::SeqCst)
    }
}

impl PointOutput for StdoutOutput {
    fn output(
        &self,
        input_layer: &InputLayer,
        dataset_id: DatasetID,
        point_range: PointRange,
        matching_indices: &[bool],
    ) -> Result<()> {
        let _span = tracy_client::span!("StdoutOutput::output");

        assert_eq!(point_range.points_in_file.len(), matching_indices.len());
        let file_point_layout = input_layer
            .get_default_point_layout_of_file(FileHandle(dataset_id, point_range.file_index))
            .context("Could not determine default PointLayout of file")?;

        let memory = if self.output_layout == file_point_layout {
            input_layer
                .get_point_data(dataset_id, point_range.clone())
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
                )
                .context(format!(
                    "Could not get point data for points {point_range} in dataset {dataset_id}"
                ))
        }?;

        let size_of_point = self.output_layout.size_of_point_entry() as usize;

        // If the data is columnar, we can't range-copy, but the current PointData interface does not have this
        // information. Indicates to me that the interface is flawed...
        match memory {
            PointData::OwnedColumnar(point_buffer) => {
                let num_matches = matching_indices.iter().filter(|m| **m).count();
                let mut interleaved_buffer = vec![0; num_matches * size_of_point];
                for (index, point_chunk) in matching_indices
                    .iter()
                    .zip(interleaved_buffer.chunks_exact_mut(size_of_point))
                    .enumerate()
                    .filter_map(|(index, (is_match, chunk))| {
                        if *is_match {
                            Some((index, chunk))
                        } else {
                            None
                        }
                    })
                {
                    point_buffer.get_point(index, point_chunk);
                }

                let mut stdout = std::io::stdout().lock();
                stdout.write_all(&interleaved_buffer)?;
                self.bytes_output
                    .fetch_add(interleaved_buffer.len(), Ordering::SeqCst);
            }
            _ => {
                let points_range = memory.get_point_range_ref(0..memory.len());

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
                self.bytes_output
                    .fetch_add(filtered_memory.len(), Ordering::SeqCst);
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
    ) -> Result<()> {
        assert_eq!(point_range.points_in_file.len(), matching_indices.len());
        let memory = input_layer
            .get_point_data(dataset_id, point_range.clone())
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
    ) -> Result<()> {
        let memory = input_layer
            .get_point_data_in_layout(dataset_id, point_range.clone(), &self.output_layout, true)
            .context("Could not get point data")?;

        // This is pretty inefficient, but I currently see no other way besides collecting the data into a new
        // buffer, since the `PointWriter::write` method expects a `BorrowedBuffer`, but we can't implement `SliceBuffer`
        // on `PointData`, since the type of the slice depends on the variant of `PointData` (either a `VectorBuffer` or
        // an `ExternalMemoryBuffer`)
        let num_matches = matching_indices.iter().copied().filter(|b| *b).count();
        let mut tmp_buffer = VectorBuffer::new_from_layout(self.output_layout.clone());
        tmp_buffer.resize(num_matches);
        let mut current_point: usize = 0;

        let mut single_point_buffer = vec![0; self.output_layout.size_of_point_entry() as usize];

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
        matching_indices: &[bool],
    ) -> Result<()> {
        let count = matching_indices.iter().filter(|b| **b).count();
        self.count.fetch_add(count, Ordering::SeqCst);
        Ok(())
    }
}
