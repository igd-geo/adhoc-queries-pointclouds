use std::{io::Write, sync::Mutex};

use anyhow::{Context, Result};
use pasture_core::{
    containers::{
        BorrowedBuffer, InterleavedBuffer, MakeBufferFromLayout, OwningBuffer, VectorBuffer,
    },
    layout::PointLayout,
};

use crate::index::{DatasetID, PointRange};

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
    // TODO Support interleaved and columnar output formats
}

impl StdoutOutput {
    pub fn new(output_layout: PointLayout) -> Self {
        Self { output_layout }
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
        assert_eq!(point_range.points_in_file.len(), matching_indices.len());
        let memory = input_layer
            .get_point_data(dataset_id, point_range.clone())
            .context(format!(
                "Could not get point data for points {point_range} in dataset {dataset_id}"
            ))?;

        if self.output_layout == *memory.point_layout() {
            let points_range = memory.get_point_range_ref(point_range.points_in_file);
            let size_of_point = self.output_layout.size_of_point_entry() as usize;
            let mut stdout = std::io::stdout().lock();
            for (_, point_memory) in points_range
                .chunks_exact(size_of_point)
                .enumerate()
                .filter(|(idx, _)| matching_indices[*idx])
            {
                stdout.write_all(point_memory)?;
            }

            Ok(())
        } else {
            unimplemented!()
        }
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
