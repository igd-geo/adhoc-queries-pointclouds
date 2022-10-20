use anyhow::Result;
use pasture_core::{containers::PointBuffer, math::AABB};

use crate::grid_sampling::SparseGrid;

pub trait PointBufferSend: PointBuffer + Send {}
impl<T: PointBuffer + Send> PointBufferSend for T {}

pub trait ResultCollector: Send {
    fn collect(&mut self, points: Box<dyn PointBufferSend>);
    fn point_count(&self) -> usize;
}

pub struct BufferCollector {
    buffers: Vec<Box<dyn PointBufferSend>>,
}

impl BufferCollector {
    pub fn new() -> Self {
        Self {
            buffers: Vec::new(),
        }
    }
}

impl ResultCollector for BufferCollector {
    fn point_count(&self) -> usize {
        self.buffers.iter().map(|buf| buf.len()).sum()
    }

    fn collect(&mut self, points: Box<dyn PointBufferSend>) {
        self.buffers.push(points);
    }
}

pub struct StdOutCollector {}

impl StdOutCollector {
    pub fn _new() -> Self {
        Self {}
    }
}

impl ResultCollector for StdOutCollector {
    fn point_count(&self) -> usize {
        0
    }

    fn collect(&mut self, points: Box<dyn PointBufferSend>) {
        unimplemented!()
    }
}

pub struct CountCollector {
    point_count: usize,
}

impl CountCollector {
    pub fn new() -> Self {
        Self { point_count: 0 }
    }
}

impl ResultCollector for CountCollector {
    fn point_count(&self) -> usize {
        self.point_count
    }

    fn collect(&mut self, points: Box<dyn PointBufferSend>) {
        self.point_count += points.len();
    }
}

pub struct GridSampledCollector {
    grid: SparseGrid,
}

impl GridSampledCollector {
    pub fn new(bounds: AABB<f64>, cell_size: f64) -> Result<Self> {
        let grid = SparseGrid::new(bounds, cell_size)?;
        Ok(Self { grid: grid })
    }
}

impl ResultCollector for GridSampledCollector {
    fn point_count(&self) -> usize {
        self.grid.points().count()
    }

    fn collect(&mut self, points: Box<dyn PointBufferSend>) {
        unimplemented!()
    }
}
