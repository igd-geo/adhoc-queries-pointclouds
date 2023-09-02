use anyhow::Result;
use pasture_core::{containers::BorrowedBuffer, math::AABB};

// use crate::grid_sampling::SparseGrid;

pub trait ResultCollector: Send {
    fn collect<'a, 'b, B: BorrowedBuffer<'a>>(&mut self, points: &'b B)
    where
        'a: 'b;
    fn point_count(&self) -> usize;
}

// pub struct BufferCollector {
//     buffers: Vec<Box<dyn PointBufferSend>>,
// }

// impl BufferCollector {
//     pub fn new() -> Self {
//         Self {
//             buffers: Vec::new(),
//         }
//     }

//     pub fn buffers(&self) -> &[Box<dyn PointBufferSend>] {
//         &self.buffers
//     }

//     pub fn as_single_buffer(&self) -> Option<InterleavedVecPointStorage> {
//         if self.buffers.is_empty() {
//             return None;
//         }

//         let layout = self.buffers[0].point_layout();
//         let mut ret = InterleavedVecPointStorage::new(layout.clone());

//         for buffer in &self.buffers {
//             ret.push(buffer.as_point_buffer());
//         }

//         Some(ret)
//     }
// }

// impl ResultCollector for BufferCollector {
//     fn point_count(&self) -> usize {
//         self.buffers.iter().map(|buf| buf.len()).sum()
//     }

//     fn collect(&mut self, points: Box<dyn PointBufferSend>) {
//         self.buffers.push(points);
//     }
// }

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

    fn collect<'a, 'b, B: BorrowedBuffer<'a>>(&mut self, points: &'b B)
    where
        'a: 'b,
    {
        todo!()
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

    fn collect<'a, 'b, B: BorrowedBuffer<'a>>(&mut self, points: &'b B)
    where
        'a: 'b,
    {
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

    fn collect<'a, 'b, B: BorrowedBuffer<'a>>(&mut self, points: &'b B)
    where
        'a: 'b,
    {
        todo!()
    }
}
