use anyhow::Result;
use pasture_core::math::AABB;

use crate::{grid_sampling::SparseGrid, points::Point};

pub trait ResultCollector {
    fn collect_one(&mut self, point: Point);
    fn points(&self) -> Option<Vec<Point>>;
    fn points_ref(&self) -> Option<&[Point]>;
    fn point_count(&self) -> usize;
}

pub struct BufferCollector {
    buffer: Vec<Point>,
}

impl BufferCollector {
    pub fn new() -> Self {
        Self { buffer: Vec::new() }
    }

    pub fn buffer(&self) -> &[Point] {
        &self.buffer[..]
    }
}

impl ResultCollector for BufferCollector {
    fn collect_one(&mut self, point: Point) {
        self.buffer.push(point);
    }

    fn points(&self) -> Option<Vec<Point>> {
        Some(self.buffer().to_vec())
    }

    fn points_ref(&self) -> Option<&[Point]> {
        Some(&self.buffer())
    }

    fn point_count(&self) -> usize {
        self.buffer().len()
    }
}

pub struct StdOutCollector {}

impl StdOutCollector {
    pub fn new() -> Self {
        Self {}
    }
}

impl ResultCollector for StdOutCollector {
    fn collect_one(&mut self, point: Point) {
        println!("Found point: {:#?}", point);
    }

    fn points(&self) -> Option<Vec<Point>> {
        None
    }

    fn points_ref(&self) -> Option<&[Point]> {
        None
    }

    fn point_count(&self) -> usize {
        0
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
    fn collect_one(&mut self, _: Point) {
        self.point_count += 1;
    }

    fn points(&self) -> Option<Vec<Point>> {
        None
    }

    fn points_ref(&self) -> Option<&[Point]> {
        None
    }

    fn point_count(&self) -> usize {
        self.point_count
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
    fn collect_one(&mut self, point: Point) {
        self.grid.insert_point(point);
    }

    fn points(&self) -> Option<Vec<Point>> {
        Some(self.grid.points().map(|p| p.clone()).collect())
    }

    fn points_ref(&self) -> Option<&[Point]> {
        None
    }

    fn point_count(&self) -> usize {
        self.grid.points().count()
    }
}
