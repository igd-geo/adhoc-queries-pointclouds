use anyhow::{anyhow, Result};
use pasture_core::{
    math::AABB,
    nalgebra::{distance_squared, Point3, Vector3},
};
use readers::Point;
use std::collections::HashMap;

pub struct SparseGrid {
    bounds: AABB<f64>,
    cell_size: f64,
    dimensions: Vector3<u64>,
    bits_per_dimension: Vector3<u64>,
    cells: HashMap<u64, Point>,
}

impl SparseGrid {
    pub fn new(bounds: AABB<f64>, cell_size: f64) -> Result<Self> {
        let extent = Vector3::new(
            bounds.max().x - bounds.min().x,
            bounds.max().y - bounds.min().y,
            bounds.max().z - bounds.min().z,
        );
        let num_cells_per_dimension = Vector3::new(
            f64::ceil(extent.x / cell_size),
            f64::ceil(extent.y / cell_size),
            f64::ceil(extent.z / cell_size),
        );
        let x_bits = f64::ceil(f64::log2(num_cells_per_dimension.x)) as u64;
        let y_bits = f64::ceil(f64::log2(num_cells_per_dimension.y)) as u64;
        let z_bits = f64::ceil(f64::log2(num_cells_per_dimension.z)) as u64;
        if (x_bits + y_bits + z_bits) as usize > (std::mem::size_of::<u64>() * 8) {
            return Err(anyhow!("Too many cells ({}*{}*{}) in SparseGrid! The number of cells exceeds the capacity of a u64 index!"));
        }

        Ok(Self {
            bounds: bounds,
            cell_size: cell_size,
            dimensions: Vector3::new(
                num_cells_per_dimension.x as u64,
                num_cells_per_dimension.y as u64,
                num_cells_per_dimension.z as u64,
            ),
            bits_per_dimension: Vector3::new(x_bits, y_bits, z_bits),
            cells: HashMap::new(),
        })
    }

    pub fn insert_point(&mut self, point: Point) -> bool {
        let rx = (point.position.x - self.bounds.min().x) * self.dimensions.x as f64
            / (self.bounds.max().x - self.bounds.min().x);
        let ry = (point.position.y - self.bounds.min().y) * self.dimensions.y as f64
            / (self.bounds.max().y - self.bounds.min().y);
        let rz = (point.position.z - self.bounds.min().z) * self.dimensions.z as f64
            / (self.bounds.max().z - self.bounds.min().z);

        let cell_x = rx as u64;
        let cell_y = ry as u64;
        let cell_z = rz as u64;

        let x_bit_mask = ((1 as u64) << self.bits_per_dimension.x) - 1;
        let y_bit_mask = ((1 as u64) << self.bits_per_dimension.y) - 1;
        let z_bit_mask = ((1 as u64) << self.bits_per_dimension.z) - 1;

        let y_bit_shift = self.bits_per_dimension.x;
        let z_bit_shift = self.bits_per_dimension.x + self.bits_per_dimension.y;
        let index = (cell_x & x_bit_mask)
            | (cell_y & y_bit_mask) << y_bit_shift
            | (cell_z & z_bit_mask) << z_bit_shift;

        match self.cells.get_mut(&index) {
            None => {
                self.cells.insert(index, point);
                true
            }
            Some(current_point) => {
                let cell_center = Point3::new(
                    (cell_x as f64 + 0.5) * self.cell_size + self.bounds.min().x,
                    (cell_y as f64 + 0.5) * self.cell_size + self.bounds.min().y,
                    (cell_z as f64 + 0.5) * self.cell_size + self.bounds.min().z,
                );
                let cur_dist_sqr = distance_squared(
                    &cell_center,
                    &Point3::new(
                        current_point.position.x,
                        current_point.position.y,
                        current_point.position.z,
                    ),
                );
                let new_dist_sqr = distance_squared(
                    &cell_center,
                    &Point3::new(point.position.x, point.position.y, point.position.z),
                );

                if new_dist_sqr < cur_dist_sqr {
                    *current_point = point;
                    true
                } else {
                    false
                }
            }
        }
    }

    pub fn cells(&self) -> impl Iterator<Item = &u64> {
        self.cells.keys()
    }

    pub fn points(&self) -> impl Iterator<Item = &Point> {
        self.cells.values()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_grid_add_one() -> Result<()> {
        let bounds = AABB::from_min_max(Point3::new(-5.0, -5.0, -5.0), Point3::new(5.0, 5.0, 5.0));
        let mut grid = SparseGrid::new(bounds, 1.0)?;

        grid.insert_point(Point {
            position: Vector3::new(-4.5, -4.6, -4.7),
            ..Default::default()
        });

        let cells = grid.cells().collect::<Vec<_>>();
        assert_eq!(cells.len(), 1);
        assert_eq!(*cells[0], 0);

        let points = grid.points().collect::<Vec<_>>();
        assert_eq!(points.len(), 1);
        assert_eq!(points[0].position.x, -4.5);
        assert_eq!(points[0].position.y, -4.6);
        assert_eq!(points[0].position.z, -4.7);

        Ok(())
    }

    #[test]
    fn test_sparse_grid_add_multiple_in_different_cells() -> Result<()> {
        let bounds = AABB::from_min_max(Point3::new(-5.0, -5.0, -5.0), Point3::new(5.0, 5.0, 5.0));
        let mut grid = SparseGrid::new(bounds, 1.0)?;

        grid.insert_point(Point {
            position: Vector3::new(-4.5, -4.6, -4.7),
            ..Default::default()
        });

        grid.insert_point(Point {
            position: Vector3::new(-3.5, -4.5, -4.4),
            ..Default::default()
        });

        let cells = grid.cells().collect::<Vec<_>>();
        assert_eq!(cells.len(), 2);
        assert_eq!(*cells[0], 0);
        assert_eq!(*cells[1], 1);

        let points = grid.points().collect::<Vec<_>>();
        assert_eq!(points.len(), 2);

        assert_eq!(points[0].position.x, -4.5);
        assert_eq!(points[0].position.y, -4.6);
        assert_eq!(points[0].position.z, -4.7);

        assert_eq!(points[1].position.x, -3.5);
        assert_eq!(points[1].position.y, -4.5);
        assert_eq!(points[1].position.z, -4.4);

        Ok(())
    }

    #[test]
    fn test_sparse_grid_add_multiple_in_same_cell() -> Result<()> {
        let bounds = AABB::from_min_max(Point3::new(-5.0, -5.0, -5.0), Point3::new(5.0, 5.0, 5.0));
        let mut grid = SparseGrid::new(bounds, 1.0)?;

        grid.insert_point(Point {
            position: Vector3::new(-4.8, -4.6, -4.7),
            ..Default::default()
        });
        grid.insert_point(Point {
            position: Vector3::new(-4.5, -4.4, -4.6),
            ..Default::default()
        });

        let cells = grid.cells().collect::<Vec<_>>();
        assert_eq!(cells.len(), 1);
        assert_eq!(*cells[0], 0);

        let points = grid.points().collect::<Vec<_>>();
        assert_eq!(points.len(), 1);

        assert_eq!(points[0].position.x, -4.5);
        assert_eq!(points[0].position.y, -4.4);
        assert_eq!(points[0].position.z, -4.6);

        Ok(())
    }
}
