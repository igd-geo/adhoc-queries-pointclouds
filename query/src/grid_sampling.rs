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
        let point_position = point.position;
        let rx = (point_position.x - self.bounds.min().x) * self.dimensions.x as f64
            / (self.bounds.max().x - self.bounds.min().x);
        let ry = (point_position.y - self.bounds.min().y) * self.dimensions.y as f64
            / (self.bounds.max().y - self.bounds.min().y);
        let rz = (point_position.z - self.bounds.min().z) * self.dimensions.z as f64
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
                let current_point_position = current_point.position;
                let cur_dist_sqr = distance_squared(
                    &cell_center,
                    &Point3::new(
                        current_point_position.x,
                        current_point_position.y,
                        current_point_position.z,
                    ),
                );
                let new_dist_sqr = distance_squared(
                    &cell_center,
                    &Point3::new(point_position.x, point_position.y, point_position.z),
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
    use std::collections::HashSet;

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

        let first_point_position = points[0].position;
        assert_eq!(first_point_position.x, -4.5);
        assert_eq!(first_point_position.y, -4.6);
        assert_eq!(first_point_position.z, -4.7);

        Ok(())
    }

    #[test]
    fn test_sparse_grid_add_multiple_in_different_cells() -> Result<()> {
        let bounds = AABB::from_min_max(Point3::new(-5.0, -5.0, -5.0), Point3::new(5.0, 5.0, 5.0));
        let mut grid = SparseGrid::new(bounds, 1.0)?;

        let expected_points = vec![
            Point {
                position: Vector3::new(-4.5, -4.6, -4.7),
                ..Default::default()
            },
            Point {
                position: Vector3::new(-3.5, -4.5, -4.4),
                ..Default::default()
            },
        ];

        for point in &expected_points {
            grid.insert_point(*point);
        }

        let cells = grid.cells().copied().collect::<HashSet<_>>();
        let expected_cells: HashSet<u64> = [0, 1].iter().copied().collect::<HashSet<_>>();
        assert_eq!(expected_cells, cells);

        // Order of the points is random, but we can't collect them in a HashSet because they also can't implement Eq...
        let points = grid.points().copied().collect::<Vec<_>>();
        assert_eq!(points.len(), 2);

        let expected_points_different_order =
            expected_points.iter().copied().rev().collect::<Vec<_>>();

        let points_are_equal =
            points == expected_points || points == expected_points_different_order;

        assert!(
            points_are_equal,
            "Expected points {:?} (in any order) but got points {:?}",
            expected_points, points
        );

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

        let first_point_position = points[0].position;
        assert_eq!(first_point_position.x, -4.5);
        assert_eq!(first_point_position.y, -4.4);
        assert_eq!(first_point_position.z, -4.6);

        Ok(())
    }
}
