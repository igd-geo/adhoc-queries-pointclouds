use anyhow::{anyhow, Result};
use pasture_core::{containers::InterleavedPointView, nalgebra::Vector3};
use pasture_io::{
    base::PointWriter,
    las::LASWriter,
    las_rs::{point::Format, Builder, Transform, Vector},
};
use std::path::Path;
use std::path::PathBuf;

use readers::Point;

pub trait PointDumper {
    fn dump_points(&mut self, points: &[Point]) -> Result<()>;
    fn num_dumped_points(&self) -> usize;
}

pub struct IgnoreDumper {
    dumped_count: usize,
}

impl IgnoreDumper {
    pub fn new() -> Self {
        IgnoreDumper { dumped_count: 0 }
    }
}

impl PointDumper for IgnoreDumper {
    fn dump_points(&mut self, points: &[Point]) -> Result<()> {
        self.dumped_count += points.len();
        Ok(())
    }

    fn num_dumped_points(&self) -> usize {
        self.dumped_count
    }
}

pub struct FileDumper {
    root_dir: PathBuf,
    file_index: usize,
    dumped_count: usize,
}

impl FileDumper {
    pub fn new(root_dir: impl AsRef<Path>) -> Result<Self> {
        let path = root_dir.as_ref();
        if !path.exists() {
            return Err(anyhow!("Path {} does not exist!", path.display()));
        }
        if !path.is_dir() {
            return Err(anyhow!("Path {} is no directory!", path.display()));
        }

        Ok(Self {
            root_dir: path.to_owned(),
            file_index: 0,
            dumped_count: 0,
        })
    }
}

impl PointDumper for FileDumper {
    fn dump_points(&mut self, points: &[Point]) -> Result<()> {
        if points.is_empty() {
            return Ok(());
        }
        let file_path = self
            .root_dir
            .join(format!("matching_points_{}.las", self.file_index));
        self.file_index += 1;

        // Calculate appropriate offset and scale parameters for the LAS header
        let pos_max = Vector3::new(f64::MAX, f64::MAX, f64::MAX);
        let pos_min = Vector3::new(f64::MIN, f64::MIN, f64::MIN);
        let min_max_position = points.iter().fold((pos_max, pos_min), |state, point| {
            let pos = point.position;
            (state.0.inf(&pos), state.1.sup(&pos))
        });
        let extent = min_max_position.1 - min_max_position.0;
        let max_extent = extent.max();
        let min_scale = max_extent / (i32::MAX as f64);
        // Round min_scale to higher next power of ten
        let mut scale = 10_f64.powf(min_scale.log10().ceil());
        // We clamp at millimeter precision to prevent underflows
        if scale < 0.001 {
            scale = 0.001;
        }

        let mut builder = Builder::from((1, 2));
        builder.point_format = Format::new(2).unwrap();
        builder.transforms = Vector {
            x: Transform {
                offset: min_max_position.0.x,
                scale,
            },
            y: Transform {
                offset: min_max_position.0.y,
                scale,
            },
            z: Transform {
                offset: min_max_position.0.z,
                scale,
            },
        };
        let header = builder.into_header().unwrap();

        println!("Writing {} points", points.len());
        let buffer = InterleavedPointView::from_slice(points);
        let mut writer = LASWriter::from_path_and_header(&file_path, header)?;
        writer.write(&buffer)?;

        self.dumped_count += points.len();

        Ok(())
    }

    fn num_dumped_points(&self) -> usize {
        self.dumped_count
    }
}
