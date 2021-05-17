use anyhow::{anyhow, Result};
use pasture_core::containers::InterleavedPointView;
use pasture_io::{base::PointWriter, las::LASWriter, las_rs::{Builder, point::Format}};
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use std::path::PathBuf;

use crate::points::Point;

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
            .join(format!("matching_points_{}.laz", self.file_index));
        self.file_index += 1;

        let mut builder = Builder::from((1, 4));
        builder.point_format = Format::new(2).unwrap();
        let header = builder.into_header().unwrap();

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
