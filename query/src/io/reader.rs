use anyhow::{anyhow, Context, Result};
use std::{ops::Range, path::Path};

use pasture_core::nalgebra::Vector3;

use super::LASReader;

/// Trait for an abstract reader that reads points from a file
pub trait PointReader {
    fn read_positions(&mut self, point_indices: Range<usize>) -> Result<Vec<Vector3<f64>>>;
    /// Read positions in local space, for any file format that stores data in some local space (mostly LAS/LAZ). If
    /// the reader does not support reading data in local space, `None` is returned
    fn read_positions_in_local_space(
        &mut self,
        point_indices: Range<usize>,
    ) -> Option<Result<Vec<Vector3<i32>>>>;
    fn read_classifications(&mut self, point_indices: Range<usize>) -> Result<Vec<u8>>;
}

/// Opens a `PointReader` to the given file
pub(crate) fn open_reader<P: AsRef<Path>>(path: P) -> Result<Box<dyn PointReader>> {
    let extension = path
        .as_ref()
        .extension()
        .ok_or_else(|| anyhow!("Can't determine file extension of file"))?;
    match extension.to_string_lossy().as_ref() {
        "las" | "LAS" => {
            let reader = LASReader::open(path).context("Can't open reader to LAS file")?;
            Ok(Box::new(reader))
        }
        _ => Err(anyhow!("Unsupported file extension")),
    }
}
