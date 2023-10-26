// mod reader;
// pub use self::reader::*;

mod las;
use std::path::Path;
use std::path::PathBuf;

use walkdir::WalkDir;

pub(crate) use self::las::*;

mod last;
pub(crate) use self::last::*;

mod laz;
pub(crate) use self::laz::*;

mod lazer;
pub(crate) use self::lazer::*;

mod input_layer;
pub use self::input_layer::*;

mod output_layer;
pub use self::output_layer::*;

mod stats;
pub use self::stats::*;

/// Returns all point cloud files in `dir`. A point cloud file is anything with the extensions "las",
/// "laz", "last", or "lazer"
pub fn get_point_files_in_path(dir: &Path) -> Vec<PathBuf> {
    WalkDir::new(dir)
        .into_iter()
        .filter_map(|p| {
            p.ok().and_then(|p| {
                let extension = p.path().extension()?.to_str()?;
                match extension.to_ascii_lowercase().as_str() {
                    "las" | "laz" | "last" | "lazer" => Some(p.path().to_owned()),
                    _ => None,
                }
            })
        })
        .collect::<Vec<_>>()
}
