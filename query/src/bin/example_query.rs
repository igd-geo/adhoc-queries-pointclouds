use std::path::{Path, PathBuf};

use anyhow::Result;
use pasture_core::nalgebra::Vector3;
use query::{
    collect_points::BufferCollector,
    index::{Position, ProgressiveIndex, QueryExpression, Value},
};
use walkdir::WalkDir;

fn get_point_files_in_path(dir: &Path) -> Vec<PathBuf> {
    WalkDir::new(dir)
        .into_iter()
        .filter_map(|p| {
            p.ok().and_then(|p| {
                let extension = p.path().extension()?.to_str()?;
                match extension {
                    "las" | "laz" | "last" | "lazer" => Some(p.path().to_owned()),
                    _ => None,
                }
            })
        })
        .collect::<Vec<_>>()
}

fn main() -> Result<()> {
    pretty_env_logger::init();

    let paths = get_point_files_in_path(Path::new(
        "/Users/pbormann/data/projects/progressive_indexing/experiment_data/doc/las",
    ));
    let query = QueryExpression::Within(
        Value::Position(Position(Vector3::new(390000.0, 130000.0, 0.0)))
            ..Value::Position(Position(Vector3::new(390500.0, 140000.0, 200.0))),
    );

    let mut progressive_index = ProgressiveIndex::new();
    let dataset_id = progressive_index.add_dataset(paths)?;

    let mut result_collector = BufferCollector::new();
    progressive_index.query(dataset_id, query, &mut result_collector)?;

    Ok(())
}
