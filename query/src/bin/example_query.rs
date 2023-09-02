use std::path::{Path, PathBuf};

use anyhow::Result;
use pasture_core::nalgebra::Vector3;
use query::{
    index::{
        AtomicExpression, Classification, CompareExpression, NoRefinementStrategy, Position,
        ProgressiveIndex, QueryExpression, Value,
    },
    io::NullOutput,
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

    let query_doc_aabb_s = QueryExpression::Atomic(AtomicExpression::Within(
        Value::Position(Position(Vector3::new(390000.0, 130000.0, 0.0)))
            ..Value::Position(Position(Vector3::new(390500.0, 140000.0, 200.0))),
    ));
    let query_doc_aabb_l = QueryExpression::Atomic(AtomicExpression::Within(
        Value::Position(Position(Vector3::new(390000.0, 130000.0, 0.0)))
            ..Value::Position(Position(Vector3::new(400000.0, 140000.0, 200.0))),
    ));
    let query_doc_aabb_xl = QueryExpression::Atomic(AtomicExpression::Within(
        Value::Position(Position(Vector3::new(389400.0, 124200.0, -94.88)))
            ..Value::Position(Position(Vector3::new(406200.0, 148200.0, 760.03))),
    ));

    let query_doc_all_buildings = QueryExpression::Atomic(AtomicExpression::Compare((
        CompareExpression::Equals,
        Value::Classification(Classification(6)),
    )));

    let query_all_buildings_within_bounds = QueryExpression::And(
        Box::new(query_doc_aabb_l.clone()),
        Box::new(query_doc_all_buildings.clone()),
    );

    // let aabb_doc_l = AABB::from_min_max(
    //     Point3::new(390000.0, 130000.0, 0.0),
    //     Point3::new(400000.0, 140000.0, 200.0),
    // );
    // let aabb_doc_xl = AABB::from_min_max(
    //     Point3::new(389400.0, 124200.0, -94.88),
    //     Point3::new(406200.0, 148200.0, 760.03),
    // );

    let mut progressive_index = ProgressiveIndex::new();
    let dataset_id = progressive_index.add_dataset(paths.as_slice())?;

    let output = NullOutput::default();
    let stats = progressive_index.query(
        dataset_id,
        query_doc_all_buildings,
        &NoRefinementStrategy,
        &output,
    )?;

    eprintln!("{}", stats);

    Ok(())
}
