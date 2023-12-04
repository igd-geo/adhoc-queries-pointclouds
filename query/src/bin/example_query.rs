use std::{
    path::{Path, PathBuf},
    time::Duration,
};

use anyhow::{bail, Context, Result};
use geo::{line_string, MultiPolygon, Polygon};
use pasture_core::nalgebra::Vector3;
use query::{
    index::{
        AlwaysRefinementStrategy, AtomicExpression, Classification, CompareExpression, DiscreteLod,
        Geometry, GpsTime, NumberOfReturns, Position, ProgressiveIndex, QueryExpression,
        ReturnNumber, Value,
    },
    io::CountOutput,
};
use shapefile::{Shape, ShapeReader};
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

fn get_query() -> QueryExpression {
    let _doc_aabb_small = QueryExpression::Atomic(AtomicExpression::Within(
        Value::Position(Position(Vector3::new(390000.0, 130000.0, 0.0)))
            ..Value::Position(Position(Vector3::new(390500.0, 140000.0, 200.0))),
    ));
    let _doc_polygon_small = QueryExpression::Atomic(AtomicExpression::Intersects(
        Geometry::Polygon(Polygon::new(
            line_string![(x: 390000.0, y: 130000.0), (x: 390500.0, y: 130000.0), (x: 390500.0, y: 140000.0), (x: 390000.0, y: 140000.0), (x: 390000.0, y: 130000.0)],
            vec![],
        )),
    ));
    let _doc_aabb_large = QueryExpression::Atomic(AtomicExpression::Within(
        Value::Position(Position(Vector3::new(390000.0, 130000.0, 0.0)))
            ..Value::Position(Position(Vector3::new(400000.0, 140000.0, 200.0))),
    ));
    let _doc_aabb_complete = QueryExpression::Atomic(AtomicExpression::Within(
        Value::Position(Position(Vector3::new(389400.0, 124200.0, -94.88)))
            ..Value::Position(Position(Vector3::new(408599.99, 148199.99, 760.03))),
    ));

    let _all_buildings = QueryExpression::Atomic(AtomicExpression::Compare((
        CompareExpression::Equals,
        Value::Classification(Classification(6)),
    )));

    let at_least_three_returns = QueryExpression::Atomic(AtomicExpression::Compare((
        CompareExpression::GreaterThan,
        Value::NumberOfReturns(NumberOfReturns(2)),
    )));

    let _third_or_higher_return = QueryExpression::Atomic(AtomicExpression::Compare((
        CompareExpression::GreaterThan,
        Value::ReturnNumber(ReturnNumber(2)),
    )));

    // Everything with at least three returns, but give us the first return. This should be the canopies
    let _maybe_vegetation = QueryExpression::And(
        Box::new(at_least_three_returns.clone()),
        Box::new(QueryExpression::Atomic(AtomicExpression::Compare((
            CompareExpression::Equals,
            Value::ReturnNumber(ReturnNumber(1)),
        )))),
    );

    let _doc_time_range_5percent = QueryExpression::Atomic(AtomicExpression::Within(
        Value::GpsTime(GpsTime(207011500.0))..Value::GpsTime(GpsTime(207012000.0)),
    ));

    let lod0 = QueryExpression::Atomic(AtomicExpression::Compare((
        CompareExpression::Equals,
        Value::LOD(DiscreteLod(0)),
    )));
    let _lod2 = QueryExpression::Atomic(AtomicExpression::Compare((
        CompareExpression::Equals,
        Value::LOD(DiscreteLod(2)),
    )));
    let _vegetation_classes = QueryExpression::Atomic(AtomicExpression::Within(
        Value::Classification(Classification(3))..Value::Classification(Classification(6)),
    ));

    // _doc_aabb_small
    lod0
    // QueryExpression::Or(
    //     Box::new(doc_time_range_5percent.clone()),
    //     Box::new(_doc_aabb_large.clone()),
    // )
}

fn main() -> Result<()> {
    pretty_env_logger::init();
    let _client = tracy_client::Client::start();
    std::thread::sleep(Duration::from_secs(2));

    let paths = get_point_files_in_path(Path::new(
        "/Users/pbormann/data/projects/progressive_indexing/experiment_data/ahn4s/las",
    ));

    let shapefile_path = Path::new(
        "/Users/pbormann/data/projects/progressive_indexing/queries/doc/doc_polygon_small_with_holes_1.shp",
    );
    // Very crude, only support reading first shape in shapefile, assuming that it is a polygon
    let shape_reader = ShapeReader::from_path(shapefile_path).context("Can't read shapefile")?;
    let shapes = shape_reader
        .read()
        .context("Failed to read shapes from shapefile")?;
    if shapes.is_empty() {
        bail!("No shapes found in shapefile");
    }
    let first_shape = &shapes[0];
    let _shapefile_query = match first_shape {
        Shape::Polygon(poly) => {
            let geo_polygon: MultiPolygon = poly.clone().into();
            let first_polygon = geo_polygon.0[0].clone();
            QueryExpression::Atomic(AtomicExpression::Intersects(Geometry::Polygon(
                first_polygon,
            )))
        }
        _ => bail!("Unsupported shape type"),
    };

    // let query_all_buildings_within_bounds = QueryExpression::And(
    //     Box::new(query_doc_aabb_l.clone()),
    //     Box::new(query_doc_all_buildings.clone()),
    // );

    // let query_all_buildings_within_shape = QueryExpression::And(
    //     Box::new(query_doc_all_buildings.clone()),
    //     Box::new(shapefile_query.clone()),
    // );

    let mut progressive_index = ProgressiveIndex::new();
    let dataset_id = progressive_index.add_dataset(paths.as_slice())?;

    let stats = progressive_index.dataset_stats(dataset_id);
    eprintln!("Stats:\n{stats}");

    // let output = LASOutput::new("ahn4s_vegetation.las", stats.point_layout())?;
    // let output = StdoutOutput::new(
    //     point_layout_from_las_point_format(&Format::new(6)?, true)?,
    //     // [
    //     //     POSITION_3D.with_custom_datatype(PointAttributeDataType::Vec3i32),
    //     //     CLASSIFICATION,
    //     // ]
    //     // .into_iter()
    //     // .collect(),
    //     true,
    // );
    let output = CountOutput::default();
    // let output = NullOutput::default();

    let query = get_query();
    eprintln!("Query: {query}");
    // for run_id in 0..5 {
    let _stats = progressive_index.query(
        dataset_id,
        query.clone(),
        &AlwaysRefinementStrategy,
        &output,
    )?;
    //     eprintln!("Run {run_id}:\n{stats}");
    // }

    Ok(())
}
