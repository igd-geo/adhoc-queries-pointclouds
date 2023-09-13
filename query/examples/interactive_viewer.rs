use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};

use domus_viewer::{
    event_loop::{get_main_loop, EventLoop},
    navigation::pan_tilt::PanTiltNavigation,
    renderers::{point_cloud::PointCloudRendererFactory, triangle::TriangleRenderFactory},
    window::WindowHandler,
};
use geo::MultiPolygon;
use pasture_core::nalgebra::{Point3, Vector3};
use query::{
    index::{AtomicExpression, Geometry, NoRefinementStrategy, ProgressiveIndex, QueryExpression},
    io::InMemoryOutput,
};
use shapefile::{Shape, ShapeReader};
use walkdir::WalkDir;

struct App;

impl WindowHandler for App {
    fn on_closed(&self) {
        get_main_loop().exit();
    }
}

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
    dotenv::dotenv().ok();
    pretty_env_logger::init();

    let paths = get_point_files_in_path(Path::new(
        "/Users/pbormann/data/projects/progressive_indexing/experiment_data/doc/las",
    ));

    let shapefile_path = Path::new(
        "/Users/pbormann/data/projects/progressive_indexing/queries/doc_polygon_large_1.shp",
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
    let shapefile_query = match first_shape {
        Shape::Polygon(poly) => {
            let geo_polygon: MultiPolygon = poly.clone().into();
            let first_polygon = geo_polygon.0[0].clone();
            QueryExpression::Atomic(AtomicExpression::Intersects(Geometry::Polygon(
                first_polygon,
            )))
        }
        _ => bail!("Unsupported shape type"),
    };

    let mut progressive_index = ProgressiveIndex::new();
    let dataset_id = progressive_index.add_dataset(paths.as_slice())?;

    let output = InMemoryOutput::default();
    let stats =
        progressive_index.query(dataset_id, shapefile_query, &NoRefinementStrategy, &output)?;

    let mut evloop = EventLoop::new();

    let window = evloop.create_window().unwrap();

    {
        let matching_points = output
            .into_single_buffer()
            .ok_or(anyhow!("No matching points"))?;

        // let mut legacy_pasture_buffer = InterleavedVecPointStorage::new();

        // window.add_system(PointCloudRendererFactory {
        //     points:
        // });
    }

    // let path = std::env::var_os("WALL_DETECT_INPUT_FILE");
    // let focus = if let Some(path) = path {
    //     let mut reader = pasture_io::las::LASReader::from_path(&path).unwrap();
    //     let nr_points = reader.remaining_points();
    //     let points = reader.read(nr_points).unwrap();
    //     let bounds = reader.header().bounds();
    //     let focus = Point3::from(
    //         Vector3::new(bounds.min.x, bounds.min.y, bounds.min.z)
    //             .lerp(&Vector3::new(bounds.max.x, bounds.max.y, bounds.max.z), 0.5),
    //     );
    //     window.add_system(PointCloudRendererFactory::new(
    //         points,
    //         Point3::new(bounds.min.x, bounds.min.y, bounds.min.z),
    //         Point3::new(bounds.max.x, bounds.max.y, bounds.max.z),
    //     ));
    //     focus
    // } else {
    //     Point3::new(0.0, 0.0, 0.0)
    // };

    window.set_handler(App);
    window.set_navigation(PanTiltNavigation {
        focus: Point3::new(0.0, 0.0, 0.0),
        radius: 50.0,
        ..Default::default()
    });
    window.add_system(TriangleRenderFactory);

    evloop.run_as_main_loop();

    Ok(())
}
