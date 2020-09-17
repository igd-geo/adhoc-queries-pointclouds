extern crate serde;

mod collect_points;
mod las;
mod math;

use clap::{App, Arg};
use memmap::MmapOptions;
use nalgebra::Vector3;
use rayon::prelude::*;
use std::convert::TryInto;
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::time::{Duration, Instant};

/**
 * Returns all input files from the given file path or directory path. Directories are not traversed recursively!
 */
fn get_all_input_files(input_path: &Path) -> Result<Vec<PathBuf>, String> {
    if !input_path.exists() {
        return Err(format!(
            "Input path {} does not exist!",
            input_path.display()
        ));
    }

    if input_path.is_file() {
        return Ok(vec![input_path.into()]);
    }

    if input_path.is_dir() {
        match fs::read_dir(input_path) {
            Ok(dir_iter) => {
                let files: Result<Vec<_>, _> = dir_iter
                    .map(|dir_entry| dir_entry.map(|entry| entry.path()))
                    .collect();
                return files.map_err(|e| format!("{}", e));
            }
            Err(why) => return Err(format!("{}", why)),
        };
    }

    Err(format!(
        "Input path {} is neither file nor directory!",
        input_path.display()
    ))
}

fn parse_aabb(aabb_str: &str) -> Result<math::AABB<f64>, String> {
    let components: Vec<&str> = aabb_str.split(";").collect();
    if components.len() != 6 {
        return Err(format!("Could not parse AABB from string \"{}\"", aabb_str));
    }

    let components_as_floats: Vec<_> = match components
        .iter()
        .map(|s| s.parse::<f64>())
        .collect::<Result<Vec<_>, _>>()
    {
        Ok(v) => v,
        Err(why) => {
            return Err(format!(
                "Could not parse AABB from string \"{}\": {}",
                aabb_str, why
            ))
        }
    };

    Ok(math::AABB::new(
        Vector3::new(
            components_as_floats[0],
            components_as_floats[1],
            components_as_floats[2],
        ),
        Vector3::new(
            components_as_floats[3],
            components_as_floats[4],
            components_as_floats[5],
        ),
    ))
}

fn search_single_file(
    path: &Path,
    query_bounds: &math::AABB<f64>,
    result_collector: &mut dyn collect_points::ResultCollector,
) -> Result<(), String> {
    let file = match fs::File::open(path) {
        Ok(f) => f,
        Err(why) => return Err(format!("Could not open file {}: {}", path.display(), why)),
    };
    let mmap = unsafe {
        match MmapOptions::new().map(&file) {
            Ok(m) => m,
            Err(why) => return Err(format!("mmap failed on file {}: {}", path.display(), why)),
        }
    };

    let header = las::try_parse_las_header(&mmap[0..227])?;
    let file_bounds = header.bounds();

    if !file_bounds.intersects(query_bounds) {
        return Ok(());
    }

    // Convert bounds of query area into integer coordinates in local space of file. This makes intersection
    // checks very fast because they can be done on integer values
    let query_bounds_local = math::AABB::<i64>::new(
        Vector3::<i64>::new(
            ((query_bounds.min.x - header.x_offset) / header.x_scale) as i64,
            ((query_bounds.min.y - header.y_offset) / header.x_scale) as i64,
            ((query_bounds.min.z - header.z_offset) / header.x_scale) as i64,
        ),
        Vector3::<i64>::new(
            ((query_bounds.max.x - header.x_offset) / header.x_scale) as i64,
            ((query_bounds.max.y - header.y_offset) / header.x_scale) as i64,
            ((query_bounds.max.z - header.z_offset) / header.x_scale) as i64,
        ),
    );

    let points_blob_start: usize = header.offset_to_point_data.try_into().unwrap();
    let points_blob_size: usize = (header.num_point_records as u64
        * header.point_record_length as u64)
        .try_into()
        .unwrap();
    let points_blob = &mmap[points_blob_start..points_blob_start + points_blob_size];

    for point_idx in 0..header.num_point_records {
        let point_offset: usize = (point_idx as u64 * header.point_record_length as u64)
            .try_into()
            .unwrap();

        let point_x = i32::from_le_bytes([
            points_blob[point_offset],
            points_blob[point_offset + 1],
            points_blob[point_offset + 2],
            points_blob[point_offset + 3],
        ]) as i64;
        if point_x < query_bounds_local.min.x || point_x > query_bounds_local.max.x {
            continue;
        }

        let point_y = i32::from_le_bytes([
            points_blob[point_offset + 4],
            points_blob[point_offset + 5],
            points_blob[point_offset + 6],
            points_blob[point_offset + 7],
        ]) as i64;
        if point_y < query_bounds_local.min.y || point_y > query_bounds_local.max.y {
            continue;
        }

        let point_z = i32::from_le_bytes([
            points_blob[point_offset + 8],
            points_blob[point_offset + 9],
            points_blob[point_offset + 10],
            points_blob[point_offset + 11],
        ]) as i64;
        if point_z < query_bounds_local.min.z || point_z > query_bounds_local.max.z {
            continue;
        }

        // let point_in_global_space = Vector3::<f64>::new(
        //     header.x_offset + (point_x as f64 * header.x_scale),
        //     header.y_offset + (point_y as f64 * header.y_scale),
        //     header.z_offset + (point_z as f64 * header.z_scale),
        // );
        result_collector.collect_one(
            &points_blob
                [points_blob_start..points_blob_start + header.point_record_length as usize],
        );
    }

    Ok(())
}

fn run_search_sequential(
    files: &Vec<PathBuf>,
    query_bounds: &math::AABB<f64>,
) -> Result<(), String> {
    let mut collector = collect_points::BufferCollector::new();

    for file in files.iter() {
        search_single_file(&file, query_bounds, &mut collector)?;
    }

    println!(
        "Found {} matching bytes ({} points)",
        collector.buffer().len(),
        collector.buffer().len() / 28
    );

    Ok(())
}

fn run_search_parallel(files: &Vec<PathBuf>, query_bounds: &math::AABB<f64>) {}

fn main() -> Result<(), String> {
    let t_start = Instant::now();

    let matches = App::new("LAS I/O experiments")
                          .version("0.1")
                          .author("Pascal Bormann <pascal.bormann@igd.fraunhofer.de>")
                          .about("LAS I/O experiments")
                          .arg(Arg::with_name("INPUT")
                               .short("i")
                               .long("input")
                               .value_name("FILE")
                               .help("Input point cloud")
                               .takes_value(true)
                            .required(true))
                          .arg(Arg::with_name("BOUNDS")
                                .long("bounds")
                               .help("Bounding box to search points in. Specify this as string \"minX,minY,minZ,maxX,maxY,maxZ\" in the target SRS of the input dataset")
                               .takes_value(true)
                               .required(true)
                            )
                          .get_matches();

    let input_dir = Path::new(matches.value_of("INPUT").unwrap());
    let input_files = get_all_input_files(input_dir)?;

    let total_file_size: u64 = input_files
        .iter()
        .map(|f| f.metadata().unwrap().len())
        .sum();
    let total_file_size_mib = total_file_size as f64 / 1048576.0;

    let aabb = parse_aabb(matches.value_of("BOUNDS").unwrap())?;

    run_search_sequential(&input_files, &aabb)?;

    let elapsed_seconds = t_start.elapsed().as_secs_f64();
    let throughput = total_file_size as f64 / elapsed_seconds;
    let throughput_mibs = throughput / 1048576.0;

    println!(
        "Searched {:.2} MiB in {:.2}s (throughput: {:.2}MiB/s",
        total_file_size_mib, elapsed_seconds, throughput_mibs
    );

    Ok(())
}
