extern crate serde;

mod collect_points;
mod las;
mod math;
mod search;

use anyhow::{anyhow, Context, Result};
use clap::{value_t, App, Arg};
use memmap::MmapOptions;
use nalgebra::Vector3;
use rayon::prelude::*;
use std::convert::TryInto;
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use crate::search::{SearchImplementation, Searcher};

/**
 * Returns all input files from the given file path or directory path. Directories are not traversed recursively!
 */
fn get_all_input_files(input_path: &Path) -> Result<Vec<PathBuf>> {
    if !input_path.exists() {
        return Err(anyhow!(
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
                return files.map_err(|e| e.into());
            }
            Err(why) => return Err(anyhow!("{}", why)),
        };
    }

    Err(anyhow!(
        "Input path {} is neither file nor directory!",
        input_path.display()
    ))
}

fn parse_aabb(aabb_str: &str) -> Result<math::AABB<f64>> {
    let components: Vec<&str> = aabb_str.split(";").collect();
    if components.len() != 6 {
        return Err(anyhow!("Could not parse AABB from string \"{}\"", aabb_str));
    }

    let components_as_floats: Vec<_> = match components
        .iter()
        .map(|s| s.parse::<f64>())
        .collect::<Result<Vec<_>, _>>()
    {
        Ok(v) => v,
        Err(why) => {
            return Err(anyhow!(
                "Could not parse AABB from string \"{}\": {}",
                aabb_str,
                why
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

fn run_search_sequential(
    files: &Vec<PathBuf>,
    searcher: &dyn Searcher,
    search_impl: SearchImplementation,
) -> Result<()> {
    let mut collector = collect_points::BufferCollector::new();

    for file in files.iter() {
        searcher.search_file(&file, &search_impl, &mut collector)?;
    }

    println!("Found {} matching points", collector.buffer().len(),);

    Ok(())
}

fn run_search_parallel(
    files: &Vec<PathBuf>,
    searcher: &dyn Searcher,
    search_impl: SearchImplementation,
) -> Result<()> {
    todo!("not implemented")
}

fn main() -> Result<()> {
    let t_start = Instant::now();

    let matches = App::new("I/O experiments")
                          .version("0.1")
                          .author("Pascal Bormann <pascal.bormann@igd.fraunhofer.de>")
                          .about("LAS I/O experiments")
                          .arg(Arg::with_name("INPUT")
                               .short("i")
                               .long("input")
                               .value_name("FILE")
                               .help("Input point cloud. Can be a single file or a directory. Directories are scanned recursively for all point cloud files with supported formats LAS, LAZ, LAST, LAZT, LASER, LAZER.")
                               .takes_value(true)
                            .required(true))
                          .arg(Arg::with_name("BOUNDS")
                                .long("bounds")
                               .help("Bounding box to search points in. Specify this as string \"minX,minY,minZ,maxX,maxY,maxZ\" in the target SRS of the input dataset")
                               .takes_value(true)
                            )
                          .arg(Arg::with_name("CLASS")
                                .long("class")
                               .help("Object class to search points for. This is a single 8-bit unsigned integer value corresponding to the valid classes in the LAS file specification")
                               .takes_value(true)
                            )
                          .arg(Arg::with_name("PARALLEL").long("parallel").short("p").help("Run search in parallel on multiple threads"))
                          .arg(Arg::with_name("OPTIMIZED").long("optimized").short("o").help("Run search with optimized implementation"))
                          .get_matches();

    let input_dir = Path::new(matches.value_of("INPUT").unwrap());
    let input_files = get_all_input_files(input_dir)?;

    let total_file_size: u64 = input_files
        .iter()
        .map(|f| f.metadata().unwrap().len())
        .sum();
    let total_file_size_mib = total_file_size as f64 / 1048576.0;
    let run_in_parallel = matches.is_present("PARALLEL");
    let run_optimized = matches.is_present("OPTIMIZED");

    let maybe_bounds = matches.value_of("BOUNDS").map(parse_aabb);
    let maybe_class = matches.value_of("CLASS").map(str::parse::<u8>);
    if maybe_bounds.is_some() && maybe_class.is_some() {
        return Err(anyhow!("Specifying BOUNDS and CLASS at the same time is invalid! Specify either BOUNDS or CLASS argument!"));
    }

    if maybe_bounds.is_none() && maybe_class.is_none() {
        return Err(anyhow!("Found neither BOUNDS nor CLASS argument but exactly one of these arguments is required!"));
    }

    let searcher: Box<dyn search::Searcher> = if maybe_bounds.is_some() {
        let bounds = maybe_bounds
            .unwrap()
            .context("Could not parse argument BOUNDS")?;
        Box::new(search::BoundsSearcher::new(bounds))
    } else {
        let class = maybe_class
            .unwrap()
            .context("Could not parse argument CLASS")?;
        Box::new(search::ClassSearcher::new(class))
    };

    let search_impl = if run_optimized {
        SearchImplementation::Optimized
    } else {
        SearchImplementation::Regular
    };

    if run_in_parallel {
        run_search_parallel(&input_files, searcher.as_ref(), search_impl)?;
    } else {
        run_search_sequential(&input_files, searcher.as_ref(), search_impl)?;
    }

    let elapsed_seconds = t_start.elapsed().as_secs_f64();
    let throughput = total_file_size as f64 / elapsed_seconds;
    let throughput_mibs = throughput / 1048576.0;

    println!(
        "Searched {:.2} MiB in {:.2}s (throughput: {:.2}MiB/s)",
        total_file_size_mib, elapsed_seconds, throughput_mibs
    );

    Ok(())
}
