extern crate serde;

mod collect_points;
mod dump_points;
mod grid_sampling;
mod las;
mod search;

use crate::collect_points::{ResultCollector};
use anyhow::{anyhow, Result};
use clap::{App, Arg};
use pasture_core::{math::AABB, nalgebra::Point3};
use pasture_io::base::{IOFactory, PointReadAndSeek};
use readers::{LASTReader, LAZERSource};
use rayon::prelude::*;
use std::{fs::File};
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::time::{Instant};

use crate::search::{SearchImplementation, Searcher};

type CollectorFactoryFn = Box<dyn Fn() -> Result<Box<dyn collect_points::ResultCollector + Send>> + Sync>;

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

fn parse_aabb(aabb_str: &str) -> Result<AABB<f64>> {
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

    Ok(AABB::from_min_max(
        Point3::new(
            components_as_floats[0],
            components_as_floats[1],
            components_as_floats[2],
        ),
        Point3::new(
            components_as_floats[3],
            components_as_floats[4],
            components_as_floats[5],
        ),
    ))
}

fn get_total_bounds(files : &[PathBuf]) -> Result<AABB<f64>> {
    let mut factory = IOFactory::default();
    factory.register_reader_for_extension("last", |path| -> Result<Box<dyn PointReadAndSeek>> {
        let file = File::open(path)?;
        let reader = LASTReader::from(file)?;
        let boxed = Box::new(reader);
        Ok(boxed)
    });
    factory.register_reader_for_extension("lazer", |path| -> Result<Box<dyn PointReadAndSeek>> {
        let file = File::open(path)?;
        let reader = LAZERSource::from(file)?;
        let boxed = Box::new(reader);
        Ok(boxed)
    });

    let file_bounds = files.iter().map(|f| -> Result<AABB<f64>> {
        let source = factory.make_reader(f)?;
        Ok(source.get_metadata().bounds().unwrap().into())
    }).collect::<Result<Vec<_>, _>>()?;

    let mut total_bounds = AABB::from_min_max_unchecked(Point3::new(f64::MAX, f64::MAX, f64::MAX), Point3::new(f64::MIN, f64::MIN, f64::MIN));
    for other_bounds in file_bounds.iter() {
        total_bounds = AABB::union(&total_bounds, other_bounds);
    }

    Ok(total_bounds)
}

fn run_search_sequential(
    files: &Vec<PathBuf>,
    searcher: &dyn Searcher,
    search_impl: SearchImplementation,
    collector_factory_fn: CollectorFactoryFn,
    point_dumper : &mut dyn dump_points::PointDumper,
) -> Result<()> {
    let mut collector = collector_factory_fn()?;

    for file in files.iter() {
        searcher.search_file(&file, &search_impl, collector.as_mut())?;
    }

    if let Some(points_ref) = collector.points_ref() {
        point_dumper.dump_points(points_ref)?;
    } else if let Some(points) = collector.points() {
        point_dumper.dump_points(points.as_slice())?;
    } else {
        println!("Found {} matching points", collector.point_count());
    }

    Ok(())
}

fn run_search_parallel(
    files: &Vec<PathBuf>,
    searcher: &(dyn Searcher + Sync),
    search_impl: SearchImplementation,
    collector_factory_fn: CollectorFactoryFn,
    point_dumper : &mut dyn dump_points::PointDumper,
) -> Result<()> {
    let results = files
        .par_iter()
        .map(|file| -> Result<Box<dyn ResultCollector + Send>> {
            let mut collector = collector_factory_fn()?;

            searcher.search_file(&file, &search_impl, collector.as_mut())?;
            Ok(collector)
        })
        .collect::<Result<Vec<_>>>();

    let separate_buffers = results?;
    let mut num_matches = None;
    for collector in separate_buffers.iter() {
        if let Some(points_ref) = collector.points_ref() {
            point_dumper.dump_points(points_ref)?;
        } else if let Some(points) = collector.points() {
            point_dumper.dump_points(points.as_slice())?;
        } else {
            match num_matches {
                Some(cur_matches) => num_matches = Some(cur_matches + collector.point_count()),
                None => num_matches = Some(collector.point_count()),
            }
        }
    }

    if let Some(matches) = num_matches {
        println!("Found {} matching points", matches);
    }

    Ok(())
}

fn is_valid_file(file: &Path) -> bool {
    file.extension().map(|ex| {
        ex == "las" || ex == "laz" || ex == "last" || ex == "lazer"
    }).unwrap_or(false)
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
                               .help("Bounding box to search points in. Specify this as string \"minX;minY;minZ;maxX;maxY;maxZ\" in the target SRS of the input dataset")
                               .takes_value(true)
                               .allow_hyphen_values(true)
                            )
                          .arg(Arg::with_name("CLASS")
                                .long("class")
                               .help("Object class to search points for. This is a single 8-bit unsigned integer value corresponding to the valid classes in the LAS file specification")
                               .takes_value(true)
                            ) 
                          .arg(Arg::with_name("OUTPUT").long("output").short("o").help("Path to output directory for storing found points in. If this parameter is omitted, only the number of matching points is reported in the command line").takes_value(true))
                          .arg(Arg::with_name("DENSITY").long("density").help("Maximum density of the resulting dataset. A density of 1 is equal to a minimum spacing of 1 meter between points.").takes_value(true))
                          .arg(Arg::with_name("PARALLEL").long("parallel").help("Run search in parallel on multiple threads"))
                          .arg(Arg::with_name("OPTIMIZED").long("optimized").help("Run search with optimized implementation"))
                          .get_matches();

    let input_dir = Path::new(matches.value_of("INPUT").unwrap());
    let input_files = get_all_input_files(input_dir)?.into_iter().filter(|f| is_valid_file(&f)).collect::<Vec<_>>();

    let maybe_output_dir = matches.value_of("OUTPUT").map(|p| Path::new(p));

    let total_file_size: u64 = input_files
        .iter()
        .map(|f| f.metadata().unwrap().len())
        .sum();
    let total_file_size_mib = total_file_size as f64 / 1048576.0;
    let run_in_parallel = matches.is_present("PARALLEL");
    let run_optimized = matches.is_present("OPTIMIZED");

    let maybe_bounds = matches.value_of("BOUNDS").map(parse_aabb).map(|res| res.expect("Could not prase argument BOUNDS"));
    let maybe_class = matches.value_of("CLASS").map(str::parse::<u8>).map(|res| res.expect("Could not prase argument CLASS"));
    let maybe_density = matches.value_of("DENSITY").map(str::parse::<f64>).map(|res| res.expect("Could not prase argument DENSITY"));
    if maybe_bounds.is_some() && maybe_class.is_some() {
        return Err(anyhow!("Specifying BOUNDS and CLASS at the same time is invalid! Specify either BOUNDS or CLASS argument!"));
    }

    if maybe_bounds.is_none() && maybe_class.is_none() {
        return Err(anyhow!("Found neither BOUNDS nor CLASS argument but exactly one of these arguments is required!"));
    }

    let searcher: Box<dyn search::Searcher + Sync> = if let Some(ref bounds) = maybe_bounds {
        Box::new(search::BoundsSearcher::new(bounds.clone()))
    } else {
        let class = maybe_class.unwrap();
        Box::new(search::ClassSearcher::new(class))
    };

    let collector_factory : CollectorFactoryFn = if let Some(density) = maybe_density {
        let cell_size = density;
        let bounds : AABB<f64> = if let Some(ref bounds) = maybe_bounds {
            bounds.clone()
        } else {
            get_total_bounds(&input_files)?
        };

        Box::new(move || {
            let collector = collect_points::GridSampledCollector::new(bounds, cell_size)?;
            Ok(Box::new(collector))
        })
    } else if maybe_output_dir.is_some() {
        Box::new(|| {
            Ok(Box::new(collect_points::BufferCollector::new()))
        })
    } else {
        Box::new(|| {
            Ok(Box::new(collect_points::CountCollector::new()))
        })
    };

    let mut point_dumper : Box<dyn dump_points::PointDumper> = match maybe_output_dir {
        Some(output_dir) => {
            let dumper = dump_points::FileDumper::new(output_dir)?;
            Box::new(dumper)
        },
        None => Box::new(dump_points::IgnoreDumper::new())
    };

    let search_impl = if run_optimized {
        SearchImplementation::Optimized
    } else {
        SearchImplementation::Regular
    };

    println!("Searching {} files...", input_files.len());

    if run_in_parallel {
        run_search_parallel(
            &input_files,
            searcher.as_ref(),
            search_impl,
            collector_factory,
            point_dumper.as_mut(),
        )?;
    } else {
        run_search_sequential(
            &input_files,
            searcher.as_ref(),
            search_impl,
            collector_factory,
            point_dumper.as_mut(),
        )?;
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
