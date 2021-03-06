use std::{
    fs::{read_dir, File},
    io::BufReader,
    path::{Path, PathBuf},
};

use anyhow::{anyhow, Context, Result};
use clap::{value_t, App, Arg};
use pasture_io::base::IOFactory;
use readers::{LASTReader, LAZERSource};

fn get_all_input_files<P: AsRef<Path>>(path: P) -> Result<Vec<PathBuf>> {
    let input_path = path.as_ref();
    if !input_path.exists() {
        return Err(anyhow!(
            "Input path {} does not exist!",
            input_path.display()
        ));
    }

    if let Ok(linked_path) = input_path.read_link() {
        return get_all_input_files(&linked_path);
    }

    if input_path.is_file() {
        return Ok(vec![input_path.into()]);
    }

    if input_path.is_dir() {
        match read_dir(input_path) {
            Ok(dir_iter) => {
                let files: Result<Vec<_>, _> = dir_iter
                    .map(|dir_entry| dir_entry.map(|entry| entry.path()))
                    .filter(|dir_entry| {
                        dir_entry
                            .as_ref()
                            .map(|entry| entry.is_file())
                            .unwrap_or(false)
                    })
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

fn get_reader_factory() -> IOFactory {
    let mut reader_factory = IOFactory::default();
    reader_factory.register_reader_for_extension("last", |path| {
        let reader = LASTReader::from(BufReader::new(File::open(path)?))?;
        Ok(Box::new(reader))
    });
    reader_factory.register_reader_for_extension("lazer", |path| {
        let reader = LAZERSource::from(BufReader::new(File::open(path)?))?;
        Ok(Box::new(reader))
    });
    reader_factory
}

fn count_points_in_dataset<P: AsRef<Path>>(path_to_dataset: P) -> Result<usize> {
    let factory = get_reader_factory();
    let files = get_all_input_files(path_to_dataset)?;

    let points_per_file = files
        .iter()
        .map(|file| -> Result<usize> {
            let reader = factory.make_reader(&file)?;
            Ok(reader.get_metadata().number_of_points().unwrap())
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(points_per_file.into_iter().sum())
}

fn main() -> Result<()> {
    let matches = App::new("Point cloud queries - count points")
                          .version("0.1")
                          .author("Pascal Bormann <pascal.bormann@igd.fraunhofer.de>")
                          .about("Counts the number of points in a given point cloud")
                          .arg(Arg::with_name("INPUT")
                               .short("i")
                               .long("input")
                               .value_name("FILE")
                               .help("Input point cloud. Can be a single file or a directory. Directories are scanned recursively for all point cloud files with supported formats LAS, LAZ, LAST, LAZT, LASER, LAZER.")
                               .takes_value(true)
                            .required(true))
                          .get_matches();

    let in_path = value_t!(matches, "INPUT", String).context("Argument 'INPUT' not found")?;
    let count = count_points_in_dataset(&in_path).context("Error while counting points")?;

    println!("{}", count);

    Ok(())
}
