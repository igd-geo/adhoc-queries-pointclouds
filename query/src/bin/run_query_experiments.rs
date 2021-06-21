use std::{
    fs::{read_dir, File},
    io::BufReader,
    path::{Path, PathBuf},
    process::{Command, ExitStatus},
    time::Instant,
};

use anyhow::{anyhow, bail, Context, Result};
use clap::{value_t, App, Arg};
use pasture_core::{
    math::AABB,
    nalgebra::{Point3, Vector3},
};
use pasture_io::base::IOFactory;
use readers::{LASTReader, LAZERSource};
use statrs::statistics::{Data, Median, Statistics};

fn reset_page_cache() -> Result<()> {
    let sync_output = Command::new("sync")
        .output()
        .context("Could not execute sync command")?;
    if !sync_output.status.success() {
        bail!("Sync command failed with exit code {}", sync_output.status);
    }

    let purge_output = Command::new("purge")
        .output()
        .context("Could not execute purge command")?;
    if !purge_output.status.success() {
        bail!(
            "Purge command failed with exit code {}",
            purge_output.status
        );
    }

    Ok(())
}

fn execute_aabb_query<P: AsRef<Path>>(
    path_to_dataset: P,
    bounds: &AABB<f64>,
    density: Option<f64>,
) -> Result<f64> {
    reset_page_cache()?;

    let bounds_str = format!(
        "{};{};{};{};{};{}",
        bounds.min().x,
        bounds.min().y,
        bounds.min().z,
        bounds.max().x,
        bounds.max().y,
        bounds.max().z
    );

    let mut args = vec![
        "-i".to_owned(),
        path_to_dataset.as_ref().display().to_string(),
        "--bounds".to_owned(),
        bounds_str,
        "--optimized".to_owned(),
        "--parallel".to_owned(),
    ];

    if let Some(actual_density) = density {
        args.push("--density".into());
        args.push(format!("{}", actual_density));
    }

    let t_start = Instant::now();
    let output = Command::new("./target/release/query")
        .args(args.as_slice())
        .output()
        .context("Could not run query executable")?;
    if output.status.success() {
        Ok(t_start.elapsed().as_secs_f64())
    } else {
        let stderr_output = String::from_utf8(output.stderr)?;
        eprintln!("{}", stderr_output);
        bail!(
            "Could not execute AABB query. Process exited with {}",
            output.status
        )
    }
}

fn execute_class_query<P: AsRef<Path>>(path_to_dataset: P, class: u8) -> Result<f64> {
    reset_page_cache()?;

    let mut args = vec![
        "-i".to_owned(),
        path_to_dataset.as_ref().display().to_string(),
        "--class".to_owned(),
        format!("{}", class),
        "--optimized".to_owned(),
        "--parallel".to_owned(),
    ];

    let t_start = Instant::now();
    let output = Command::new("./target/release/query")
        .args(args.as_slice())
        .output()
        .context("Could not run query executable")?;
    if output.status.success() {
        Ok(t_start.elapsed().as_secs_f64())
    } else {
        let stderr_output = String::from_utf8(output.stderr)?;
        eprintln!("{}", stderr_output);
        bail!(
            "Could not execute class query. Process exited with {}",
            output.status
        )
    }
}

fn run_aabb_experiments<P: AsRef<Path>>(in_path: P, num_runs: usize) -> Result<()> {
    let file_extensions = vec![
        "las", //"laz",
        "last",
        //"lazer"
    ];

    let aabb_navvis_s = AABB::from_min_max(Point3::new(0.0, 0.0, 0.0), Point3::new(2.0, 2.0, 2.0));
    let aabb_navvis_l =
        AABB::from_min_max(Point3::new(0.0, 0.0, 0.0), Point3::new(20.0, 20.0, 5.0));
    let aabb_navvis_xl = AABB::from_min_max(
        Point3::new(-23.108, -21.261, -10.029),
        Point3::new(28.588, 27.123, 5.959),
    );

    let aabb_doc_s = AABB::from_min_max(
        Point3::new(390000.0, 130000.0, 0.0),
        Point3::new(390500.0, 140000.0, 200.0),
    );
    let aabb_doc_l = AABB::from_min_max(
        Point3::new(390000.0, 130000.0, 0.0),
        Point3::new(400000.0, 140000.0, 200.0),
    );
    let aabb_doc_xl = AABB::from_min_max(
        Point3::new(389400.0, 124200.0, -94.88),
        Point3::new(406200.0, 148200.0, 760.03),
    );

    //S = ~35M matches
    let aabb_ca13_s = AABB::from_min_max(
        Point3::new(665000.0, 3910000.0, 0.0),
        Point3::new(705000.0, 3950000.0, 480.0),
    );
    //L = ~500M matches
    let aabb_ca13_l = AABB::from_min_max(
        Point3::new(665000.0, 3910000.0, 0.0),
        Point3::new(710000.0, 3950000.0, 480.0),
    );
    //XL = ~2.6B matches (all)
    let aabb_ca13_xl = AABB::from_min_max(
        Point3::new(643431.76, 3883547.565, -46194.145),
        Point3::new(736910.93, 3977026.735, 47285.025),
    );

    struct AABBExperimentInput {
        pub dataset_name: &'static str,
        pub bounds_name: &'static str,
        pub bounds: AABB<f64>,
        pub density: Option<f64>,
    }

    let aabb_experiment_inputs = vec![
        AABBExperimentInput {
            dataset_name: "navvis3",
            bounds_name: "s",
            bounds: aabb_navvis_s.clone(),
            density: None,
        },
        AABBExperimentInput {
            dataset_name: "navvis3",
            bounds_name: "s",
            bounds: aabb_navvis_s.clone(),
            density: Some(0.1),
        },
        AABBExperimentInput {
            dataset_name: "navvis3",
            bounds_name: "l",
            bounds: aabb_navvis_l.clone(),
            density: None,
        },
        AABBExperimentInput {
            dataset_name: "navvis3",
            bounds_name: "l",
            bounds: aabb_navvis_l.clone(),
            density: Some(0.1),
        },
        AABBExperimentInput {
            dataset_name: "navvis3",
            bounds_name: "xl",
            bounds: aabb_navvis_xl.clone(),
            density: None,
        },
        AABBExperimentInput {
            dataset_name: "navvis3",
            bounds_name: "xl",
            bounds: aabb_navvis_xl.clone(),
            density: Some(0.1),
        },
        //doc
        AABBExperimentInput {
            dataset_name: "doc",
            bounds_name: "s",
            bounds: aabb_doc_s.clone(),
            density: None,
        },
        AABBExperimentInput {
            dataset_name: "doc",
            bounds_name: "s",
            bounds: aabb_doc_s.clone(),
            density: Some(25.0),
        },
        AABBExperimentInput {
            dataset_name: "doc",
            bounds_name: "l",
            bounds: aabb_doc_l.clone(),
            density: None,
        },
        AABBExperimentInput {
            dataset_name: "doc",
            bounds_name: "l",
            bounds: aabb_doc_l.clone(),
            density: Some(25.0),
        },
        AABBExperimentInput {
            dataset_name: "doc",
            bounds_name: "xl",
            bounds: aabb_doc_xl.clone(),
            density: None,
        },
        AABBExperimentInput {
            dataset_name: "doc",
            bounds_name: "xl",
            bounds: aabb_doc_xl.clone(),
            density: Some(25.0),
        },
        //ca13
        AABBExperimentInput {
            dataset_name: "ca13",
            bounds_name: "s",
            bounds: aabb_ca13_s.clone(),
            density: None,
        },
        AABBExperimentInput {
            dataset_name: "ca13",
            bounds_name: "s",
            bounds: aabb_ca13_s.clone(),
            density: Some(100.0),
        },
        AABBExperimentInput {
            dataset_name: "ca13",
            bounds_name: "l",
            bounds: aabb_ca13_l.clone(),
            density: None,
        },
        AABBExperimentInput {
            dataset_name: "ca13",
            bounds_name: "l",
            bounds: aabb_ca13_l.clone(),
            density: Some(100.0),
        },
        AABBExperimentInput {
            dataset_name: "ca13",
            bounds_name: "xl",
            bounds: aabb_ca13_xl.clone(),
            density: None,
        },
        AABBExperimentInput {
            dataset_name: "ca13",
            bounds_name: "xl",
            bounds: aabb_ca13_xl.clone(),
            density: Some(100.0),
        },
    ];

    for data in aabb_experiment_inputs.iter() {
        for extension in file_extensions.iter() {
            eprintln!(
                "Experiment {}_{}_{}...",
                data.dataset_name, data.bounds_name, extension
            );
            let file_path = format!(
                "{}/{}/{}",
                in_path.as_ref().display(),
                data.dataset_name,
                extension
            );
            let runtimes = (0..num_runs)
                .map(|_| execute_aabb_query(&file_path, &data.bounds, data.density.clone()))
                .collect::<Result<Vec<_>>>()?;
            let runtimes_data = Data::new(runtimes.clone());
            println!(
                "{}_{}_{}_{};{};{};{}",
                data.dataset_name,
                data.bounds_name,
                if data.density.is_some() {
                    "lod"
                } else {
                    "full"
                },
                extension,
                (&runtimes).mean(),
                runtimes_data.median(),
                (&runtimes).std_dev()
            );
        }
    }

    Ok(())
}

fn run_class_experiments<P: AsRef<Path>>(in_path: P, num_runs: usize) -> Result<()> {
    let file_extensions = vec![
        "las", //"laz",
        "last",
        //"lazer",
    ];

    struct ClassExperimentInput {
        pub dataset_name: &'static str,
        pub class_name: &'static str,
        pub class: u8,
    }

    let experiment_data = vec![
        ClassExperimentInput {
            dataset_name: "doc",
            class_name: "building",
            class: 6,
        },
        ClassExperimentInput {
            dataset_name: "doc",
            class_name: "noclass",
            class: 19,
        },
        ClassExperimentInput {
            dataset_name: "ca13",
            class_name: "building",
            class: 6,
        },
        ClassExperimentInput {
            dataset_name: "ca13",
            class_name: "noclass",
            class: 19,
        },
    ];

    for data in experiment_data.iter() {
        for extension in file_extensions.iter() {
            eprintln!(
                "Experiment {}_{}_{}...",
                data.dataset_name, data.class_name, extension
            );
            let file_path = format!(
                "{}/{}/{}",
                in_path.as_ref().display(),
                data.dataset_name,
                extension
            );
            let runtimes = (0..num_runs)
                .map(|_| execute_class_query(&file_path, data.class))
                .collect::<Result<Vec<_>>>()?;
            let runtimes_data = Data::new(runtimes.clone());
            println!(
                "{}_{}_{};{};{};{}",
                data.dataset_name,
                data.class_name,
                extension,
                (&runtimes).mean(),
                runtimes_data.median(),
                (&runtimes).std_dev()
            );
        }
    }

    Ok(())
}

fn main() -> Result<()> {
    let matches = App::new("Point cloud queries - run experiments")
        .version("0.1")
        .author("Pascal Bormann <pascal.bormann@igd.fraunhofer.de>")
        .about("Runs the point cloud query experiments")
        .arg(
            Arg::with_name("INPUT")
                .short("i")
                .long("input")
                .value_name("DIRECTORY")
                .help("Root directory in which all datasets can be found")
                .takes_value(true)
                .required(true),
        )
        .get_matches();

    let in_path = value_t!(matches, "INPUT", String).context("Argument 'INPUT' not found")?;

    eprintln!("Running experiments... Output is: experiment_name;mean;median;stddev with runtimes in seconds");

    //run_aabb_experiments(&in_path, 5)?;
    run_class_experiments(&in_path, 5)?;

    Ok(())
}
