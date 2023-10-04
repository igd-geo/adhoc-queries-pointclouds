use std::{
    borrow::Cow,
    cmp::min,
    collections::HashSet,
    fmt::Display,
    fs::File,
    io::Cursor,
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{Context, Result};
use clap::Parser;
use experiment_archiver::{Experiment, VariableTemplate};
use memmap::Mmap;
use pasture_core::containers::{BorrowedBuffer, InterleavedBuffer, OwningBuffer, VectorBuffer};
use pasture_io::{
    base::{GenericPointReader, PointReader},
    las::LASReader,
    las_rs::Read,
};
use rand::{thread_rng, Rng};

const VARIABLE_DATASET: VariableTemplate = VariableTemplate::new(
    Cow::Borrowed("Dataset"),
    Cow::Borrowed("The dataset used in the experiment"),
    Cow::Borrowed("none"),
);
const VARIABLE_ACCESS_PATTERN: VariableTemplate = VariableTemplate::new(Cow::Borrowed("Access Pattern"), Cow::Borrowed("The access pattern defines which points are actually read within a given range of points in a file"), Cow::Borrowed("text"));

const VARIABLE_TOOL: VariableTemplate = VariableTemplate::new(
    Cow::Borrowed("Tool / crate"),
    Cow::Borrowed("For which tool/crate is this measurement?"),
    Cow::Borrowed("text"),
);
const VARIABLE_RUNTIME: VariableTemplate = VariableTemplate::new(
    Cow::Borrowed("Runtime"),
    Cow::Borrowed("The runtime of the tool/crate"),
    Cow::Borrowed("ms"),
);
const VARIABLE_NR_POINTS_TOTAL : VariableTemplate = VariableTemplate::new(Cow::Borrowed("Number of points (total)"), Cow::Borrowed("The total number of points that were read. This is the (combined) size of all ranges of points that the readers had to touch in the dataset in order to read points with the current access pattern. The actual number of matching points might be less than this number"), Cow::Borrowed("number"));
const VARIABLE_NR_POINTS_MATCHING: VariableTemplate = VariableTemplate::new(
    Cow::Borrowed("Number of points (matching)"),
    Cow::Borrowed("The number of matching points based on the access pattern"),
    Cow::Borrowed("number"),
);
const VARIABLE_MACHINE: VariableTemplate = VariableTemplate::new(
    Cow::Borrowed("Machine"),
    Cow::Borrowed("The machine that the experiment is run on"),
    Cow::Borrowed("text"),
);

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    input_file: PathBuf,
    #[arg(short, long, default_value_t = 1_000_000)]
    count: usize,
}

#[derive(Debug, Clone, Copy)]
enum AccessPattern {
    Random,
}

impl AccessPattern {
    pub fn all() -> Vec<AccessPattern> {
        vec![AccessPattern::Random]
    }
}

impl Display for AccessPattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AccessPattern::Random => write!(f, "Random"),
        }
    }
}

fn gen_sample_from_access_pattern(pattern: AccessPattern, count: usize) -> Vec<bool> {
    match pattern {
        AccessPattern::Random => gen_random_point_sample(count),
    }
}

fn gen_random_point_sample(count: usize) -> Vec<bool> {
    let mut rng = thread_rng();
    (0..count).map(|_| rng.gen()).collect()
}

fn read_with_pasture(indices: &[bool], file: &Path, with_filtering: bool) -> Result<()> {
    let file = File::open(file).context("Can't open file")?;
    let file_mmap = unsafe { Mmap::map(&file).context("Can't mmap file")? };
    let mut reader = LASReader::from_read(Cursor::new(file_mmap), false, false)
        .context("Failed to create LASReader")?;

    let mut buffer =
        VectorBuffer::with_capacity(indices.len(), reader.get_default_point_layout().clone());
    reader
        .read_into(&mut buffer, indices.len())
        .context("Failed to read points with pasture")?;

    if with_filtering {
        let mut filtered_data =
            VectorBuffer::with_capacity(indices.len(), buffer.point_layout().clone());

        for (index, keep) in indices.iter().enumerate() {
            if !keep {
                continue;
            }

            // Safe because both buffers share the same point layout
            unsafe {
                filtered_data.push_points(buffer.get_point_ref(index));
            }
        }
    }

    // Make sure this doesn't get optimized away, even though we don't do anything with the data
    // TODO std::hint::black_box is unstable, but I don't want to go to nightly only for this...
    // unsafe { black_box(filtered_data) };

    Ok(())
}

fn read_with_las_crate(indices: &[bool], file: &Path) -> Result<()> {
    let file = File::open(file).context("Can't open file")?;
    let file_mmap = unsafe { Mmap::map(&file).context("Can't mmap file")? };
    let mut las_reader = pasture_io::las_rs::Reader::new(Cursor::new(file_mmap))
        .context("Can't create LAS reader")?;

    let points = las_reader
        .points()
        .take(indices.len())
        .enumerate()
        .filter(|(idx, _)| indices[*idx])
        .map(|(_, point)| point.unwrap())
        .collect::<Vec<_>>();
    // println!("{}", points.len());
    drop(points);

    Ok(())
}

// fn read_with_extractor(indices: &[bool], file: &Path) -> Result<()> {
//     let file = File::open(file).context("Can't open file")?;
//     let file_mmap = unsafe { Mmap::map(&file).context("Can't mmap file")? };
//     let file_data: &[u8] = &file_mmap;

//     let raw_header =
//         Header::read_from(&mut Cursor::new(file_data)).context("Can't read LAS header")?;

//     let extractor = LASExtractor;
//     let num_matches = indices.iter().filter(|b| **b).count();
//     let runtime_tracker = BlockQueryRuntimeTracker::default();
//     let result = extractor
//         .extract_data(
//             &mut Cursor::new(file_data),
//             &raw_header,
//             PointRange {
//                 file_index: 0,
//                 points_in_file: 0..indices.len(),
//             },
//             indices,
//             num_matches,
//             &runtime_tracker,
//         )
//         .context("Failed to extract data")?;

//     drop(result);

//     Ok(())
// }

fn main() -> Result<()> {
    dotenv::dotenv().context("Failed to initialize with .env file")?;
    pretty_env_logger::init();

    let args = Args::parse();

    if args
        .input_file
        .extension()
        .expect("Could not get file extension")
        .to_string_lossy()
        != "las"
    {
        panic!("Currently only LAS files are supported!");
    }

    let machine = std::env::var("MACHINE").context("To run experiments, please set the 'MACHINE' environment variable to the name of the machine that you are running this experiment on. This is required so that experiment data can be mapped to the actual machine that ran the experiment. This will typically be the name or system configuration of the computer that runs the experiment.")?;

    let variables = [
        VARIABLE_DATASET.clone(),
        VARIABLE_ACCESS_PATTERN.clone(),
        VARIABLE_TOOL.clone(),
        VARIABLE_RUNTIME.clone(),
        VARIABLE_NR_POINTS_TOTAL.clone(),
        VARIABLE_NR_POINTS_MATCHING.clone(),
        VARIABLE_MACHINE.clone(),
    ]
    .into_iter()
    .collect::<HashSet<VariableTemplate>>();

    let mut experiment = Experiment::new(
        "Pointcloud read performance of Rust crates".into(),
        "Measures the performance of reading point clouds using various Rust crates and various access patterns. An access pattern refers to the indices of the points that should be read and would be something that comes out of a query, e.g. reading all points, reading only points within specific bounds, reading points with a certain LOD etc. In this experiment, the access patterns are modeled using a bunch of predefined patterns instead of running actual queries.".into(),
        "Pascal Bormann".into(),
        variables,
    ).context("Failed to setup experiment")?;
    experiment.set_autolog_runs(true);

    // TODO Generate a bunch of sequences of matching points and extract the data for these sequences, once
    // using the LAS data extractor in `search` and once by using pasture_io
    // The structure of the sequence will matter, we can do some variations, e.g. full random, sequences that
    // match some query, different densities. Some work is necessary to figure out good test scenarios that
    // correspond to actual real-world use cases
    // --> This could be a nice experiment for my diss

    let point_count = {
        let reader =
            GenericPointReader::open_file(&args.input_file).context("Could not open file")?;
        let metadata = reader.get_metadata();
        metadata
            .number_of_points()
            .expect("Only file formats that know their point counts are supported")
    };
    let count = min(point_count, args.count);
    let dataset = args.input_file.display().to_string();

    for access_pattern in AccessPattern::all() {
        let sample = gen_sample_from_access_pattern(access_pattern, count);
        let num_matches = sample.iter().filter(|b| **b).count();

        let run_experiment = |f: Box<dyn FnOnce() -> Result<()>>, tool_name| -> Result<()> {
            experiment
                .run(|context| {
                    let time = {
                        let timer = Instant::now();
                        f().context("Reading failed")?;
                        timer.elapsed()
                    };

                    context.add_value_by_name(
                        VARIABLE_ACCESS_PATTERN.name(),
                        access_pattern.to_string(),
                    );
                    context.add_value_by_name(VARIABLE_DATASET.name(), &dataset);
                    context.add_value_by_name(VARIABLE_NR_POINTS_MATCHING.name(), num_matches);
                    context.add_value_by_name(VARIABLE_NR_POINTS_TOTAL.name(), count);
                    context.add_value_by_name(VARIABLE_RUNTIME.name(), time.as_millis());
                    context.add_value_by_name(VARIABLE_TOOL.name(), tool_name);
                    context.add_value_by_name(VARIABLE_MACHINE.name(), &machine);

                    Ok(())
                })
                .context("Failed to run experiment")?;
            Ok(())
        };

        run_experiment(
            Box::new(|| read_with_pasture(&sample, &args.input_file, false)),
            "pasture (unfiltered)",
        )?;

        run_experiment(
            Box::new(|| read_with_pasture(&sample, &args.input_file, true)),
            "pasture (filtered)",
        )?;

        // run_experiment(
        //     Box::new(|| read_with_extractor(&sample, &args.input_file)),
        //     "LASExtractor from query crate",
        // )?;

        run_experiment(
            Box::new(|| read_with_las_crate(&sample, &args.input_file)),
            "las-rs",
        )?;
    }

    Ok(())
}
