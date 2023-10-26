// Gathers statistics about the I/O performance for reading point cloud data in LAS, LAZ, LAST, and LAZER formats
// on the machine that it is run on. Stores the statistics in an internal format in the default config dir

use std::{
    collections::HashMap,
    fs::{File, OpenOptions},
    hint::black_box,
    io::{BufReader, BufWriter, Cursor, SeekFrom},
    path::Path,
    time::Instant,
};

use anyhow::{bail, Context, Result};
use io::{
    last::{las_to_last, LASTReader},
    lazer::{LazerReader, LazerWriter},
};
use log::info;
use lz4::EncoderBuilder;
use pasture_core::{
    containers::{HashMapBuffer, VectorBuffer},
    nalgebra::Vector3,
};
use pasture_io::{
    base::{PointReader, PointWriter, SeekToPoint},
    las::{
        LASReader, LASWriter, LasPointFormat0, LasPointFormat1, LasPointFormat2, LasPointFormat3,
        LasPointFormat4, LasPointFormat5, LasPointFormat6, LasPointFormat7, LasPointFormat8,
    },
    las_rs::{point::Format, Builder},
};
use query::io::{FileFormat, IOMethod, IOStats, IOStatsParameters};
use rand::{thread_rng, Rng};
use statrs::statistics::{Data, Median};

const LAS_PATH: &str = "test_points.las";
const LAZ_PATH: &str = "test_points.laz";
const LAST_PATH: &str = "test_points.last";
const LAZER_PATH: &str = "test_points.lazer";
const NUM_TEST_POINTS: usize = 1_000_000;
const ITERATIONS: usize = 10;

fn gen_random_points_in_format(point_record_format: Format) -> Result<VectorBuffer> {
    let mut rng = thread_rng();
    let buffer = match point_record_format.to_u8()? {
        0 => (0..NUM_TEST_POINTS)
            .map(|_| LasPointFormat0 {
                classification: rng.gen_range(0..15),
                edge_of_flight_line: rng.gen_range(0..2),
                intensity: rng.gen(),
                number_of_returns: 0,
                point_source_id: rng.gen(),
                position: Vector3::new(
                    rng.gen_range(-1000.0..1000.0),
                    rng.gen_range(-1000.0..1000.0),
                    rng.gen_range(-1000.0..1000.0),
                ),
                return_number: 0,
                scan_angle_rank: rng.gen(),
                scan_direction_flag: rng.gen_range(0..2),
                user_data: rng.gen(),
            })
            .collect(),
        1 => (0..NUM_TEST_POINTS)
            .map(|_| LasPointFormat1 {
                classification: rng.gen_range(0..15),
                edge_of_flight_line: rng.gen_range(0..2),
                intensity: rng.gen(),
                number_of_returns: 0,
                point_source_id: rng.gen(),
                position: Vector3::new(
                    rng.gen_range(-1000.0..1000.0),
                    rng.gen_range(-1000.0..1000.0),
                    rng.gen_range(-1000.0..1000.0),
                ),
                return_number: 0,
                scan_angle_rank: rng.gen(),
                scan_direction_flag: rng.gen_range(0..2),
                user_data: rng.gen(),
                gps_time: rng.gen(),
            })
            .collect(),
        2 => (0..NUM_TEST_POINTS)
            .map(|_| LasPointFormat2 {
                classification: rng.gen_range(0..15),
                edge_of_flight_line: rng.gen_range(0..2),
                intensity: rng.gen(),
                number_of_returns: 0,
                point_source_id: rng.gen(),
                position: Vector3::new(
                    rng.gen_range(-1000.0..1000.0),
                    rng.gen_range(-1000.0..1000.0),
                    rng.gen_range(-1000.0..1000.0),
                ),
                return_number: 0,
                scan_angle_rank: rng.gen(),
                scan_direction_flag: rng.gen_range(0..2),
                user_data: rng.gen(),
                color_rgb: Vector3::new(rng.gen(), rng.gen(), rng.gen()),
            })
            .collect(),
        3 => (0..NUM_TEST_POINTS)
            .map(|_| LasPointFormat3 {
                classification: rng.gen_range(0..15),
                edge_of_flight_line: rng.gen_range(0..2),
                intensity: rng.gen(),
                number_of_returns: 0,
                point_source_id: rng.gen(),
                position: Vector3::new(
                    rng.gen_range(-1000.0..1000.0),
                    rng.gen_range(-1000.0..1000.0),
                    rng.gen_range(-1000.0..1000.0),
                ),
                return_number: 0,
                scan_angle_rank: rng.gen(),
                scan_direction_flag: rng.gen_range(0..2),
                user_data: rng.gen(),
                color_rgb: Vector3::new(rng.gen(), rng.gen(), rng.gen()),
                gps_time: rng.gen(),
            })
            .collect(),
        4 => (0..NUM_TEST_POINTS)
            .map(|_| LasPointFormat4 {
                classification: rng.gen_range(0..15),
                edge_of_flight_line: rng.gen_range(0..2),
                intensity: rng.gen(),
                number_of_returns: 0,
                point_source_id: rng.gen(),
                position: Vector3::new(
                    rng.gen_range(-1000.0..1000.0),
                    rng.gen_range(-1000.0..1000.0),
                    rng.gen_range(-1000.0..1000.0),
                ),
                return_number: 0,
                scan_angle_rank: rng.gen(),
                scan_direction_flag: rng.gen_range(0..2),
                user_data: rng.gen(),
                byte_offset_to_waveform_data: 0,
                gps_time: rng.gen(),
                return_point_waveform_location: rng.gen(),
                wave_packet_descriptor_index: 0,
                waveform_packet_size: 0,
                waveform_parameters: Vector3::new(rng.gen(), rng.gen(), rng.gen()),
            })
            .collect(),
        5 => (0..NUM_TEST_POINTS)
            .map(|_| LasPointFormat5 {
                classification: rng.gen_range(0..15),
                edge_of_flight_line: rng.gen_range(0..2),
                intensity: rng.gen(),
                number_of_returns: 0,
                point_source_id: rng.gen(),
                position: Vector3::new(
                    rng.gen_range(-1000.0..1000.0),
                    rng.gen_range(-1000.0..1000.0),
                    rng.gen_range(-1000.0..1000.0),
                ),
                return_number: 0,
                scan_angle_rank: rng.gen(),
                scan_direction_flag: rng.gen_range(0..2),
                user_data: rng.gen(),
                gps_time: rng.gen(),
                color_rgb: Vector3::new(rng.gen(), rng.gen(), rng.gen()),
                byte_offset_to_waveform_data: 0,
                return_point_waveform_location: rng.gen(),
                wave_packet_descriptor_index: 0,
                waveform_packet_size: 0,
                waveform_parameters: Vector3::new(rng.gen(), rng.gen(), rng.gen()),
            })
            .collect(),
        6 => (0..NUM_TEST_POINTS)
            .map(|_| LasPointFormat6 {
                classification: rng.gen_range(0..15),
                edge_of_flight_line: rng.gen_range(0..2),
                intensity: rng.gen(),
                number_of_returns: 0,
                point_source_id: rng.gen(),
                position: Vector3::new(
                    rng.gen_range(-1000.0..1000.0),
                    rng.gen_range(-1000.0..1000.0),
                    rng.gen_range(-1000.0..1000.0),
                ),
                return_number: 0,
                classification_flags: rng.gen_range(0..15),
                scan_angle: rng.gen(),
                scanner_channel: rng.gen_range(0..15),
                scan_direction_flag: rng.gen_range(0..2),
                user_data: rng.gen(),
                gps_time: rng.gen(),
            })
            .collect(),
        7 => (0..NUM_TEST_POINTS)
            .map(|_| LasPointFormat7 {
                classification: rng.gen_range(0..15),
                edge_of_flight_line: rng.gen_range(0..2),
                intensity: rng.gen(),
                number_of_returns: 0,
                point_source_id: rng.gen(),
                position: Vector3::new(
                    rng.gen_range(-1000.0..1000.0),
                    rng.gen_range(-1000.0..1000.0),
                    rng.gen_range(-1000.0..1000.0),
                ),
                return_number: 0,
                classification_flags: rng.gen_range(0..15),
                scan_angle: rng.gen(),
                scanner_channel: rng.gen_range(0..15),
                scan_direction_flag: rng.gen_range(0..2),
                user_data: rng.gen(),
                gps_time: rng.gen(),
                color_rgb: Vector3::new(rng.gen(), rng.gen(), rng.gen()),
            })
            .collect(),
        8 => (0..NUM_TEST_POINTS)
            .map(|_| LasPointFormat8 {
                classification: rng.gen_range(0..15),
                edge_of_flight_line: rng.gen_range(0..2),
                intensity: rng.gen(),
                number_of_returns: 0,
                point_source_id: rng.gen(),
                position: Vector3::new(
                    rng.gen_range(-1000.0..1000.0),
                    rng.gen_range(-1000.0..1000.0),
                    rng.gen_range(-1000.0..1000.0),
                ),
                return_number: 0,
                classification_flags: rng.gen_range(0..15),
                scan_angle: rng.gen(),
                scanner_channel: rng.gen_range(0..15),
                scan_direction_flag: rng.gen_range(0..2),
                user_data: rng.gen(),
                gps_time: rng.gen(),
                color_rgb: Vector3::new(rng.gen(), rng.gen(), rng.gen()),
                nir: rng.gen(),
            })
            .collect(),
        other => bail!("Unsupported point record format {other}"),
    };
    Ok(buffer)
}

fn gen_test_files_in_format(point_record_format: Format) -> Result<()> {
    let points = gen_random_points_in_format(point_record_format).with_context(|| {
        format!(
            "Failed to create random test points in point record format {point_record_format:?}"
        )
    })?;
    let header = {
        let mut builder = Builder::from((1, 4));
        builder.point_format = point_record_format.clone();
        builder.into_header().context("Can't create LAS header")?
    };
    {
        let mut las_writer = LASWriter::from_writer_and_header(
            BufWriter::new(File::create(LAS_PATH).context("Could not create LAS file")?),
            header.clone(),
            false,
        )
        .context("Could not create LAS writer")?;
        las_writer
            .write(&points)
            .context("Failed to write points to LAS file")?;
        las_writer.flush().context("Failed to flush LAS points")?;
    }
    {
        let mut laz_writer = LASWriter::from_writer_and_header(
            BufWriter::new(File::create(LAZ_PATH).context("Could not create LAZ file")?),
            header.clone(),
            true,
        )
        .context("Could not create LAZ writer")?;
        laz_writer
            .write(&points)
            .context("Failed to write points to LAZ file")?;
        laz_writer.flush().context("Failed to flush LAZ points")?;
    }
    {
        let las_file_length = Path::new(LAS_PATH).metadata()?.len();
        let las_reader = BufReader::new(File::open(LAS_PATH)?);
        let last_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(LAST_PATH)?;
        last_file.set_len(las_file_length)?;
        let mut last_mmap =
            unsafe { memmap2::MmapMut::map_mut(&last_file).context("Failed to mmap LAST file")? };
        las_to_last(las_reader, Cursor::new(&mut last_mmap[..]))
            .context("Failed to write LAST file")?;
    }
    {
        let mut lazer_writer = LazerWriter::new(
            BufWriter::new(File::create(LAZER_PATH).context("Could not create LAZER file")?),
            header.clone(),
            EncoderBuilder::new(),
        )
        .context("Could not create LAZER writer")?;
        lazer_writer
            .write(&points)
            .context("Failed to write LAZER points")?;
        lazer_writer
            .flush()
            .context("Failed to flush LAZER writer")?;
    }
    Ok(())
}

fn remove_test_files() -> Result<()> {
    for path in [LAS_PATH, LAST_PATH, LAZ_PATH, LAZER_PATH] {
        let path = Path::new(path);
        if !path.exists() {
            continue;
        }
        std::fs::remove_file(path)
            .with_context(|| format!("Failed to remove file {}", path.display()))?;
    }
    Ok(())
}

fn bench_las(format_number: u8) -> Result<Vec<(IOStatsParameters, f64)>> {
    let file_stats = {
        let mut reader = LASReader::from_path(LAS_PATH, true)?;
        let runtimes = (0..ITERATIONS)
            .map(|_| -> Result<f64> {
                let timer = Instant::now();
                reader.seek_point(SeekFrom::Start(0))?;
                let data = reader.read::<VectorBuffer>(NUM_TEST_POINTS)?;
                black_box(data);
                let time = timer.elapsed();
                let mpts = (NUM_TEST_POINTS as f64 / 1e6) / time.as_secs_f64();
                Ok(mpts)
            })
            .collect::<Result<Vec<_>>>()?;
        let runtime_data = Data::new(runtimes);
        let median_mpts = runtime_data.median();
        (
            IOStatsParameters {
                file_format: FileFormat::LAS,
                io_method: IOMethod::File,
                point_record_format: format_number,
            },
            median_mpts,
        )
    };
    let mmap_stats = {
        let file = File::open(LAS_PATH)?;
        // There is a big difference between calling `mmap` on every loop iteration vs. calling it once outside of the loop
        // For the ad-hoc query engine, the latter is a more reasonable use-case, as the input layer will try to keep data
        // mapped as long as it doesn't run out of memory
        let las_mmap = unsafe { memmap2::Mmap::map(&file)? };
        let runtimes = (0..ITERATIONS)
            .map(|_| -> Result<f64> {
                let timer = Instant::now();
                let mut reader = LASReader::from_read(Cursor::new(&las_mmap[..]), false, true)?;
                let data = reader.read::<VectorBuffer>(NUM_TEST_POINTS)?;
                black_box(data);
                let time = timer.elapsed();
                let mpts = (NUM_TEST_POINTS as f64 / 1e6) / time.as_secs_f64();
                Ok(mpts)
            })
            .collect::<Result<Vec<_>>>()?;
        let runtime_data = Data::new(runtimes);
        let median_mpts = runtime_data.median();
        (
            IOStatsParameters {
                file_format: FileFormat::LAS,
                io_method: IOMethod::Mmap,
                point_record_format: format_number,
            },
            median_mpts,
        )
    };
    Ok(vec![file_stats, mmap_stats])
}

fn bench_laz(format_number: u8) -> Result<Vec<(IOStatsParameters, f64)>> {
    let file_stats = {
        let mut reader = LASReader::from_path(LAZ_PATH, true)?;
        let runtimes = (0..ITERATIONS)
            .map(|_| -> Result<f64> {
                let timer = Instant::now();
                reader.seek_point(SeekFrom::Start(0))?;
                let data = reader.read::<VectorBuffer>(NUM_TEST_POINTS)?;
                black_box(data);
                let time = timer.elapsed();
                let mpts = (NUM_TEST_POINTS as f64 / 1e6) / time.as_secs_f64();
                Ok(mpts)
            })
            .collect::<Result<Vec<_>>>()?;
        let runtime_data = Data::new(runtimes);
        let median_mpts = runtime_data.median();
        (
            IOStatsParameters {
                file_format: FileFormat::LAZ,
                io_method: IOMethod::File,
                point_record_format: format_number,
            },
            median_mpts,
        )
    };
    let mmap_stats = {
        let file = File::open(LAZ_PATH)?;
        // There is a big difference between calling `mmap` on every loop iteration vs. calling it once outside of the loop
        // For the ad-hoc query engine, the latter is a more reasonable use-case, as the input layer will try to keep data
        // mapped as long as it doesn't run out of memory
        let las_mmap = unsafe { memmap2::Mmap::map(&file)? };
        let runtimes = (0..ITERATIONS)
            .map(|_| -> Result<f64> {
                let timer = Instant::now();
                let mut reader = LASReader::from_read(Cursor::new(&las_mmap[..]), true, true)?;
                let data = reader.read::<VectorBuffer>(NUM_TEST_POINTS)?;
                black_box(data);
                let time = timer.elapsed();
                let mpts = (NUM_TEST_POINTS as f64 / 1e6) / time.as_secs_f64();
                Ok(mpts)
            })
            .collect::<Result<Vec<_>>>()?;
        let runtime_data = Data::new(runtimes);
        let median_mpts = runtime_data.median();
        (
            IOStatsParameters {
                file_format: FileFormat::LAZ,
                io_method: IOMethod::Mmap,
                point_record_format: format_number,
            },
            median_mpts,
        )
    };
    Ok(vec![file_stats, mmap_stats])
}

fn bench_last(format_number: u8) -> Result<Vec<(IOStatsParameters, f64)>> {
    let file_stats = {
        let mut reader = LASTReader::from_read(BufReader::new(File::open(LAST_PATH)?))?;
        let runtimes = (0..ITERATIONS)
            .map(|_| -> Result<f64> {
                let timer = Instant::now();
                reader.seek_point(SeekFrom::Start(0))?;
                let data = reader.read::<HashMapBuffer>(NUM_TEST_POINTS)?;
                black_box(data);
                let time = timer.elapsed();
                let mpts = (NUM_TEST_POINTS as f64 / 1e6) / time.as_secs_f64();
                Ok(mpts)
            })
            .collect::<Result<Vec<_>>>()?;
        let runtime_data = Data::new(runtimes);
        let median_mpts = runtime_data.median();
        (
            IOStatsParameters {
                file_format: FileFormat::LAST,
                io_method: IOMethod::File,
                point_record_format: format_number,
            },
            median_mpts,
        )
    };
    let mmap_stats = {
        let file = File::open(LAST_PATH)?;
        // There is a big difference between calling `mmap` on every loop iteration vs. calling it once outside of the loop
        // For the ad-hoc query engine, the latter is a more reasonable use-case, as the input layer will try to keep data
        // mapped as long as it doesn't run out of memory
        let last_mmap = unsafe { memmap2::Mmap::map(&file)? };
        let runtimes = (0..ITERATIONS)
            .map(|_| -> Result<f64> {
                let timer = Instant::now();
                let mut reader = LASTReader::from_read(Cursor::new(&last_mmap[..]))?;
                let data = reader.read::<HashMapBuffer>(NUM_TEST_POINTS)?;
                black_box(data);
                let time = timer.elapsed();
                let mpts = (NUM_TEST_POINTS as f64 / 1e6) / time.as_secs_f64();
                Ok(mpts)
            })
            .collect::<Result<Vec<_>>>()?;
        let runtime_data = Data::new(runtimes);
        let median_mpts = runtime_data.median();
        (
            IOStatsParameters {
                file_format: FileFormat::LAST,
                io_method: IOMethod::Mmap,
                point_record_format: format_number,
            },
            median_mpts,
        )
    };
    Ok(vec![file_stats, mmap_stats])
}

fn bench_lazer(format_number: u8) -> Result<Vec<(IOStatsParameters, f64)>> {
    let file_stats = {
        let mut reader = LazerReader::new(BufReader::new(File::open(LAZER_PATH)?))?;
        let runtimes = (0..ITERATIONS)
            .map(|_| -> Result<f64> {
                let timer = Instant::now();
                reader.seek_point(SeekFrom::Start(0))?;
                let data = reader.read::<HashMapBuffer>(NUM_TEST_POINTS)?;
                black_box(data);
                let time = timer.elapsed();
                let mpts = (NUM_TEST_POINTS as f64 / 1e6) / time.as_secs_f64();
                Ok(mpts)
            })
            .collect::<Result<Vec<_>>>()?;
        let runtime_data = Data::new(runtimes);
        let median_mpts = runtime_data.median();
        (
            IOStatsParameters {
                file_format: FileFormat::LAZER,
                io_method: IOMethod::File,
                point_record_format: format_number,
            },
            median_mpts,
        )
    };
    let mmap_stats = {
        let file = File::open(LAZER_PATH)?;
        // There is a big difference between calling `mmap` on every loop iteration vs. calling it once outside of the loop
        // For the ad-hoc query engine, the latter is a more reasonable use-case, as the input layer will try to keep data
        // mapped as long as it doesn't run out of memory
        let lazer_mmap = unsafe { memmap2::Mmap::map(&file)? };
        let runtimes = (0..ITERATIONS)
            .map(|_| -> Result<f64> {
                let timer = Instant::now();
                let mut reader = LazerReader::new(Cursor::new(&lazer_mmap[..]))?;
                let data = reader.read::<HashMapBuffer>(NUM_TEST_POINTS)?;
                black_box(data);
                let time = timer.elapsed();
                let mpts = (NUM_TEST_POINTS as f64 / 1e6) / time.as_secs_f64();
                Ok(mpts)
            })
            .collect::<Result<Vec<_>>>()?;
        let runtime_data = Data::new(runtimes);
        let median_mpts = runtime_data.median();
        (
            IOStatsParameters {
                file_format: FileFormat::LAZER,
                io_method: IOMethod::Mmap,
                point_record_format: format_number,
            },
            median_mpts,
        )
    };
    Ok(vec![file_stats, mmap_stats])
}

fn main() -> Result<()> {
    pretty_env_logger::init();

    let mut stat_entries = HashMap::new();

    for format_number in 0..=8 {
        info!("Benchmarking LAS point record format {format_number}...");
        let format = Format::new(format_number)?;
        scopeguard::defer! {
            remove_test_files().expect("Failed to remove test files");
        }
        gen_test_files_in_format(format.clone()).context("Failed to create test files")?;

        let las_stats = bench_las(format_number)?;
        info!("{:?}: {:.3}Mpts/s", las_stats[0].0, las_stats[0].1);
        info!("{:?}: {:.3}Mpts/s", las_stats[1].0, las_stats[1].1);
        las_stats.into_iter().for_each(|(key, val)| {
            stat_entries.insert(key, val);
        });

        let laz_stats = bench_laz(format_number)?;
        info!("{:?}: {:.3}Mpts/s", laz_stats[0].0, laz_stats[0].1);
        info!("{:?}: {:.3}Mpts/s", laz_stats[1].0, laz_stats[1].1);
        laz_stats.into_iter().for_each(|(key, val)| {
            stat_entries.insert(key, val);
        });

        let last_stats = bench_last(format_number)?;
        info!("{:?}: {:.3}Mpts/s", last_stats[0].0, last_stats[0].1);
        info!("{:?}: {:.3}Mpts/s", last_stats[1].0, last_stats[1].1);
        last_stats.into_iter().for_each(|(key, val)| {
            stat_entries.insert(key, val);
        });

        let lazer_stats = bench_lazer(format_number)?;
        info!("{:?}: {:.3}Mpts/s", lazer_stats[0].0, lazer_stats[0].1);
        info!("{:?}: {:.3}Mpts/s", lazer_stats[1].0, lazer_stats[1].1);
        lazer_stats.into_iter().for_each(|(key, val)| {
            stat_entries.insert(key, val);
        });
    }

    let io_stats = IOStats::new(stat_entries);
    io_stats
        .store_to_config()
        .context("Failed to write IOStats to config directory")?;

    Ok(())
}
