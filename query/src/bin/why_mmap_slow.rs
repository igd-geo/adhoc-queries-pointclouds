use std::{
    fs::File,
    hint::black_box,
    io::{BufReader, Cursor, Read},
    time::Instant,
};

use anyhow::{Context, Result};
use clap::Parser;
use memmap2::Advice;
use pasture_core::containers::VectorBuffer;
use pasture_io::{base::PointReader, las::LASReader};

const PATH: &str =
    "/Users/pbormann/data/projects/progressive_indexing/experiment_data/doc/las/1321_1.las";

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, short, default_value_t = false)]
    use_mmap: bool,
}

fn read_points(args: &Args) -> Result<()> {
    let points = if args.use_mmap {
        let file = File::open(PATH).context("failed to open file")?;
        let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
        mmap.advise(Advice::sequential())?;
        let mut reader = LASReader::from_read(Cursor::new(&mmap[..]), false, true)
            .context("failed to open LAS reader")?;
        let points = reader
            .read::<VectorBuffer>(reader.remaining_points())
            .context("failed to read points")?;
        points
    } else {
        let file = File::open(PATH).context("failed to open file")?;
        let mut reader = LASReader::from_read(BufReader::new(file), false, true)
            .context("failed to open LAS reader")?;
        let points = reader
            .read::<VectorBuffer>(reader.remaining_points())
            .context("failed to read points")?;
        points
    };

    black_box(points);

    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();

    let timer = Instant::now();

    let mut file = File::open(PATH)?;
    let len = file.metadata()?.len() as usize;

    // let mut data = Vec::with_capacity(len);
    // unsafe {
    //     data.set_len(len);
    // }
    // {
    //     file.read_exact(&mut data)?;
    // }
    // black_box(data);
    let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
    mmap.advise(Advice::sequential())?;
    // // data.copy_from_slice(&mmap[..]);
    for b in mmap.iter() {
        black_box(*b);
    }
    // black_box(data);

    eprintln!("{}ms", timer.elapsed().as_millis());

    Ok(())
}
