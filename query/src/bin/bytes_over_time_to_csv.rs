use anyhow::{bail, Context, Result};

fn main() -> Result<()> {
    println!("Iteration,Time,Match percentage");
    let mut cur_iteration = 0;
    loop {
        let mut line = String::new();
        let len = std::io::stdin().read_line(&mut line)?;
        if len == 0 {
            break;
        }

        cur_iteration += 1;

        // Remove all braces so we have one large string of comma-separated values
        let comma_separated_values = line.replace("(", "").replace(")", "");
        let values = comma_separated_values
            .split(",")
            .map(|val| {
                val.trim()
                    .parse::<f64>()
                    .with_context(|| format!("Can't parse value {val} as f64"))
            })
            .collect::<Result<Vec<_>>>()
            .context("Failed to parse values as f64")?;
        if values.len() % 2 != 0 {
            bail!("Number of values in an input row must be even");
        }
        if values.is_empty() {
            bail!("Input row must not be empty");
        }
        let max_bytes = values.last().unwrap();
        // Each pair of values is a row in the CSV output
        for chunk in values.chunks_exact(2) {
            let bytes_percentage = 100.0 * (chunk[1] / max_bytes);
            println!("{cur_iteration},{},{}", chunk[0], bytes_percentage);
        }
    }

    Ok(())
}
