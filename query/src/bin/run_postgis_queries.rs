use std::{fmt::Display, time::Instant};

use anyhow::{bail, Context, Result};
use clap::{Parser, ValueEnum};
use exar::{
    experiment::{ExperimentInstance, ExperimentVersion},
    variable::GenericValue,
};
use log::info;
use postgres::{Client, Config, NoTls};

struct NamedQuery {
    name: &'static str,
    query: String,
}

#[derive(ValueEnum, Copy, Clone, Debug)]
enum Dataset {
    Doc,
    CA13,
    AHN4S,
}

impl Dataset {
    fn table_name(&self) -> &'static str {
        match self {
            Dataset::Doc => "doc",
            Dataset::CA13 => "ca13",
            Dataset::AHN4S => "ahn4s",
        }
    }

    fn queries(&self, output_format: &str) -> Vec<NamedQuery> {
        match self {
            Dataset::Doc => get_queries_doc_patches(output_format),
            Dataset::AHN4S => get_queries_ahn4s_patches(output_format),
            Dataset::CA13 => get_queries_ca13_patches(output_format),
        }
    }
}

impl Display for Dataset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Dataset::Doc => write!(f, "DoC"),
            Dataset::CA13 => write!(f, "CA13"),
            Dataset::AHN4S => write!(f, "AHN4S"),
        }
    }
}

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    dataset: Dataset,
    /// Name of a specific query to execute
    query_name: Option<String>,
}

fn get_queries_doc_patches(output_format: &str) -> Vec<NamedQuery> {
    // For quickly testing we can include a limit string
    const _LIMIT: &str = "LIMIT 1";
    let limit_str = "";

    let small_rect_query = format!(
        "SELECT {output_format} 
        FROM (
            SELECT pa AS patches 
            FROM doc
            WHERE PC_Intersects(
                ST_Transform((SELECT geom FROM doc_shapes WHERE name='small_rect'), 4329),
                pa
            ) {limit_str}
        ) AS subquery"
    );
    let small_polygon_query = format!(
        "SELECT {output_format} 
        FROM (
            SELECT pa AS patches 
            FROM doc
            WHERE PC_Intersects(
                ST_Transform((SELECT geom FROM doc_shapes WHERE name='small_polygon'), 4329),
                pa
            ) {limit_str}
        ) AS subquery"
    );
    let small_polygon_with_holes_query = format!(
        "SELECT {output_format} 
        FROM (
            SELECT pa AS patches 
            FROM doc 
            WHERE PC_Intersects(
                ST_Transform((SELECT geom FROM doc_shapes WHERE name='small_polygon_with_holes'), 4329),
                pa
            ) {limit_str}
        ) AS subquery");
    let large_rect_query = format!(
        "SELECT {output_format} 
        FROM (
            SELECT pa AS patches 
            FROM doc 
            WHERE PC_Intersects(
                ST_Transform((SELECT geom FROM doc_shapes WHERE name='large_rect'), 4329),
                pa
            ) {limit_str}
        ) AS subquery"
    );
    let large_polygon_query = format!(
        "SELECT {output_format} 
        FROM (
            SELECT pa AS patches 
            FROM doc 
            WHERE PC_Intersects(
                ST_Transform((SELECT geom FROM doc_shapes WHERE name='large_polygon'), 4329),
                pa
            ) {limit_str}
        ) AS subquery"
    );

    let bounds_none_query = format!(
        "SELECT {output_format} 
        FROM (
            SELECT pa AS patches 
            FROM doc 
            WHERE PC_Intersects(ST_MakeEnvelope(0, 0, 0, 0, 4329), pa) {limit_str}
        ) AS subquery"
    );
    let bounds_small_query = format!(
        "SELECT {output_format} 
        FROM (
            SELECT pa AS patches 
            FROM doc 
            WHERE PC_Intersects(ST_MakeEnvelope(390000, 130000, 390500, 140000, 4329), pa) {limit_str}
        ) AS subquery"
    );
    let bounds_large_query = format!(
        "SELECT {output_format} 
        FROM (
            SELECT pa AS patches 
            FROM doc 
            WHERE PC_Intersects(ST_MakeEnvelope(390000, 130000, 400000, 140000, 4329), pa) {limit_str}
        ) AS subquery"
    );
    let bounds_all_query = format!(
        "SELECT {output_format} 
        FROM (
            SELECT pa AS patches 
            FROM doc 
            WHERE PC_Intersects(ST_MakeEnvelope(0, 0, 1000000, 1000000, 4329), pa) {limit_str}
        ) AS subquery"
    );

    let buildings_query = format!(
        "SELECT {output_format} 
        FROM (
            SELECT PC_FilterEquals(pa, 'Classification', 6) AS patches 
            FROM doc {limit_str}
        ) AS subquery"
    );
    let vegetation_query = format!(
        "SELECT {output_format} 
        FROM (
            SELECT PC_FilterBetween(pa, 'Classification', 2, 6) AS patches 
            FROM doc {limit_str}
        ) AS subquery"
    );
    let first_returns_query = format!(
        "SELECT {output_format} 
        FROM (
            SELECT PC_FilterEquals(pa, 'ReturnNumber', 1) AS patches 
            FROM doc {limit_str}
        ) AS subquery"
    );
    let canopies_estimate = format!(
        "SELECT {output_format}
        FROM (
            SELECT PC_FilterGreaterThan(PC_FilterEquals(pa, 'ReturnNumber', 1), 'NumberOfReturns', 1) AS patches 
	        FROM doc {limit_str}
        ) AS subquery"
    );

    // LOD queries seem impossible with pgPointclouds...

    vec![
        NamedQuery {
            name: "AABB small",
            query: bounds_small_query,
        },
        NamedQuery {
            name: "AABB large",
            query: bounds_large_query,
        },
        NamedQuery {
            name: "AABB full",
            query: bounds_all_query,
        },
        NamedQuery {
            name: "AABB none",
            query: bounds_none_query,
        },
        NamedQuery {
            name: "Rect small",
            query: small_rect_query,
        },
        NamedQuery {
            name: "Polygon small",
            query: small_polygon_query,
        },
        NamedQuery {
            name: "Rect large",
            query: large_rect_query,
        },
        NamedQuery {
            name: "Polygon large",
            query: large_polygon_query,
        },
        NamedQuery {
            name: "Polygon holes",
            query: small_polygon_with_holes_query,
        },
        NamedQuery {
            name: "Buildings",
            query: buildings_query,
        },
        NamedQuery {
            name: "Vegetation",
            query: vegetation_query,
        },
        NamedQuery {
            name: "First return",
            query: first_returns_query,
        },
        NamedQuery {
            name: "Canopies estimate",
            query: canopies_estimate,
        },
    ]
}

fn get_queries_ahn4s_patches(output_format: &str) -> Vec<NamedQuery> {
    // For quickly testing we can include a limit string
    const _LIMIT: &str = "LIMIT 1";
    let limit_str = "";

    let small_rect_query = format!(
        "SELECT {output_format} 
        FROM (
            SELECT pa AS patches 
            FROM ahn4s
            WHERE PC_Intersects(
                ST_Transform((SELECT geom FROM ahn4s_shapes WHERE name='Rect small'), 4329),
                pa
            ) {limit_str}
        ) AS subquery"
    );
    let small_polygon_query = format!(
        "SELECT {output_format} 
        FROM (
            SELECT pa AS patches 
            FROM ahn4s
            WHERE PC_Intersects(
                ST_Transform((SELECT geom FROM ahn4s_shapes WHERE name='Polygon small'), 4329),
                pa
            ) {limit_str}
        ) AS subquery"
    );
    let small_polygon_with_holes_query = format!(
        "SELECT {output_format} 
        FROM (
            SELECT pa AS patches 
            FROM ahn4s 
            WHERE PC_Intersects(
                ST_Transform((SELECT geom FROM ahn4s_shapes WHERE name='Polygon holes'), 4329),
                pa
            ) {limit_str}
        ) AS subquery"
    );
    let large_rect_query = format!(
        "SELECT {output_format} 
        FROM (
            SELECT pa AS patches 
            FROM ahn4s 
            WHERE PC_Intersects(
                ST_Transform((SELECT geom FROM ahn4s_shapes WHERE name='Rect large'), 4329),
                pa
            ) {limit_str}
        ) AS subquery"
    );
    let large_polygon_query = format!(
        "SELECT {output_format} 
        FROM (
            SELECT pa AS patches 
            FROM ahn4s 
            WHERE PC_Intersects(
                ST_Transform((SELECT geom FROM ahn4s_shapes WHERE name='Polygon large'), 4329),
                pa
            ) {limit_str}
        ) AS subquery"
    );

    let bounds_none_query = format!(
        "SELECT {output_format} 
        FROM (
            SELECT pa AS patches 
            FROM ahn4s 
            WHERE PC_Intersects(ST_MakeEnvelope(0, 0, 0, 0, 4329), pa) {limit_str}
        ) AS subquery"
    );
    let bounds_small_query = format!(
        "SELECT {output_format} 
        FROM (
            SELECT pa AS patches 
            FROM ahn4s 
            WHERE PC_Intersects(ST_MakeEnvelope(122000, 481250, 122500, 482500, 4329), pa) {limit_str}
        ) AS subquery"
    );
    let bounds_large_query = format!(
        "SELECT {output_format} 
        FROM (
            SELECT pa AS patches 
            FROM ahn4s 
            WHERE PC_Intersects(ST_MakeEnvelope(122000, 481250, 122500, 482500, 4329), pa) {limit_str}
        ) AS subquery"
    );
    let bounds_all_query = format!(
        "SELECT {output_format} 
        FROM (
            SELECT pa AS patches 
            FROM ahn4s 
            WHERE PC_Intersects(ST_MakeEnvelope(120000, 481250, 125000, 487500, 4329), pa) {limit_str}
        ) AS subquery"
    );

    let buildings_query = format!(
        "SELECT {output_format} 
        FROM (
            SELECT PC_FilterEquals(pa, 'Classification', 6) AS patches 
            FROM ahn4s {limit_str}
        ) AS subquery"
    );
    let buildings_in_small_polygon_query = format!(
        "SELECT {output_format} 
        FROM (
            SELECT PC_FilterEquals(pa, 'Classification', 6) AS patches 
            WHERE PC_Intersects(
                ST_Transform((SELECT geom FROM ahn4s_shapes WHERE name='Polygon small'), 4329),
                pa
            )
            FROM ahn4s {limit_str}
        ) AS subquery"
    );
    let vegetation_query = format!(
        "SELECT {output_format} 
        FROM (
            SELECT PC_FilterBetween(pa, 'Classification', 2, 6) AS patches 
            FROM ahn4s {limit_str}
        ) AS subquery"
    );
    let first_returns_query = format!(
        "SELECT {output_format} 
        FROM (
            SELECT PC_FilterEquals(pa, 'ReturnNumber', 1) AS patches 
            FROM ahn4s {limit_str}
        ) AS subquery"
    );
    let canopies_estimate = format!(
        "SELECT {output_format}
        FROM (
            SELECT PC_FilterGreaterThan(PC_FilterEquals(pa, 'ReturnNumber', 1), 'NumberOfReturns', 1) AS patches 
	        FROM ahn4s {limit_str}
        ) AS subquery"
    );

    // LOD queries seem impossible with pgPointclouds...

    vec![
        NamedQuery {
            name: "AABB small",
            query: bounds_small_query,
        },
        NamedQuery {
            name: "AABB large",
            query: bounds_large_query,
        },
        NamedQuery {
            name: "AABB full",
            query: bounds_all_query,
        },
        NamedQuery {
            name: "AABB none",
            query: bounds_none_query,
        },
        NamedQuery {
            name: "Rect small",
            query: small_rect_query,
        },
        NamedQuery {
            name: "Polygon small",
            query: small_polygon_query,
        },
        NamedQuery {
            name: "Rect large",
            query: large_rect_query,
        },
        NamedQuery {
            name: "Polygon large",
            query: large_polygon_query,
        },
        NamedQuery {
            name: "Polygon holes",
            query: small_polygon_with_holes_query,
        },
        NamedQuery {
            name: "Buildings",
            query: buildings_query,
        },
        NamedQuery {
            name: "Buildings in small polygon",
            query: buildings_in_small_polygon_query,
        },
        NamedQuery {
            name: "Vegetation",
            query: vegetation_query,
        },
        NamedQuery {
            name: "First return",
            query: first_returns_query,
        },
        NamedQuery {
            name: "Canopies estimate",
            query: canopies_estimate,
        },
    ]
}

fn get_queries_ca13_patches(output_format: &str) -> Vec<NamedQuery> {
    // For quickly testing we can include a limit string
    const _LIMIT: &str = "LIMIT 1";
    let limit_str = "";

    let small_rect_query = format!(
        "SELECT {output_format} 
        FROM (
            SELECT pa AS patches 
            FROM ca13s
            WHERE PC_Intersects(
                ST_Transform((SELECT geom FROM ca13s_shapes WHERE name='Rect small'), 4329),
                pa
            ) {limit_str}
        ) AS subquery"
    );
    let small_polygon_query = format!(
        "SELECT {output_format} 
        FROM (
            SELECT pa AS patches 
            FROM ca13s
            WHERE PC_Intersects(
                ST_Transform((SELECT geom FROM ca13s_shapes WHERE name='Polygon small'), 4329),
                pa
            ) {limit_str}
        ) AS subquery"
    );
    let small_polygon_with_holes_query = format!(
        "SELECT {output_format} 
        FROM (
            SELECT pa AS patches 
            FROM ca13s 
            WHERE PC_Intersects(
                ST_Transform((SELECT geom FROM ca13s_shapes WHERE name='Polygon holes'), 4329),
                pa
            ) {limit_str}
        ) AS subquery"
    );
    let large_rect_query = format!(
        "SELECT {output_format} 
        FROM (
            SELECT pa AS patches 
            FROM ca13s 
            WHERE PC_Intersects(
                ST_Transform((SELECT geom FROM ca13s_shapes WHERE name='Rect large'), 4329),
                pa
            ) {limit_str}
        ) AS subquery"
    );
    let large_polygon_query = format!(
        "SELECT {output_format} 
        FROM (
            SELECT pa AS patches 
            FROM ca13s 
            WHERE PC_Intersects(
                ST_Transform((SELECT geom FROM ca13s_shapes WHERE name='Polygon large'), 4329),
                pa
            ) {limit_str}
        ) AS subquery"
    );

    let bounds_none_query = format!(
        "SELECT {output_format} 
        FROM (
            SELECT pa AS patches 
            FROM ca13s 
            WHERE PC_Intersects(ST_MakeEnvelope(0, 0, 0, 0, 4329), pa) {limit_str}
        ) AS subquery"
    );
    let bounds_small_query = format!(
        "SELECT {output_format} 
        FROM (
            SELECT pa AS patches 
            FROM ca13s 
            WHERE PC_Intersects(ST_MakeEnvelope(734000.0, 3889087.89, 735000.00, 3905000.0, 4329), pa) {limit_str}
        ) AS subquery"
    );
    let bounds_large_query = format!(
        "SELECT {output_format} 
        FROM (
            SELECT pa AS patches 
            FROM ca13s 
            WHERE PC_Intersects(ST_MakeEnvelope(715932.19, 3889087.89, 736910.93, 3905000.0, 4329), pa) {limit_str}
        ) AS subquery"
    );
    let bounds_all_query = format!(
        "SELECT {output_format} 
        FROM (
            SELECT pa AS patches 
            FROM ca13s 
            WHERE PC_Intersects(ST_MakeEnvelope(715932.19, 3889087.89, 736910.93, 3909670.85, 4329), pa) {limit_str}
        ) AS subquery"
    );

    let buildings_query = format!(
        "SELECT {output_format} 
        FROM (
            SELECT PC_FilterEquals(pa, 'Classification', 6) AS patches 
            FROM ca13s {limit_str}
        ) AS subquery"
    );
    let vegetation_query = format!(
        "SELECT {output_format} 
        FROM (
            SELECT PC_FilterBetween(pa, 'Classification', 2, 6) AS patches 
            FROM ca13s {limit_str}
        ) AS subquery"
    );
    let first_returns_query = format!(
        "SELECT {output_format} 
        FROM (
            SELECT PC_FilterEquals(pa, 'ReturnNumber', 1) AS patches 
            FROM ca13s {limit_str}
        ) AS subquery"
    );
    let canopies_estimate = format!(
        "SELECT {output_format}
        FROM (
            SELECT PC_FilterGreaterThan(PC_FilterEquals(pa, 'ReturnNumber', 1), 'NumberOfReturns', 1) AS patches 
	        FROM ca13s {limit_str}
        ) AS subquery"
    );

    // LOD queries seem impossible with pgPointclouds...

    vec![
        NamedQuery {
            name: "AABB small",
            query: bounds_small_query,
        },
        NamedQuery {
            name: "AABB large",
            query: bounds_large_query,
        },
        NamedQuery {
            name: "AABB full",
            query: bounds_all_query,
        },
        NamedQuery {
            name: "AABB none",
            query: bounds_none_query,
        },
        NamedQuery {
            name: "Rect small",
            query: small_rect_query,
        },
        NamedQuery {
            name: "Polygon small",
            query: small_polygon_query,
        },
        NamedQuery {
            name: "Rect large",
            query: large_rect_query,
        },
        NamedQuery {
            name: "Polygon large",
            query: large_polygon_query,
        },
        NamedQuery {
            name: "Polygon holes",
            query: small_polygon_with_holes_query,
        },
        NamedQuery {
            name: "Buildings",
            query: buildings_query,
        },
        NamedQuery {
            name: "Vegetation",
            query: vegetation_query,
        },
        NamedQuery {
            name: "First return",
            query: first_returns_query,
        },
        NamedQuery {
            name: "Canopies estimate",
            query: canopies_estimate,
        },
    ]
}

fn connect_to_db(config: &Config) -> Result<Client> {
    let client = config.connect(NoTls).context(format!(
        "Could not connect to postgres DB with config {:?}",
        config
    ))?;
    Ok(client)
}

fn assure_index_exists(tablename: &str, client: &mut Client) -> Result<()> {
    let rows = client.query(
        "SELECT indexdef
                FROM pg_indexes
                WHERE schemaname = 'public' AND tablename = $1",
        &[&tablename],
    )?;
    // Looking for a gist index using the envelope geometries of the patches
    let has_index = rows.iter().any(|row| {
        let index_definition: &str = row.get(0);
        index_definition.contains("USING gist (pc_envelopegeometry(pa))")
    });
    if !has_index {
        bail!("Missing spatial index on table {tablename}");
    }
    Ok(())
}

fn run_query(
    query: &NamedQuery,
    experiment_instance: ExperimentInstance<'_>,
    result_is_match_count: bool,
    client: &mut Client,
) -> Result<()> {
    let (time, num_matches, num_bytes) = {
        let t_start = Instant::now();
        if result_is_match_count {
            let matching_rows = client.query(&query.query, &[]).with_context(|| {
                format!(
                    "Query \"{}\" failed (Full query: {})",
                    query.name, query.query
                )
            })?;
            let elapsed = t_start.elapsed();
            let num_matches = matching_rows[0].try_get::<_, i64>("sum").unwrap_or(0) as usize;
            (elapsed, num_matches, 0)
        } else {
            let mut reader = client
                .copy_out(&format!("COPY ({}) TO STDOUT (FORMAT binary)", query.query))
                .with_context(|| {
                    format!(
                        "Query \"{}\" failed (Full query: {})",
                        query.name, query.query
                    )
                })?;
            // let mut data = vec![];
            // let num_bytes = reader.read_to_end(&mut data)?;
            // let mut data_after_header = Cursor::new(&data[19..]);
            // let num_fields = data_after_header.read_u16::<BigEndian>()?;
            // println!("Num fields: {num_fields}");
            // for field_idx in 0..num_fields {
            //     let field_length = data_after_header.read_u32::<BigEndian>()?;
            //     println!("Field {field_idx}: {field_length}B");
            //     data_after_header.seek(SeekFrom::Current(field_length as i64))?;
            // }
            let num_bytes = std::io::copy(&mut reader, &mut std::io::sink())?;
            let elapsed = t_start.elapsed();
            // TODO Calculate num matches from num_bytes written. This is not trivial: We should use BINARY format
            // so that we are as close as possible to what the ad-hoc query engine does, but in BINARY format the
            // actual size of the tuples (which are `double precision[]` types) is implementation-defined...
            (elapsed, 0, num_bytes)
        }
    };

    info!(
        "{} - {:.3}s - {} matches",
        query.name,
        time.as_secs_f64(),
        num_matches
    );

    experiment_instance.run(|context| -> Result<()> {
        context.add_measurement("Runtime", GenericValue::Numeric(time.as_secs_f64()));
        context.add_measurement("Bytes written", GenericValue::Numeric(num_bytes as f64));
        context.add_measurement("Match count", GenericValue::Numeric(num_matches as f64));
        // Impossible to get the number of queried points I believe...
        context.add_measurement("Queried points count", GenericValue::Numeric(f64::NAN));

        Ok(())
    })?;

    Ok(())
}

fn main() -> Result<()> {
    pretty_env_logger::init();

    let args = Args::parse();

    let machine = std::env::var("MACHINE").context("To run experiments, please set the 'MACHINE' environment variable to the name of the machine that you are running this experiment on. This is required so that experiment data can be mapped to the actual machine that ran the experiment. This will typically be the name or system configuration of the computer that runs the experiment.")?;

    let mut postgis_config = Client::configure();
    postgis_config
        .host(&std::env::var("PC_HOST").expect("Missing environment variable PC_HOST"))
        .port(
            std::env::var("PC_PORT")
                .expect("Missing environment variable PC_PORT")
                .as_str()
                .parse()?,
        )
        .user(&std::env::var("PC_USER").expect("Missing environment variable PC_USER"))
        .password(std::env::var("PC_PASSWORD").expect("Missing environment variable PC_PASSWORD"))
        .dbname(&std::env::var("PC_DBNAME").expect("Missing environment variable PC_DBNAME"));

    let mut client = connect_to_db(&postgis_config)?;
    assure_index_exists(args.dataset.table_name(), &mut client)?;

    let output_formats = [
        ("Point counts", "SUM(PC_NumPoints(patches))"), 
        ("All (default)", "PC_Get(PC_Explode(patches))"),
        ("Positions", "(PC_Get(PC_Explode(patches), 'X'), PC_Get(PC_Explode(patches), 'Y'), PC_Get(PC_Explode(patches), 'Z'))"),
        ("Positions, classifications, intensities", "(PC_Get(PC_Explode(patches), 'X'), PC_Get(PC_Explode(patches), 'Y'), PC_Get(PC_Explode(patches), 'Z'), PC_Get(PC_Explode(patches), 'Classification'), PC_Get(PC_Explode(patches), 'Intensity'))"),
    ];

    let experiment_description = include_str!("yaml/ad_hoc_queries.yaml");
    let experiment = ExperimentVersion::from_yaml_str(experiment_description)
        .context("Could not get current version of experiment")?;

    for (output_format_label, output_format) in output_formats {
        let mut queries = args.dataset.queries(output_format);
        if let Some(query_name) = args.query_name.as_ref() {
            queries.retain(|q| q.name == query_name);
        }
        for query in &queries {
            let experiment_instance = experiment.make_instance([
                ("Dataset", GenericValue::String(args.dataset.to_string())),
                ("Machine", GenericValue::String(machine.clone())),
                (
                    "System",
                    GenericValue::String("pgPointclouds (patches)".to_string()),
                ),
                ("Query", GenericValue::String(query.name.to_string())),
                (
                    "Output attributes",
                    GenericValue::String(output_format_label.to_string()),
                ),
                ("Purge cache", GenericValue::Bool(false)),
            ])?;
            let result_is_match_count = output_format_label == "Point counts";
            run_query(
                query,
                experiment_instance,
                result_is_match_count,
                &mut client,
            )
            .with_context(|| format!("Query {} failed", query.name))?;
        }
    }

    Ok(())
}
