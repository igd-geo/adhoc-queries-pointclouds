use std::time::Instant;

use anyhow::{Context, Result};
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

// const DOC_NUM_MPOINTS: f64 = 876.0;
// const CA13_NUM_MPOINTS: f64 = 2608.0;

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

fn connect_to_db(config: &Config) -> Result<Client> {
    let client = config.connect(NoTls).context(format!(
        "Could not connect to postgres DB with config {:?}",
        config
    ))?;
    Ok(client)
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
    // This assumes that there are three tables in the database named 'navvis', 'doc' and 'ca13' for the three tests
    // datasets referenced in the paper

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

    let output_formats = [
        ("Point counts", "SUM(PC_NumPoints(patches))"), 
        ("All (default)", "PC_Get(PC_Explode(patches))"),
        ("Positions", "(PC_Get(PC_Explode(patches), 'X'), PC_Get(PC_Explode(patches), 'Y'), PC_Get(PC_Explode(patches), 'Z'))"),
        ("Positions, classifiations, intensites", "(PC_Get(PC_Explode(patches), 'X'), PC_Get(PC_Explode(patches), 'Y'), PC_Get(PC_Explode(patches), 'Z'), PC_Get(PC_Explode(patches), 'Classification'), PC_Get(PC_Explode(patches), 'Intensity'))"),
    ];

    let experiment_description = include_str!("yaml/ad_hoc_queries.yaml");
    let experiment = ExperimentVersion::from_yaml_str(experiment_description)
        .context("Could not get current version of experiment")?;

    for (output_format_label, output_format) in output_formats {
        let doc_patches_queries = get_queries_doc_patches(output_format);
        for query in &doc_patches_queries {
            let experiment_instance = experiment.make_instance([
                ("Dataset", GenericValue::String(format!("DoC"))),
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

    // let navvis_bounds = Bounds {
    //     s: "'SRID=4329;POLYGON((0.0 0.0, 0.0 2.0, 2.0 2.0, 2.0 0.0, 0.0 0.0))'::geometry".into(),
    //     l: "'SRID=4329;POLYGON((0.0 0.0, 0.0 20.0, 20.0 20.0, 20.0 0.0, 0.0 0.0))'::geometry".into(),
    //     xl: "'SRID=4329;POLYGON((-23.108 -21.261, -23.108 27.123, 28.588 27.123, 28.588 -21.261, -23.108 -21.261))'::geometry".into(),
    // };
    // let doc_bounds = Bounds {
    //     s: "'SRID=4329;POLYGON((390000 130000, 390000 140000, 390500 140000, 390500 130000, 390000 130000))'::geometry".into(),
    //     l: "'SRID=4329;POLYGON((390000 130000, 390000 140000, 400000 140000, 400000 130000, 390000 130000))'::geometry".into(),
    //     xl: "'SRID=4329;POLYGON((389400 124200, 389400 148200, 406200 148200, 406200 124200, 389400 124200))'::geometry".into(),
    // };
    // let ca13_bounds = Bounds {
    //     s: "'SRID=4329;POLYGON((665000 3910000, 665000 3950000, 705000 3950000, 705000 3910000, 665000 3910000))'::geometry".into(),
    //     l: "'SRID=4329;POLYGON((665000 3910000, 665000 3950000, 710000 3950000, 710000 3910000, 665000 3910000))'::geometry".into(),
    //     xl: "'SRID=4329;POLYGON((643431.76 3883547.565, 643431.76 3977026.735, 736910.93 3977026.735, 736910.93 3883547.565, 643431.76 3883547.565))'::geometry".into(),
    // };

    // run_spatial_queries(&navvis_bounds, "navvis", NAVVIS_NUM_MPOINTS, &client).await?;
    // run_spatial_queries(&doc_bounds, "doc", DOC_NUM_MPOINTS, &client).await?;
    // run_spatial_queries(&ca13_bounds, "ca13", CA13_NUM_MPOINTS, &client).await?;

    // run_class_queries("doc", DOC_NUM_MPOINTS, &client).await?;
    // run_class_queries("ca13", CA13_NUM_MPOINTS, &client).await?;

    Ok(())
}
