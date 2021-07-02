use std::time::Instant;

use anyhow::Result;
use clap::{App, Arg};
use statrs::statistics::{Data, Median, Statistics};
use tokio::task::JoinHandle;
use tokio_postgres::{Client, NoTls};

struct PostGISConfig {
    pub host: String,
    pub username: String,
    pub database: String,
    pub password: String,
}

struct Bounds {
    s: String,
    l: String,
    xl: String,
}

const NAVVIS_NUM_MPOINTS: f64 = 56.2;
const DOC_NUM_MPOINTS: f64 = 854.0;
const CA13_NUM_MPOINTS: f64 = 2608.0;

async fn connect_to_db(config: &PostGISConfig) -> Result<(Client, JoinHandle<()>)> {
    let connect_string = format!(
        "host={} dbname={} user={} password={}",
        config.host, config.database, config.username, config.password
    );

    let (client, connection) = tokio_postgres::connect(&connect_string, NoTls).await?;

    // The connection object performs the actual communication with the database,
    // so spawn it off to run on its own.
    let join_handle = tokio::spawn(async move {
        if let Err(e) = connection.await {
            eprintln!("connection error: {}", e);
        }
    });

    Ok((client, join_handle))
}

async fn get_avg_query_runtime(
    query: &str,
    client: &Client,
    iterations: usize,
) -> Result<Vec<f64>> {
    let mut timings = vec![];
    for _ in 0..iterations {
        let t_start = Instant::now();
        client.query(query, &[]).await?;
        timings.push(t_start.elapsed().as_secs_f64());
    }
    Ok(timings)
}

async fn run_spatial_queries(
    bounds: &Bounds,
    dataset: &str,
    num_megapoints: f64,
    client: &Client,
) -> Result<()> {
    // Hardcoded queries with the three bounding boxes S, L, XL (in 2D)

    let query_patches_s = format!(
        "SELECT SUM(PC_NumPoints(pa)) FROM {} WHERE PC_Intersects({}, pa);",
        dataset, bounds.s,
    );
    let query_patches_l = format!(
        "SELECT SUM(PC_NumPoints(pa)) FROM {} WHERE PC_Intersects({}, pa);",
        dataset, bounds.l,
    );
    let query_patches_xl = format!(
        "SELECT SUM(PC_NumPoints(pa)) FROM {} WHERE PC_Intersects({}, pa);",
        dataset, bounds.xl,
    );

    let query_points_s = format!(
        "SELECT SUM(PC_NumPoints(PC_Intersection(
                pa,
                {bounds}
        ))) 
        FROM 
        (
        SELECT * FROM {name} WHERE 
            PC_Intersects(
                pa,
                {bounds}
            )
        ) AS nested;",
        bounds = bounds.s,
        name = dataset,
    );
    let query_points_l = format!(
        "SELECT SUM(PC_NumPoints(PC_Intersection(
                pa,
                {bounds}
        ))) 
        FROM 
        (
        SELECT * FROM {name} WHERE 
            PC_Intersects(
                pa,
                {bounds}
            )
        ) AS nested;",
        bounds = bounds.l,
        name = dataset,
    );
    let query_points_xl = format!(
        "SELECT SUM(PC_NumPoints(PC_Intersection(
                pa,
                {bounds}
        ))) 
        FROM 
        (
        SELECT * FROM {name} WHERE 
            PC_Intersects(
                pa,
                {bounds}
            )
        ) AS nested;",
        bounds = bounds.xl,
        name = dataset,
    );

    let iterations = 5;

    // We explicitly run all futures AFTER each other using await, because we don't want to measure how long
    // 5 queries in parallel take, but how long a single query takes on average

    let patches_s_timings = get_avg_query_runtime(&query_patches_s, client, iterations).await?;
    let patches_l_timings = get_avg_query_runtime(&query_patches_l, client, iterations).await?;
    let patches_xl_timings = get_avg_query_runtime(&query_patches_xl, client, iterations).await?;

    let patches_s_median = Data::new(patches_s_timings.clone()).median();
    let patches_s_stddev = (&patches_s_timings).std_dev();
    let patches_l_median = Data::new(patches_l_timings.clone()).median();
    let patches_l_stddev = (&patches_l_timings).std_dev();
    let patches_xl_median = Data::new(patches_xl_timings.clone()).median();
    let patches_xl_stddev = (&patches_xl_timings).std_dev();

    // Print runtimes for S, L, XL and throughput in Mpts/s for S, L, XL. Print it so that we can easily insert it into a Latex table :)
    println!(
        "Patches: {:.2} $\\pm$ {:.2} & {:.2} $\\pm$ {:.2} & {:.2} $\\pm$ {:.2} & {:.2} & {:.2} & {:.2}",
        patches_s_median,
        patches_s_stddev,
        patches_l_median,
        patches_l_stddev,
        patches_xl_median,
        patches_xl_stddev,
        num_megapoints / patches_s_median,
        num_megapoints / patches_l_median,
        num_megapoints / patches_xl_median,
    );

    let points_s_timings = get_avg_query_runtime(&query_points_s, client, iterations).await?;
    let points_l_timings = get_avg_query_runtime(&query_points_l, client, iterations).await?;
    let points_xl_timings = get_avg_query_runtime(&query_points_xl, client, iterations).await?;

    let points_s_median = Data::new(points_s_timings.clone()).median();
    let points_s_stddev = (&points_s_timings).std_dev();
    let points_l_median = Data::new(points_l_timings.clone()).median();
    let points_l_stddev = (&points_l_timings).std_dev();
    let points_xl_median = Data::new(points_xl_timings.clone()).median();
    let points_xl_stddev = (&points_xl_timings).std_dev();

    println!(
        "Points: {:.2} $\\pm$ {:.2} & {:.2} $\\pm$ {:.2} & {:.2} $\\pm$ {:.2} & {:.2} & {:.2} & {:.2}",
        points_s_median,
        points_s_stddev,
        points_l_median,
        points_l_stddev,
        points_xl_median,
        points_xl_stddev,
        num_megapoints / points_s_median,
        num_megapoints / points_l_median,
        num_megapoints / points_xl_median,
    );

    Ok(())
}

async fn run_class_queries(dataset: &str, num_megapoints: f64, client: &Client) -> Result<()> {
    let query_existing_class = format!(
        "SELECT SUM(PC_NumPoints(pc_filterequals)) FROM
            (SELECT PC_FilterEquals(pa, 'Classification', {}) FROM {}) AS filtered;",
        6, dataset
    );
    let query_non_existing_class = format!(
        "SELECT SUM(PC_NumPoints(pc_filterequals)) FROM
            (SELECT PC_FilterEquals(pa, 'Classification', {}) FROM {}) AS filtered;",
        19, dataset
    );

    let iterations = 5;
    let existing_runtime =
        get_avg_query_runtime(query_existing_class.as_str(), client, iterations).await?;
    let non_existing_runtime =
        get_avg_query_runtime(query_non_existing_class.as_str(), client, iterations).await?;

    let existing_median = Data::new(existing_runtime.clone()).median();
    let existing_stddev = (&existing_runtime).std_dev();
    let non_existing_median = Data::new(non_existing_runtime.clone()).median();
    let non_existing_stddev = (&non_existing_runtime).std_dev();

    println!(
        "Class: {:.2} $\\pm$ {:.2} & {:.2} $\\pm$ {:.2} & {:.2} & {:.2}",
        existing_median,
        existing_stddev,
        non_existing_median,
        non_existing_stddev,
        num_megapoints / existing_median,
        num_megapoints / non_existing_median,
    );

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    // This assumes that there are three tables in the database named 'navvis', 'doc' and 'ca13' for the three tests
    // datasets referenced in the paper
    let matches = App::new("PostGIS queries")
        .version("0.1")
        .author("Pascal Bormann <pascal.bormann@igd.fraunhofer.de>")
        .about("PostGIS queries")
        .arg(
            Arg::with_name("HOST")
                .long("host")
                .help("PostGIS DB hostname")
                .takes_value(true)
                .required(true),
        )
        .arg(
            Arg::with_name("DBNAME")
                .long("dbname")
                .help("Name of the PostGIS database")
                .takes_value(true)
                .required(true),
        )
        .arg(
            Arg::with_name("USER")
                .long("user")
                .help("Username for PostGIS DB")
                .takes_value(true)
                .required(true),
        )
        .arg(
            Arg::with_name("PASSWORD")
                .long("pwd")
                .help("Password for PostGIS DB")
                .takes_value(true)
                .required(true),
        )
        .get_matches();

    let postgis_config = PostGISConfig {
        database: matches.value_of("DBNAME").unwrap().into(),
        host: matches.value_of("HOST").unwrap().into(),
        password: matches.value_of("PASSWORD").unwrap().into(),
        username: matches.value_of("USER").unwrap().into(),
    };

    let (client, _join_handle) = connect_to_db(&postgis_config).await?;

    let navvis_bounds = Bounds {
        s: "'SRID=4329;POLYGON((0.0 0.0, 0.0 2.0, 2.0 2.0, 2.0 0.0, 0.0 0.0))'::geometry".into(),
        l: "'SRID=4329;POLYGON((0.0 0.0, 0.0 20.0, 20.0 20.0, 20.0 0.0, 0.0 0.0))'::geometry".into(),
        xl: "'SRID=4329;POLYGON((-23.108 -21.261, -23.108 27.123, 28.588 27.123, 28.588 -21.261, -23.108 -21.261))'::geometry".into(),
    };
    let doc_bounds = Bounds {
        s: "'SRID=4329;POLYGON((390000 130000, 390000 140000, 390500 140000, 390500 130000, 390000 130000))'::geometry".into(),
        l: "'SRID=4329;POLYGON((390000 130000, 390000 140000, 400000 140000, 400000 130000, 390000 130000))'::geometry".into(),
        xl: "'SRID=4329;POLYGON((389400 124200, 389400 148200, 406200 148200, 406200 124200, 389400 124200))'::geometry".into(),
    };
    let ca13_bounds = Bounds {
        s: "'SRID=4329;POLYGON((695000 3917500, 695000 3920000, 696500 3920000, 696500 3917500, 695000 3917500))'::geometry".into(),
        l: "'SRID=4329;POLYGON((679000 3925000, 679000 3935000, 691000 3935000, 691000 3925000, 679000 3925000))'::geometry".into(),
        xl: "'SRID=4329;POLYGON((643431.76 3883547.565, 643431.76 3977026.735, 736910.93 3977026.735, 736910.93 3883547.565, 643431.76 3883547.565))'::geometry".into(),
    };

    //run_queries(&navvis_bounds, "navvis", 56.2, &client).await?;
    run_spatial_queries(&doc_bounds, "doc", DOC_NUM_MPOINTS, &client).await?;

    //run_spatial_queries(&ca13_bounds, "ca13", CA13_NUM_MPOINTS, &client).await?;

    //run_class_queries("doc", DOC_NUM_MPOINTS, &client).await?;
    //run_class_queries("ca13", CA13_NUM_MPOINTS, &client).await?;

    Ok(())
}
