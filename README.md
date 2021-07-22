# Ad-hoc queries on point cloud data

Command line application for running ad-hoc queries on point cloud data. An ad-hoc query is a query on unindexed data, in the case of point clouds LAS or LAZ files. Using this application, a point cloud dataset of one or more files can be searched using a number of predefined query types, outputting the number of matching points or the matching data as an LAS file. 

## Usage

Requires a recent Rust installation (>= 1.51). Run using `cargo run --release`, which will show you the command line usage. As an example, here is a bounding-box query on an LAS dataset:
```
cargo run --release -- -i /path/to/dataset --bounds 10;10;10;20;20;20 --optimized --parallel
```

For uncompressed file formats, the option `--optimized` uses a faster algorithm, so you will want that option on most of the time. It does nothing for compressed files (LAZ, LAZER) though! If you have a dataset with multiple files, the `--parallel` option will run the query on as many threads as there are files, or logical cores, whichever is smaller. 

Bounding boxes are specified as a string of the form `minx;miny;minz;maxx;maxy;maxz` in the coordinate space of the point cloud.

Other query types are:
*  By object class, using the `--class` parameter. Object classes are specified according to the LAS specification. This query mode is mutually exclusive from the other query modes
*  With maximum density, using the `--density` parameter. A density of `X` means that at most one point per `X^3`m^3 will be returned from the query, using simple grid-center sampling. Max-density querying can be combined with bounding box querying

## Reproducing measurements from the paper "Executing ad-hoc queries on large geospatial data sets without acceleration structures"

This repository contains all relevant code to reproduce the experiments form the section "Querying point cloud data" of the paper "Executing ad-hoc queries on large geospatial data sets without acceleration structures". To run the experiments, the following prerequisites are required:

*  Download one or all of the reference datasets
    *  The `navvis` data can be obtained from [here](https://www.navvis.com/resources/specifications/navvis-m6-sample-data)
    *  The `doc` data can be obtained from [here](https://registry.opendata.aws/dc-lidar/). Use the 2018 dataset!
    *  The `ca13` data can be obtained from [here](http://opentopo.sdsc.edu/lidarDataset?opentopoID=OTLAS.032013.26910.2)

The experiments use different file formats, so next you have to convert the dataset(s) into the different file formats. You can use [this project](https://github.com/igd-geo/pointcloud-format-conversions) for the conversion. All experiments expect a single dataset to be available in the following formats: `LAS`, `LAZ`, `LAST` and `LAZER`. The linked project can convert from `LAS` or `LAZ` to any of the other formats. Make sure you end up with the following directory structure:

```
- dataset_name (e.g. doc)
    - LAS
        - file1.las
        - file2.las
        - ...
    - LAZ
        - file1.laz
        - file2.laz
        - ...
    - LAST
        - file1.last
        - ...
    - LAZER
        - file1.lazer
        - ...
```

You can then run the experiments on a dataset, or run custom queries. First, make sure you built this project in release mode:
```
cargo build --release
```

This gives you an executable called `query` under `target/release`. This executable can be used to run single queries on a dataset. Run `./query --help` to get information about the usage. A simple bounding box query on the `ca13` dataset would look like this:

```
query -i /path/to/ca13/LAST --bounds 665000;3910000;0;705000;3950000;480 --optimized --parallel
```

Running the query like this will print out the number of matching points that fall into the given bounding box. You can also write the matching points to disk like this:

```
query -i /path/to/ca13/LAST --bounds 665000;3910000;0;705000;3950000;480 --optimized --parallel -o /path/to/output/directory
```

Since the data is read from potentially multiple threads at once, multiple files might be written. 

### Running the query experiments on raw data

To run the experiments from the paper, you can use the `run_query_experiments` binary. It uses the `query` binary internally, so *make sure you have the `query` binary built!* Then simply execute:

```
cargo run --release --bin run_query_experiments --input /path/to/dataset --experiment ID
```

There are 5 possible experiments. Experiment IDs 1 to 3 run bounding box queries on the `navvis`, `doc` and `ca13` datasets, respectively. These correspond to experiments 1 and 2 in the paper, as they include both regular bounding box queries as well as max-density queries. Experiment IDs 4 and 5 run class queries on the `doc` and `ca13` datasets, respectively. These correspond to experiment 3 in the paper. **Make sure you call each experiment with the correct dataset!** So to run e.g. experiment 2 (bounds query on `doc`), you would call:

```
cargo run --release --bin run_query_experiments --input /path/to/doc --experiment 2
```

This will print out the runtimes of the `S`, `L` and `XL` bounding box queries, each time once without max-density and once with max-density. All experiments are run multiple times, as described in the paper. 

### Running the query experiments on a PostGIS database

For running the queries on a PostGIS database, as described in the paper, you first have to set up a PostGIS database with the `pgPointcloud` extension. An installation guide is available on the [pgPointcloud homepage](https://pgpointcloud.github.io/pointcloud/quickstart.html). Perhaps the simplest way is to download and run the Docker container: `docker pull pgpointcloud/pointcloud`. 

The experiments expect specific names for the tables that hold the test data. The `navvis` data has to be stored in a table named `navvis`, the `doc` data in a table named `doc` and the `ca13` data in a table named `ca13`. In the paper, we performed the upload using PDAL in the way described [here](https://pgpointcloud.github.io/pointcloud/quickstart.html#running-a-pipeline). After all data has been uploaded (which can take many hours!), make sure you create spatial indexes for each of the tables, by running the following SQL statement on the database:

```
CREATE INDEX spatial_idx on navvis using GIST (Geometry(pa));
```

By default, the patches column is named `pa`, but this might differ for you. 

Then you can execute the PostGIS queries like so:

```
cargo run --release --bin run_postgis_queries -- dbname NAME_OF_DB --host HOST_IP --pwd PASSWORD --user USER
```

Provide the appropriate credentials for your database and this should execute all queries as stated in the paper. 