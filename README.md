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

## Notes

This project contains a lot of experiment code for the geo query case-study paper. This includes two new file formats derived from the LAS format (LAST and LAZER), for which readers exist in the `readers` sub-project. Under `query/src/bin` are some executables which run experimental queries on different file formats, as well as on a PostGIS database with the `pgPointclouds` extension. 