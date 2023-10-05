#!/bin/sh

EX_DATA_ROOT_DIR=$1

cd "${0%/*}" && cd ..
cargo build --release --bin optimized_reader_performance

sudo ./target/release/optimized_reader_performance --purge-cache $EX_DATA_ROOT_DIR/doc/las/1321.las &&
sudo ./target/release/optimized_reader_performance --purge-cache $EX_DATA_ROOT_DIR/doc/last/1321.last &&
sudo ./target/release/optimized_reader_performance $EX_DATA_ROOT_DIR/doc/las/1321.las &&
sudo ./target/release/optimized_reader_performance $EX_DATA_ROOT_DIR/doc/last/1321.last