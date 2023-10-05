#!/bin/sh

EX_DATA_ROOT_DIR=$1

cd "${0%/*}" && cd ..
cargo build --release --bin reader_performance

sudo ./target/release/reader_performance --purge-cache $EX_DATA_ROOT_DIR/doc/las/1321.las &&
sudo ./target/release/reader_performance --purge-cache $EX_DATA_ROOT_DIR/doc/last/1321.last &&
sudo ./target/release/reader_performance --purge-cache $EX_DATA_ROOT_DIR/doc/laz/1321.laz &&
sudo ./target/release/reader_performance --purge-cache $EX_DATA_ROOT_DIR/doc/lazer/1321.lazer &&
sudo ./target/release/reader_performance $EX_DATA_ROOT_DIR/doc/las/1321.las &&
sudo ./target/release/reader_performance $EX_DATA_ROOT_DIR/doc/last/1321.last &&
sudo ./target/release/reader_performance $EX_DATA_ROOT_DIR/doc/laz/1321.laz &&
sudo ./target/release/reader_performance $EX_DATA_ROOT_DIR/doc/lazer/1321.lazer