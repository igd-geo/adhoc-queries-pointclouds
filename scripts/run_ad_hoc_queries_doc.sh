#!/bin/sh

# config for the experiment database running on an OpenStack VM
OPENSTACK_POSTGRES_HOST=10.52.43.92
OPENSTACK_POSTGRES_PORT=5432
OPENSTACK_POSTGRES_USER=postgres
OPENSTACK_POSTGRES_PASSWORD=lvivwue4539ca
OPENSTACK_POSTGRES_DB=experiments
OPENSTACK_POSTGRES_SCHEMA=data

export PSQL_USER=$OPENSTACK_POSTGRES_USER
export PSQL_PWD=$OPENSTACK_POSTGRES_PASSWORD
export PSQL_HOST=$OPENSTACK_POSTGRES_HOST
export PSQL_PORT=$OPENSTACK_POSTGRES_PORT
export PSQL_DBNAME=$OPENSTACK_POSTGRES_DB
export PSQL_DBSCHEMA=$OPENSTACK_POSTGRES_SCHEMA
export RUST_BACKTRACE=1
export RUST_LOG=info

cd "${0%/*}" && cd ..
cargo build --release --bin query_experiments_wip
# args: data path (experiment data root), shapefile path
sudo ./target/release/query_experiments_wip $1 $2 > /dev/null