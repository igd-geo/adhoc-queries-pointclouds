#!/bin/sh

# config for the experiment database running locally in docker
# DOCKER_POSTGRES_USER=postgres
# DOCKER_POSTGRES_PASSWORD=2XWVhtvi
# DOCKER_POSTGRES_HOST=0.0.0.0
# DOCKER_POSTGRES_PORT=16378
# DOCKER_POSTGRES_DB=experiment_base
# DOCKER_POSTGRES_SCHEMA=public

# config for the experiment database running together with the pointcloud database. Seems like I set up to DBs...
DOCKER_POSTGRES_DB=experiments
DOCKER_POSTGRES_HOST=0.0.0.0
DOCKER_POSTGRES_PORT=14587
DOCKER_POSTGRES_USER=postgres
DOCKER_POSTGRES_PASSWORD=test123
DOCKER_POSTGRES_SCHEMA=public

# config for the experiment database running on an OpenStack VM
OPENSTACK_POSTGRES_HOST=10.52.43.92
OPENSTACK_POSTGRES_PORT=5432
OPENSTACK_POSTGRES_USER=postgres
OPENSTACK_POSTGRES_PASSWORD=lvivwue4539ca
OPENSTACK_POSTGRES_DB=experiments
OPENSTACK_POSTGRES_SCHEMA=data

# config for the pgPointclouds database containing the point cloud data
PC_DBNAME=postgres
PC_HOST=0.0.0.0
PC_PORT=14587
PC_USER=postgres
PC_PASSWORD=test123

export PSQL_USER=$DOCKER_POSTGRES_USER
export PSQL_PWD=$DOCKER_POSTGRES_PASSWORD
export PSQL_HOST=$DOCKER_POSTGRES_HOST
export PSQL_PORT=$DOCKER_POSTGRES_PORT
export PSQL_DBNAME=$DOCKER_POSTGRES_DB
export PSQL_DBSCHEMA=$DOCKER_POSTGRES_SCHEMA
export PC_DBNAME=$PC_DBNAME
export PC_HOST=$PC_HOST
export PC_PORT=$PC_PORT
export PC_USER=$PC_USER
export PC_PASSWORD=$PC_PASSWORD
export RUST_BACKTRACE=1
export RUST_LOG=info
# export MACHINE=pc3018
# run locally just to test that queries are correct
# export EXAR_LOCAL=1

cd "${0%/*}" && cd ..
cargo build --release --bin run_postgis_queries &&
./target/release/run_postgis_queries $1