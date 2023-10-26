#!/bin/sh

# config for the experiment database running on an OpenStack VM
OPENSTACK_POSTGRES_HOST=10.52.43.92
OPENSTACK_POSTGRES_PORT=5432
OPENSTACK_POSTGRES_USER=postgres
OPENSTACK_POSTGRES_PASSWORD=lvivwue4539ca
OPENSTACK_POSTGRES_DB=experiments
OPENSTACK_POSTGRES_SCHEMA=data

MACBOOK_POSTGRES_DB=experiments
MACBOOK_POSTGRES_HOST=0.0.0.0
MACBOOK_POSTGRES_PORT=14587
MACBOOK_POSTGRES_USER=postgres
MACBOOK_POSTGRES_PASSWORD=test123
MACBOOK_POSTGRES_SCHEMA=public

export PSQL_USER=$MACBOOK_POSTGRES_USER
export PSQL_PWD=$MACBOOK_POSTGRES_PASSWORD
export PSQL_HOST=$MACBOOK_POSTGRES_HOST
export PSQL_PORT=$MACBOOK_POSTGRES_PORT
export PSQL_DBNAME=$MACBOOK_POSTGRES_DB
export PSQL_DBSCHEMA=$MACBOOK_POSTGRES_SCHEMA
export RUST_BACKTRACE=1
export RUST_LOG=info
# `mmap` is delicate. It is definitely slower than `read` in most cases on macOS, but has the benefit of being lock-free
# whereas the file-based readers use locking. For single files, this is super slow of course, as it kills all parallelism
# so sometimes it makes sense to force the usage of `mmap`
# export FORCE_MMAP=1

cd "${0%/*}" && cd ..
cargo build --release --bin adaptive_indexing_experiment
sudo -E ./target/release/adaptive_indexing_experiment > /dev/null