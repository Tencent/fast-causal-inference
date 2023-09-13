#!/bin/bash
set -e
set -o pipefail

base_path=$(
  cd "$(dirname "$0")/.."
  pwd
)
cd ${base_path}

init_dir() {
  if [ ! -d "clickhouse/data" ]; then
    mkdir -p clickhouse/data
  fi

  if [ ! -d "clickhouse/logs" ]; then
    mkdir -p clickhouse/logs
  fi

  if [ ! -d "clickhouse/initdb" ]; then
    mkdir -p clickhouse/initdb
    if [ -f "examples/test_data_small.sql" ]; then
      mv examples/test_data_small.sql clickhouse/initdb
    fi
  fi

  if [ ! -d "jupyter" ]; then
    mkdir -p jupyter
    if [ -f "examples/all_in_sql_demo.ipynb" ]; then
      mv examples/all_in_sql_demo.ipynb jupyter/
    fi
    if [ -f "conf/package_conf.yaml" ]; then
      mv conf/package_conf.yaml jupyter/
    fi
  fi

  if [ ! -d "mysql/initdb" ]; then
    mkdir -p mysql/initdb
    if [ -f "examples/schema.sql" ]; then
      mv examples/schema.sql mysql/initdb
    fi
  fi
}

deploy() {
  docker-compose up -d
}

init_dir
deploy
