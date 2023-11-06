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
    if [ ! -f "clickhouse/initdb/test_data_small.sql" ]; then
      cp examples/test_data_small.sql clickhouse/initdb/
    fi
  fi

  if [ ! -d "jupyter" ]; then
    mkdir -p jupyter
    if [ -f "examples/demo.ipynb" ]; then
      cp examples/demo.ipynb jupyter/
      chmod u-w jupyter/demo.ipynb
    fi
    if [ ! -d "jupyter/.jupyter" ]; then
      mkdir -p jupyter/.jupyter
    fi
    if [ ! -f "jupyter/.jupyter/jupyter_server_config.py" ]; then
      cp conf/jupyter_server_config.py jupyter/.jupyter/
    fi
    if [ ! -d "jupyter/.jupyter/custom" ]; then
      mkdir -p jupyter/.jupyter/custom
    fi
    if [ ! -f "jupyter/.jupyter/custom/custom.js" ]; then
      cp conf/jupyter_custom.js jupyter/.jupyter/custom/custom.js
    fi
    if [ -f "conf/package_conf.yaml" ]; then
      cp conf/package_conf.yaml jupyter/.jupyter/conf.yaml
    fi
  fi

  if [ ! -d "mysql/initdb" ]; then
    mkdir -p mysql/initdb
    if [ -f "examples/schema.sql" ]; then
      cp examples/schema.sql mysql/initdb
    fi
  fi
}

deploy() {
  docker-compose up -d
}

init_dir
deploy
