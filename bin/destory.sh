#!/bin/bash
set -e
set -o pipefail

base_path=$(
  cd "$(dirname "$0")/.."
  pwd
)
cd ${base_path}

# docker-compose down
rm -rf clickhouse
rm -rf jupyter
rm -rf mysql
rm -rf starrocks
docker rmi sqlgateway:1.0
docker rmi docker.io/jupyter/fastcausalinference-notebook:latest
