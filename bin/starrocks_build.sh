#!/bin/bash
set -e
set -o pipefail

history_path=`pwd`
base_path=$(cd $(dirname $0)/..; pwd)
cd $base_path
if [ ! -f "$base_path/contrib/starrocks/LICENSE" ];then
  echo "fetch contrib submodule"
  git submodule update --init --recursive contrib/starrocks
fi
cd contrib/starrocks
git checkout 3.1.1
cd $base_path
cp -rf $base_path/src/udf/starrocks/* $base_path/contrib/starrocks/
echo "build starrocks in docker"
DOCKER_BUILDKIT=1 docker build --rm=true --build-arg builder=starrocks/dev-env-ubuntu:3.1.1 -f docker/starrocks/build.Dockerfile -t artifacts-ubuntu:3.1.1 .
echo "build starrocks success"
echo "pack deploy docker"
DOCKER_BUILDKIT=1 docker build -f docker/starrocks/deploy.Dockerfile -t fastcausalinference/starrocks-server:latest .
cd ${history_path}
echo "build success"
