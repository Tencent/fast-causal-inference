#!/bin/bash
set -e
set -o pipefail

history_path=`pwd`
base_path=$(cd $(dirname $0)/..; pwd)
cd $base_path
if [ ! -f "$base_path/contrib/ClickHouse/LICENSE" ];then
  echo "fetch contrib submodule"
  git submodule update --init --recursive
fi
cp -f $base_path/src/udf/ClickHouse/src/AggregateFunctions/* $base_path/contrib/ClickHouse/src/AggregateFunctions/
cd $base_path/contrib/ClickHouse/; mkdir -p build
export CC=clang-16
export CXX=clang++-16
cmake -S . -B build
cd build; ninja clickhouse
rm -f clickhouse
mv ./programs/clickhouse $base_path/clickhouse
cd ${history_path}
echo "build success"
# select * from system.functions where is_aggregate=1;