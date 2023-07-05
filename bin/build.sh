#!/bin/bash
set -e
history_path=`pwd`
base_path=$(cd $(dirname $0)/..; pwd)
cd $base_path
if [ ! -f "$base_path/contrib/ClickHouse/LICENSE" ];then
  echo "fetch contrib submodule"
  git submodule update --init --recursive
fi
cp $base_path/src/udf/ClickHouse/src/AggregateFunctions/* $base_path/contrib/ClickHouse/src/AggregateFunctions/
cd $base_path/contrib/ClickHouse/; mkdir -p build
cmake -S . -B build
cd build; ninja clickhouse
mv ./programs/clickhouse $base_path/clickhouse
cd ${history_path}
echo "build success"