#!/bin/bash
set -e
set -o pipefail

history_path=`pwd`
base_path=$(cd $(dirname $0)/..; pwd)

# python sdk package
cd $base_path/src/package_util/python/causal_inference
python3 setup.py clean --all
python3 setup.py sdist build
version=`cat setup.py |grep version|awk -F "[=']" '{print $3}'`
if [ -f "$base_path/lib/fast-causal-inference.tar.gz" ];then
  rm -f "$base_path/lib/fast-causal-inference.tar.gz"
fi
mv dist/fast-causal-inference-${version}.tar.gz $base_path/lib/fast-causal-inference.tar.gz

# sqlgateway
cd $base_path/src/sql_gateway/calcite/core
../gradlew clean
../gradlew assemble
if [ -e $base_path/src/sql_gateway/sqlgateway/src/main/resources/lib/calcite-core-1.36.0-SNAPSHOT.jar ]; then
    echo "The file exists."
    rm -f $base_path/src/sql_gateway/sqlgateway/src/main/resources/lib/calcite-core-1.36.0-SNAPSHOT.jar
fi
mv $base_path/src/sql_gateway/calcite/core/build/libs/calcite-core-1.36.0-SNAPSHOT.jar $base_path/src/sql_gateway/sqlgateway/src/main/resources/lib/
cd $base_path/src/sql_gateway/sqlgateway
mvn clean package
if [ -f "$base_path/lib/sqlgateway-0.0.1-SNAPSHOT.jar" ];then
  rm -f "$base_path/lib/sqlgateway-0.0.1-SNAPSHOT.jar"
fi
mv $base_path/src/sql_gateway/sqlgateway/target/sqlgateway-0.0.1-SNAPSHOT.jar $base_path/lib
cd $base_path