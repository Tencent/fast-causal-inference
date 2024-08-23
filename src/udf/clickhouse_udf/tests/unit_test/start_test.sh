client=~/workspace/mmdcch/build/programs/clickhouse-client

$client -nm < unit_test_tbl.sql
# input test data
$client --query "INSERT INTO default.causal_inference_test FORMAT CSV" < unit_test_data

#$client -nm < unit_test_udf.sql

udf_test="udf_test" 

sql_suffix=".sql"

reference_suffix=".reference"

for sql_file in $udf_test/*$sql_suffix; do
    file_name=${sql_file%$sql_suffix}
    ref_file="$file_name.reference"
    if [ -f "$ref_file" ]; then
        $client -nm < "$sql_file" > $file_name".result"
        diff $file_name".result" $ref_file
        if [ $? -eq 0 ]; then
            echo  "\033[32mTest $file_name passed\033[0m"
        else
            echo  "\033[31mTest $file_name failed\033[0m"
        fi
        rm $file_name".result"
    fi
done
