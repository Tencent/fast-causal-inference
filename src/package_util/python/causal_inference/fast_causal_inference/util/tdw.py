import os
import time
from concurrent.futures import (
    ThreadPoolExecutor,
    wait,
    ALL_COMPLETED,
    as_completed,
)
from fast_causal_inference.common import handle_exception
from fast_causal_inference.common import get_context

from .clickhouse_utils import ClickHouseUtils
from .starrocks_utils import StarRocksUtils



class TDWUtils(object):
    DEFAULT_TDW_USER = "guest"
    DEFAULT_TDW_PASSWORD = "guest"
    MAX_FILE_COUNT = 3600
    MAX_LENGTH = 50  # GB

    def __init__(self, spark=None):
        from pyspark.sql import SparkSession

        self.name_space = get_context().project_conf["tdw"]["hdfs_namespace"]
        self.base_hdfs_path = get_context().project_conf["tdw"]["hdfs_base_path"]

        if spark:
            self.spark = spark
        else:
            self.spark = (
                SparkSession.builder.enableHiveSupport()
                .config("spark.driver.maxResultSize", "0")
                .config("spark.speculation", "true")
                .config("spark.blacklist.enabled", "true")
                .config("spark.sql.broadcastTimeout", "800")
                .config("spark.task.maxFailures", "7")
                .config("dfs.ha.zkfc.nn.http.timeout.ms", "66000")
                .getOrCreate()
                # .config("spark.driver.memory", "2g")
                # .config("spark.executor.cores", 2)
                # .config("spark.executor.memory", "4g")
            )
        if os.getenv("JUPYTERHUB_USER"):
            TDWUtils.DEFAULT_TDW_USER = "tdw_" + os.getenv("JUPYTERHUB_USER")
        else:
            TDWUtils.DEFAULT_TDW_USER = "tdw_" + os.getenv("USER")
        TDWUtils.DEFAULT_TDW_PASSWORD = os.getenv("TDW_PASSWORD")

    def get_tdw_provider(self, db):
        from pytoolkit import TDWSQLProvider

        return TDWSQLProvider(
            self.spark,
            user=TDWUtils.DEFAULT_TDW_USER,
            passwd=TDWUtils.DEFAULT_TDW_PASSWORD,
            group="tl",
            db=db,
        )

    def get_tdw_util(self, db):
        from pytoolkit import TDWUtil

        return TDWUtil(TDWUtils.DEFAULT_TDW_USER, TDWUtils.DEFAULT_TDW_PASSWORD, db)

    def get_table_meta(self, tdw_database_name, tdw_table_name):
        tdw_util = self.get_tdw_util(tdw_database_name)
        tblInfo = tdw_util.getTableInfo(tdw_table_name)
        partitions = list()
        for partition_cell in tblInfo.partitions:
            partitions.append(partition_cell.name)
        get_context().logger.debug(
            tdw_database_name
            + "."
            + tdw_table_name
            + ",colNames="
            + str(tblInfo.colNames)
            + ",colTypes="
            + str(tblInfo.colTypes)
            + ",partitions="
            + str(partitions)
        )
        return tblInfo.colNames, tblInfo.colTypes, partitions

    @classmethod
    def field_type_map(self, col_type, olap="clickhouse"):
        if olap.lower() == "clickhouse":
            if col_type == "bigint":
                col_ch_type = "Int64"
            elif col_type == "int":
                col_ch_type = "Int32"
            elif col_type == "double":
                col_ch_type = "Float64"
            elif col_type == "float":
                col_ch_type = "Float32"
            elif col_type == "string":
                col_ch_type = "String"
            elif col_type == "decimal(20,0)":
                col_ch_type = "UInt32"
            elif col_type == "date":
                col_ch_type = "Date"
            elif col_type == "timestamp":
                col_ch_type = "DateTime"
            else:
                raise Exception(col_type + " col_type is not support")
            return col_ch_type
        elif olap.lower() == "starrocks":
            if col_type == "bigint":
                col_ch_type = "bigint"
            elif col_type == "int":
                col_ch_type = "int"
            elif col_type == "double":
                col_ch_type = "double"
            elif col_type == "float":
                col_ch_type = "double"
            elif col_type == "string":
                col_ch_type = "varchar"
            elif col_type == "decimal(20,0)":
                col_ch_type = "bigint"
            elif col_type == "date":
                col_ch_type = "Date"
            elif col_type == "timestamp":
                col_ch_type = "DateTime"
            else:
                raise Exception(col_type + " col_type is not support")
            return col_ch_type
        else:
            raise Exception(f"Unsupported olap engine `{olap}`")

    @classmethod
    def get_sql_statement(self, col_names, col_types, olap="clickhouse"):
        colname_statement = ""
        col_statement = ""
        col_nullable_statement = ""
        col_if_statement = ""
        for col_index in range(len(col_names)):
            col_name = col_names[col_index]
            col_type = col_types[col_index]
            col_ch_type = self.field_type_map(col_type, olap)
            get_context().logger.debug(
                "col_name="
                + col_name
                + ",col_type="
                + col_type
                + ",col_ch_type="
                + col_ch_type
            )
            colname_statement += col_name + ","
            col_statement += col_name + " " + col_ch_type + ","
            col_nullable_statement += col_name + " Nullable(" + col_ch_type + "),"
            if col_ch_type == "String":
                col_if_statement += (
                    "if ("
                    + col_name
                    + " is null,'null',"
                    + col_name
                    + ") as "
                    + col_name
                    + ","
                )
            else:
                col_if_statement += (
                    "if ("
                    + col_name
                    + " is null, 0, "
                    + col_name
                    + ") as "
                    + col_name
                    + ","
                )
        colname_statement = colname_statement[:-1]
        col_statement = col_statement[:-1]
        col_nullable_statement = col_nullable_statement[:-1]
        col_if_statement = col_if_statement[:-1]
        get_context().logger.debug("colname_statement=" + colname_statement)
        get_context().logger.debug("col_statement=" + col_statement)
        get_context().logger.debug("col_nullable_statement=" + col_nullable_statement)
        get_context().logger.debug("col_if_statement=" + col_if_statement)
        return (
            colname_statement,
            col_statement,
            col_nullable_statement,
            col_if_statement,
        )

    @classmethod
    def get_hdfs_location(
        self, base_location, tdw_partition_list=None, postfix=None, table_format="orc"
    ):
        table_format = table_format.lower()
        if table_format == "text":
            table_format = ""
        if postfix:
            postfix = postfix + "*." + table_format + "*"
        else:
            postfix = "*." + table_format + "*"
        if tdw_partition_list:
            if tdw_partition_list.__len__() > 1:
                location = (
                    base_location + "/{" + ",".join(tdw_partition_list) + "}/" + postfix
                )
            else:
                location = base_location + "/" + tdw_partition_list[0] + "/" + postfix
        else:
            location = base_location + "/" + postfix
        get_context().logger.debug("location=" + location)
        return location

    # insert one node
    @classmethod
    def tdw_2_clickhouse(
        self,
        tdw_database_name,
        tdw_table_name,
        clickhouse_table_name,
        spark,
        cmk=None,
        tdw_partition_list=None,
        is_auto_create=True,
    ):
        from fast_causal_inference.util import get_user

        user = get_user()
        if not cmk:
            raise Exception("please input cmk arg")
        get_context().logger.debug("user=" + user + ",cmk=" + cmk)
        get_context().logger.debug("done thive/hive meta data")
        # check partition input
        if tdw_partition_list:
            for tdw_partition in tdw_partition_list:
                if tdw_partition not in partitions:
                    raise Exception(
                        tdw_partition
                        + " not exist partition in tdw table "
                        + tdw_table_name
                    )
        file_count, length = TDWUtils.hdfs_summary(spark, location)
        if (
            file_count > TDWUtils.MAX_FILE_COUNT
            or length > TDWUtils.MAX_LENGTH * 1000 * 1000 * 1000
        ):
            raise Exception(
                "tdw table too big, file_count>"
                + str(TDWUtils.MAX_FILE_COUNT)
                + " or  length > "
                + str(TDWUtils.MAX_LENGTH)
                + ", file_count="
                + str(file_count)
                + ",length="
                + str(length)
            )
        (
            colname_statement,
            col_statement,
            col_nullable_statement,
            col_if_statement,
        ) = self.get_sql_statement(col_names, col_types)
        clickhouse_utils = ClickHouseUtils()
        if clickhouse_table_name not in clickhouse_utils.show_tables():
            if is_auto_create:
                get_context().logger.debug("auto create table")
                clickhouse_utils.create_table(clickhouse_table_name, col_statement)
            else:
                raise Exception("clickhouse table is not exist")
        else:
            get_context().logger.debug("clickhouse table is exist")
        # create hdfs external table
        external_table_name = clickhouse_table_name + "_hdfs_" + str(int(time.time()))
        clickhouse_utils.create_table(
            table_name=external_table_name,
            col_statement=col_nullable_statement,
            type="hdfs",
            location=self.get_hdfs_location(
                location, tdw_partition_list, "part-", table_format
            ),
            format=table_format,
        )
        get_context().logger.debug(
            "tdw rows = " + str(clickhouse_utils.table_rows(external_table_name))
        )
        clickhouse_utils.insert_table(
            clickhouse_table_name,
            external_table_name,
            colname_statement,
            col_if_statement,
        )
        clickhouse_utils.close()

    @classmethod
    def get_metainfo(
        self, tdw_database_name, tdw_table_name, tdw_partition_list, spark, cmk=None
    ):

        user = get_user()
        if not cmk:
            raise Exception("please input cmk arg")
        get_context().logger.debug("user=" + user + ",cmk=" + cmk)
        get_context().logger.debug(
            "col_names="
            + str(col_names)
            + ",col_types="
            + str(col_types)
            + ",partitions="
            + str(partitions)
            + "location="
            + str(location)
            + "table_format="
            + str(table_format)
        )
        get_context().logger.debug("done thive/hive meta data")

        total_file_count = 0
        total_length = 0
        # check partition input
        if tdw_partition_list:
            for tdw_partition in tdw_partition_list:
                if tdw_partition not in partitions:
                    raise Exception(
                        tdw_partition
                        + " not exist partition in tdw table "
                        + tdw_table_name
                    )
                else:
                    file_count, length = TDWUtils.hdfs_summary(
                        spark, location + "/" + tdw_partition
                    )
                    get_context().logger.debug(
                        "tdw_partition="
                        + tdw_partition
                        + "file_count="
                        + str(file_count)
                        + ",length="
                        + str(length)
                    )
                    total_file_count += file_count
                    total_length += length
        else:
            file_count, length = TDWUtils.hdfs_summary(spark, location)
            get_context().logger.debug(
                "file_count=" + str(file_count) + ",length=" + str(length)
            )
            total_file_count += file_count
            total_length += length

        if (
            total_file_count > TDWUtils.MAX_FILE_COUNT
            or total_length > TDWUtils.MAX_LENGTH * 1000 * 1000 * 1000
        ):
            raise Exception(
                "tdw table too big, total_file_count>"
                + str(TDWUtils.MAX_FILE_COUNT)
                + " or total_length > "
                + str(TDWUtils.MAX_LENGTH)
                + ", total_file_count="
                + str(total_file_count)
                + ",total_length="
                + str(total_length)
            )
        return col_names, col_types, location, table_format

    # insert everyone node
    @classmethod
    def tdw_2_clickhouse_distribute(
        self,
        tdw_database_name,
        tdw_table_name,
        clickhouse_table_name,
        spark,
        cmk=None,
        tdw_partition_list=None,
        clickhouse_partition_column=None,
        clickhouse_primary_column=None,
        is_auto_create=True,
        location=None,
        table_format=None,
        col_statement_sql=None,
        sql=None,
    ):
        if not location and not table_format:
            col_names, col_types, location, table_format = self.get_metainfo(
                tdw_database_name, tdw_table_name, tdw_partition_list, spark, cmk
            )
            (
                colname_statement,
                col_statement,
                col_nullable_statement,
                col_if_statement,
            ) = self.get_sql_statement(col_names, col_types)
        else:
            (
                colname_statement,
                col_statement,
                col_nullable_statement,
                col_if_statement,
            ) = (None, col_statement_sql, col_statement_sql, None)

        clickhouse_utils = ClickHouseUtils()
        prefix = "part-??"
        hdfs_path_list = list()
        for i in range(clickhouse_utils.cluster_hosts_len):
            hdfs_path_list.append(list())
        for i in range(1000):
            if i < 10:
                hdfs_path_list[i % clickhouse_utils.cluster_hosts_len].append(
                    "00" + str(i)
                )
            elif i < 100:
                hdfs_path_list[i % clickhouse_utils.cluster_hosts_len].append(
                    "0" + str(i)
                )
            else:
                hdfs_path_list[i % clickhouse_utils.cluster_hosts_len].append(str(i))
        hdfs_path = list()
        for i in hdfs_path_list:
            hdfs_path.append(prefix + "{" + ",".join(i) + "}")
        get_context().logger.debug(
            "hdfs_path_list=" + str(hdfs_path_list) + ",hdfs_path=" + str(hdfs_path)
        )
        get_context().logger.debug("running, please wait")

        if clickhouse_table_name not in clickhouse_utils.show_tables():
            if is_auto_create:
                get_context().logger.debug("auto create table")
                clickhouse_utils.create_table(
                    clickhouse_table_name,
                    col_statement,
                    cluster=clickhouse_utils.CLUSTER,
                    partition_column=clickhouse_partition_column,
                    primary_column=clickhouse_primary_column,
                )
            else:
                raise Exception("clickhouse table is not exist")
        else:
            get_context().logger.debug("clickhouse table is exist")

        def clickhouse_node_job(
            clickhouse_utils,
            clickhouse_table_name,
            external_table_name,
            colname_statement,
            col_if_statement,
            col_nullable_statement,
            postfix,
            base_location,
            table_format,
            sql,
        ):
            # create hdfs external table
            location = TDWUtils.get_hdfs_location(
                base_location, tdw_partition_list, postfix, table_format
            )
            get_context().logger.debug(
                str(clickhouse_utils.host) + ", location=" + str(location)
            )
            clickhouse_utils.create_table(
                table_name=external_table_name,
                col_statement=col_nullable_statement,
                type="hdfs",
                location=location,
                format=table_format,
            )
            get_context().logger.debug(
                str(clickhouse_utils.host)
                + ", external_table_name="
                + external_table_name
                + ", location="
                + str(location)
                + ",tdw rows = "
                + str(clickhouse_utils.table_rows(external_table_name))
            )
            if sql:
                start = time.perf_counter()
                get_context().logger.debug("insert into sql=" + str(sql))
                clickhouse_utils.execute(sql)
                end = time.perf_counter()
                get_context().logger.debug(
                    "insert into sql done time cost: {0} Seconds".format(
                        str(end - start)
                    )
                )
            else:
                clickhouse_utils.insert_table(
                    clickhouse_table_name,
                    external_table_name,
                    colname_statement,
                    col_if_statement,
                )
            get_context().logger.debug(
                str(clickhouse_utils.host)
                + ", insert done, table_name="
                + clickhouse_table_name
                + ", location="
                + str(location)
                + ",tdw rows = "
                + str(clickhouse_utils.table_rows(clickhouse_table_name))
            )
            clickhouse_utils.close()

        # 线程提交后都会执行结束
        with ThreadPoolExecutor(max_workers=min(16, clickhouse_utils.cluster_hosts_len)) as pool:
            all_task = list()
            for i in range(clickhouse_utils.cluster_hosts_len):
                clickhouse_utils = ClickHouseUtils(
                    host=clickhouse_utils.cluster_hosts[i]
                )
                external_table_name = clickhouse_table_name + "_hdfs_local"
                future = pool.submit(
                    clickhouse_node_job,
                    clickhouse_utils,
                    clickhouse_table_name + "_local",
                    external_table_name,
                    colname_statement,
                    col_if_statement,
                    col_nullable_statement,
                    hdfs_path[i],
                    location,
                    table_format,
                    sql,
                )
                future.add_done_callback(handle_exception)
                all_task.append(future)
            for future in as_completed(all_task, timeout=30 * 60):
                if future.exception():
                    raise Exception("子进程异常")
            # wait(all_task, timeout= 30 * 60, return_when=FIRST_EXCEPTION)  # 当任何 future 完成时返回, 如果未来没有引发异常, 那么它相当于 ALL_COMPLETED
            # for subtask in all_task:
            #     if subtask.exception():
            #         raise Exception("子进程异常")
        table_row = clickhouse_utils.table_rows(clickhouse_table_name)
        if table_row > 0:
            # 插入分布式副本表 表同步可能会有延迟, 导致table row变小
            get_context().logger.debug(
                "write clickhouse success, table rows number =" + str(table_row)
            )
        else:
            get_context().logger.debug(
                "write clickhouse result is empty, please check source tdw table whether or not empty,"
                " or call Xhelper"
            )
        clickhouse_utils.close()

    @classmethod
    def create_table_by_dataframe(
        self,
        dataframe,
        table_name,
        partition_column,
        primary_column,
        is_auto_create,
        mode,
        is_cluster,
        olap="clickhouse",
    ):
        get_context().logger.debug(dataframe.dtypes)
        col_names = list()
        col_types = list()
        for row in dataframe.dtypes:
            col_name = row[0]
            col_names.append(col_name)
            col_type = row[1]
            col_types.append(col_type)

        (
            colname_statement,
            col_statement,
            col_nullable_statement,
            col_if_statement,
        ) = self.get_sql_statement(col_names, col_types, olap)
        if olap.lower() == "clickhouse":
            clickhouse_utils = ClickHouseUtils()
            if table_name not in clickhouse_utils.show_tables():
                if is_auto_create:
                    get_context().logger.info("clickhouse table auto create table")
                    if is_cluster:
                        clickhouse_utils.create_table(
                            table_name,
                            col_statement,
                            cluster=clickhouse_utils.CLUSTER,
                            partition_column=partition_column,
                            primary_column=primary_column,
                        )
                    else:
                        clickhouse_utils.create_table(
                            table_name,
                            col_statement,
                            partition_column=partition_column,
                            primary_column=primary_column,
                        )
                else:
                    raise Exception("clickhouse table is not exist")
            else:
                get_context().logger.debug(
                    "clickhouse table is exist, " + mode + " data"
                )
        elif olap.lower() == "starrocks":
            starrocks = StarRocksUtils()
            if table_name not in starrocks.show_tables():
                if is_auto_create:
                    get_context().logger.info("starrocks table auto create table")
                    starrocks.create_table(
                        table_name,
                        col_statement,
                        primary_column=primary_column,
                    )
                else:
                    raise Exception("starrocks table is not exist")
            else:
                get_context().logger.debug(
                    "starrocks table is exist, " + mode + " data"
                )

    # multi partition write only one node
    @classmethod
    def datafame_2_clickhouse(
        self,
        dataframe,
        clickhouse_table_name,
        clickhouse_partition_column,
        mode="append",
        is_auto_create=True,
        num_partitions=5,
        batch_size=100000,
    ):
        num = dataframe.count()
        get_context().logger.info("dataframe count=" + str(num))
        if num > ClickHouseUtils.MAX_ROWS:
            raise Exception(
                "dataframe table rows num too big, >"
                + str(ClickHouseUtils.MAX_ROWS)
                + " not support"
            )
        self.create_table_by_dataframe(
            dataframe,
            clickhouse_table_name,
            clickhouse_partition_column,
            None,
            is_auto_create,
            mode,
            is_cluster=False,
        )
        dataframe.write.mode(mode).option("batchsize", str(batch_size)).option(
            "numPartitions", str(num_partitions)
        ).jdbc(
            url=ClickHouseUtils().get_jdbc_connect_string(),
            table=clickhouse_table_name,
            properties=ClickHouseUtils().get_jdbc_properties(),
        )

    # multi partition write everyone node, as distribute clickhouse table
    @classmethod
    def datafame_2_clickhouse_distribute(
        self,
        dataframe,
        clickhouse_table_name,
        clickhouse_partition_column,
        clickhouse_primary_column,
        mode="append",
        is_auto_create=True,
        num_partitions=5,
        batch_size=100000,
    ):
        from pyspark.storagelevel import StorageLevel
        from pyspark.sql import functions
        from pyspark.sql import types

        dataframe.persist(StorageLevel.MEMORY_AND_DISK)
        num = dataframe.count()
        get_context().logger.info("dataframe count=" + str(num))
        if num > ClickHouseUtils.MAX_ROWS:
            raise Exception(
                "dataframe table rows num too big, >"
                + str(ClickHouseUtils.MAX_ROWS)
                + " not support"
            )
        self.create_table_by_dataframe(
            dataframe,
            clickhouse_table_name,
            clickhouse_partition_column,
            clickhouse_primary_column,
            is_auto_create,
            mode,
            is_cluster=True,
        )
        clickhouse_utils = ClickHouseUtils()
        cluster_hosts_len = clickhouse_utils.cluster_hosts_len
        cluster_hosts = clickhouse_utils.cluster_hosts
        clickhouse_utils.close()
        if clickhouse_primary_column:
            def calc_bucket_id(uin):
                return hash(str(uin)) % cluster_hosts_len

            calc_bucket_id_udf = functions.udf(calc_bucket_id, types.IntegerType())
            bucket_dataframe = dataframe.withColumn(
                "bucket_key", calc_bucket_id_udf(clickhouse_primary_column)
            )
            # dataframe = dataframe.withColumn("bucket_key", functions.col(bucket_column) % 40)
            split_datasets = list()
            for i in range(cluster_hosts_len):
                dataset = bucket_dataframe.where(functions.col("bucket_key") == i)
                split_datasets.append(dataset)
        else:
            weights = list()
            for i in range(cluster_hosts_len):
                weights.append(1.0)
            # split dataframe
            split_datasets = dataframe.randomSplit(weights)

        insert_counts = list()

        def dataframe_cache(dataset):
            dataset.persist()
            insert_counts.append(dataset.count())

        def dataframe_write_node(raw_dataset, url):
            if "bucket_key" in raw_dataset.schema.names:
                dataset = raw_dataset.drop("bucket_key")
            else:
                dataset = raw_dataset
            dataset.write.mode(mode).option("batchsize", str(batch_size)).option(
                "numPartitions", str(num_partitions)
            ).jdbc(
                url=url,
                table=clickhouse_table_name + "_local",
                properties=ClickHouseUtils().get_jdbc_properties(),
            )
            raw_dataset.unpersist()

        all_task = list()
        with ThreadPoolExecutor(max_workers=min(16, cluster_hosts_len)) as pool:
            for dataset in split_datasets:
                future = pool.submit(dataframe_cache, dataset)
                future.add_done_callback(handle_exception)
                all_task.append(future)
            wait(all_task, timeout=None, return_when=ALL_COMPLETED)
            get_context().logger.debug("insert counts list=" + str(insert_counts))
            for insert_count in insert_counts:
                if insert_count > ClickHouseUtils.MAX_ROWS / cluster_hosts_len:
                    raise Exception("count is too large")
            dataframe.unpersist()
            all_task.clear()
            get_context().logger.debug("all dataset cached")
            num = 0
            for dataset in split_datasets:
                url = (
                    "jdbc:clickhouse://"
                    + cluster_hosts[num]
                    + ":"
                    + str(clickhouse_utils.DEFAULT_HTTP_PORT)
                    + "/"
                    + clickhouse_utils.DEFAULT_DATABASE
                    + ClickHouseUtils.JDBC_ARGS
                )
                future = pool.submit(dataframe_write_node, dataset, url)
                future.add_done_callback(handle_exception)
                all_task.append(future)
                num = num + 1
                # print("url=", url, ",num=", num)
            wait(all_task, timeout=None, return_when=ALL_COMPLETED)

    @classmethod
    def datafame_2_starrocks(
        self,
        dataframe,
        starrocks_table_name,
        starrocks_partition_column,
        starrocks_primary_column,
        mode="append",
        is_auto_create=True,
        batch_size=500000,
    ):
        from pyspark.sql.functions import monotonically_increasing_id
        from pyspark.sql.functions import lit
        from datetime import datetime

        # 获取当前日期和时间，不包含微秒
        current_datetime = datetime.now().replace(microsecond=0)

        num = dataframe.count()
        get_context().logger.info(f"dataframe count={num}")
        if num > StarRocksUtils.MAX_ROWS:
            raise Exception(
                "dataframe table rows num too big, >"
                + str(StarRocksUtils.MAX_ROWS)
                + " not support"
            )
        self.create_table_by_dataframe(
            dataframe,
            starrocks_table_name,
            starrocks_partition_column,
            starrocks_primary_column,
            is_auto_create,
            mode,
            is_cluster=False,
            olap="starrocks",
        )
        sr_utils = StarRocksUtils.get_default_sr_utils()
        finalize_df = dataframe.withColumn(
            "tmptbl_id_col_", monotonically_increasing_id()
        ).withColumn("tmptbl_day_col_", lit(str(current_datetime)))
        finalize_df.write.format("starrocks").option(
            "starrocks.fenodes", f"{sr_utils.DEFAULT_HOST}:{sr_utils.DEFAULT_HTTP_PORT}"
        ).option("user", sr_utils.DEFAULT_USER).option(
            "password", sr_utils.DEFAULT_PASSWORD
        ).option(
            "starrocks.table.identifier",
            f"{sr_utils.DEFAULT_DATABASE}.{starrocks_table_name}",
        ).option(
            "starrocks.batch.size", batch_size
        ).mode(
            "append"
        ).save()

    def select_limit(self, tdw_database_name, tdw_table_name, pri_parts):
        tdw = self.get_tdw_provider(tdw_database_name)
        df = tdw.table(tdw_table_name, priParts=pri_parts)
        df.createOrReplaceTempView(tdw_table_name + "_view")
        df2 = self.spark.sql(
            """
            select
                  *
            from
                  %s
            limit 10
            """
            % (tdw_table_name + "_view")
        )
        return df2

    def create_table(self, tdw_database_name, tdw_table_name, col_list):
        tdw_util = self.get_tdw_util(tdw_database_name)
        from pytoolkit import TableDesc

        table_desc = (
            TableDesc()
            .setTblName(tdw_table_name)
            .setCols(col_list)
            .setComment(tdw_table_name)
            .setCompress(False)
            .setFieldDelimiter(",")
            .setFileFormat("textfile")
        )
        tdw_util.createTable(table_desc)

    def table_exits(self, tdw_database_name, tdw_table_name):
        tdw_util = self.get_tdw_util(tdw_database_name)
        return tdw_util.tableExist(tdw_table_name)

    def hdfs_move(self, raw_dfs_path, target_dfs_path):
        jFileSystemClass = (
            self.spark.sparkContext._gateway.jvm.org.apache.hadoop.fs.FileSystem
        )
        hadoop_configuration = self.spark.sparkContext._jsc.hadoopConfiguration()
        # change defaultFS
        hadoop_configuration.set("fs.defaultFS", self.name_space)
        dfs_file_system = jFileSystemClass.get(hadoop_configuration)
        jPathClass = self.spark.sparkContext._gateway.jvm.org.apache.hadoop.fs.Path
        raw_path = jPathClass(raw_dfs_path)
        target_path = jPathClass(target_dfs_path)
        dfs_file_system.rename(raw_path, target_path)
        # dfs_file_system.exists(del_path_obj)
        # dfs_file_system.delete(del_path_obj)

    def hdfs_mkdir_and_chmod(self, dfs_path):
        jFileSystemClass = (
            self.spark.sparkContext._gateway.jvm.org.apache.hadoop.fs.FileSystem
        )
        hadoop_configuration = self.spark.sparkContext._jsc.hadoopConfiguration()
        # change defaultFS
        hadoop_configuration.set("fs.defaultFS", self.name_space)
        dfs_file_system = jFileSystemClass.get(hadoop_configuration)
        jPathClass = self.spark.sparkContext._gateway.jvm.org.apache.hadoop.fs.Path
        jdfs_path = jPathClass(dfs_path)
        dfs_file_system.mkdirs(jdfs_path)
        # jShortClass = self.spark.sparkContext._gateway.jvm.java.lang.Short
        # jFileUtilClass = self.spark.sparkContext._gateway.jvm.org.apache.hadoop.fs.FileUtil
        # jFsPermissionClass = self.spark.sparkContext._gateway.jvm.org.apache.hadoop.fs.permission.FsPermission
        # jFsActionClass = self.spark.sparkContext._gateway.jvm.org.apache.hadoop.fs.permission.FsAction
        # for i in range(777):
        #     dfs_file_system.mkdirs(jPathClass("/user/_hdfs_export_2"),
        #     jFsPermissionClass(jShortClass(i)))
        # jFileUtilClass.chmod("" + dfs_path, "0777")

    @classmethod
    def hdfs_summary(self, spark, dfs_path):
        jFileSystemClass = (
            spark.sparkContext._gateway.jvm.org.apache.hadoop.fs.FileSystem
        )
        hadoop_configuration = spark.sparkContext._jsc.hadoopConfiguration()
        # change defaultFS
        hadoop_configuration.set(
            "fs.defaultFS", dfs_path[: dfs_path.find("/", "hdfs://".__len__())]
        )
        dfs_file_system = jFileSystemClass.get(hadoop_configuration)
        jPathClass = spark.sparkContext._gateway.jvm.org.apache.hadoop.fs.Path
        jdfs_path = jPathClass(dfs_path)
        return (
            dfs_file_system.getContentSummary(jdfs_path).getFileCount(),
            dfs_file_system.getContentSummary(jdfs_path).getLength(),
        )

    def close(self):
        if self.spark:
            self.spark.stop()


if __name__ == "__main__":
    tdw_utils = TDWUtils()
    tdw_utils.close()

