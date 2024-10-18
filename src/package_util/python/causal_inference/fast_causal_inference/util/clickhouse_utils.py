import csv
import math
import warnings
from time import perf_counter as _perf_counter
import random
import time
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
from fast_causal_inference.common import handle_exception
from fast_causal_inference.common import get_context
from fast_causal_inference.dataframe.provider import FCIProvider

import datetime

"""
dataframe和tdw默认操作的是Clickhouse分布式表
reference: https://clickhouse.com/docs/en/integrations/python
"""


class ClickHouseUtils(object):
    # 5min
    JDBC_ARGS = "?socket_timeout=7203000&max_execution_time=7202&compress=0"
    MAX_ROWS = 160 * 10000 * 10000
    MAX_CSV_ROWS = 1000 * 10000
    MAX_VIEW_MATERIALIZE_ROWS = MAX_CSV_ROWS
    MAX_EXECUTION_TIME = 15 * 60

    def __init__(self, host=None, database=None, rand=False, device_id=None, provider=FCIProvider("global")):
        self._provider = provider
        PROJECT_CONF = self._provider.get_project_conf()
        if not device_id:
            device_id = PROJECT_CONF["datasource"][0]["device_id"]

        device_info_dict = self.get_device_info_dict()

        self.DEFAULT_DATABASE = device_info_dict[device_id]["clickhouse_database"]

        clickhouse_host_list = device_info_dict[device_id]["clickhouse_launch_host"].split(",")
        self.DEFAULT_HOST = clickhouse_host_list[random.randint(0, len(clickhouse_host_list) - 1)]

        self.DEFAULT_PORT = device_info_dict[device_id]["clickhouse_port"]
        self.DEFAULT_HTTP_PORT = device_info_dict[device_id]["clickhouse_http_port"]
        self.DEFAULT_USER = device_info_dict[device_id]["clickhouse_user"]
        self.DEFAULT_PASSWORD = device_info_dict[device_id]["clickhouse_password"]
        self.CLUSTER = device_info_dict[device_id]["clickhouse_cluster_name"]
        self.DEFAULT_TTL_DAY = device_info_dict[device_id]["ttl"]
        if (
            "ip_mapping" in device_info_dict[device_id]
            and device_info_dict[device_id]["ip_mapping"].__len__() > 0
        ):
            self.IP_MAPPING = device_info_dict[device_id]["ip_mapping"][0]
        else:
            self.IP_MAPPING = None
        self.JDBC_PROPERTIES = self.get_jdbc_properties(device_id)
        if not database:
            database = self.DEFAULT_DATABASE
        if not host:
            self.host = self.DEFAULT_HOST
        else:
            self.host = host
        # connect_timeout. Default is 10 seconds.
        # send_receive_timeout. Default is 300 seconds.
        # sync_request_timeout. Default is 5 seconds.
        settings = {"connect_timeout": 120}
        self.client = None
        self.client.execute("set distributed_ddl_task_timeout = 1800")
        self.client.execute("set max_execution_time = 1800")
        self.client.execute("set max_parser_depth = 3000")
        if self.CLUSTER:
            self.cluster_hosts = self.system_clusters(self.CLUSTER)
            self.cluster_hosts_len = self.cluster_hosts.__len__()
            if rand:
                self.close()
                self.host = self.cluster_hosts[
                    random.randint(0, self.cluster_hosts_len - 1)
                ]
                self.client = None

        from fast_causal_inference.util import SqlGateWayConn

        self.sql_instance = SqlGateWayConn(device_id=device_id, db_name=database)

    def get_device_info_dict(self):
        PROJECT_CONF = self._provider.get_project_conf()
        device_info_dict = dict()
        for datasource in PROJECT_CONF["datasource"]:
            device_info_dict[datasource["device_id"]] = datasource
        return device_info_dict

    def get_jdbc_connect_string(self, database=None, device_id=None):
        PROJECT_CONF = self._provider.get_project_conf()
        if not device_id:
            device_id = PROJECT_CONF["datasource"][0]["device_id"]

        device_info_dict = self.get_device_info_dict()
        if not database:
            database = device_info_dict[device_id]["clickhouse_database"]
        return (
            "jdbc:clickhouse://"
            + device_info_dict[device_id]["clickhouse_launch_host"]
            + ":"
            + str(device_info_dict[device_id]["clickhouse_http_port"])
            + "/"
            + database
            + ClickHouseUtils.JDBC_ARGS
        )

    def get_jdbc_properties(self, device_id=None):
        PROJECT_CONF = self._provider.get_project_conf()
        if not device_id:
            device_id = PROJECT_CONF["datasource"][0]["device_id"]

        device_info_dict = self.get_device_info_dict()
        return {
            "driver": "com.clickhouse.jdbc.ClickHouseDriver",
            "user": device_info_dict[device_id]["clickhouse_user"],
            "password": device_info_dict[device_id]["clickhouse_password"],
            "socket_timeout": "7203000",
            "max_execution_time": "7202",
            "compress": "0",
        }

    def get_jdbc_connect_strings(self, database=None):
        if not database:
            database = self.DEFAULT_DATABASE
        jdbc_strings = list()
        for host in self.cluster_hosts:
            jdbc_strings.append(
                (
                    "jdbc:clickhouse://"
                    + host
                    + ":"
                    + str(self.DEFAULT_HTTP_PORT)
                    + "/"
                    + database
                    + ClickHouseUtils.JDBC_ARGS,
                    host,
                )
            )
        return jdbc_strings

    def execute(self, sql, values=None):
        if values:
            get_context().logger.debug(self.host + ",sql=" + sql + ",values ...")
            return self.client.execute(sql, values)
        else:
            get_context().logger.debug(self.host + ",sql=" + sql)
            return self.client.execute(sql)

    def sqlgateway_execute(self, sql, is_calcite_parse=False):
        get_context().logger.debug("sqlgateway, sql=" + sql)
        return self.sql_instance.sql(
            sql=sql, is_calcite_parse=is_calcite_parse, is_dataframe=False
        )

    def execute_with_progress(self, sql):
        progress = self.client.execute_with_progress(sql)
        timeout = 20
        started_at = datetime.now()
        for num_rows, total_rows in progress:
            get_context().logger.debug(
                "num_rows=" + str(num_rows) + ",total_rows=" + str(total_rows)
            )
            if total_rows:
                done = float(num_rows) / total_rows
            else:
                done = total_rows
            now = datetime.now()
            elapsed = (now - started_at).total_seconds()
            # Cancel query if it takes more than 20 seconds
            # to process 50% of rows.
            if elapsed > timeout and done < 0.5:
                self.client.cancel()
                break
        else:  # 循环整体结束后要执行的else代码, 如果是break退出的循环, 则else代码不执行
            rv = progress.get_result()  # 阻塞式
            get_context().logger.debug(rv)

    def show_tables(self):
        ch_tables_list = list()
        sql = "show tables"
        for table_cell in self.execute(sql):
            ch_tables_list.append(table_cell[0])
        get_context().logger.debug("ch_tables_list=" + str(ch_tables_list))
        return ch_tables_list

    def show_create_tables(self, clickhouse_table_name):
        sql = "show create table " + clickhouse_table_name
        try:
            desc = self.execute(sql)
        except Exception as e:
            if "doesn't exist" in repr(e):
                raise Exception("table is not exist, please check")
        return desc[0][0]

    def system_clusters(self, clickhouse_cluster):
        ch_hosts_list = list()
        sql = (
            "select host_address from system.clusters where cluster='"
            + clickhouse_cluster
            + "'"
        )
        for host in self.execute(sql):
            if self.IP_MAPPING and host[0] in self.IP_MAPPING:
                real_ip = self.IP_MAPPING[host[0]]
            else:
                real_ip = host[0]
            ch_hosts_list.append(real_ip)
        get_context().logger.debug("ch_hosts_list=" + str(ch_hosts_list))
        return ch_hosts_list

    def table_rows(self, clickhouse_table_name, database=None):
        if not database:
            database = self.DEFAULT_DATABASE
        sql = (
            "select count(*) from "
            + database
            + "."
            + clickhouse_table_name
            + " SETTINGS max_execution_time = "
            + str(ClickHouseUtils.MAX_EXECUTION_TIME)
        )
        num = self.execute(sql)[0][0]
        get_context().logger.debug(self.host + ",num=" + str(num))
        return num

    """
    columns python 列名和类型
    columns={
                'uin': int,
                'numbera': float,
                'numberb': float,
                'time': lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
            }
    """

    def csv_2_clickhouse(
        self,
        csv_file_abs_path,
        clickhouse_table_name,
        columns,
        clickhouse_database_name=None,
        is_auto_create=True,
    ):
        if not columns:
            # 类型推断
            import pandas as pd

            df = pd.read_csv(csv_file_abs_path)
            columns = dict()
            for column_name, column_type in df.dtypes.to_dict().items():
                if "Unnamed: 0" == column_name:
                    column_name = "id"
                if "int" in column_type.name:
                    columns[column_name] = int
                elif "float" in column_type.name:
                    columns[column_name] = float
                else:
                    columns[column_name] = str
            get_context().logger.debug(columns)
        if not clickhouse_database_name:
            clickhouse_database_name = self.DEFAULT_DATABASE

        def iter_csv(filename):
            with open(filename, "r") as f:
                reader = csv.DictReader(f)
                for line in reader:
                    res = dict()
                    for k, v in line.items():
                        if not k:
                            k = "id"
                        if k in columns:
                            try:
                                if v == None:
                                    v = ""
                                if v == '' and (columns[k] == int or columns[k] == float):
                                    v = 0
                                value = columns[k](v)
                            except ValueError:
                                print(f"Error converting value {v} to type {columns[k]}")
                        else:
                            value = v
                        res[k] = value
                    yield res

        sql_statement = ""
        type_map = {int: "Int64", float: "Float64", str: "String"}
        for k, v in columns.items():
            sql_statement += k + " " + type_map[v] + ","
        sql_statement = sql_statement[:-1]
        if is_auto_create:
            clickhouse_utils = ClickHouseUtils(provider=self._provider)
            self.create_table(
                clickhouse_table_name,
                sql_statement,
                type="local_without_ttl_id",
                cluster=clickhouse_utils.CLUSTER,
                primary_column="tuple()",
                database_name=clickhouse_database_name,
            )
        self.execute(
            "INSERT INTO "
            + clickhouse_database_name
            + "."
            + clickhouse_table_name
            + " VALUES",
            iter_csv(csv_file_abs_path),
        )
        self.close()

    def clickhouse_2_csv(
        self, clickhouse_table_name, csv_file_abs_path, clickhouse_database_name=None
    ):
        if not clickhouse_database_name:
            clickhouse_database_name = self.DEFAULT_DATABASE
        if self.table_rows(clickhouse_table_name) > ClickHouseUtils.MAX_CSV_ROWS:
            raise Exception("table rows too large")
        with open(csv_file_abs_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    column[0]
                    for column in self.execute(
                        "DESC " + clickhouse_database_name + "." + clickhouse_table_name
                    )
                ]
            )
            for row in self.execute(
                "SELECT * FROM "
                + clickhouse_database_name
                + "."
                + clickhouse_table_name
            ):
                writer.writerow(row)
        self.close()

    def create_global_view(self, executed_sql):
        clickhouse_database_name = self.DEFAULT_DATABASE
        view_name = "df_table_temp_view_" + str(int(time.time()))
        sql = "CREATE VIEW " + clickhouse_database_name + "." + view_name + " on cluster " + self.CLUSTER + " AS " + executed_sql
        self.execute(sql)
        return view_name

    def clickhouse_2_pandas(
        self, executed_sql, clickhouse_database_name=None
    ):
        if not clickhouse_database_name:
            clickhouse_database_name = self.DEFAULT_DATABASE
        import pandas as pd
        #if self.table_rows(clickhouse_table_name) > ClickHouseUtils.MAX_CSV_ROWS:
        #    raise Exception("table rows too large")

        # create view
        view_name = "df_table_temp_view_" + str(int(time.time()))
        sql = "CREATE VIEW " + clickhouse_database_name + "." + view_name + " on cluster " + self.CLUSTER + " AS " + executed_sql
        self.execute(sql)
        
        csv_name = "/tmp/" + view_name + ".csv"
        self.clickhouse_2_csv(view_name, csv_name, clickhouse_database_name)
        return pd.read_csv(csv_name)

    def create_table(
        self,
        table_name,
        col_statement,
        type="local",
        format="ORC",
        location=None,
        cluster=None,
        partition_column=None,
        primary_column=None,
        database_name=None,
    ):
        timestamp_start = _perf_counter()
        if not database_name:
            database_name = self.DEFAULT_DATABASE
        if partition_column:
            partition_column = " PARTITION BY " + partition_column
        else:
            partition_column = ""
        if primary_column:
            primary_column = " ORDER BY " + primary_column
            default_key = ""
            default_primary = ""
        else:
            primary_column = " ORDER BY id "
            default_key = "`id` UUID,"
            default_primary = "`id` UUID DEFAULT generateUUIDv4(),"
        if type == "local":
            if cluster:
                # sql = """
                #          CREATE TABLE IF NOT EXISTS %s.%s on cluster %s (`id` UUID DEFAULT generateUUIDv4(), %s,
                #          `day_` Date DEFAULT toDate(now()))
                #          ENGINE = ReplicatedMergeTree('/clickhouse/tables/replicated/{layer}-{shard}/%s', '{replica}')
                #          ORDER BY id TTL (day_ + toIntervalDay(7)) + toIntervalHour(7)
                #       """ % (database_name, table_name + "_local", cluster, col_statement, table_name + "_local")
                sql = """
                         CREATE TABLE IF NOT EXISTS %s.%s on cluster %s (%s %s, `day_` Date DEFAULT toDate(now())) 
                         ENGINE = MergeTree() %s %s TTL (day_ + toIntervalDay(%s)) + toIntervalHour(7)
                      """ % (
                    database_name,
                    table_name + "_local",
                    cluster,
                    default_primary,
                    col_statement,
                    partition_column,
                    primary_column,
                    self.DEFAULT_TTL_DAY,
                )
                get_context().logger.debug("sql=" + str(sql))
                self.execute(sql)
                sql = """
                         CREATE TABLE IF NOT EXISTS %s.%s on cluster %s (%s %s, `day_` Date) 
                         ENGINE = Distributed(%s, %s, %s, rand())
                      """ % (
                    database_name,
                    table_name,
                    cluster,
                    default_key,
                    col_statement,
                    cluster,
                    database_name,
                    table_name + "_local",
                )
                get_context().logger.debug("sql=" + str(sql))
                self.execute(sql)
            else:
                sql = """
                         CREATE TABLE IF NOT EXISTS %s.%s (%s %s, `day_` Date DEFAULT toDate(now())) 
                         ENGINE = MergeTree() %s ORDER BY id TTL (day_ + toIntervalDay(%s)) + toIntervalHour(7)
                      """ % (
                    database_name,
                    table_name,
                    default_primary,
                    col_statement,
                    partition_column,
                    self.DEFAULT_TTL_DAY,
                )
                get_context().logger.debug("sql=" + str(sql))
                self.execute(sql)
        elif type == "hdfs":
            sql = """
                         CREATE TABLE IF NOT EXISTS %s.%s (%s) ENGINE = HDFS('%s', %s) 
                  """ % (
                database_name,
                table_name,
                col_statement,
                location,
                format,
            )
            get_context().logger.debug(
                "external table sql=" + str(sql) + " " + str(self.client.settings)
            )
            self.execute(sql)
        elif type == "memory":
            sql = """
                         CREATE TABLE IF NOT EXISTS %s.%s (%s) Engine = Memory
                  """ % (
                database_name,
                table_name,
                col_statement,
            )
            get_context().logger.debug("sql=" + str(sql))
            self.execute(sql)
        elif type == "local_without_ttl_id":
            sql = """
                         CREATE TABLE IF NOT EXISTS %s.%s on cluster %s (%s) 
                         ENGINE = MergeTree() order by tuple()
                      """ % (
                database_name,
                table_name + "_local",
                cluster,
                col_statement,
            )
            get_context().logger.debug("sql=" + str(sql))
            self.execute(sql)
            sql = """
                         CREATE TABLE IF NOT EXISTS %s.%s on cluster %s (%s) 
                         ENGINE = Distributed(%s, %s, %s, rand())
                      """ % (
                database_name,
                table_name,
                cluster,
                col_statement,
                cluster,
                database_name,
                table_name + "_local",
            )
            get_context().logger.debug("sql=" + str(sql))
            self.execute(sql)

        else:
            raise Exception("type value exception")
        timestamp_end = _perf_counter()
        get_context().logger.debug(
            "create table done, "
            + "time cost: %s Seconds" % (timestamp_end - timestamp_start)
        )

    def insert_table(
        self,
        clickhouse_table_name,
        external_table_name,
        col_name_statement,
        col_if_statement,
    ):
        start = time.perf_counter()
        sql = """
                insert into %s.%s(%s) select %s from %s.%s SETTINGS max_execution_time = %s
                """ % (
            self.DEFAULT_DATABASE,
            clickhouse_table_name,
            col_name_statement,
            col_if_statement,
            self.DEFAULT_DATABASE,
            external_table_name,
            str(ClickHouseUtils.MAX_EXECUTION_TIME),
        )
        get_context().logger.debug(self.host + ", insert into sql=" + str(sql))
        self.execute(sql)
        end = time.perf_counter()
        get_context().logger.debug(
            self.host
            + ", insert into sql done time cost: "
            + str(end - start)
            + " Seconds"
        )

    def get_table_meta(self, clickhouse_table_name, database=None):
        if not database:
            database = self.DEFAULT_DATABASE
        desc_table = self.execute("desc " + database + "." + clickhouse_table_name)
        field_names = list()
        field_types = list()
        field_raw_types = list()
        for field in desc_table:
            name = field[0]
            if name == "id" or name == "day_":
                continue
            type = field[1]
            field_names.append(name)
            field_types.append(self.field_type_map(type))
            field_raw_types.append(type)
        get_context().logger.debug(field_names)
        get_context().logger.debug(field_types)
        get_context().logger.debug(field_raw_types)
        return field_names, field_types, field_raw_types

    def is_distribute_table(self, clickhouse_table_name, database=None):
        if not database:
            database = self.DEFAULT_DATABASE
        show_create_table = self.execute(
            "show create table " + database + "." + clickhouse_table_name
        )
        if "ENGINE = Distributed" in show_create_table[0][0]:
            return True
        else:
            return False

    """
    clickhouse field type trans tdw field type
    """

    def field_type_map(self, col_type):
        if "Nullable" in col_type:
            col_type = col_type["Nullable(".__len__() : -1]
        if col_type == "Int64":
            col_trans_type = "bigint"
        elif col_type == "Int32":
            col_trans_type = "int"
        elif col_type == "Int8":
            col_trans_type = "int"
        elif col_type == "UInt8":
            col_trans_type = "int"
        elif col_type == "UInt32":
            col_trans_type = "bigint"
        elif col_type == "UInt64":
            col_trans_type = "bigint"
        elif col_type == "UInt16":
            col_trans_type = "int"
        elif col_type == "Int16":
            col_trans_type = "int"
        elif col_type == "Float64":
            col_trans_type = "double"
        elif col_type == "Float32":
            col_trans_type = "float"
        elif col_type == "String":
            col_trans_type = "string"
        elif col_type == "Date":
            col_trans_type = "date"
        elif col_type == "DateTime":
            col_trans_type = "timestamp"
        else:
            raise Exception(col_type + " col_type is not support")
        return col_trans_type

    def get_sql_statement(self, col_names, col_tdw_types, col_clickhouse_types):
        create_clickhouse_sql_statement = ""
        create_tdw_sql_statement = ""
        for col_index in range(len(col_names)):
            col_name = col_names[col_index]
            col_clickhouse_type = col_clickhouse_types[col_index]
            col_tdw_type = col_tdw_types[col_index]
            get_context().logger.debug(
                "col_name="
                + str(col_name)
                + ",col_clickhouse_type="
                + str(col_clickhouse_type)
                + ",col_tdw_type="
                + str(col_tdw_type)
            )
            create_clickhouse_sql_statement += (
                col_name + " " + col_clickhouse_type + ","
            )
            create_tdw_sql_statement += (
                col_name + " " + col_tdw_type + " comment '" + col_name + "',"
            )
        create_clickhouse_sql_statement = create_clickhouse_sql_statement[:-1]
        create_tdw_sql_statement = create_tdw_sql_statement[:-1]
        return create_clickhouse_sql_statement, create_tdw_sql_statement

    @classmethod
    def clickhouse_2_tdw(
        cls,
        clickhouse_table_name,
        tdw_database_name,
        tdw_table_name,
        spark_session,
        cmk=None,
        is_auto_create=True,
        provider=FCIProvider("global")
    ):
        clickhouse_utils = ClickHouseUtils(provider=provider)
        num = clickhouse_utils.table_rows(clickhouse_table_name)
        get_context().logger.info("clickhouse table count=" + str(num))
        if num > ClickHouseUtils.MAX_ROWS:
            raise Exception(
                "clickhouse table rows num too big, >"
                + str(ClickHouseUtils.MAX_ROWS)
                + " not support"
            )
        field_names, field_types, field_raw_types = clickhouse_utils.get_table_meta(
            clickhouse_table_name
        )
        col_list = list()
        for i in range(field_names.__len__()):
            col_list.append([field_names[i], field_types[i], field_names[i]])
        get_context().logger.debug(col_list)
        sql_statement, create_tdw_sql_statement = clickhouse_utils.get_sql_statement(
            field_names, field_types, field_raw_types
        )
        from fast_causal_inference.util import TDWUtils

        tdw_utils = TDWUtils(get_context().spark_session)

        export_hdfs_path = tdw_utils.base_hdfs_path + tdw_table_name + "_hdfs_export"
        tdw_utils = TDWUtils(spark_session)
        if tdw_utils.table_exits(tdw_database_name, tdw_table_name):
            raise Exception("tdw table is already exist")
        tdw_utils.hdfs_mkdir_and_chmod(tdw_utils.name_space + export_hdfs_path)
        if is_auto_create:
            from fast_causal_inference.util import get_user

            user = get_user()
            if not cmk:
                raise Exception("please input cmk arg")
            get_context().logger.debug("user=" + user + ",cmk=" + cmk)
            sql = """
                            set `supersql.datasource.default`=hive_online_internal;
                            set `supersql.bypass.forceAll`=true;
                            use %s;
                            CREATE EXTERNAL TABLE IF NOT EXISTS %s(
                                %s
                            )
                            STORED AS PARQUET
                            LOCATION '%s'
                            """ % (
                tdw_database_name,
                tdw_table_name,
                create_tdw_sql_statement,
                tdw_utils.name_space + export_hdfs_path,
            )
            get_context().logger.debug(sql)
        if clickhouse_utils.is_distribute_table(
            clickhouse_table_name=clickhouse_table_name
        ):
            # distribute table
            def clickhouse_node_job(
                clickhouse_utils, clickhouse_hdfs_table_name, sql_statement, file_name
            ):
                get_context().logger.debug(clickhouse_utils.host)
                location_file = (
                    tdw_utils.base_hdfs_path + tdw_table_name + "_" + file_name
                )
                get_context().logger.debug(
                    "clickhouse_node_job, location_file=" + location_file
                )
                clickhouse_utils.create_table(
                    clickhouse_hdfs_table_name,
                    sql_statement,
                    type="hdfs",
                    format="Parquet",
                    location=tdw_utils.name_space + location_file,
                )
                clickhouse_utils.insert_table(
                    clickhouse_hdfs_table_name,
                    clickhouse_table_name + "_local",
                    ",".join(field_names),
                    ",".join(field_names),
                )
                get_context().logger.debug(
                    "insert success, sink clickhouse_hdfs_table_name="
                    + clickhouse_hdfs_table_name
                    + ",source clickhouse_table_name="
                    + clickhouse_table_name
                )
                get_context().logger.debug(
                    "location_file="
                    + location_file
                    + ",export_hdfs_path="
                    + export_hdfs_path
                )
                tdw_utils.hdfs_move(location_file, export_hdfs_path + "/" + file_name)
                clickhouse_utils.close()

            with ThreadPoolExecutor(
                max_workers=min(16, clickhouse_utils.cluster_hosts_len)
            ) as pool:
                all_task = list()
                for i in range(clickhouse_utils.cluster_hosts_len):
                    timestamp = str(int(time.time()))
                    file_name = (
                        "part"
                        + "_"
                        + clickhouse_utils.cluster_hosts[i].replace(".", "_")
                        + "_"
                        + timestamp
                        + ".parquet"
                    )
                    clickhouse_hdfs_table_name = (
                        clickhouse_table_name + "_hdfs_export_" + timestamp
                    )
                    future = pool.submit(
                        clickhouse_node_job,
                        ClickHouseUtils(host=clickhouse_utils.cluster_hosts[i], provider=provider),
                        clickhouse_hdfs_table_name,
                        sql_statement,
                        file_name,
                    )
                    future.add_done_callback(handle_exception)
                    all_task.append(future)
                wait(all_task, timeout=None, return_when=ALL_COMPLETED)
        else:
            # local table
            timestamp = str(int(time.time()))
            location_file = tdw_utils.base_hdfs_path + "part" + timestamp + ".parquet"
            clickhouse_hdfs_table_name = (
                clickhouse_table_name + "_hdfs_export_" + timestamp
            )
            clickhouse_utils.create_table(
                table_name=clickhouse_hdfs_table_name,
                col_statement=sql_statement,
                type="hdfs",
                format="Parquet",
                location=tdw_utils.name_space + location_file,
            )
            clickhouse_utils.insert_table(
                clickhouse_hdfs_table_name,
                clickhouse_table_name,
                ",".join(field_names),
                ",".join(field_names),
            )
            get_context().logger.debug(location_file)
            get_context().logger.debug(export_hdfs_path)
            tdw_utils.hdfs_move(
                location_file, export_hdfs_path + "/part" + timestamp + ".parquet"
            )
        clickhouse_utils.close()

    @classmethod
    def clickhouse_2_tdw_v2(
        self,
        session,
        clickhouse_table,
        tdw_database,
        tdw_table,
        tdw_user,
        tdw_passward,
        group,
        is_drop_table=False,
        overwrite=True,
        priPart=None,
        provider=FCIProvider("global")
    ):
        clickhouse_utils = ClickHouseUtils(provider=provider)
        num = clickhouse_utils.table_rows(clickhouse_table)
        get_context().logger.info("clickhouse table count=" + str(num))
        if num > ClickHouseUtils.MAX_ROWS:
            raise Exception(
                "clickhouse table rows num too big, >"
                + str(ClickHouseUtils.MAX_ROWS)
                + " not support"
            )
        field_names, field_types, field_raw_types = clickhouse_utils.get_table_meta(
            clickhouse_table
        )
        from pytoolkit import TDWSQLProvider, TDWUtil, TableDesc

        # 将 field_names, field_types 转为 [[name, type, name], ...]
        col_list = list()
        for i in range(field_names.__len__()):
            col_list.append([field_names[i], field_types[i], field_names[i]])
        tdw = TDWUtil(
            user=tdw_user, passwd=tdw_passward, dbName=tdw_database, group=group
        )
        table_desc = (
            TableDesc()
            .setTblName(tdw_table)
            .setCols(col_list)
            .setComment("all in sql to tdw")
        )
        if is_drop_table:
            tdw.dropTable(tdw_table)
        tdw.createTable(table_desc)

        SPARK_SESSION = session

        spark_df = self.clickhouse_2_dataframe(SPARK_SESSION, clickhouse_table, provider=provider)
        spark_df = spark_df.select(field_names)
        tdw = TDWSQLProvider(
            SPARK_SESSION,
            user=tdw_user,
            passwd=tdw_passward,
            db=tdw_database,
            group=group,
        )
        tdw.saveToTable(
            df=spark_df, tblName=tdw_table, overwrite=overwrite, priPart=priPart
        )

    # select from distribute table
    @classmethod
    def clickhouse_2_dataframe(
        self,
        spark,
        clickhouse_table_name,
        clickhouse_database_name=None,
        batch_size=100000,
        provider=FCIProvider("global")
    ):
        clickhouse_utils = ClickHouseUtils(provider=provider)
        if not clickhouse_database_name:
            clickhouse_database_name = clickhouse_utils.DEFAULT_DATABASE

        num = clickhouse_utils.table_rows(
            clickhouse_table_name, clickhouse_database_name
        )
        get_context().logger.debug("clickhouse table count=" + str(num))
        if num == 0:
            raise Exception("clickhouse table rows is empty")
        elif num > ClickHouseUtils.MAX_ROWS:
            raise Exception(
                "clickhouse table rows num too big, >"
                + str(ClickHouseUtils.MAX_ROWS)
                + " not support"
            )

        columns = clickhouse_utils.execute(
            "desc " + clickhouse_database_name + "." + clickhouse_table_name
        )
        get_context().logger.debug(columns)
        part_column = ""
        if columns.__len__() > 0:
            for column in columns:
                if "Int" in column[1] or "Float" in column[1]:
                    part_column = column[0]
                if part_column == "uin":
                    break
        get_context().logger.debug(part_column)
        if part_column:
            # 默认并发为10
            quantiles = clickhouse_utils.execute(
                'SELECT quantiles(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)("'
                + part_column
                + '") AS quantiles FROM  '
                + clickhouse_database_name
                + "."
                + clickhouse_table_name
            )
            get_context().logger.debug(quantiles)
        clickhouse_utils.close()
        if part_column:
            predicates = list()
            for index, quantile in enumerate(quantiles[0][0]):
                get_context().logger.debug(str(index) + " " + str(quantile))
                if index == 0:
                    # 左闭右开
                    predicates.append(part_column + " < " + str(quantile))
                elif index >= 8:
                    predicates.append(
                        part_column
                        + " >= "
                        + str(pre_quantile)
                        + " and "
                        + part_column
                        + " < "
                        + str(quantile)
                    )
                    predicates.append(part_column + " >= " + str(quantile))
                else:
                    predicates.append(
                        part_column
                        + " >= "
                        + str(pre_quantile)
                        + " and "
                        + part_column
                        + " < "
                        + str(quantile)
                    )
                pre_quantile = quantile
            get_context().logger.debug(predicates)
            return spark.read.option("batch_size", batch_size).jdbc(
                url=ClickHouseUtils(provider=provider).get_jdbc_connect_string(clickhouse_database_name),
                table=clickhouse_table_name,
                predicates=predicates,
                properties=ClickHouseUtils(provider=provider).get_jdbc_properties(),
            )
        else:
            return spark.read.option("batch_size", batch_size).jdbc(
                url=ClickHouseUtils(provider=provider).get_jdbc_connect_string(clickhouse_database_name),
                table=clickhouse_table_name,
                properties=ClickHouseUtils(provider=provider).get_jdbc_properties(),
            )

    # select from everyone node
    @classmethod
    def clickhouse_2_dataframe_distribute(
        self, spark, clickhouse_table_name, partition_num, clickhouse_database_name, provider=FCIProvider("global")
    ):
        dataframe = None
        global_clickhouse_utils = ClickHouseUtils(provider=provider)
        for jdbc_string, host in global_clickhouse_utils.get_jdbc_connect_strings(
            clickhouse_database_name
        ):
            get_context().logger.debug("jdbc_string=" + jdbc_string + ",host=" + host)
            predicates = list()
            clickhouse_utils = ClickHouseUtils(host=host, provider=provider)
            num = clickhouse_utils.table_rows(
                clickhouse_table_name, clickhouse_database_name
            )
            get_context().logger.debug("clickhouse table count=" + str(num))
            if num > ClickHouseUtils.MAX_ROWS:
                raise Exception(
                    "clickhouse table rows num too big, >"
                    + str(ClickHouseUtils.MAX_ROWS)
                    + " not support"
                )
            step = math.floor(num / partition_num + 1)
            get_context().logger.debug("num=" + num + ",step=" + str(step))
            for i in range(partition_num - 1):
                predicates.append("1 = 1 limit " + str(step * i) + ", " + str(step))
            predicates.append(
                "1 = 1 limit "
                + str(step * (partition_num - 1))
                + ", "
                + str(num - step * (partition_num - 1))
            )
            get_context().logger.debug("predicates=" + str(predicates))
            clickhouse_utils.close()
            if dataframe:
                dataframe = dataframe.union(
                    spark.read.jdbc(
                        url=jdbc_string,
                        table=clickhouse_table_name,
                        predicates=predicates,
                        properties=ClickHouseUtils(provider=provider).get_jdbc_properties(),
                    )
                )
            else:
                dataframe = spark.read.jdbc(
                    url=jdbc_string,
                    table=clickhouse_table_name,
                    predicates=predicates,
                    properties=ClickHouseUtils(provider=provider).get_jdbc_properties(),
                )
        global_clickhouse_utils.close()
        return dataframe

    """
    table_name: 表示创建表的名称
    select_statement: 建表的具体select sql
    is_physical_table:  创建的是物理表还是视图,  [默认为视图] 物理表占用实际存储，如果数据量过大(> 1亿)请考虑集群存储占用，视图为虚拟表，不实际占用存储
    is_distributed_create: 是否分布式创建 [默认为True]
                           (1 如果为True则建表下发到每个worker node执行, 后续针对该表查询会利用分布式多机处理能力,
                           (2 如果为False则建表仅在单节点创建, 后续针对该表查询仅会使用单节点处理计算(请注意单节点内存使用,可能会因为超限导致执行失败);
                           注: 有些sql为多机处理后生成的结果，则应指定为False
    """

    @classmethod
    def create_view_v2(
        self,
        table_name,
        select_statement,
        is_physical_table=False,
        is_distributed_create=True,
        origin_table_name=None,
        is_agg_status=False,
        provider=FCIProvider("global")
    ):
        if is_agg_status:
            is_physical_table = True

        clickhouse_utils = ClickHouseUtils(provider=provider)
        database = clickhouse_utils.DEFAULT_DATABASE

        calcite_select_sql = select_statement

        # 校验语句合法性  及识别table
        explain_result = clickhouse_utils.execute("explain " + calcite_select_sql)
        tables = dict()
        table = ""
        for explain_cell in explain_result:
            ast_record = explain_cell[0].strip()
            if ast_record.startswith("ReadFromMergeTree"):
                table = ast_record.replace("ReadFromMergeTree (", "").replace(")", "")
            if table and ast_record.startswith("ReadFromRemote"):
                tables[table.replace("_local", "")] = table
                table = ""
        get_context().logger.debug(str(tables))
        if origin_table_name and origin_table_name.find(".") == -1:
            tables = {
                database
                + "."
                + origin_table_name: database
                + "."
                + origin_table_name
                + "_local"
            }

        if is_distributed_create:
            for table_k, table_v in tables.items():
                # 删除 table_k, table_v 的数据库名
                if calcite_select_sql.lower().find('union all') == -1:
                    calcite_select_sql = calcite_select_sql.replace(
                        " " + table_k[table_k.index(".") + 1 :], " " + table_k
                    )
                    table_k = table_k[table_k.index(".") + 1 :]
                    table_v = table_v[table_v.index(".") + 1 :]
                    calcite_select_sql = calcite_select_sql[::-1].replace(
                        table_k[::-1], table_v[::-1], 1
                    )[::-1]
                else:
                    # replace all if union all
                    calcite_select_sql = calcite_select_sql.replace(table_k, table_v)

        if is_agg_status:
            calcite_select_sql += " limit 0 "

        # 创建表
        if not is_physical_table:
            if is_distributed_create and clickhouse_utils.CLUSTER:
                sql = (
                    "CREATE VIEW "
                    + database
                    + "."
                    + table_name
                    + "_local"
                    + " on cluster "
                    + clickhouse_utils.CLUSTER
                    + " AS ("
                    + calcite_select_sql
                    + ")"
                )
            else:
                sql = (
                    "CREATE VIEW "
                    + database
                    + "."
                    + table_name
                    + " AS ("
                    + calcite_select_sql
                    + ")"
                )
        else:
            if is_distributed_create and clickhouse_utils.CLUSTER:
                sql = (
                    "CREATE TABLE "
                    + database
                    + "."
                    + table_name
                    + "_local"
                    + " on cluster "
                    + clickhouse_utils.CLUSTER
                    + " ENGINE = MergeTree ORDER BY tuple() AS ("
                    + calcite_select_sql
                    + ")"
                )
            else:
                sql = (
                    "CREATE TABLE "
                    + database
                    + "."
                    + table_name
                    + " ENGINE = MergeTree ORDER BY tuple() AS ("
                    + calcite_select_sql
                    + ")"
                )

        get_context().logger.debug("insert table running")
        get_context().logger.debug(sql)
        start = time.perf_counter()

        clickhouse_utils.execute(sql)
        end = time.perf_counter()
        get_context().logger.debug(
            "insert table done" + "time cost: " + str(end - start) + " Seconds"
        )
        # fci_get_context().logger.info(database + "." + table_name)

        if is_distributed_create and clickhouse_utils.CLUSTER:
            sql = (
                "CREATE TABLE "
                + database
                + "."
                + table_name
                + " on cluster "
                + clickhouse_utils.CLUSTER
                + " as "
                + table_name
                + "_local"
                + " ENGINE = Distributed("
                + clickhouse_utils.CLUSTER
                + ", "
                + database
                + ", "
                + table_name
                + "_local"
                + ", rand())"
            )
            clickhouse_utils.execute(sql)
            get_context().logger.debug(sql)
            get_context().logger.debug("materialize done")

        if is_agg_status:
            sql = "INSERT INTO " + database + "." + table_name + " " + calcite_select_sql.replace('_local', '').replace('limit 0','')
            clickhouse_utils.execute(sql)

    def __materialize_table(
        self,
        clickhouse_utils,
        select_sql,
        sql_table_name,
        database,
        clickhouse_view_name,
        primary_column,
        is_use_local,
    ):
        get_context().logger.debug("materialize view doing")
        if is_use_local and self.CLUSTER:
            clickhouse_view_name_real = clickhouse_view_name + "_local"
            if sql_table_name:
                if isinstance(sql_table_name, list):
                    for sql_table_name_cell in sql_table_name:
                        select_sql = select_sql.replace(
                            sql_table_name, sql_table_name_cell + "_local"
                        )
                else:
                    select_sql = select_sql.replace(
                        sql_table_name, sql_table_name + "_local"
                    )
        else:
            clickhouse_view_name_real = clickhouse_view_name
        if sql_table_name:
            select_sql = select_sql.replace(
                sql_table_name, database + "." + sql_table_name
            )

        calcite_select_sql = clickhouse_utils.sqlgateway_execute(
            select_sql, is_calcite_parse=True
        )
        if "error message" in calcite_select_sql:
            raise Exception(
                    "sql execute or sqlparse is error, please check, info:"
                + calcite_select_sql
            )

        if is_use_local and self.CLUSTER:
            sql = (
                "CREATE TABLE "
                + clickhouse_view_name_real
                + " on cluster "
                + self.CLUSTER
                + " ENGINE = MergeTree ORDER BY "
                + primary_column
                + " AS "
                + calcite_select_sql
            )
        else:
            sql = (
                "CREATE TABLE "
                + clickhouse_view_name_real
                + " ENGINE = MergeTree ORDER BY "
                + primary_column
                + " AS "
                + calcite_select_sql
            )
        get_context().logger.debug("insert table running")
        get_context().logger.debug(sql)
        start = time.perf_counter()
        clickhouse_utils.execute(sql)
        end = time.perf_counter()
        get_context().logger.debug(
            "insert table done" + "time cost: " + str(end - start) + " Seconds"
        )

        if is_use_local and self.CLUSTER:
            sql = (
                "DROP VIEW if exists "
                + clickhouse_view_name
                + " on cluster "
                + self.CLUSTER
            )
            get_context().logger.debug(sql)
            clickhouse_utils.execute(sql)
            sql = (
                "CREATE TABLE "
                + clickhouse_view_name
                + " on cluster "
                + self.CLUSTER
                + " as "
                + clickhouse_view_name
                + "_local"
                + " ENGINE = Distributed("
                + self.CLUSTER
                + ", "
                + database
                + ", "
                + clickhouse_view_name
                + "_local"
                + ", rand())"
            )
            clickhouse_utils.execute(sql)
            get_context().logger.debug(sql)
            get_context().logger.debug("materialize view done")

    """
    primary_column 是clickhouse MergeTree表引擎用户排序的列，在该列实现跳数索引，查询频率大的列建议放在前面做索引
    is_force_materialize 是否强制将视图物化为物理表, 会执行计算落表
    is_sql_complete sql_statement字段是否提供的是完整的sql声明
    is_use_local 是否要转化为本地表的方式执行
    """

    @classmethod
    def create_view(
        self,
        clickhouse_view_name,
        sql_statement,
        sql_table_name=None,
        sql_where=None,
        sql_group_by=None,
        sql_limit=None,
        primary_column="tuple()",
        database=None,
        is_force_materialize=False,
        is_sql_complete=False,
        is_use_local=True,
        provider=FCIProvider("global")
    ):
        warnings.warn(
            "create_view function is deprecated, please use create_view_v2",
            DeprecationWarning,
        )
        clickhouse_utils = ClickHouseUtils(provider=provider)
        if not database:
            database = clickhouse_utils.DEFAULT_DATABASE
        if is_sql_complete:
            select_sql = sql_statement
        else:
            if sql_limit:
                sql_limit = " LIMIT " + str(sql_limit)
            else:
                sql_limit = ""
            if sql_where:
                sql_where = " WHERE " + sql_where
            else:
                sql_where = ""
            if sql_group_by:
                sql_group_by = " GROUP BY " + sql_group_by
            else:
                sql_group_by = ""
            select_sql = (
                " SELECT "
                + sql_statement
                + " FROM "
                + sql_table_name
                + " \n"
                + sql_where
                + sql_group_by
                + sql_limit
            )
        select_sql = select_sql.replace("{database}", database)
        get_context().logger.debug("raw sql = \n" + select_sql)
        # fci_get_context().logger.info(database + "." + str(sql_table_name))

        if is_force_materialize:
            clickhouse_utils.__materialize_table(
                clickhouse_utils,
                select_sql,
                sql_table_name,
                database,
                clickhouse_view_name,
                primary_column,
                is_use_local,
            )
        else:
            if clickhouse_utils.CLUSTER:
                sql = (
                    "CREATE VIEW "
                    + database
                    + "."
                    + clickhouse_view_name
                    + " on cluster "
                    + clickhouse_utils.CLUSTER
                    + " AS "
                    + select_sql
                )
            else:
                sql = (
                    "CREATE VIEW "
                    + database
                    + "."
                    + clickhouse_view_name
                    + " AS "
                    + select_sql
                )
            clickhouse_utils.execute(sql)
            fields = list()
            for i in clickhouse_utils.execute(
                "DESC " + database + "." + clickhouse_view_name
            ):
                fields.append(i[0])
            get_context().logger.debug("view table fields =" + str(fields))
            if primary_column != "tuple()" and primary_column not in fields:
                raise Exception("primary_column set not valid")
            rows_number = clickhouse_utils.table_rows(
                clickhouse_view_name, database=database
            )
            get_context().logger.debug("view rows number is " + str(rows_number))
            if rows_number < ClickHouseUtils.MAX_VIEW_MATERIALIZE_ROWS:
                clickhouse_utils.execute(
                    "DROP VIEW " + database + "." + clickhouse_view_name
                )
                clickhouse_utils.__materialize_table(
                    clickhouse_utils,
                    select_sql,
                    sql_table_name,
                    database,
                    clickhouse_view_name,
                    primary_column,
                    is_use_local,
                )
            get_context().logger.debug("create view success")
        clickhouse_utils.close()

    @classmethod
    def drop_view(self, clickhouse_view_name, database=None, provider=FCIProvider("global")):
        clickhouse_utils = ClickHouseUtils(provider=provider)
        if not database:
            database = clickhouse_utils.DEFAULT_DATABASE
        try:
            table_desc = clickhouse_utils.show_create_tables(clickhouse_view_name)
        except Exception:
            return

        def handle_exception_execute(sql):
            try:
                clickhouse_utils.execute(sql)
            except Exception as e:
                get_context().logger.error(e)

        if "CREATE VIEW " in table_desc:
            if clickhouse_utils.CLUSTER:
                sql = (
                    "DROP VIEW if exists "
                    + database
                    + "."
                    + clickhouse_view_name
                    + " on cluster "
                    + clickhouse_utils.CLUSTER
                )
            else:
                sql = "DROP VIEW if exists " + database + "." + clickhouse_view_name
            handle_exception_execute(sql)
        else:
            if clickhouse_utils.CLUSTER:
                sql = (
                    "DROP TABLE if exists "
                    + database
                    + "."
                    + clickhouse_view_name
                    + " on cluster "
                    + clickhouse_utils.CLUSTER
                )
                handle_exception_execute(sql)
                sql = (
                    "DROP TABLE if exists "
                    + database
                    + "."
                    + clickhouse_view_name
                    + "_local on cluster "
                    + clickhouse_utils.CLUSTER
                )
                handle_exception_execute(sql)
            else:
                sql = "DROP TABLE if exists " + database + "." + clickhouse_view_name
                handle_exception_execute(sql)
                sql = (
                    "DROP TABLE if exists "
                    + database
                    + "."
                    + clickhouse_view_name
                    + "_local"
                )
                handle_exception_execute(sql)
        clickhouse_utils.close()

    @classmethod
    def drop_partition(
        self, clickhouse_view_name, clickhouse_partition_name, database=None, provider=FCIProvider("global"
    )):
        clickhouse_utils = ClickHouseUtils(provider=provider)
        if not database:
            database = clickhouse_utils.DEFAULT_DATABASE
        if clickhouse_utils.CLUSTER:
            sql = (
                "alter table "
                + database
                + "."
                + clickhouse_view_name
                + "_local on cluster "
                + clickhouse_utils.CLUSTER
                + " drop partition "
                + clickhouse_partition_name
            )
        else:
            sql = (
                "alter table "
                + database
                + "."
                + clickhouse_view_name
                + "_local drop partition "
                + clickhouse_partition_name
            )
        clickhouse_utils.execute(sql)
        clickhouse_utils.close()

    def close(self):
        if self.client:
            self.client.disconnect()

    @classmethod
    def clickhouse_create_view(
        cls,
        clickhouse_view_name,
        sql_statement,
        sql_table_name=None,
        sql_where=None,
        sql_group_by=None,
        sql_limit=None,
        primary_column="tuple()",
        is_force_materialize=False,
        is_sql_complete=False,
        is_use_local=True,
        provider=FCIProvider("global")
    ):
        """
        创建实验指标明细视图
        sql_statement, sql_table_name, sql_where, sql_group_by, sql_limit 会组成完整sql
        默认采用视图/物理表策略： 当明细数据量小于100w行则直接物化为物理表，超过100w行则默认以视图形式提供，可通过is_force_materialize=True
        强制物化为物理表
        :param clickhouse_view_name: clikchouse视图/物理表名称
        :param sql_statement:  查询子语句
        :param sql_table_name:  查询表
        :param sql_where:      where子句
        :param sql_group_by:   group by 子句
        :param sql_limit:      limit子句
        :param primary_column:  指定导入时候的分bucket的列名, clickhouse colocate join维度列需要预先分bucket
        :param is_force_materialize:   为True则强制物化为物理表，请注意磁盘存储空间占用
        :return:
        """
        import warnings

        warnings.warn(
            "clickhouse_create_view function is deprecated, please use clickhouse_create_view_v2",
            DeprecationWarning,
        )
        # fci_get_context().logger.info("running, please wait")
        start = _perf_counter()
        from fast_causal_inference.util.clickhouse_utils import ClickHouseUtils

        ClickHouseUtils.create_view(
            clickhouse_view_name,
            sql_statement,
            sql_table_name,
            sql_where,
            sql_group_by,
            sql_limit=sql_limit,
            primary_column=primary_column,
            is_force_materialize=is_force_materialize,
            is_sql_complete=is_sql_complete,
            is_use_local=is_use_local,
            provider=provider,
        )
        end = _perf_counter()
        get_context().logger.debug("done" + "time cost: %s Seconds" % (end - start))

    @classmethod
    def clickhouse_create_view_v2(
        cls,
        table_name,
        select_statement,
        is_physical_table=False,
        is_distributed_create=True,
        origin_table_name=None,
        is_agg_status=False,
        provider = FCIProvider("global")
    ):
        """
        table_name: 表示创建表的名称
        select_statement: 建表的具体select sql
        is_physical_table:  创建的是物理表还是视图,  [默认为视图] 物理表占用实际存储，如果数据量过大(> 1亿)请考虑集群存储占用，视图为虚拟表，不实际占用存储
        is_distributed_create: 是否分布式创建 [默认为True]
                            (1 如果为True则建表下发到每个worker node执行, 后续针对该表查询会利用分布式多机处理能力,
                            (2 如果为False则建表仅在单节点创建, 后续针对该表查询仅会使用单节点处理计算(请注意单节点内存使用,可能会因为超限导致执行失败);
                            注: 有些sql为多机处理后生成的结果，则应指定为False
        """
        # get_context().logger.info("creating view, please wait")
        start = _perf_counter()

        ClickHouseUtils(provider=provider).create_view_v2(
            table_name,
            select_statement,
            is_physical_table,
            is_distributed_create,
            origin_table_name,
            is_agg_status,
            provider=provider,
        )
        end = _perf_counter()
        get_context().logger.debug("done" + "time cost: %s Seconds" % (end - start))

    @classmethod
    def clickhouse_drop_view(cls, clickhouse_view_name, provider = FCIProvider("global")):
        """
        在集群删除视图/物理表
        :param clickhouse_view_name:  视图或物理表名称
        :return:
        """
        # get_context().logger.info("running, please wait")
        start = _perf_counter()
        from fast_causal_inference.util.clickhouse_utils import ClickHouseUtils

        ClickHouseUtils.drop_view(clickhouse_view_name, provider=provider)
        end = _perf_counter()
        get_context().logger.debug("done" + "time cost: %s Seconds" % (end - start))

    @classmethod
    def clickhouse_drop_partition(cls, clickhouse_view_name, clickhouse_partition_name, provider = FCIProvider("global")):
        """
        在集群删除分区
        :param clickhouse_view_name:  物理表名称
        :param clickhouse_partition_name:  分区名称
        :return:
        """
        # get_context().logger.info("running, please wait")
        start = _perf_counter()


        ClickHouseUtils.drop_partition(clickhouse_view_name, clickhouse_partition_name, provider=provider)
        end = _perf_counter()
        get_context().logger.debug("done" + "time cost: %s Seconds" % (end - start))


if __name__ == "__main__":
    clickhouse_utils = ClickHouseUtils()
    clickhouse_utils.close()


