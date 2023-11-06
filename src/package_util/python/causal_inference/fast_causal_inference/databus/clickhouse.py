import csv
import math
from time import perf_counter as _perf_counter
import random
import time
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
from ..common.exception import handle_exception
from ..util.utils import get_user

from clickhouse_driver import Client
from .. import logger
from .. import PROJECT_CONF
import datetime
from ..all_in_sql import AllInSqlConn

"""
dataframe和tdw默认操作的是Clickhouse分布式表
reference: https://clickhouse.com/docs/en/integrations/python
"""

class ClickHouseUtils(object):
    # 5min
    JDBC_ARGS = "?socket_timeout=7203000&max_execution_time=7202&compress=0"
    MAX_ROWS = 150 * 10000 * 10000
    MAX_CSV_ROWS = 1000 * 10000
    MAX_VIEW_MATERIALIZE_ROWS = MAX_CSV_ROWS
    MAX_EXECUTION_TIME = 15 * 60

    def __init__(self, host=None, database=None, rand=False, device_id=None):
        if not device_id:
            device_id = PROJECT_CONF['datasource'][0]['device_id']

        device_info_dict = ClickHouseUtils.get_device_info_dict()

        self.DEFAULT_DATABASE = device_info_dict[device_id]["clickhouse_database"]
        self.DEFAULT_HOST = device_info_dict[device_id]["clickhouse_launch_host"]
        self.DEFAULT_PORT = device_info_dict[device_id]["clickhouse_port"]
        self.DEFAULT_HTTP_PORT = device_info_dict[device_id]["clickhouse_http_port"]
        self.DEFAULT_USER = device_info_dict[device_id]["clickhouse_user"]
        self.DEFAULT_PASSWORD = device_info_dict[device_id]["clickhouse_password"]
        self.CLUSTER = device_info_dict[device_id]["clickhouse_cluster_name"]
        self.DEFAULT_TTL_DAY = device_info_dict[device_id]["ttl"]
        self.JDBC_PROPERTIES = ClickHouseUtils.get_jdbc_properties(device_id)
        if not database:
            database = self.DEFAULT_DATABASE
        if not host:
            self.host = self.DEFAULT_HOST
        else:
            self.host = host
        # connect_timeout. Default is 10 seconds.
        # send_receive_timeout. Default is 300 seconds.
        # sync_request_timeout. Default is 5 seconds.
        settings = {'connect_timeout': 120}
        self.client = Client(host=self.host, port=self.DEFAULT_PORT,
                             database=database, user=self.DEFAULT_USER,
                             password=self.DEFAULT_PASSWORD, settings=settings)
        if self.CLUSTER:
            self.cluster_hosts = self.system_clusters(self.CLUSTER)
            self.cluster_hosts_len = self.cluster_hosts.__len__()
            if rand:
                self.close()
                self.host = self.cluster_hosts[random.randint(0, self.cluster_hosts_len - 1)]
                self.client = Client(host=self.host, port=self.DEFAULT_PORT,
                                     database=database, user=self.DEFAULT_USER,
                                     password=self.DEFAULT_PASSWORD, settings=settings)


        self.sql_instance = AllInSqlConn(device_id=device_id, db_name=database)

    @classmethod
    def get_device_info_dict(self):
        device_info_dict = dict()
        for datasource in PROJECT_CONF['datasource']:
            device_info_dict[datasource['device_id']] = datasource
        return device_info_dict

    @classmethod
    def get_jdbc_connect_string(self, database=None, device_id=None):
        if not device_id:
            device_id = PROJECT_CONF['datasource'][0]['device_id']

        device_info_dict = ClickHouseUtils.get_device_info_dict()
        if not database:
            database = device_info_dict[device_id]["clickhouse_database"]
        return "jdbc:clickhouse://" + device_info_dict[device_id]["clickhouse_launch_host"] + ":" + str(
            device_info_dict[device_id]["clickhouse_http_port"]) + "/" + database + ClickHouseUtils.JDBC_ARGS

    @classmethod
    def get_jdbc_properties(self, device_id=None):
        if not device_id:
            device_id = PROJECT_CONF['datasource'][0]['device_id']

        device_info_dict = ClickHouseUtils.get_device_info_dict()
        return {
            "driver": "com.clickhouse.jdbc.ClickHouseDriver",
            "user": device_info_dict[device_id]["clickhouse_user"],
            "password": device_info_dict[device_id]["clickhouse_password"],
            "socket_timeout": "7203000",
            "max_execution_time": "7202",
            "compress": "0"}

    def get_jdbc_connect_strings(self, database=None):
        if not database:
            database = self.DEFAULT_DATABASE
        jdbc_strings = list()
        for host in self.cluster_hosts:
            jdbc_strings.append(("jdbc:clickhouse://" + host + ":" + str(
                self.DEFAULT_HTTP_PORT) + "/" + database + ClickHouseUtils.JDBC_ARGS, host))
        return jdbc_strings

    def execute(self, sql, values=None):
        if values:
            logger.debug(self.host + ",sql=" + sql + ",values ...")
            return self.client.execute(sql, values)
        else:
            logger.debug(self.host + ",sql=" + sql)
            return self.client.execute(sql)

    def sqlgateway_execute(self, sql, is_calcite_parse=False):
        logger.debug("sqlgateway, sql=" + sql)
        return self.sql_instance.sql(sql=sql, is_calcite_parse=is_calcite_parse, is_dataframe=False)

    def execute_with_progress(self, sql):
        progress = self.client.execute_with_progress(sql)
        timeout = 20
        started_at = datetime.now()
        for num_rows, total_rows in progress:
            logger.debug("num_rows=" + str(num_rows) + ",total_rows=" + str(total_rows))
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
            logger.debug(rv)

    def show_tables(self):
        ch_tables_list = list()
        sql = "show tables"
        for table_cell in self.execute(sql):
            ch_tables_list.append(table_cell[0])
        logger.debug("ch_tables_list=" + str(ch_tables_list))
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
        sql = "select host_address from system.clusters where cluster='" + clickhouse_cluster + "'"
        for host in self.execute(sql):
            ch_hosts_list.append(host[0])
        logger.debug("ch_hosts_list=" + str(ch_hosts_list))
        return ch_hosts_list

    def table_rows(self, clickhouse_table_name, database=None):
        if not database:
            database = self.DEFAULT_DATABASE
        sql = "select count(*) from " + database + "." + clickhouse_table_name + " SETTINGS max_execution_time = " + str(ClickHouseUtils.MAX_EXECUTION_TIME)
        num = self.execute(sql)[0][0]
        logger.debug(self.host + ",num=" + str(num))
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

    def csv_2_clickhouse(self, csv_file_abs_path, clickhouse_table_name, columns,
                         clickhouse_database_name=None, is_auto_create=True):
        if not columns:
            #类型推断
            import pandas as pd
            df = pd.read_csv(csv_file_abs_path)
            columns = dict()
            for column_name, column_type in df.dtypes.to_dict().items():
                if 'Unnamed: 0' == column_name:
                    column_name = "id"
                if "int" in column_type.name:
                    columns[column_name] = int
                elif "float" in column_type.name:
                    columns[column_name] = float
                else:
                    columns[column_name] = str
            logger.debug(columns)
        if not clickhouse_database_name:
            clickhouse_database_name = self.DEFAULT_DATABASE
        def iter_csv(filename):
            with open(filename, 'r') as f:
                reader = csv.DictReader(f)
                for line in reader:
                    res = dict()
                    for k, v in line.items():
                        if not k:
                            k = "id"
                        if k in columns:
                            value = columns[k](v)
                        else:
                            value = v
                        res[k] = value
                    yield res

        sql_statement = ""
        type_map = {
            int: "Int64",
            float: "Float64",
            str: "String"
        }
        for k, v in columns.items():
            sql_statement += k + " " + type_map[v] + ","
        sql_statement = sql_statement[:-1]
        if is_auto_create:
            self.create_table(clickhouse_table_name, sql_statement, type="memory",
                              database_name=clickhouse_database_name)
        self.execute('INSERT INTO ' + clickhouse_database_name + '.' + clickhouse_table_name + ' VALUES',
                     iter_csv(csv_file_abs_path))
        self.close()

    def clickhouse_2_csv(self, clickhouse_table_name, csv_file_abs_path, clickhouse_database_name=None):
        if not clickhouse_database_name:
            clickhouse_database_name = self.DEFAULT_DATABASE
        if self.table_rows(clickhouse_table_name) > ClickHouseUtils.MAX_CSV_ROWS:
            raise Exception("table rows too large")
        with open(csv_file_abs_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([column[0] for column in
                             self.execute('DESC ' + clickhouse_database_name + '.' + clickhouse_table_name)])
            for row in self.execute('SELECT * FROM ' + clickhouse_database_name + '.' + clickhouse_table_name):
                writer.writerow(row)
        self.close()

    def create_table(self, table_name, col_statement, type="local", format="ORC", location=None, cluster=None,
                     partition_column=None, primary_column=None, database_name=None):
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
                database_name, table_name + "_local", cluster, default_primary, col_statement, partition_column,
                primary_column, self.DEFAULT_TTL_DAY)
                logger.debug("sql=" + str(sql))
                self.execute(sql)
                sql = """
                         CREATE TABLE IF NOT EXISTS %s.%s on cluster %s (%s %s, `day_` Date) 
                         ENGINE = Distributed(%s, %s, %s, rand())
                      """ % (
                    database_name, table_name, cluster, default_key, col_statement, cluster, database_name,
                    table_name + "_local")
                logger.debug("sql=" + str(sql))
                self.execute(sql)
            else:
                sql = """
                         CREATE TABLE IF NOT EXISTS %s.%s (%s %s, `day_` Date DEFAULT toDate(now())) 
                         ENGINE = MergeTree() %s ORDER BY id TTL (day_ + toIntervalDay(%s)) + toIntervalHour(7)
                      """ % (database_name, table_name, default_primary, col_statement, partition_column,
                             self.DEFAULT_TTL_DAY)
                logger.debug("sql=" + str(sql))
                self.execute(sql)
        elif type == "hdfs":
            sql = """
                         CREATE TABLE IF NOT EXISTS %s.%s (%s) ENGINE = HDFS('%s', %s) 
                  """ % (database_name, table_name, col_statement, location, format)
            logger.debug("external table sql=" + str(sql) + " " + str(self.client.settings))
            self.execute(sql)
        elif type == "memory":
            sql = """
                         CREATE TABLE IF NOT EXISTS %s.%s (%s) Engine = Memory
                  """ % (database_name, table_name, col_statement)
            logger.debug("sql=" + str(sql))
            self.execute(sql)
        else:
            raise Exception("type value exception")
        timestamp_end = _perf_counter()
        logger.debug("create table done, " + 'time cost: %s Seconds' % (timestamp_end - timestamp_start))

    def insert_table(self, clickhouse_table_name, external_table_name, col_name_statement, col_if_statement):
        start = time.perf_counter()
        sql = """
                insert into %s.%s(%s) select %s from %s.%s SETTINGS max_execution_time = %s
                """ % (
            self.DEFAULT_DATABASE, clickhouse_table_name, col_name_statement, col_if_statement,
            self.DEFAULT_DATABASE, external_table_name, str(ClickHouseUtils.MAX_EXECUTION_TIME))
        logger.debug(self.host + ", insert into sql=" + str(sql))
        self.execute(sql)
        end = time.perf_counter()
        logger.debug(self.host + ", insert into sql done time cost: " + str(end - start) + " Seconds")

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
        logger.debug(field_names)
        logger.debug(field_types)
        logger.debug(field_raw_types)
        return field_names, field_types, field_raw_types

    def is_distribute_table(self, clickhouse_table_name, database=None):
        if not database:
            database = self.DEFAULT_DATABASE
        show_create_table = self.execute("show create table " + database + "." + clickhouse_table_name)
        if "ENGINE = Distributed" in show_create_table[0][0]:
            return True
        else:
            return False

    """
    clickhouse field type trans tdw field type
    """

    def field_type_map(self, col_type):
        if "Nullable" in col_type:
            col_type = col_type["Nullable(".__len__():-1]
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
            logger.debug("col_name=" + str(col_name) + ",col_clickhouse_type=" + str(
                col_clickhouse_type) + ",col_tdw_type=" + str(col_tdw_type))
            create_clickhouse_sql_statement += col_name + " " + col_clickhouse_type + ","
            create_tdw_sql_statement += col_name + " " + col_tdw_type + " comment '" + col_name + "',"
        create_clickhouse_sql_statement = create_clickhouse_sql_statement[:-1]
        create_tdw_sql_statement = create_tdw_sql_statement[:-1]
        return create_clickhouse_sql_statement, create_tdw_sql_statement

    @classmethod
    def clickhouse_2_tdw(self, clickhouse_table_name, tdw_database_name, tdw_table_name, spark_session, cmk=None,
                         is_auto_create=True):
        clickhouse_utils = ClickHouseUtils()
        num = clickhouse_utils.table_rows(clickhouse_table_name)
        logger.info("clickhouse table count=" + str(num))
        if num > ClickHouseUtils.MAX_ROWS:
            raise Exception("clickhouse table rows num too big, >" + str(ClickHouseUtils.MAX_ROWS) + " not support")
        field_names, field_types, field_raw_types = clickhouse_utils.get_table_meta(clickhouse_table_name)
        col_list = list()
        for i in range(field_names.__len__()):
            col_list.append([field_names[i], field_types[i], field_names[i]])
        logger.debug(col_list)
        sql_statement, create_tdw_sql_statement = clickhouse_utils.get_sql_statement(field_names, field_types,
                                                                                     field_raw_types)
        from .tdw import TDWUtils
        export_hdfs_path = TDWUtils.BASE_HDFS_PATH + tdw_table_name + "_hdfs_export"
        tdw_utils = TDWUtils(spark_session)
        if tdw_utils.table_exits(tdw_database_name, tdw_table_name):
            raise Exception("tdw table is already exist")
        tdw_utils.hdfs_mkdir_and_chmod(TDWUtils.NAME_SPACE + export_hdfs_path)
        if is_auto_create:
            from ..common.idex import IdexUtils
            user = get_user()
            if not cmk:
                raise Exception("please input cmk arg")
            logger.debug("user=" + user + ",cmk=" + cmk)
            idex_utils = IdexUtils(user=user, cmk=cmk)
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
                tdw_database_name, tdw_table_name, create_tdw_sql_statement, TDWUtils.NAME_SPACE + export_hdfs_path)
            logger.debug(sql)
            idex_utils.run_sql(sql)
        if clickhouse_utils.is_distribute_table(clickhouse_table_name=clickhouse_table_name):
            # distribute table
            def clickhouse_node_job(clickhouse_utils, clickhouse_hdfs_table_name, sql_statement, file_name):
                logger.debug(clickhouse_utils.host)
                location_file = TDWUtils.BASE_HDFS_PATH + tdw_table_name + "_" + file_name
                logger.debug("clickhouse_node_job, location_file=" + location_file)
                clickhouse_utils.create_table(clickhouse_hdfs_table_name, sql_statement, type="hdfs", format="Parquet",
                                              location=TDWUtils.NAME_SPACE + location_file)
                clickhouse_utils.insert_table(clickhouse_hdfs_table_name, clickhouse_table_name + "_local",
                                              ",".join(field_names), ",".join(field_names))
                logger.debug(
                    "insert success, sink clickhouse_hdfs_table_name=" + clickhouse_hdfs_table_name
                    + ",source clickhouse_table_name=" + clickhouse_table_name)
                logger.debug("location_file=" + location_file + ",export_hdfs_path=" + export_hdfs_path)
                tdw_utils.hdfs_move(location_file, export_hdfs_path + "/" + file_name)
                clickhouse_utils.close()

            with ThreadPoolExecutor(max_workers=clickhouse_utils.cluster_hosts_len) as pool:
                all_task = list()
                for i in range(clickhouse_utils.cluster_hosts_len):
                    timestamp = str(int(time.time()))
                    file_name = "part" + "_" + clickhouse_utils.cluster_hosts[i].replace(".", "_") \
                                + "_" + timestamp + ".parquet"
                    clickhouse_hdfs_table_name = clickhouse_table_name + "_hdfs_export_" + timestamp
                    future = pool.submit(clickhouse_node_job,
                                         ClickHouseUtils(host=clickhouse_utils.cluster_hosts[i]),
                                         clickhouse_hdfs_table_name, sql_statement, file_name)
                    future.add_done_callback(handle_exception)
                    all_task.append(future)
                wait(all_task, timeout=None, return_when=ALL_COMPLETED)
        else:
            # local table
            timestamp = str(int(time.time()))
            location_file = TDWUtils.BASE_HDFS_PATH + "part" + timestamp + ".parquet"
            clickhouse_hdfs_table_name = clickhouse_table_name + "_hdfs_export_" + timestamp
            self.create_table(clickhouse_hdfs_table_name, sql_statement, type="hdfs",
                              format="Parquet", location=TDWUtils.NAME_SPACE + location_file)
            self.insert_table(clickhouse_hdfs_table_name, clickhouse_table_name, ",".join(field_names),
                              ",".join(field_names))
            logger.debug(location_file)
            logger.debug(export_hdfs_path)
            tdw_utils.hdfs_move(location_file, export_hdfs_path + "/part" + timestamp + ".parquet")
        clickhouse_utils.close()

    # select from distribute table
    @classmethod
    def clickhouse_2_dataframe(self, spark, clickhouse_table_name, partition_num, clickhouse_database_name=None, batch_size=100000):
        predicates = list()
        clickhouse_utils = ClickHouseUtils()
        if not clickhouse_database_name:
            clickhouse_database_name = clickhouse_utils.DEFAULT_DATABASE
        num = clickhouse_utils.table_rows(clickhouse_table_name, clickhouse_database_name)
        logger.debug("clickhouse table count=" + str(num))
        if num == 0:
            raise Exception("clickhouse table rows is empty")
        elif num > ClickHouseUtils.MAX_ROWS:
            raise Exception("clickhouse table rows num too big, >" + str(ClickHouseUtils.MAX_ROWS) + " not support")
        step = math.floor(num / partition_num + 1)
        logger.debug("step=" + str(step))
        for i in range(partition_num - 1):
            predicates.append("1 = 1 limit " + str(step * i) + ", " + str(step))
        predicates.append(
            "1 = 1 limit " + str(step * (partition_num - 1)) + ", " + str(num - step * (partition_num - 1)))
        logger.debug("predicates=" + str(predicates))
        clickhouse_utils.close()
        return spark.read.option("batch_size", batch_size).jdbc(url=ClickHouseUtils.get_jdbc_connect_string(clickhouse_database_name),
                               table=clickhouse_table_name, predicates=predicates,
                               properties=ClickHouseUtils.get_jdbc_properties())

    # select from everyone node
    @classmethod
    def clickhouse_2_dataframe_distribute(self, spark, clickhouse_table_name, partition_num, clickhouse_database_name):
        dataframe = None
        global_clickhouse_utils = ClickHouseUtils()
        for jdbc_string, host in global_clickhouse_utils.get_jdbc_connect_strings(clickhouse_database_name):
            logger.debug("jdbc_string=" + jdbc_string + ",host=" + host)
            predicates = list()
            clickhouse_utils = ClickHouseUtils(host=host)
            num = clickhouse_utils.table_rows(clickhouse_table_name, clickhouse_database_name)
            logger.debug("clickhouse table count=" + str(num))
            if num > ClickHouseUtils.MAX_ROWS:
                raise Exception("clickhouse table rows num too big, >" + str(ClickHouseUtils.MAX_ROWS) + " not support")
            step = math.floor(num / partition_num + 1)
            logger.debug("num=" + num + ",step=" + str(step))
            for i in range(partition_num - 1):
                predicates.append("1 = 1 limit " + str(step * i) + ", " + str(step))
            predicates.append(
                "1 = 1 limit " + str(step * (partition_num - 1)) + ", " + str(num - step * (partition_num - 1)))
            logger.debug("predicates=" + str(predicates))
            clickhouse_utils.close()
            if dataframe:
                dataframe = dataframe.union(spark.read.jdbc(url=jdbc_string, table=clickhouse_table_name,
                                                            predicates=predicates,
                                                            properties=ClickHouseUtils.get_jdbc_properties()))
            else:
                dataframe = spark.read.jdbc(url=jdbc_string, table=clickhouse_table_name,
                                            predicates=predicates,
                                            properties=ClickHouseUtils.get_jdbc_properties())
        global_clickhouse_utils.close()
        return dataframe

    def __materialize_table(self, clickhouse_utils, select_sql, sql_table_name, database, clickhouse_view_name,
                            primary_column, is_use_local):
        logger.debug("materialize view doing")
        if is_use_local:
            clickhouse_view_name_real = clickhouse_view_name + "_local"
            select_sql = select_sql.replace(sql_table_name, sql_table_name + "_local")
        else:
            clickhouse_view_name_real = clickhouse_view_name
        if sql_table_name:
            select_sql = select_sql.replace(sql_table_name, database + "." + sql_table_name)

        calcite_select_sql = clickhouse_utils.sqlgateway_execute(select_sql, is_calcite_parse=True)
        if "error message" in calcite_select_sql:
            raise Exception("sql execute or sqlparse is error, please check, info:" + calcite_select_sql)

        if is_use_local and self.CLUSTER:
            sql = "CREATE TABLE " + clickhouse_view_name_real + " on cluster " + self.CLUSTER \
                  + " ENGINE = MergeTree ORDER BY " + primary_column + " AS " + calcite_select_sql
        else:
            sql = "CREATE TABLE " + clickhouse_view_name_real \
                  + " ENGINE = MergeTree ORDER BY " + primary_column + " AS " + calcite_select_sql
        logger.debug("insert table running")
        logger.debug(sql)
        start = time.perf_counter()
        clickhouse_utils.execute(sql)
        end = time.perf_counter()
        logger.debug("insert table done" + 'time cost: ' + str(end - start) + ' Seconds')

        if is_use_local:
            if self.CLUSTER:
                sql = "DROP VIEW if exists " + clickhouse_view_name + " on cluster " + self.CLUSTER
            else:
                sql = "DROP VIEW if exists " + clickhouse_view_name
            logger.debug(sql)
            clickhouse_utils.execute(sql)
            if self.CLUSTER:
                sql = "CREATE TABLE " + clickhouse_view_name + " on cluster " + self.CLUSTER \
                      + " as " + clickhouse_view_name + "_local" \
                      + " ENGINE = Distributed(" + self.CLUSTER + ", " + database + ", " + clickhouse_view_name \
                      + "_local" + ", rand())"
                clickhouse_utils.execute(sql)
                logger.debug(sql)
                logger.debug("materialize view done")


    """
    primary_column 是clickhouse MergeTree表引擎用户排序的列，在该列实现跳数索引，查询频率大的列建议放在前面做索引
    is_force_materialize 是否强制将视图物化为物理表, 会执行计算落表
    is_sql_complete sql_statement字段是否提供的是完整的sql声明
    is_use_local 是否要转化为本地表的方式执行
    """
    @classmethod
    def create_view(self, clickhouse_view_name, sql_statement, sql_table_name=None, sql_where=None, sql_group_by=None,
                    sql_limit=None, primary_column="tuple()", database=None, is_force_materialize=False, is_sql_complete=False, is_use_local=True):
        clickhouse_utils = ClickHouseUtils()
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
            select_sql = " SELECT " + sql_statement + " FROM " + sql_table_name + " \n" + sql_where + sql_group_by + sql_limit
        select_sql = select_sql.replace('{database}', database)
        logger.debug("raw sql = \n" + select_sql)
        if sql_limit and "LIMIT" not in sql_limit.upper():
            select_sql_example = select_sql + " LIMIT 100"
        else:
            select_sql_example = select_sql

        # 判断sql语句是否有效,可执行
        try:
            logger.debug("check sql whether valid, now...")
            sqlgateway_res = clickhouse_utils.sqlgateway_execute(select_sql_example)
            if "error message" in sqlgateway_res:
                raise Exception("sql execute or sqlparse is error, please check, info:" + sqlgateway_res)
        except Exception:
            clickhouse_utils.close()
            raise Exception("sql execute or sqlparse is error, please check")

        if is_force_materialize:
            clickhouse_utils.__materialize_table(clickhouse_utils, select_sql, sql_table_name, database,
                                                 clickhouse_view_name, primary_column, is_use_local)
        else:
            if clickhouse_utils.CLUSTER:
                sql = "CREATE VIEW " + database + "." + clickhouse_view_name + " on cluster " + clickhouse_utils.CLUSTER \
                      + " AS " + select_sql
            else:
                sql = "CREATE VIEW " + database + "." + clickhouse_view_name + " AS " + select_sql
            clickhouse_utils.execute(sql)
            fields = list()
            for i in clickhouse_utils.execute('DESC ' + database + '.' + clickhouse_view_name):
                fields.append(i[0])
            logger.debug("view table fields =" + str(fields))
            if primary_column != 'tuple()' and primary_column not in fields:
                raise Exception("primary_column set not valid")
            rows_number = clickhouse_utils.table_rows(clickhouse_view_name, database=database)
            logger.debug("view rows number is " + str(rows_number))
            if rows_number < ClickHouseUtils.MAX_VIEW_MATERIALIZE_ROWS:
                clickhouse_utils.execute("DROP VIEW " + database + "." + clickhouse_view_name)
                clickhouse_utils.__materialize_table(clickhouse_utils, select_sql, sql_table_name, database,
                                                     clickhouse_view_name, primary_column, is_use_local)
            logger.debug("create view success")
        clickhouse_utils.close()

    @classmethod
    def drop_view(self, clickhouse_view_name, database=None):
        clickhouse_utils = ClickHouseUtils()
        if not database:
            database = clickhouse_utils.DEFAULT_DATABASE
        try:
            table_desc = clickhouse_utils.show_create_tables(clickhouse_view_name)
        except Exception:
            return
        if "CREATE VIEW " in table_desc:
            if clickhouse_utils.CLUSTER:
                sql = "DROP VIEW if exists " + database + "." + clickhouse_view_name + " on cluster " \
                      + clickhouse_utils.CLUSTER
            else:
                sql = "DROP VIEW if exists " + database + "." + clickhouse_view_name
            clickhouse_utils.execute(sql)
        else:
            if clickhouse_utils.CLUSTER:
                sql = "DROP TABLE if exists " + database + "." + clickhouse_view_name + " on cluster " \
                    + clickhouse_utils.CLUSTER
                clickhouse_utils.execute(sql)
                sql = "DROP TABLE if exists " + database + "." + clickhouse_view_name + "_local on cluster " \
                    + clickhouse_utils.CLUSTER
                clickhouse_utils.execute(sql)
            else:
                sql = "DROP TABLE if exists " + database + "." + clickhouse_view_name
                clickhouse_utils.execute(sql)
                sql = "DROP TABLE if exists " + database + "." + clickhouse_view_name + "_local"
                clickhouse_utils.execute(sql)
        clickhouse_utils.close()

    @classmethod
    def drop_partition(self, clickhouse_view_name, clickhouse_partition_name, database=None):
        clickhouse_utils = ClickHouseUtils()
        if not database:
            database = clickhouse_utils.DEFAULT_DATABASE
        if clickhouse_utils.CLUSTER:
            sql = "alter table " + database + "." + clickhouse_view_name + "_local on cluster " + clickhouse_utils.CLUSTER \
                  + " drop partition " + clickhouse_partition_name
        else:
            sql = "alter table " + database + "." + clickhouse_view_name + "_local drop partition " + clickhouse_partition_name
        clickhouse_utils.execute(sql)
        clickhouse_utils.close()

    def close(self):
        if self.client:
            self.client.disconnect()


if __name__ == '__main__':
    clickhouse_utils = ClickHouseUtils()
    clickhouse_utils.close()
