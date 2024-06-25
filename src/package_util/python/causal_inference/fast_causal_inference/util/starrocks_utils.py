import csv
from time import perf_counter as _perf_counter
import pymysql

from fast_causal_inference.common import get_context
from datetime import datetime

from fast_causal_inference.util.sqlgateway import SqlGateWayConn


"""
Starrocks utils: jdbc connector, and other utils.
"""


class StarRocksUtils(object):
    # 5min
    JDBC_ARGS = "?charset=utf8"
    MAX_ROWS = 160 * 10000 * 10000
    MAX_CSV_ROWS = 1000 * 10000
    MAX_VIEW_MATERIALIZE_ROWS = MAX_CSV_ROWS
    MAX_EXECUTION_TIME = 15 * 60

    def __init__(self, host=None, database=None, rand=False, device_id=None):
        if device_id is None:
            device_id = self._find_valid_device()

        device_info_dict = StarRocksUtils.get_device_info_dict()

        self.DEFAULT_DATABASE = device_info_dict[device_id]["starrocks_database"]
        self.DEFAULT_HOST = device_info_dict[device_id]["starrocks_launch_host"]
        self.DEFAULT_PORT = device_info_dict[device_id]["starrocks_port"]
        self.DEFAULT_HTTP_PORT = device_info_dict[device_id]["starrocks_http_port"]
        self.DEFAULT_USER = device_info_dict[device_id]["starrocks_user"]
        self.DEFAULT_PASSWORD = device_info_dict[device_id]["starrocks_password"]
        self.DEFAULT_TTL_DAY = device_info_dict[device_id]["ttl"]
        if (
            "ip_mapping" in device_info_dict[device_id]
            and device_info_dict[device_id]["ip_mapping"].__len__() > 0
        ):
            self.IP_MAPPING = device_info_dict[device_id]["ip_mapping"][0]
        else:
            self.IP_MAPPING = None
        self.JDBC_PROPERTIES = StarRocksUtils.get_jdbc_properties(device_id)
        if not database:
            database = self.DEFAULT_DATABASE
        if not host:
            self.host = self.DEFAULT_HOST
        else:
            self.host = host
        get_context().logger.debug(
            f"try to connnect to user/passwd/host/port/db({self.DEFAULT_USER}/{self.DEFAULT_PASSWORD}/{self.host}/{self.DEFAULT_PORT}/{database})"
        )

        self._db = pymysql.connect(
            host=self.host,
            user=self.DEFAULT_USER,
            passwd=self.DEFAULT_PASSWORD,
            db=database,
            port=self.DEFAULT_PORT,
        )
        self.sql_instance = SqlGateWayConn(device_id=device_id, db_name=database)

    def execute(self, sql):
        cursor = self._db.cursor()
        get_context().logger.debug(f"Execute sql:\n{sql}")
        try:
            # 执行sql语句
            cursor.execute(sql)
            # 提交到数据库执行
            self._db.commit()
            return cursor.fetchall()
        except Exception as e:
            # 如果发生错误则回滚
            self._db.rollback()
            raise Exception(f"Unable to exec sql:\n{sql}\nError Msg:{e}")

    @classmethod
    def _find_valid_device(cls):
        device_id = None
        datasource = get_context().project_conf["datasource"]
        for device in datasource:
            if device.get("starrocks_database") is not None:
                device_id = device.get("device_id")
                break
        if device_id is None:
            raise Exception(f"Unable to get valid starrocks datasource.")
        return device_id

    @classmethod
    def get_device_info_dict(self):
        device_info_dict = dict()
        for datasource in get_context().project_conf["datasource"]:
            device_info_dict[datasource["device_id"]] = datasource
        return device_info_dict

    @classmethod
    def get_jdbc_connect_string(self, database=None, device_id=None):
        if not device_id:
            device_id = self._find_valid_device()

        device_info_dict = StarRocksUtils.get_device_info_dict()
        if not database:
            database = device_info_dict[device_id]["starrocks_database"]
        return (
            "jdbc:mysql://"
            + device_info_dict[device_id]["starrocks_launch_host"]
            + ":"
            + str(device_info_dict[device_id]["starrocks_port"])
            + "/"
            + database
            + StarRocksUtils.JDBC_ARGS
        )

    @classmethod
    def get_jdbc_properties(self, device_id=None):
        if not device_id:
            device_id = self._find_valid_device()

        device_info_dict = StarRocksUtils.get_device_info_dict()
        return {
            "driver": "com.mysql.jdbc.Driver",
            "user": device_info_dict[device_id]["starrocks_user"],
            "password": device_info_dict[device_id]["starrocks_password"],
        }

    @classmethod
    def get_default_sr_utils(cls):
        return StarRocksUtils()

    def get_jdbc_connect_strings(self, database=None):
        if not database:
            database = self.DEFAULT_DATABASE
        jdbc_strings = list()
        for host in self.cluster_hosts:
            jdbc_strings.append(
                (
                    "jdbc:mysql://"
                    + host
                    + ":"
                    + str(self.DEFAULT_HTTP_PORT)
                    + "/"
                    + database
                    + StarRocksUtils.JDBC_ARGS,
                    host,
                )
            )
        return jdbc_strings

    def show_tables(self):
        ch_tables_list = list()
        sql = "show tables"
        for table_cell in self.execute(sql):
            ch_tables_list.append(table_cell[0])
        get_context().logger.debug("ch_tables_list=" + str(ch_tables_list))
        return ch_tables_list

    def show_create_table(self, starrocks_table_name):
        sql = "show create table " + starrocks_table_name
        try:
            desc = self.execute(sql)
            print(desc[0][1])
        except Exception as e:
            if "doesn't exist" in repr(e):
                raise Exception("table is not exist, please check")

    def table_rows(self, starrocks_table_name, database=None):
        if not database:
            database = self.DEFAULT_DATABASE
        sql = "select count(*) from " + database + "." + starrocks_table_name
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

    def csv_2_starrocks(
        self,
        csv_file_abs_path,
        starrocks_table_name,
        columns,
        starrocks_database_name=None,
        is_auto_create=True,
    ):
        if not columns:
            # 类型推断
            import pandas as pd

            df = pd.read_csv(csv_file_abs_path)
            columns = dict()
            for column_name, column_type in df.dtypes.to_dict().items():
                if "Unnamed: 0" == column_name:
                    column_name = "tmptbl_id_col_"
                if "int" in column_type.name:
                    columns[column_name] = int
                elif "float" in column_type.name:
                    columns[column_name] = float
                elif "double" in column_type.name:
                    columns[column_name] = float
                else:
                    columns[column_name] = str
            get_context().logger.debug(columns)
        if not starrocks_database_name:
            starrocks_database_name = self.DEFAULT_DATABASE

        def iter_csv(filename):
            with open(filename, "r") as f:
                reader = csv.DictReader(f)
                for line in reader:
                    res = dict()
                    for k, v in line.items():
                        if not k:
                            k = "tmptbl_id_col_"
                        if k in columns:
                            value = columns[k](v)
                        else:
                            value = v
                        res[k] = value
                    yield res

        sql_statement = ""
        type_map = {int: "bigint", float: "double", str: "varchar"}
        for k, v in columns.items():
            sql_statement += k + " " + type_map[v] + ","
        sql_statement = sql_statement[:-1]
        if is_auto_create:
            self.create_table(
                starrocks_table_name,
                sql_statement,
                database_name=starrocks_database_name,
            )
        self.execute(
            "INSERT INTO "
            + starrocks_database_name
            + "."
            + starrocks_table_name
            + " VALUES",
            iter_csv(csv_file_abs_path),
        )

    def create_table(
        self,
        table_name,
        col_statement,
        primary_column=None,
        database_name=None,
    ):
        timestamp_start = _perf_counter()
        if not database_name:
            database_name = self.DEFAULT_DATABASE
        if primary_column is None:
            default_primary = "`tmptbl_id_col_` BIGINT NOT NULL"
            primary_column = "`tmptbl_id_col_`"
        else:
            default_primary = primary_column
        sql = """
                    CREATE TABLE IF NOT EXISTS %s.%s (%s, %s, `tmptbl_day_col_` DATETIME DEFAULT CURRENT_TIMESTAMP) 
                    ENGINE=OLAP
                    DUPLICATE KEY(%s) 
                    PARTITION BY RANGE(tmptbl_day_col_)()
                    DISTRIBUTED BY HASH(%s)
                    PROPERTIES(
                        "dynamic_partition.enable" = "true",
                        "dynamic_partition.time_unit" = "DAY",
                        "dynamic_partition.start" = "-%s",
                        "dynamic_partition.end" = "%s",
                        "dynamic_partition.prefix" = "p",
                        "dynamic_partition.history_partition_num" = "0"
                    )
                """ % (
            database_name,
            table_name,
            default_primary,
            col_statement,
            primary_column,
            primary_column,
            self.DEFAULT_TTL_DAY,
            self.DEFAULT_TTL_DAY,
        )
        get_context().logger.debug("sql=" + str(sql))
        self.execute(sql)
        timestamp_end = _perf_counter()
        get_context().logger.debug(
            "create table done, "
            + "time cost: %s Seconds" % (timestamp_end - timestamp_start)
        )

    def get_table_meta(self, starrocks_table_name, database=None):
        if not database:
            database = self.DEFAULT_DATABASE
        desc_table = self.execute("desc " + database + "." + starrocks_table_name)
        field_names = list()
        field_types = list()
        field_raw_types = list()
        for field in desc_table:
            name = field[0]
            type = field[1]
            field_names.append(name)
            field_types.append(self.field_type_map(type))
            field_raw_types.append(type)
        get_context().logger.debug(field_names)
        get_context().logger.debug(field_types)
        get_context().logger.debug(field_raw_types)
        return field_names, field_types, field_raw_types

    """
    starrocks field type trans tdw field type
    """

    @classmethod
    def field_type_map(cls, col_type):
        if "Nullable" in col_type:
            col_type = col_type["Nullable(".__len__() : -1]
        if col_type == "bigint":
            col_trans_type = "bigint"
        elif col_type == "int":
            col_trans_type = "int"
        elif col_type == "tinyint":
            col_trans_type = "int"
        elif col_type == "smallint":
            col_trans_type = "int"
        elif col_type == "double":
            col_trans_type = "float"
        elif col_type == "String":
            col_trans_type = "string"
        elif "varchar" in col_type:
            col_trans_type = "string"
        elif col_type in ["Date", "date"]:
            col_trans_type = "date"
        elif col_type in ["DateTime", "datetime"]:
            col_trans_type = "timestamp"
        elif "decimal" in col_type:
            col_trans_type = "double"
        elif "int" in col_type:
            col_trans_type = "int"
        else:
            raise Exception(col_type + " col_type is not support")
        return col_trans_type

    @classmethod
    def spark_type_to_starrocks(cls, col_type):
        if "Nullable" in col_type:
            col_type = col_type["Nullable(".__len__() : -1]
        if col_type == "Int64":
            col_trans_type = "bigint"
        elif col_type == "int":
            col_trans_type = "int"
        elif col_type == "tinyint":
            col_trans_type = "int"
        elif col_type == "smallint":
            col_trans_type = "int"
        elif col_type == "double":
            col_trans_type = "float"
        elif col_type == "String":
            col_trans_type = "string"
        elif "varchar" in col_type:
            col_trans_type = "string"
        elif col_type == "Date":
            col_trans_type = "date"
        elif col_type == "DateTime":
            col_trans_type = "timestamp"
        else:
            raise Exception(col_type + " col_type is not support")
        return col_trans_type

    def get_sql_statement(self, col_names, col_tdw_types, col_starrocks_types):
        create_starrocks_sql_statement = ""
        create_tdw_sql_statement = ""
        for col_index in range(len(col_names)):
            col_name = col_names[col_index]
            col_starrocks_type = col_starrocks_types[col_index]
            col_tdw_type = col_tdw_types[col_index]
            get_context().logger.debug(
                "col_name="
                + str(col_name)
                + ",col_starrocks_type="
                + str(col_starrocks_type)
                + ",col_tdw_type="
                + str(col_tdw_type)
            )
            create_starrocks_sql_statement += col_name + " " + col_starrocks_type + ","
            create_tdw_sql_statement += (
                col_name + " " + col_tdw_type + " comment '" + col_name + "',"
            )
        create_starrocks_sql_statement = create_starrocks_sql_statement[:-1]
        create_tdw_sql_statement = create_tdw_sql_statement[:-1]
        return create_starrocks_sql_statement, create_tdw_sql_statement

    @classmethod
    def starrocks_2_dataframe(
        self,
        spark,
        starrocks_table_name,
        starrocks_database_name=None,
        batch_size=100000,
    ):
        sr_utils = StarRocksUtils()
        if not starrocks_database_name:
            starrocks_database_name = sr_utils.DEFAULT_DATABASE

        num = sr_utils.table_rows(starrocks_table_name, starrocks_database_name)
        get_context().logger.debug("starrocks table count=" + str(num))
        if num == 0:
            raise Exception("starrocks table rows is empty")
        elif num > StarRocksUtils.MAX_ROWS:
            raise Exception(
                "starrocks table rows num too big, >"
                + str(StarRocksUtils.MAX_ROWS)
                + " not support"
            )

        columns = sr_utils.execute(
            "desc " + starrocks_database_name + "." + starrocks_table_name
        )
        get_context().logger.debug(columns)
        return (
            spark.read.format("starrocks")
            .option(
                "starrocks.fenodes",
                f"{sr_utils.DEFAULT_HOST}:{sr_utils.DEFAULT_HTTP_PORT}",
            )
            .option("user", sr_utils.DEFAULT_USER)
            .option("password", sr_utils.DEFAULT_PASSWORD)
            .option(
                "starrocks.table.identifier",
                f"{starrocks_database_name}.{starrocks_table_name}",
            )
            .option("starrocks.batch.size", batch_size)
            .load()
        )

    def close(self):
        if self._db:
            self._db.close()

    def csv_2_starrocks(
        self,
        csv_file_abs_path,
        starrocks_table_name,
        columns,
        starrocks_database_name=None,
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
        if not starrocks_database_name:
            starrocks_database_name = self.DEFAULT_DATABASE

        def iter_csv(filename):
            with open(filename, "r") as f:
                reader = csv.DictReader(f)
                for idx, line in enumerate(reader):
                    res = dict()
                    for k, v in line.items():
                        if not k:
                            k = "id"
                        if k in columns:
                            value = columns[k](v)
                        else:
                            value = v
                        res[k] = value
                    if len(res) == 0:
                        continue
                    res["tmptbl_id_col_"] = idx
                    res["tmptbl_day_col_"] = (
                        "'" + str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + "'"
                    )
                    yield res

        sql_statement = ""
        type_map = {int: "bigint", float: "double", str: "String"}
        is_str = {}
        for k, v in columns.items():
            sql_statement += k + " " + type_map[v] + ","
            if v == str:
                is_str[k] = True
        sql_statement = sql_statement[:-1]
        if is_auto_create:
            self.create_table(
                starrocks_table_name,
                sql_statement,
                database_name=starrocks_database_name,
            )
        values = []
        cols = []
        for row in iter_csv(csv_file_abs_path):
            if len(cols) == 0:
                for col in row:
                    cols.append(col)
            assert len(row) == len(cols)
            values.append(
                "("
                + ", ".join(
                    map(
                        lambda col: str(row[col])
                        if is_str.get(col) is None
                        else f"'{str(row[col])}'",
                        cols,
                    )
                )
                + ")"
            )
        self.execute(
            "INSERT INTO "
            + starrocks_database_name
            + "."
            + starrocks_table_name
            + "("
            + ", ".join(cols)
            + ")"
            + " VALUES"
            + ",\n".join(values)
        )
        self.close()

    
    def sqlgateway_execute(self, sql, is_calcite_parse=False):
        get_context().logger.debug("sqlgateway, sql=" + sql)
        return self.sql_instance.sql(
            sql=sql, is_calcite_parse=is_calcite_parse, is_dataframe=False
        )

    def create_view(self, view_name, sql_statement, is_table=False):
        if "error message" in sql_statement:
            raise Exception(
                "sql execute or sqlparse is error, please check, info:"
                + sql_statement
            )
        self.execute(f"""
            create {'view' if not is_table else 'table'} {view_name} as {sql_statement}
        """)

    def starrocks_2_pandas(
        self, starrocks_table_name, starrocks_database_name=None
    ):
        if not starrocks_database_name:
            starrocks_database_name = self.DEFAULT_DATABASE
        import pandas as pd
        if self.table_rows(starrocks_table_name) > StarRocksUtils.MAX_CSV_ROWS:
            raise Exception("table rows too large")
        
        csv_name = "/tmp/" + starrocks_table_name + ".csv"
        self.starrocks_2_csv(starrocks_table_name, csv_name, starrocks_database_name)
        return pd.read_csv(csv_name)


    def starrocks_2_csv(self, starrocks_table_name, csv_file_abs_path, starrocks_database_name=None):
        if not starrocks_database_name:
            starrocks_database_name = self.DEFAULT_DATABASE
        if self.table_rows(starrocks_table_name) > StarRocksUtils.MAX_CSV_ROWS:
            raise Exception("table rows too large")
        with open(csv_file_abs_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    column[0]
                    for column in self.execute(
                        "DESC " + starrocks_database_name + "." + starrocks_table_name
                    )
                ]
            )
            for row in self.execute(
                "SELECT * FROM "
                + starrocks_database_name
                + "."
                + starrocks_table_name
            ):
                writer.writerow(row)
        self.close()

    @classmethod
    def starrocks_2_tdw(
        self,
        session,
        starrocks_table,
        tdw_database,
        tdw_table,
        tdw_user,
        tdw_passward,
        group,
        overwrite=True,
        priPart=None,
    ):
        sr = StarRocksUtils()
        num = sr.table_rows(starrocks_table)
        get_context().logger.info("starrocks table count=" + str(num))
        if num > sr.MAX_ROWS:
            raise Exception(
                "starrocks table rows num too big, >"
                + str(sr.MAX_ROWS)
                + " not support"
            )
        field_names, field_types, field_raw_types = sr.get_table_meta(
            starrocks_table
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
        tdw.createTable(table_desc)

        SPARK_SESSION = session

        spark_df = self.starrocks_2_dataframe(SPARK_SESSION, starrocks_table)
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
