__all__ = [
    "readClickHouse",
    "readStarRocks",
    "readTdw",
    "readSparkDf",
    "readCsv",
    "DataFrame",
]

import random
from typing import Tuple

import pandas
import requests
import time

from fast_causal_inference.dataframe.format import format_db_result,convert_to_numeric
from fast_causal_inference.dataframe.format import testResult
from fast_causal_inference.dataframe import ais_dataframe_pb2 as DfPb
import re
import base64
from google.protobuf import json_format
import json
from google.protobuf.json_format import Parse
import copy
from fast_causal_inference.dataframe.df_base import (
    DfColumnNode,
    OlapEngineType,
    DfContext,
    DfColumnInternalNode,
    DfColumnLeafNode,
)
from fast_causal_inference.dataframe.functions import DfFnColWrapper
from fast_causal_inference.dataframe import functions as AisF
from fast_causal_inference.util.data_transformer import (
    clickhouse_2_csv,
    starrocks_2_csv,
    clickhouse_2_dataframe,
    dataframe_2_clickhouse,
    dataframe_2_starrocks,
    csv_2_clickhouse,
    csv_2_starrocks,
    starrocks_2_dataframe,
    clickhouse_2_pandas,
    starrocks_2_pandas,
)
from fast_causal_inference.dataframe import regression as AisRegressionF
from fast_causal_inference.dataframe import statistics as AisStatisticsF
from fast_causal_inference.util import get_user, ClickHouseUtils
from fast_causal_inference.common import get_context
import fast_causal_inference.lib.tools as AisTools
from fast_causal_inference.util.starrocks_utils import StarRocksUtils
from fast_causal_inference.dataframe.provider import FCIProvider


"""
syntax = "proto3";

enum ColumnType {
  Unknown = 0;
  String = 1;
  Int = 2;
  Float = 3;
  Bool = 4;
  Date = 5;
  DateTime = 6;
  Time = 7;
  UUID = 8;
  Array = 9;
}

enum TaskType {
  TASK_TYPE_FILL_SCHEMA = 0;
}

message Column {
  string name = 1;
  string alias = 2;
  ColumnType type = 3;
}

message Limit {
  int64 limit = 1;
  int64 offset = 2;
}

enum SourceType {
  ClickHouse = 0;
}

message ClickHouseSource {
  string table_name = 1;
  string database = 2;
}

message Source {
  SourceType type = 1;
  ClickHouseSource clickhouse = 2;
}

message Order {
  Column column = 1;
  bool desc = 2;
}

message DataFrame {
  repeated Column columns = 1;
  repeated string filters = 2;
  repeated Column group_by = 3;
  repeated Order order_by = 4;
  Limit limit = 5;
  Source source = 6;
};

message DataFrameRequest {
  DataFrame df = 1;
  TaskType task_type = 2;
  string rtx = 3;
  int64 device_id = 4;
}

"""


def getSuperName(column):
    if isinstance(column, DfPb.Column):
        return column.alias if column.alias else column.name
    else:
        raise Exception("type error")


class DataFrame:
    """
    This class is used to create a DataFrame object.
    """
    def __init__(self, olap_engine=OlapEngineType.CLICKHOUSE, provider=FCIProvider("global")):
        datasource = dict()
        self._provider = provider
        PROJECT_CONF = self._provider.project_conf

        for cell in PROJECT_CONF["datasource"]:
            datasource[cell["device_id"]] = cell
        self.device_id = None
        for device in datasource:
            if datasource[device].get(str(olap_engine) + "_database") is not None:
                self.device_id = device
                self.database = datasource[device][str(olap_engine) + "_database"]
                break
        if self.device_id is None:
            raise Exception(f"Unable to get any device of engine({olap_engine}).")
        self.data = DfPb.DataFrame()
        if olap_engine == OlapEngineType.STARROCKS:
            self.data.source.type = 1
        self.url = (
            PROJECT_CONF["sqlgateway"]["url"]
            + PROJECT_CONF["sqlgateway"]["dataframe_path"]
        )
        print(f"url: {self.url}")
        self.rtx = get_user()
        self._ctx = DfContext(engine=olap_engine, dataframe=self)
        self._select_list = []
        self._name_dict = {}
        self._need_agg = False
        self._origin_table_name = None

    def serialize(self):
        return self.data.SerializeToString()

    def serializeBase64(self):
        return base64.b64encode(self.serialize()).decode()

    def serializeJson(self):
        return json_format.MessageToJson(self.data)

    def deserialize(self, data):
        self.data.ParseFromString(data)

    def fill_result(self, limit = 200):
        new_df = copy.deepcopy(self)
        new_df.data.limit.limit = limit
        new_df._finalize()
        new_df.data.result = ""
        new_df.task_type = DfPb.TaskType.EXECUTE
        new_df.execute()
        return new_df

    def toPandas(self):
        """
        This function is used to convert the result of the dataframe to pandas.DataFrame
        """
        res = self.fill_result(500).data.result
        res = format_db_result(list(eval(res)))
        if res.shape[0] < 500:
            return convert_to_numeric(res)

        execute_sql = self.getExecutedSql()
        if self.engine == OlapEngineType.CLICKHOUSE:
            res = clickhouse_2_pandas(execute_sql, do_log=False)
        elif self.engine == OlapEngineType.STARROCKS:
            new_df = self.materializedView(is_physical_table=True, is_temp=True)
            res = starrocks_2_pandas(starrocks_table_name=new_df.getTableName(), do_log=False)
        else:
            raise Exception(f"olap `{self.engine} not supported.")
        return convert_to_numeric(res)

    @property
    def dtypes(self):
        if self.engine != OlapEngineType.CLICKHOUSE and self.engine != OlapEngineType.STARROCKS:
            raise Exception(f"Olap engine `{self.engine}` not supported.")

        try:
            table_view = self.materializedView(is_temp=True, is_physical_table=False)
            table_name = table_view.getTableName()
            if self.engine == OlapEngineType.CLICKHOUSE:
                field_names, field_types, _ = ClickHouseUtils(provider=self._provider).get_table_meta(table_name)
            elif self.engine == OlapEngineType.STARROCKS:
                field_names, field_types, _ = StarRocksUtils().get_table_meta(table_name)
            return pandas.Series(field_types, index=field_names)
        except Exception as e:
            get_context().logger.debug(e)
            columns = self.columns
            return pandas.Series(["Unknown"] * len(columns), index=columns)
    
    @property
    def columns(self):
        self._finalize()
        return [col.alias if col.alias is not None and col.alias else col.name for col in self.data.columns]

    @property
    def engine(self):
        return self._ctx.engine

    def __deepcopy__(self, memo):
        # 使用object.__new__()方法创建一个新的A类实例，而不调用__init__方法
        new_instance = object.__new__(DataFrame)
        # 将新实例的引用添加到memo字典中，以防止无限递归
        memo[id(self)] = new_instance
        # 使用vars()函数获取对象的所有属性，并将它们复制到新对象中
        for key, value in vars(self).items():
            if key in ("_provider"):
                setattr(new_instance, key, value)
            else:
                setattr(new_instance, key, copy.deepcopy(value, memo))
        new_instance._ctx.dataframe = new_instance
        return new_instance

    @property
    def sql_conn(self):
        from fast_causal_inference.util.sqlgateway import SqlGateWayConn

        return SqlGateWayConn(olap=str(self.engine))

    @property
    def olap_utils(self):
        if self.engine == OlapEngineType.CLICKHOUSE:
            return ClickHouseUtils(provider=self._provider)
        elif self.engine == OlapEngineType.STARROCKS:
            return StarRocksUtils()
        else:
            raise Exception("not supported")

    def _is_column(self, column, name):
        if name is None:
            return False
        if isinstance(column, DfColumnNode):
            return column.alias == name or column.sql(ctx=self._ctx) == name
        raise Exception(f"type({type(column)}) is not DfColumn")

    def _get_col_name(self, column):
        if isinstance(column, DfColumnLeafNode):
            return column.alias if column.alias else column.sql(ctx=self._ctx)
        if isinstance(column, DfColumnInternalNode):
            return column.alias
        raise Exception(
            f"type({type(column)}) is not DfColumnLeafNode|DfColumnInternalNode"
        )

    def printSchema(self):
        """
        This function is used to print the schema of the dataframe
        """
        pd = pandas.DataFrame(columns=["name", "type"])
        dt = self.dtypes
        for column in dt.index:
            pd = pd.append({"name": column, "type": dt[column]}, ignore_index=True)
        print(pd)

    def __getitem__(self, key):
        for column in self._select_list:
            if column.alias == key or column.sql(self._ctx) == key:
                return AisF.col(column)
        raise Exception("column %s not found" % key)

    def debug(self):
        self._finalize()
        print(self.data.__str__())
        all_names = dict()
        for name in self._name_dict:
            all_names[name] = self._name_dict[name].sql(self._ctx)
        print(all_names)

    def _finalize(self):
        del self.data.columns[:]
        for col in self.data.group_by:
            self.data.columns.append(col)
        for col in self._select_list:
            self.data.columns.append(
                DfPb.Column(name=col.sql(ctx=self._ctx), alias=col.alias)
            )

    def __str__(self):
        new_df = self.fill_result(self.data.limit.limit if self.data.limit.limit else 200)
        res = format_db_result(list(eval(new_df.data.result)))
        pandas.set_option('display.width', 150)
        pandas.set_option('display.max_columns', None)
        # pandas 增加换行的宽度


        if res.shape[0] == 1 and res.shape[1] == 1:
            return res.values[0][0]
        return res.__str__()

    def __repr__(self):
        return self.__str__()

    def brief(self):
        sql = [column.sql(self._ctx) for column in self._select_list]
        alias = [column.alias if column.alias else "-" for column in self._select_list]
        try:
            dtype = list(self.dtypes)
        except Exception as e:
            from fast_causal_inference.common import get_context
            get_context().logger.info(e)
            dtype = ["Unkown"] * len(sql)

        df = pandas.DataFrame({"sql": sql, "alias": alias, "type": dtype})

        df.index = df.index + 1

        return df


    def first(self):
        """
        This function is used to get the first row of the dataframe

        >>> df.first()

        """
        new_df = copy.deepcopy(self)
        new_df.data.limit.limit = 1
        return new_df

    def head(self, n=5):
        """
        This function is used to get the first n rows of the dataframe

        >>> df.head(3)

        """
        new_df = copy.deepcopy(self)
        new_df.data.limit.limit = n
        return new_df

    def take(self, n=5):
        """
        This function is used to get the first n rows of the dataframe

        >>> df.head(3)
        """
        new_df = copy.deepcopy(self)
        new_df.data.limit.limit = n
        return new_df

    def show(self):
        """
        Prints the DataFrame, equivalent to print(dataframe).

        >>> df.head(3).show()
        """

        print(self.__str__())

    # 增加检验逻辑
    # 这里用的是 alias 的名字，需要替换回来, 下面的 groupBy, orderBy 也是一样
    def where(self, filter):
        """
        Filters rows using the given condition.

        >>> df.where("column1 > 1").show()
        """
        filter = f"( {filter} )"
        filter = self._add_space(filter)
        for alias in self._name_dict:
            if alias is None or alias == "":
                continue
            filter = filter.replace(
                " " + alias + " ",
                " " + self._name_dict[alias].sql(self._ctx) + " ",
            )
        new_df = copy.deepcopy(self)
        if new_df._has_agg():
            new_df = new_df.materializedView(is_temp=True)
        if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", filter) != None:
            check_exist = False
            for column in new_df._select_list:
                super_name = self._get_col_name(column)
                if super_name in filter:
                    check_exist = True
                    break
            if not check_exist:
                raise Exception("the column of filter is not exist" % filter)

        new_df.data.filters.append(filter)
        return new_df

    def filter(self, filter):
        """
        Alias for the 'where' function. Filters rows using the given condition.

        >>> df.filter("column1 > 1").show()
        """
        return self.where(filter)

    @staticmethod
    def _expand_args(*args):
        expand_args = []
        for arg in args:
            if isinstance(arg, list):
                for sub_arg in arg:
                    expand_args.append(sub_arg)
            else:
                expand_args.append(arg)
        return expand_args

    def select(self, *args):
        """
        Selects specified columns from the DataFrame and returns a new DataFrame.
        >>> new_df = df.select('column1', 'column2')
        >>> new_df = df.select(['column1', 'column2'])
        """
        args = DataFrame._expand_args(*args)
        new_df = copy.deepcopy(self)
        new_df.checkColumn(*args)
        new_select_list = []
        for arg in args:
            new_select_list.append(new_df._expand_expr(arg))
        new_df._select_list = new_select_list
        return new_df

    def drop(self, *args):
        """
        Drops specified columns from the DataFrame and returns a new DataFrame.

        >>> new_df = df.drop('column1', 'column2')
        >>> new_df = df.drop(['column1', 'column2'])
        """
        args = DataFrame._expand_args(*args)
        new_df = copy.deepcopy(self)
        new_df.checkColumn(*args)
        for i in range(len(new_df._select_list) - 1, -1, -1):
            name = self._get_col_name(new_df._select_list[i])
            is_exist = False
            for arg in args:
                if name == arg:
                    is_exist = True
                    break
            if is_exist:
                del new_df._select_list[i]
        return new_df

    def withColumn(self, new_column, func):
        """
        This function adds a new column to the DataFrame.

        Example
        --------

        .. code-block:: python

            import fast_causal_inference.dataframe.functions as Fn
            import fast_causal_inference.dataframe.statistics as S

            df1 = df.select('x1','x2','x3', 'numerator')

            # Method 1: Select columns through df's index: df['col']
            df1 = df1.withColumn('new_col', Fn.sqrt(df1['numerator']))
            df1.show()

            # Method 2: Select columns directly through string: Fn.col('new_col')
            df1 = df1.withColumn('new_col2', Fn.pow('new_col', 2))
            df1.show()

            df1 = df1.withColumn('new_col3', Fn.col('new_col') * Fn.col('new_col'))
            df1.show()

            # Add constant
            df2 = df1.withColumn('c1', Fn.lit(1))
            df2 = df2.withColumn('c2', Fn.lit('1'))
            df2.show()

            # Nesting
            df2 = df1.withColumn('c1', Fn.pow(Fn.sqrt(Fn.sqrt(Fn.col('x1'))), 2))
            df2.show()

            # +-*/% operations
            df2 = df1.withColumn('c1', 22 + df1['x1'] / 2 + 2 / df1['x2'] * df1['x3'] % 2 - (df1['x2']))
            df2.show()
            df2 = df1.withColumn('c1', Fn.col('x1 + x2 * x3 + x3'))
            df2.show()

            # if
            df2 = df1.withColumn('cc1', 'if(x1 > 0, 1, -1)')
            df2.show()
            df2 = df1.withColumn('cc1', Fn.If('x1 > 0',1,-1))
            df2.show()
        """
        new_df = copy.deepcopy(self)
        for idx in range(len(new_df._select_list)):
            if new_df._select_list[idx].alias == new_column or new_df._select_list[idx].sql(self._ctx) == new_column:
                del new_df._select_list[idx]
                break
        if isinstance(func, str):
            func = f"( {func} )"
            func = new_df._add_space(func)
            for alias in new_df._name_dict:
                if alias is None or alias == "":
                    continue
                func = func.replace(
                    " " + alias + " ",
                    " " + new_df._name_dict[alias].sql(new_df._ctx) + " ",
                )
            new_df._select_list.append(DfColumnLeafNode(func, new_column))
            new_df._name_dict[new_column] = new_df._select_list[-1]
            new_df._name_dict[
                new_df._select_list[-1].sql(new_df._ctx)
            ] = new_df._select_list[-1]
        elif isinstance(func, DfFnColWrapper):
            new_df = new_df._apply_func(func.alias(new_column), keep_old_cols=True)
        else:
            raise Exception(f"unsupported type `{type(func)}`, which should be str or func.")
        return new_df

    @classmethod
    def _add_space(cls, text):
        # 匹配操作符和括号
        pattern = r'(,|>=|<=|==|!=|>|<|\+|\-|\*|\/|\%|\(|\)|=)'
        # 使用正则表达式替换，在操作符和括号周围添加空格
        spaced_expression = re.sub(pattern, r' \1 ', text)
        # 匹配连续的空格并替换为单个空格
        spaced_expression = re.sub(r'\s+', ' ', spaced_expression)
        
        return " " + spaced_expression + " "

    def _expand_expr(self, expr):
        if not isinstance(expr, str):
            raise Exception(f"Logical Error: expr(`{expr}`) is expected to be str.")
        expr = self._add_space(expr)
        origin_expr = expr
        for alias in self._name_dict:
            if alias is None or alias == "":
                continue
            expr = expr.replace(
                " " + alias + " ", " " + self._name_dict[alias].sql(self._ctx) + " "
            )
        expr = expr.replace(" ", "")
        origin_expr = origin_expr.replace(" ", "")
        alias = origin_expr if origin_expr != expr else None
        self._name_dict[origin_expr] = DfColumnLeafNode(expr, alias=alias)
        self._name_dict[expr] = DfColumnLeafNode(expr, alias=alias)
        return DfColumnLeafNode(expr, alias=alias)

    def _assert_alias_not_exists(self, alias):
        if alias is None:
            return
        column: DfColumnNode
        for index, column in enumerate(self._select_list):
            if column.alias == alias or column.sql(self._ctx) == alias:
                get_context().logger.error(f"alias `{alias}` already exists in {index + 1}-th column: "
                                           f"`{column.sql(self._ctx, show_alias=True)}`.")
                raise Exception(f"alias `{alias}` already exists")

    def withColumnRenamed(self, old_name, new_name):
        """
        Returns a new DataFrame by renaming an existing column.

        >>> df.withColumnRenamed("column1", "new_column1").show()

        """
        new_df = copy.deepcopy(self)
        new_df.checkColumn(old_name)

        new_df._assert_alias_not_exists(new_name)

        count = 0
        for column in new_df._select_list:
            if column.alias == old_name or new_df._get_col_name(column) == old_name:
                column.alias = new_name
                count += 1
        if count == 0:
            get_context().logger.error(f"name `{old_name}` not found.")
            raise Exception(f"name `{old_name}` not found.")
        if count > 1:
            get_context().logger.error(f"name `{old_name}` occured multiple times.")
            raise Exception(f"name `{old_name}` occured multiple times.")
        return new_df

    @classmethod
    def is_numeric_type(cls, type):
        return type in ["int", "bigint", "float", "double", "decimal"]

    def describe(self, cols="*"):
        """
        Returns the summary statistics for the columns in the DataFrame.
        
        Example
        ----------

        ::

            df.describe()
            df.describe(['x1','x2'])
            #       count       avg       std       min  quantile_0.25  quantile_0.5  quantile_0.75  quantile_0.90  quantile_0.99      max
            # x1  10000.0 -0.018434  0.987606 -3.740101       -0.68929     -0.028665       0.654210       1.274144       2.321097  3.80166
            # x2  10000.0  0.021976  1.986209 -8.893264       -1.28461      0.015400       1.357618       2.583523       4.725829  7.19662
            
        """
        import fast_causal_inference.dataframe.functions as Fn

        types = self.dtypes
        if cols == "*":
            cols = list(types.index)
        if not(isinstance(cols, list) and all(isinstance(col, str) for col in cols)):
            raise Exception(f"cols `{cols}` is not a List[str]")

        numeric_cols = []
        for col in cols:
            if not (types[col] in ["int", "bigint", "float", "double", "decimal"]):
                print(f'Ignoring column `{col}`, whose type `{types[col]}` is not numeric.')
                continue
            numeric_cols.append(col)
        
        if not numeric_cols:
            print("No numeric column.")
            raise Exception("No numeric column.")

        funcs = [
            (lambda _, idx: Fn.count().alias(f"count_{idx}"), "count"),
            (lambda col, idx: Fn.avg(col).alias(f"avg_{idx}"), "avg"),
            (lambda col, idx: Fn.stddevSamp(col).alias(f"std_{idx}"), "std"),
            (lambda col, idx: Fn.min(col).alias(f"min_{idx}"), "min"),
            (lambda col, idx: Fn.quantile(col, level=0.25).alias(f"quantile_25_{idx}"), "quantile_0.25"),
            (lambda col, idx: Fn.quantile(col, level=0.5).alias(f"quantile_50_{idx}"), "quantile_0.50"),
            (lambda col, idx: Fn.quantile(col, level=0.75).alias(f"quantile_75_{idx}"), "quantile_0.75"),
            (lambda col, idx: Fn.quantile(col, level=0.90).alias(f"quantile_90_{idx}"), "quantile_0.90"),
            (lambda col, idx: Fn.quantile(col, level=0.99).alias(f"quantile_99_{idx}"), "quantile_0.99"),
            (lambda col, idx: Fn.max(col).alias("max").alias(f"max_{idx}"), "max"),
        ]
        aggs = [
            func(col, idx)
            for idx, col in enumerate(numeric_cols)
            for func, _ in funcs
        ]
        df = self.agg(*aggs).toPandas()

        num_cols = len(funcs)
        num_rows = len(numeric_cols)
        new_data = [[df.iloc[0][i * num_cols + j] for j in range(num_cols)] for i in range(num_rows)]
        new_df = pandas.DataFrame(new_data, columns=[alias for _, alias in funcs], index=numeric_cols)

        return new_df

    def join(lhs, rhs, *exprs, type="inner"):
        """
        Join two DataFrames based on the given expression and join type.

        Parameters
        ----------
        :param rhs: DataFrame. The right dataframe to join.
        :param exprs: str. The join expressions, e.g., 'lhs.id = rhs.id', 'treatment'.
        :param type: str. The join type, one of the following: ['inner'; 'outer', 'full', 'fullouter', 'full_outer'; 'leftouter', 'left', 'left_outer'; 'rightouter', 'right', 'right_outer'; 'leftsemi', 'semi', 'left_semi'; 'leftanti', 'anti', 'left_anti'; 'cross']

        Returns
        -------
        A new DataFrame containing the joined data.

        Notes
        -------
        Each column in the lift and right tables must have its own name or alias.
        Join expression should be a column name like 'id' or an expression like 'lhs.${col1} = rhs.${col2}'.

        Example
        -------
        >>> import fast_causal_inference
        >>> allinsql_provider = fast_causal_inference.FCIProvider("all_in_sql_guest")
        >>> df = allinsql_provider.readStarRocks("test_data_small")
        >>> df1 = df.select("id", "x1", "treatment")
        >>> df2 = df.select("id", "x2", "treatment")
        >>> df3 = df1.join(df2, "id", "lhs.treatment = rhs.treatment", type="inner")
        >>> df3.dtypes
        x1           double
        treatment    bigint
        id           string
        x2           double
        dtype: object
        """
        supported_join_types = [
            "inner",
            "outer", 
            "full", 
            "fullouter", 
            "full_outer",
            "leftouter", 
            "left", 
            "left_outer",
            "rightouter", 
            "right", 
            "right_outer",
            "leftsemi", 
            "left_semi", 
            "semi",
            "leftanti", 
            "left_anti", 
            "anti",
            "cross",
        ]
        type = type.lower().replace("_", "")
        if type == "inner":
            join_type = "inner"
        elif type in ["outer", "full", "fullouter"]:
            join_type = "full outer"
        elif type in ["leftouter", "left"]:
            join_type = "left outer"
        elif type in ["rightouter", "right"]:
            join_type = "right outer"
        elif type in ["leftsemi", "semi"]:
            join_type = "left semi"
        elif type in ["leftanti", "anti"]:
            join_type = "left anti"
        elif type == "cross":
            join_type = "cross"
        else:
            raise Exception(f"Invalid join type `{type}`. Supported join types are {supported_join_types}")

        if lhs.engine != rhs.engine:
            raise Exception("Cannot join table from different datasource.")
        
        if lhs.engine == OlapEngineType.CLICKHOUSE:
            join_type = "global " + join_type

        for expr in exprs:
            if not isinstance(expr, str) or not expr:
                raise Exception(f"type of expr({type(expr)}) can only be str, please use statement like 'lhs.id = rhs.id'.")
        lhs_view = lhs.materializedView(is_temp=True)
        rhs_view = rhs.materializedView(is_temp=True)
        lhs_table = lhs_view.getTableName()
        rhs_table = rhs_view.getTableName()

        def is_valid_variable_name(s):
            # 变量名必须以字母或下划线开头
            # 后续字符可以是字母、数字或下划线
            pattern = r'^[a-zA-Z_][a-zA-Z0-9_]*$'
            return bool(re.match(pattern, s))

        for col in lhs_view.columns + rhs_view.columns:
            if not is_valid_variable_name(col):
                raise Exception(f"Invalid column `{col}`.\n"
                                f"For join operation, every column in each table should be a column name or alias"
                                f", but `{col}` is a function call.")

        lc = lhs_view.columns
        rc = rhs_view.columns
        columns = ['lhs.' + col for col in lc]
        for col in rc:
            if col in lc:
                continue
            columns.append("rhs." + col)

        cols = []
        for expr in exprs:
            if expr and is_valid_variable_name(expr):
                expr = f"lhs.{expr} = rhs.{expr}"
            cols.append(expr)

        joined_table_name = lhs.createTableName()
        join_sql = f"""
            select 
                {', '.join(columns)}
            from 
                    {lhs.database}.{lhs_table} as lhs 
                {join_type} join 
                    {rhs.database}.{rhs_table} as rhs
                {'on' if cols else ''} 
                    {' and '.join(cols)}
        """
        lhs._provider.context.logger.info("join sql:" + join_sql)
        if lhs.engine == OlapEngineType.CLICKHOUSE:
            ClickHouseUtils.clickhouse_create_view_v2(
                table_name=joined_table_name,
                select_statement=join_sql + "limit 0",
                is_physical_table=True,
                is_distributed_create=True,
                provider=lhs._provider
            )
            clickhouse_utils = ClickHouseUtils()
            sql = f"""insert into  {joined_table_name} {join_sql}"""
            clickhouse_utils.execute(sql)
            flush_sql = f"SYSTEM FLUSH DISTRIBUTED {joined_table_name}"
            clickhouse_utils.execute(flush_sql)
            return lhs._provider.readClickHouse(joined_table_name)
        else:
            StarRocksUtils().create_view(joined_table_name, join_sql, True)
            return readOlap(joined_table_name, olap='starrocks')

    def sample(self, fraction):
        """
        Sample a fraction of rows or some rows without replacement from the DataFrame.

        Parameters
        ----------

        fraction : float
            If fraction is in [0, 1], it will return fraction*df.count() rows.
            If fraction is > 1, fraction means the sample size and it will return fraction rows.

        Returns
        -------

        DataFrame
            A DataFrame with sampled rows.

        Exampes
        --------

        >>> df1 = df.sample(1000)
        >>> df1.count().show()

        >>> df2 = df.sample(0.5)
        >>> df2.count().show()
        """
        import fast_causal_inference.dataframe.functions as Fn

        new_df = self.materializedView(is_temp=True, is_physical_table=True)
        row_num = int(new_df.count().__str__())
        if fraction <= 1 and fraction >= 0:
            pass
        elif fraction > 1:
            if fraction > row_num:
                raise Exception("fraction should be less than row number")
            fraction = fraction / row_num
        else:
            raise Exception("fraction should be in [0, 1] or > 1")

        temp_df = new_df
        temp_df = temp_df.withColumn("_rand_value", Fn.rand_cannonical()).materializedView(is_temp=True)
        temp_df = temp_df.where(f"_rand_value < {fraction}")
        return temp_df.drop("_rand_value").materializedView(is_physical_table=True)

    def split(self, test_size=0.5):
        """
        This function splits the DataFrame into two DataFrames.

        >>> df_train, df_test = df.split(0.5)
        >>> print(df_train.count())
        >>> print(df_test.count())
        """
        import fast_causal_inference.dataframe.functions as Fn

        new_df = self.materializedView(is_temp=True, is_physical_table=True)
        temp_df = new_df
        temp_df = temp_df.withColumn("_rand_value", Fn.rand_cannonical()).materializedView(is_temp=True)
        test_df = temp_df.where(f"_rand_value < {test_size}")
        test_df = test_df.drop("_rand_value").materializedView(is_physical_table=True)
        train_df = temp_df.where(f"_rand_value > {test_size}")
        train_df = train_df.drop("_rand_value").materializedView(is_physical_table=True)
        return train_df, test_df

    def orderBy(self, *args):
        """
        Orders the DataFrame by the specified columns and returns a new DataFrame.

        >>> import fast_causal_inference.dataframe.functions as Fn
        >>> new_df = df.orderBy('column1', Fn.desc('column2'))
        >>> new_df = df.orderBy(['column1', 'column2'])
        """
        args = DataFrame._expand_args(*args)
        # clear self.data.order_by
        new_df = copy.deepcopy(self)

        del new_df.data.order_by[:]
        for arg in args:
            if isinstance(arg, str):
                order = DfPb.Order()
                order.column.name = arg
                new_df.data.order_by.append(order)
            elif isinstance(arg, DfPb.Order):
                new_df.data.order_by.append(arg)
        return new_df

    def groupBy(self, *args):
        """
        Groups the DataFrame by the specified columns and returns a new DataFrame.Exception: If the DataFrame is already in `need_agg` state.


        Example:
        ----------------
        .. code-block:: python

             import fast_causal_inference
            allinsql_provider = fast_causal_inference.FCIProvider(database="all_in_sql_guest")
            # 这是一个模拟的实验明细表！可以用该表来模拟实验分析
            df = allinsql_provider.readClickHouse(f'expt_detail_test_appid')

        >>> # 看每个实验组的样本量
        >>> df.groupBy('groupname').count()
        groupname count()
        0        A1    2429
        1        B1    5094
        2        A2    2477

        >>> # 看每个实验组的分子分母之和
        >>> import fast_causal_inference.dataframe.functions as  Fn
        >>> df.groupBy('groupname').agg(Fn.sum('numerator').alias('numerator_sum'),
                                        Fn.sum('demonimator').alias('demonimator_sum'),
                                        Fn.sum('numerator_pre').alias('numerator_pre_sum'),
                                        Fn.sum('demonimator_pre').alias('demonimator_pre_sum'))
        groupname    numerator_sum  denominator_sum numerator_pre_sum denominator_pre_sum
        0        A1  11205.519002735  14308.027372613   10818.744789206     14362.799162067
        1        B1  100540.30311176  40452.337655937    23096.87560799     30776.559776899
        2        A2  11853.108720046  14715.205784573   11084.367641361     14769.032577094


        >>> # 下钻分析，groupBy首次命中日期，去看不同天命中实验的用户的策略效果是否有所不同
        >>> df.groupBy('first_hit_ds').xexpt_ttest_2samp('numerator', 'denominator', "if(groupid=12343,'B','A')", uin = 'rand()', metric_type = 'avg').show()
        first_hit_ds groupname   denominator     numerator       mean  std_samp diff_relative              95%_relative_CI   p-value t-statistic  \
        0     20240102         A  12692.309491  13013.602024   1.025314  1.756287                                                                    
        1     20240102         B   5270.784857   6501.560937   1.233509  2.151984    20.305499%      [13.900858%,26.710141%]  0.000000    6.216446   
        2     20240103         A  11217.541056  -1787.581977  -0.159356  1.248860                                                                    
        3     20240103         B    195.396019     77.324041   0.395730  1.328970  -348.330767%  [-230.430534%,-466.231000%]  0.000000    5.794129   
        4     20240101         A    5113.38261  11832.607676   2.314047  3.443390                                                                    
        5     20240101         B  34986.156779  93961.418134   2.685674  5.632780    16.059621%      [11.247994%,20.871248%]  0.000000    6.543296   

            diff               95%_CI     power recommend_samples        mde  
        0                                                                        
        1  0.208195  [0.142527,0.273863]  0.052688           1765574   0.091511  
        2                                                                        
        3  0.555086  [0.367205,0.742967]  0.050008          22171277  -1.684254  
        4                                                                        
        5  0.371627  [0.260284,0.482970]  0.054768           6616698   0.068761  



        """
        args = DataFrame._expand_args(*args)
        if self._need_agg:
            raise Exception(
                "Dataframe is already in `need_agg` state, cannot apply group by."
            )
        new_df = self._transform_to_nested()
        new_df._need_agg = True
        for arg in args:
            if arg not in self._name_dict:
                raise Exception(f"Unable to find column named `{arg}`.")
            column = DfPb.Column()
            column.name = arg
            new_df.data.group_by.append(column)
        return new_df

    def agg(self, *args):
        # 支持如下两种语法
        # df.agg(avg("A"), sum("B").alias).show()
        # df.agg({"A": "avg", "B": "sum"}).show()
        new_df = copy.deepcopy(self)
        if len(args) == 0:
            raise Exception("Nothing to agg.")
        if len(args) == 1 and isinstance(args[0], dict):
            args = list(map(lambda key: getattr(AisF, args[0][key])(key), args[0]))
        if not all(map(lambda arg: isinstance(arg, DfFnColWrapper), args)):
            raise Exception(f"{args} is not List[DfFnColWrapper].")
        new_df = new_df._apply_func(*args)
        if len(args) == 1 and not isinstance(args[0].fn, (AisRegressionF.AggStochasticLogisticRegressionDfFunction, AisRegressionF.AggStochasticLinearRegressionDfFunction, AisStatisticsF.BootStrapDfFunction)):
            return self._try_to_trival(new_df)
        if len(args) == 1 and isinstance(args[0], AisStatisticsF.BootStrapDfFunction):
            return self._handle_boot_strap(new_df)
        return new_df

        # aggregate functions

    def sum(self, column):
        return self._try_to_trival(self._apply_func(AisF.sum(column)))

    def avg(self, column):
        return self._try_to_trival(self._apply_func(AisF.avg(column)))

    def count(self, *, expr="*"):
        return self._try_to_trival(self._apply_func(AisF.count(expr=expr)))

    def max(self, column):
        return self._try_to_trival(self._apply_func(AisF.max(column)))

    def min(self, column):
        return self._try_to_trival(self._apply_func(AisF.min(column)))

    def any(self, column):
        return self._try_to_trival(self._apply_func(AisF.any(column)))

    def stddevPop(self, column):
        return self._try_to_trival(self._apply_func(AisF.stddevPop(column)))

    def stddevSamp(self, column):
        return self._try_to_trival(self._apply_func(AisF.stddevSamp(column)))

    def varPop(self, column):
        return self._try_to_trival(self._apply_func(AisF.varPop(column)))

    def varSamp(self, column):
        return self._try_to_trival(self._apply_func(AisF.varSamp(column)))

    def corr(self, x, y):
        return self._try_to_trival(self._apply_func(AisF.corr(x, y)))

    def covarPop(self, x, y):
        return self._try_to_trival(self._apply_func(AisF.covarPop(x, y)))

    def covarSamp(self, x, y):
        return self._try_to_trival(self._apply_func(AisF.covarSamp(x, y)))

    def anyLast(self, x, y):
        return self._try_to_trival(self._apply_func(AisF.anyLast(x, y)))

    def anyMin(self, x, y):
        return self._try_to_trival(self._apply_func(AisF.anyMin(x, y)))

    def anyMax(self, x, y):
        return self._try_to_trival(self._apply_func(AisF.anyMax(x, y)))

    def kolmogorov_smirnov_test(self, x, y):
        return self._try_to_trival(self._apply_func(AisStatisticsF.kolmogorov_smirnov_test(x, y)))

    def student_ttest(self, x, y):
        return self._try_to_trival(self._apply_func(AisStatisticsF.student_ttest(x, y)))

    def welch_ttest(self, x, y):
        return self._try_to_trival(self._apply_func(AisStatisticsF.welch_ttest(x, y)))

    def mean_z_test(
        self,
        sample_data,
        sample_index,
        population_variance_x,
        population_variance_y,
        confidence_level,
    ):
        return self._try_to_trival(
            self._apply_func(
                AisStatisticsF.mean_z_test(
                    sample_data,
                    sample_index,
                    population_variance_x,
                    population_variance_y,
                    confidence_level,
                )
            )
        )

    def quantile(self, x, level=None, exact=False):
        if level is None:
            raise Exception("param `level` is not set")
        return self._try_to_trival(self._apply_func(AisF.quantile(x, level=level, exact=exact)))

    def quantiles(self, x, *levels, exact=False):
        return self._try_to_trival(self._apply_func(AisF.quantiles(x, *levels, exact=exact)))

    # all in sql functions
    def delta_method(self, column, std=False):
        """
        Compute the delta method on the given expression.

        :param expr: Form like f (avg(x1), avg(x2), ...) , f is the complex function expression, x1 and x2 are column names, the columns involved here must be numeric.
        :type expr: str, optional
        :param std: Whether to return standard deviation, default is True.
        :type std: bool, optional

        :return: DataFrame contains the following columns: var or std computed by delta_method.
        :rtype: DataFrame

        Example
        ----------

        .. code-block:: python

            import fast_causal_inference
            allinsql_provider = fast_causal_inference.FCIProvider(database="all_in_sql_guest")
            # 这是一个模拟的实验明细表！可以用该表来模拟实验分析
            df = allinsql_provider.readClickHouse(f'expt_detail_test_appid')

        >>> # 看实验组的指标均值的方差（人均指标）
        >>> df.filter("groupname='B1'").delta_method('avg(numerator)',std=False)
        0.03447617786500999


        >>> # 看每个实验组的指标均值的方差（人均指标）
        >>> df.groupBy('groupname').delta_method('avg(numerator)',std=False)
        groupname Deltamethod('x1', true)(numerator)
        0        A1                         .118752394
        1        B1                         .185677618
        2        A2                         .118817924

        >>> # 看每个实验组的指标均值的标准差（比例指标）
        >>> df.groupBy('groupname').delta_method('avg(numerator)/avg(denominator)',std=True)
        groupname Deltamethod('x1/x2', true)(numerator, denominator)
        0        A1                                         .022480953
        1        B1                                         .029102234
        2        A2                                         .022300609

        """
        return self._try_to_trival(self._apply_func(AisStatisticsF.delta_method(expr=column, std=std)))

    def ttest_1samp(self, Y, alternative="two-sided", mu=0, X=""):
        """
        This function is used to calculate the t-test for the mean of one group of scores. It returns the calculated t-statistic and the two-tailed p-value.

        :param Y: str, form like f (avg(x1), avg(x2), ...), f is the complex function expression, x1 and x2 are column names, the columns involved here must be numeric.
        :type Y: str, required
        :param alternative: str, use 'two-sided' for two-tailed test, 'greater' for one-tailed test in the positive direction, and 'less' for one-tailed test in the negative direction.
        :type alternative: str, optional
        :param mu: the mean of the null hypothesis.
        :type mu: float, optional
        :param X: str, an expression used as continuous covariates for CUPED variance reduction. It follows the regression approach and can be a simple form like 'avg(x1)/avg(x2)','avg(x3)','avg(x1)/avg(x2)+avg(x3)'.
        :type X: str, optional
        :return: a testResult object that contains p-value, statistic, stderr and confidence interval. 
        :rtype: testResult

        Example:
        ----------------
        .. code-block:: python

            import fast_causal_inference
            allinsql_provider = fast_causal_inference.FCIProvider(database="all_in_sql_guest")
            # 这是一个模拟的实验明细表！可以用该表来模拟实验分析
            df = allinsql_provider.readClickHouse(f'expt_detail_test_appid')

        >>> # 均值指标检验，检验实验组均值是否=0
        >>> df.filter('groupid=12343').ttest_1samp('avg(numerator)', alternative = 'two-sided',mu=0)
            estimate    stderr t-statistic   p-value      lower      upper
        0  19.737005  0.185678  106.297168  0.000000  19.372997  20.101013

        >>> # 比率指标检验，检验实验组指标是否=0
        >>> df.filter('groupid=12343').ttest_1samp('avg(numerator)/avg(denominator)', alternative = 'two-sided',mu=0)
        estimate    stderr t-statistic   p-value     lower     upper
        0  2.485402  0.029102   85.402433  0.000000  2.428349  2.542454

        >>> # 下钻分析，groupBy首次命中日期，去看不同天命中实验的用户的策略效果是否有所不同
        >>> df.groupBy('first_hit_ds').filter('groupid=12343').ttest_1samp('avg(numerator)', alternative = 'two-sided',mu=0)
        first_hit_ds   estimate    stderr t-statistic   p-value      lower      upper
        0     20240102   7.296926  0.063260  115.347550  0.000000   7.172769   7.421083
        1     20240103   1.137118  0.194346    5.851008  0.000000   0.749203   1.525034
        2     20240101  22.723438  0.201318  112.873450  0.000000  22.328747  23.118130


        """

        ## 自版本2.5.34 更新 - test result 为object，可以获取属性
        test_result = self._try_to_trival(self._apply_func(AisStatisticsF.ttest_1samp(Y, alternative, mu, X)))
        return testResult(test_result)

    def ttest_2samp(self, Y, index, alternative="two-sided", X="",pse=''):
        """
        This function is used to calculate the t-test for the means of two independent samples of scores. It returns the calculated t-statistic and the two-tailed p-value.

        :param Y: str, form like f (avg(x1), avg(x2), ...), f is the complex function expression, x1 and x2 are column names, the columns involved here must be numeric.
        :type Y: str, required
        :param index: str, the treatment variable.
        :type index: str, required
        :param alternative: str, use 'two-sided' for two-tailed test, 'greater' for one-tailed test in the positive direction, and 'less' for one-tailed test in the negative direction.
        :type alternative: str, optional
        :param X: str, an expression used as continuous covariates for CUPED variance reduction. It follows the regression approach and can be a simple form like 'avg(x1)/avg(x2)','avg(x3)','avg(x1)/avg(x2)+avg(x3)'.
        :type X: str, optional
        :param pse: str, an expression used as discrete covariates for post-stratification variance reduction. It involves grouping by a covariate, calculating variances separately, and then weighting them. It can be any complex function form, such as 'x_cat1'.
        :type pse: str, optional
        :return: a testResult object that contains p-value, statistic, stderr and confidence interval. 
        :rtype: testResult

        Example:
        ----------------

        .. code-block:: python
        
            import fast_causal_inference
            allinsql_provider = fast_causal_inference.FCIProvider(database="all_in_sql_guest")
            # 这是一个模拟的实验明细表！可以用该表来模拟实验分析
            df = allinsql_provider.readClickHouse(f'expt_detail_test_appid')

        >>> # 均值指标检验
        >>> df.ttest_2samp('avg(numerator)', 'if(groupid=12343,1,0)', alternative = 'two-sided')
            mean0      mean1   estimate    stderr t-statistic   p-value      lower      upper
        0  4.700087  19.737005  15.036918  0.203794   73.784922  0.000000  14.637441  15.436395


        >>> # 比例指标检验，用历史数据做CUPED方差削减
        >>> df.ttest_2samp('avg(numerator)/avg(denominator)', 'if(groupid=12343,1,0)', alternative = 'two-sided', X = 'avg(numerator_pre)/avg(denominator_pre)')
            mean0     mean1  estimate    stderr t-statistic   p-value     lower     upper
        0  0.793732  2.486118  1.692386  0.026685   63.419925  0.000000  1.640077  1.744694


        >>> # 比例指标检验，用首次命中日期做后分层（PSE）方差削减
        >>> df.ttest_2samp('avg(numerator)/avg(denominator)', 'if(groupid=12343,1,0)', alternative = 'two-sided', pse = 'first_hit_ds')
            mean0     mean1  estimate    stderr t-statistic   p-value     lower     upper
        0  1.432746  1.792036  0.359289  0.038393    9.358290  0.000000  0.284032  0.434547


        >>> # 下钻分析，groupBy首次命中日期，去看不同天命中实验的用户的策略效果是否有所不同
        >>> df.groupBy('first_hit_ds').ttest_2samp('avg(numerator)', 'if(groupid=12343,1,0)', alternative = 'two-sided')
        first_hit_ds      mean0      mean1  estimate    stderr t-statistic   p-value     lower     upper
        0     20240102   6.170508   7.296926  1.126418  0.076667   14.692392  0.000000  0.976093  1.276742
        1     20240103  -0.925249   1.137118  2.062368  0.205304   10.045430  0.000000  1.659735  2.465000
        2     20240101  13.679315  22.723438  9.044123  0.231238   39.111779  0.000000  8.590796  9.497451
        """

        ## 自版本2.5.34 更新 - test result 为object，可以获取属性
        test_result = self._try_to_trival(
            self._apply_func(
                AisStatisticsF.ttest_2samp(Y, index, alternative=alternative, X=X,pse=pse)
            )
        )
        return testResult(test_result)

    def xexpt_ttest_2samp(
        self,
        numerator,
        denominator,
        index,
        uin,
        metric_type="avg",
        group_buckets=[1,1],
        alpha=0.05,
        MDE=0.005,
        power=0.8,
        X="",
    ):
        """
        微信实验平台计算两个独立样本得分均值的t检验的函数说明，它返回计算出的t统计量和双尾p值，统计效力（power）,建议样本量，最小检验差异（MDE）


        参数
        ----------

            numerator : str
                列名，分子，可以使用SQL表达式，该列必须是数值型。

            denominator : str
                列名，分母，可以使用SQL表达式，该列必须是数值型。

            index : str
                列名。该列名用于区分对照组和实验组。
                该列应仅包含两个不同的值，这些值将根据它们的字符顺序用于确定对照组和实验组。
                例如，如果可能的值是0或1，那么0将被视为对照组。 比如"if(groupid=12343,1,0)"
                同样，如果值是'A'和'B'，那么'A'将被视为对照组。 比如"if(groupid=12343,'B','A')"

            uin : str
                列名，用于分桶样本，可以包含SQL表达式，数据类型为int64。如果需要随机分桶，可以使用'rand()'函数。

            metric_type : str, 可选
                'avg'用于均值指标/比例指标的检验，avg(num)/avg(demo)，默认为'avg'。
                'sum'用于SUM指标的检验，此时分母可以省略或为1

            group_buckets : list, 可选
                每个组的流量桶数，仅当metric_type='sum'时有效，是用来做SUM指标检验的
                默认为[1,1]，对应的是实验组：对照组的流量比例

            alpha : float, 可选
                显著性水平，默认为0.05。

            MDE : float, 可选
                最小测试差异，默认为0.005。

            power : float, 可选
                统计功效，默认为0.8。

            X : str, 可选
                用作CUPED方差减少的连续协变量的表达式。
                它遵循回归方法，可以是像'avg(x1)/avg(x2)'，
                'avg(x3)'，'avg(x1)/avg(x2)+avg(x3)'这样的简单形式。


        Returns
        -------

        DataFrame
            一个包含以下列的DataFrame：
            groupname : str
                组的名称。
            numerator : float
                该组分子求和
            denominator : float, 可选
                该组分母求和。仅当metric_type='avg'时出现。
            numerator_pre : float, 可选
                实验前该组分子求和。仅当metric_type='avg'时出现。
            denominator_pre : float, 可选
                实验前该组分母求和。仅当metric_type='avg'时出现。
            mean : float, 可选
                该组度量的平均值。仅当metric_type='avg'时出现。
            std_samp : float, 可选
                该组度量的样本标准差。仅当metric_type='avg'时出现。
            ratio : float, 可选
                分组桶的比率。仅当metric_type='sum'时出现。
            diff_relative : float
                两组之间的相对差异。
            95%_relative_CI : tuple
                相对差异的95%置信区间。
            p-value : float
                计算出的p值。
            t-statistic : float
                计算出的t统计量。
            power : float
                测试的统计功效，即在原假设为假时正确拒绝原假设的概率。功效是基于提供的`mde`（最小可检测效应）、`std_samp`（度量的标准差）和样本大小计算的。
            recommend_samples : int
                推荐的样本大小，以达到检测指定`mde`（最小可检测效应）的所需功效，给定`std_samp`（度量的标准差）。
            mde : float
                设计测试以检测的最小可检测效应大小，基于输入的`power`水平、`std_samp`（度量的标准差）和样本大小计算得出。

        Example:
        ----------

        .. code-block:: python
 
            import fast_causal_inference
            allinsql_provider = fast_causal_inference.FCIProvider(database="all_in_sql_guest")
            # 这是一个模拟的实验明细表！可以用该表来模拟实验分析
            df = allinsql_provider.readClickHouse(f'expt_detail_test_appid')

        >>> # SUM指标检验， metric_type = 'sum', group_buckets需要填写实验组和对照组的流量之比
        >>> df.xexpt_ttest_2samp('numerator', 'denominator', "if(groupid=12343,'B','A')", uin = 'rand()', metric_type = 'sum', group_buckets=[1,1]).show()
        groupname ratio      numerator diff_relative            95%_relative_CI   p-value t-statistic     power recommend_samples       mde
        0         A     1   23058.627723                                                                                                     
        1         B     1  100540.303112   336.020323%  [321.350645%,350.690001%]  0.000000   44.899925  0.050511          23944279  0.209664

        >>> # SUM指标检验 并做CUPED， metric_type = 'sum', group_buckets需要填写实验组和对照组的流量之比
        >>> df.xexpt_ttest_2samp('numerator', 'denominator', "if(groupid=12343,1,0)", uin = 'rand()', metric_type = 'sum', group_buckets=[1,1], X = 'avg(numerator_pre)').show()
        groupname ratio      numerator diff_relative            95%_relative_CI   p-value t-statistic     power recommend_samples       mde
        0         A     1   23058.627723                                                                                                     
        1         B     1  100540.303112   181.046828%  [174.102340%,187.991317%]  0.000000   51.103577  0.052285          13279441  0.099253

        >>> # 均值指标或者比例指标的检验，metric_type = 'avg',
        >>> df.xexpt_ttest_2samp('numerator', 'denominator', "if(groupid=12343,'B','A')", uin = 'rand()', metric_type = 'avg').show()
        groupname   denominator      numerator      mean  std_samp diff_relative            95%_relative_CI   p-value t-statistic      diff  \
        0         A  29023.233157   23058.627723  0.794489  2.879344                                                                            
        1         B  40452.337656  100540.303112  2.485402  5.612429   212.830364%  [204.781181%,220.879546%]  0.000000   51.830152  1.690913   
                        95%_CI     power recommend_samples       mde  
        0                                                             
        1  [1.626963,1.754863]  0.051700          21414774  0.115042  

        >>> # 均值指标检验，因为均值指标分母是1，因此'denominator'可以用'1'代替
        >>> df.xexpt_ttest_2samp('numerator', '1', "if(groupid=12343,'B','A')", uin = 'rand()', metric_type = 'avg').show()
        groupname   denominator      numerator      mean  std_samp diff_relative            95%_relative_CI   p-value t-statistic      diff  \
        0         A  29023.233157   23058.627723  0.794489  2.879344                                                                            
        1         B  40452.337656  100540.303112  2.485402  5.612429   212.830364%  [204.781181%,220.879546%]  0.000000   51.830152  1.690913   
                        95%_CI     power recommend_samples       mde  
        0                                                             
        1  [1.626963,1.754863]  0.051700          21414774  0.115042  

        >>> # 均值指标或者比例指标的检验，CUPED，metric_type = 'avg'
        >>> df.xexpt_ttest_2samp('numerator', 'denominator', "if(groupid=12343,'B','A')", uin = 'rand()', metric_type = 'avg', X = 'avg(numerator_pre)/avg(denominator_pre)').show()
        groupname   denominator      numerator denominator_pre numerator_pre      mean  std_samp diff_relative            95%_relative_CI   p-value  \
        0         A  29023.233157   23058.627723    29131.831739  21903.112431  0.793621  1.257317                                                      
        1         B  40452.337656  100540.303112    30776.559777  23096.875608  2.486223  4.701808   213.275723%  [207.220762%,219.330684%]  0.000000   
        t-statistic      diff               95%_CI     power recommend_samples       mde  
        0                                                                                   
        1   69.044764  1.692601  [1.644548,1.740655]  0.053007          12118047  0.086540  

        >>> # 也可以用df.agg(S.xexpt_ttest_2samp()的形式
        >>> import fast_causal_inference.dataframe.statistics as S
        >>> df.agg(S.xexpt_ttest_2samp('numerator', 'denominator',  "if(groupid=12343,'B','A')", uin = 'rand()', metric_type = 'avg', alpha = 0.05, MDE = 0.005, power = 0.8, X = 'avg(numerator_pre)')).show()
        groupname   denominator      numerator      mean  std_samp diff_relative            95%_relative_CI   p-value t-statistic      diff  \
        0         A  29023.233157   23058.627723  0.980173  2.316314                                                                            
        1         B  40452.337656  100540.303112  2.661489  5.929225   171.532540%  [165.040170%,178.024910%]  0.000000   51.789763  1.681316   
                        95%_CI     power recommend_samples       mde  
        0                                                             
        1  [1.617679,1.744952]  0.052615          13932096  0.092791  

        >>> # 下钻分析，groupBy首次命中日期，去看不同天命中实验的用户的策略效果是否有所不同
        >>> df.groupBy('first_hit_ds').xexpt_ttest_2samp('numerator', 'denominator', "if(groupid=12343,'B','A')", uin = 'rand()', metric_type = 'avg').show()
        first_hit_ds groupname   denominator     numerator       mean  std_samp diff_relative              95%_relative_CI   p-value t-statistic  \
        0     20240102         A  12692.309491  13013.602024   1.025314  1.756287                                                                    
        1     20240102         B   5270.784857   6501.560937   1.233509  2.151984    20.305499%      [13.900858%,26.710141%]  0.000000    6.216446   
        2     20240103         A  11217.541056  -1787.581977  -0.159356  1.248860                                                                    
        3     20240103         B    195.396019     77.324041   0.395730  1.328970  -348.330767%  [-230.430534%,-466.231000%]  0.000000    5.794129   
        4     20240101         A    5113.38261  11832.607676   2.314047  3.443390                                                                    
        5     20240101         B  34986.156779  93961.418134   2.685674  5.632780    16.059621%      [11.247994%,20.871248%]  0.000000    6.543296   
            diff               95%_CI     power recommend_samples        mde  
        0                                                                        
        1  0.208195  [0.142527,0.273863]  0.052688           1765574   0.091511  
        2                                                                        
        3  0.555086  [0.367205,0.742967]  0.050008          22171277  -1.684254  
        4                                                                        
        5  0.371627  [0.260284,0.482970]  0.054768           6616698   0.068761  



        """


        return self._try_to_trival(
            self._apply_func(
                AisStatisticsF.xexpt_ttest_2samp(
                    numerator,
                    denominator,
                    index,
                    uin,
                    metric_type,
                    group_buckets,
                    alpha,
                    MDE,
                    power,
                    X,
                )
            )
        )

    def mann_whitney_utest(
        self,
        sample_data,
        sample_index,
        alternative="two-sided",
        continuity_correction=1,
    ):
        """

         This function is used to calculate the Mann-Whitney U test. It returns the calculated U-statistic and the two-tailed p-value.

        :param sample_data: column name, the numerator of the metric, can use SQL expression, the column must be numeric.
        :type sample_data: str, required
        :param sample_index: column name, the index to represent the control group and the experimental group, 1 for the experimental group and 0 for the control group.
        :type sample_index: str, required
        :param alternative:
            'two-sided': the default value, two-sided test.
            'greater': one-tailed test in the positive direction.
            'less': one-tailed test in the negative direction.
        :type alternative: str, optional
        :param continuous_correction: bool, default 1, whether to apply continuity correction.
        :type continuous_correction: bool, optional

        :return: Tuple with two elements:
        U-statistic: Float64.
        p-value: Float64.

        Example:
        ----------------

        .. code-block:: python

            import fast_causal_inference
            allinsql_provider = fast_causal_inference.FCIProvider(database="all_in_sql_guest")
            df = allinsql_provider.readClickHouse(f'expt_detail_test_appid')

        >>> # 返回的结果是 [U统计量,P值]
        >>> df.mann_whitney_utest('numerator', 'if(groupid=12343,1,0)').show()
        [2349578.0, 0.0]
        >>> df.agg(S.mann_whitney_utest('numerator', 'if(groupid=12343,1,0)')).show()
        [2349578.0, 0.0]

        """
        test_result = self._try_to_trival(
                self._apply_func(
                AisStatisticsF.mann_whitney_utest(
                    sample_data, sample_index, alternative, continuity_correction
                )
            )
        )
        return testResult(test_result)

    def srm(self, x, groupby, ratio="[1,1]"):
        return self._try_to_trival(self._apply_func(AisStatisticsF.srm(x, groupby, ratio)))

    def ols(self, column, use_bias=True):
        return self._try_to_trival(self._apply_func(AisRegressionF.ols(column, use_bias)))

    def wls(self, column, weight, use_bias=True):
        return self._try_to_trival(self._apply_func(AisRegressionF.wls(column, weight, use_bias)))

    def stochastic_linear_regression(
        self, expr, learning_rate=0.00001, l1=0.1, batch_size=15, method="SGD"
    ):
        return self._apply_func(
            AisRegressionF.stochastic_linear_regression(
                expr, learning_rate, l1, batch_size, method
            )
        )

    def stochastic_logistic_regression(
        self, expr, learning_rate=0.00001, l1=0.1, batch_size=15, method="SGD"
    ):
        return self._apply_func(
            AisRegressionF.stochastic_logistic_regression(
                expr, learning_rate, l1, batch_size, method
            )
        )

    def matrix_multiplication(self, *args, std=False, invert=False):
        return self._try_to_trival(
            self._apply_func(
                AisStatisticsF.matrix_multiplication(*args, std=std, invert=invert)
            )
        )

    def did(self, Y, treatment, time, *X):
        return self._try_to_trival(self._apply_func(AisRegressionF.did(Y, treatment, time, *X)))

    def iv_regression(self, formula):
        return self._try_to_trival(self._apply_func(AisRegressionF.iv_regression(formula)))

    def _handle_boot_strap(self, df):
        sql = df.getExecutedSql()
        result = df.olap_utils.execute(sql)[0][0]
        if isinstance(result[0], str):
            # TODO(@cooperxiong): format output
            # result = format_db_result(result)
            return df
        return result

    def boot_strap(self, func, resample_frac=1, n_resamples=100):
        return self._handle_boot_strap(self._apply_func(AisStatisticsF.boot_strap(func, resample_frac, n_resamples)))

    def permutation(self, func, permutation_num, *col):
        return self._try_to_trival(self._apply_func(AisStatisticsF.permutation(func, permutation_num, *col)))

    def checkColumn(self, *args, throw_exception=True):
        for arg in args:
            if arg not in self._name_dict:
                if throw_exception == True:
                    raise Exception("column %s not found" % arg)
                else:
                    return False
        return True

    # 优先提取 alias
    def getAliasOrName(self):
        alias_or_name = []
        for column in self._select_list:
            alias_or_name.append(self._get_col_name(column))
        return alias_or_name

    def _find_column(self, name):
        if self._name_dict.get(name) is not None:
            # print(f"find column `{name}`: {self._name_dict.get(name).sql(self._ctx)}")
            return self._name_dict.get(name)
        for column in self._select_list:
            col = column.find_column(name)
            if col is not None:
                # print(f"find column `{name}`: {col.sql(self._ctx)}")
                return col
        # print(f"cannot find column `{name}`, using self.")
        return DfColumnLeafNode(name)

    def _unwrap(self, fn_wrapper: DfFnColWrapper):
        if not isinstance(fn_wrapper, DfFnColWrapper):
            raise Exception(f"func({type(DfFnColWrapper)}) should be DfFnColWrapper!")
        fn, params, cols = fn_wrapper.fn, fn_wrapper.params, fn_wrapper.columns
        alias = fn.alias
        columns = []
        for col in cols:
            if isinstance(col, int) or isinstance(col, float) or isinstance(col, bool):
                from_col = DfColumnLeafNode(str(col))
            elif isinstance(col, str):
                from_col = copy.deepcopy(self._find_column(col))
                self._name_dict[col] = from_col
            elif isinstance(col, DfColumnNode):
                from_col = copy.deepcopy(col)
                if col.alias is not None:
                    self._name_dict[col.alias] = from_col
            elif isinstance(col, DfFnColWrapper):
                from_col = self._unwrap(col)
                if from_col.alias is not None:
                    self._name_dict[from_col.alias] = from_col
            else:
                raise Exception(
                    f"type of col({type(col)}) is neither str nor DfColumn."
                )
            if not isinstance(from_col, DfColumnNode):
                # we can only apply function on DfColumn but no others
                raise Exception(f"Logical Error: {type(from_col)} is not DfColumn.")
            self._name_dict[from_col.sql(self._ctx)] = from_col
            columns.append(from_col)
        new_col = DfColumnInternalNode(fn, params, columns, alias)
        if alias:
            self._name_dict[alias] = new_col
        self._name_dict[new_col.sql(self._ctx)] = new_col
        return new_col

    def _apply_func(self, *fn_wrappers: Tuple[DfFnColWrapper], keep_old_cols=False):
        new_df = copy.deepcopy(self)
        if any(map(lambda fn_wrapper: fn_wrapper.has_agg_func(), fn_wrappers)):
            if not new_df._need_agg:
                new_df = self._transform_to_nested()
                new_df._need_agg = True

        new_df._ctx.dataframe = copy.deepcopy(self)

        if not keep_old_cols:
            new_df._select_list = []
        for fn_wrapper in fn_wrappers:
            if not isinstance(fn_wrapper, DfFnColWrapper):
                raise Exception("fn_wrapper should be a DfFnColWrapper object!")
            new_col: DfColumnNode = new_df._unwrap(fn_wrapper)
            if fn_wrapper.has_agg_func():
                new_col.is_agg = True
            new_df._assert_alias_not_exists(new_col.alias)
            new_df._select_list.append(new_col)

        if any(map(lambda fn_wrapper: fn_wrapper.has_agg_func(), fn_wrappers)):
            new_df._need_agg = False

        new_df._ctx.dataframe = new_df
        return new_df

    def _has_agg(self):
        return any(map(lambda col: col.is_agg, self._select_list))

    def _try_to_trival(self, new_df):
        if len(new_df.data.group_by) == 0 and len(new_df._select_list) == 1:
            if new_df._select_list[0].alias is None:
                new_df._select_list[0].alias = "__result__"
            sql = new_df.getExecutedSql()
            result = new_df.olap_utils.execute(sql)
            if len(result) == 1 and len(result[0]) == 1:
                result = result[0][0]
                if isinstance(result, (int, float, list)):
                    return result
        return new_df

    def _set_cte(self, cte):
        self.data.cte = cte

    def _transform_to_nested(self):
        new_df = copy.deepcopy(self)
        subquery = new_df.getExecutedSql()
        del new_df.data.columns[:]
        del new_df.data.filters[:]
        del new_df.data.group_by[:]
        del new_df.data.order_by[:]
        new_df.data.limit.limit = 0
        del new_df._select_list[:]
        new_df.data.cte = ""

        for col_name in new_df._name_dict:
            new_df._name_dict[col_name] = DfColumnLeafNode(col_name)

        if new_df.data.source.type == DfPb.SourceType.ClickHouse:
            new_df.data.source.clickhouse.table_name = subquery
            new_df.data.source.clickhouse.database = "Nested"
        elif new_df.data.source.type == DfPb.SourceType.StarRocks:
            new_df.data.source.starrocks.table_name = subquery
            new_df.data.source.starrocks.database = "Nested"
        else:
            raise Exception("not support source type")
        return new_df

    def union(self, df, is_physical_table=True):
        """
        This function is used to union two DataFrames.
        The two DataFrames must have the same number of columns, and the columns must have the same names and order.
        >>> df1 = df1.union(df2)
        """
        if len(self._select_list) != len(df._select_list):
            raise Exception("the length of select list not match")

        for i in range(len(self._select_list)):
            name1 = self._select_list[i].alias
            if name1 == None or name1 == "":
                name1 = self._select_list[i].sql(self._ctx)
            name2 = df._select_list[i].alias
            if name2 == None or name2 == "":
                name2 = df._select_list[i].sql(df._ctx)

            if name1 != name2:
                raise Exception("the column name not match, %s != %s" % (name1, name2))

        new_df = copy.deepcopy(self)
        del new_df.data.columns[:]
        del new_df.data.filters[:]
        del new_df.data.group_by[:]
        del new_df.data.order_by[:]
        new_df.data.limit.limit = 0
        del new_df._select_list[:]
        new_df.data.cte = ""

        subquery1 = self.getExecutedSql()
        subquery2 = df.getExecutedSql()
        if new_df.data.source.type == DfPb.SourceType.ClickHouse:
            new_df.data.source.clickhouse.table_name = f"({subquery1}) union all ({subquery2})"
            new_df.data.source.clickhouse.database = "Nested"
        elif new_df.data.source.type == DfPb.SourceType.StarRocks:
            new_df.data.source.starrocks.table_name = f"({subquery1}) union all ({subquery2})"
            new_df.data.source.starrocks.database = "Nested"
        new_df._select_list = []
        new_df._name_dict = {}
        for col in self._select_list:
            name = col.alias
            if name == None or name == "":
                name = col.sql(self._ctx)
            new_df._select_list.append(new_df._expand_expr(name))

        new_df._finalize()
        return new_df.materializedView(is_physical_table=is_physical_table)

    # parse executed sql, remove limit
    def getExecutedSql(self):
        new_df = self.fill_result(1)
        sql = new_df.data.execute_sql
        sql = re.sub(r"LIMIT.*$", "", sql, flags=re.IGNORECASE)
        return sql

    # is_temp = True: 每天凌晨 2 点自动删除
    def materializedView(
        self,
        is_physical_table=False,
        is_distributed_create=True,
        is_temp=False,
        table_name=None,
    ):
        get_context().set_project_conf(self._provider.project_conf)
        materialized_sql = self.getExecutedSql()
        if table_name is not None:
            new_df_name = table_name
        else:
            new_df_name = DataFrame.createTableName(is_temp)

        origin_table_name = self.getTableName()
        if self._origin_table_name is not None:
            origin_table_name = self._origin_table_name

        if self.engine == OlapEngineType.CLICKHOUSE:
            ClickHouseUtils.clickhouse_create_view_v2(
                table_name=new_df_name,
                select_statement=materialized_sql,
                origin_table_name=origin_table_name,
                is_physical_table=is_physical_table,
                is_distributed_create=is_distributed_create,
                is_agg_status=(self.data.source.clickhouse.database == "Nested"),
                provider=self._provider
                )
            return self._provider.readClickHouse(new_df_name)
        elif self.engine == OlapEngineType.STARROCKS:
            StarRocksUtils().create_view(
                view_name=new_df_name,
                sql_statement=materialized_sql,
                is_table=is_physical_table,
            )
            return readStarRocks(new_df_name)
        else:
            raise Exception(f"Olap engine `{self.engine}` not supported.")

    def createDfRequest(self):
        df_req = DfPb.DataFrameRequest()
        df_req.df.CopyFrom(self.data)
        df_req.task_type = self.task_type
        df_req.rtx = self.rtx
        df_req.device_id = self.device_id
        df_req.database = self.database
        return df_req

    def createDataFrameRequestBase64(self):
        df_req = self.createDfRequest()
        return base64.b64encode(df_req.SerializeToString()).decode()

    def createDfRequestJson(self):
        df_req = self.createDfRequest()
        return json_format.MessageToJson(df_req)

    def createCurlRequest(self):
        return (
            'curl  -H "Content-Type: application/json" -X POST -d \''
            + self.createDfRequestJson()
            + "' "
            + self.url
            + "/json"
        )

    def execute(self, retry_times=1):
        from fast_causal_inference.common import get_context

        logger = get_context().logger
        while retry_times > 0:
            try:
                json_body = self.createDfRequestJson()
                logger.debug("url= " + self.url + ",data= " + json_body)
                resp = requests.post(
                    self.url,
                    data=json_body.encode("utf-8"),
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                    },
                )
                logger.debug("response=" + resp.text)
                df_resp = Parse(resp.text, DfPb.DataFrameResponse())
                if df_resp.status == DfPb.RetStatus.FAIL:
                    print("Error: ")
                    print(df_resp.msg)
                elif df_resp.status == DfPb.RetStatus.SUCC:
                    self.data = df_resp.df
                    if not self._select_list:
                        self._select_list = [
                            DfColumnLeafNode(
                                column_name=col.name, alias=col.alias, type=col.type
                            )
                            for col in df_resp.df.columns
                        ]
                        for col in self._select_list:
                            self._name_dict[col.sql(self._ctx)] = col
                return
            except Exception as e:
                time.sleep(1)
                retry_times -= 1
                if retry_times == 0:
                    self._provider.context.logger.error(f"Exception occurred when executing.\n")
                    raise e

    @staticmethod
    def createTableName(is_temp=False):
        table_name = "df_table_"
        if is_temp:
            table_name += "temp_"
        table_name += time.strftime("%Y%m%d%H%M%S", time.localtime()) + "_"
        table_name += str(random.randint(100000, 999999))
        return table_name

    def fill_column_info(self):
        self.task_type = DfPb.TaskType.FILL_SCHEMA
        self.execute()

    # 如果有嵌套子查询，获取最内层的表名
    def getTableName(self):
        '''
        Obtain the temporary table name for the current dataframe operation. Note that this table may be cleaned up later. If you wish to use it for the long term, please persist it as a permanent table using df.toClickHouse(table)        
        
        Example
        ----------
        
        >>> df.getTableName() # 得到df对应的临时表
        # df_table_20240610220812_600001 
        >>> df.toClickHouse('permanent_table') # 导出指定的表名称
        >>> # 下次可以避免再导入，直接读取“permanent_table”表做分析
        >>> import fast_causal_inference
        >>> allinsql_provider = fast_causal_inference.FCIProvider("all_in_sql_guest")
        >>> df.readClickHouse('permanent_table')
        '''
        if self.engine == OlapEngineType.CLICKHOUSE:
            return self.data.source.clickhouse.table_name
        elif self.engine == OlapEngineType.STARROCKS:
            return self.data.source.starrocks.table_name
        else:
            raise Exception(f"Olap engine `{self.engine}` not supported.")

    def toCsv(self, csv_file_abs_path):
        """
        Convert the data from ClickHouse table to a CSV file.

        >>> df.toCsv("/path/to/output.csv") # 把数据导出到当前文件所在目录下的output.csv文件里，覆盖写入。
        """

        if self.engine == OlapEngineType.CLICKHOUSE:
            global_view_name = ClickHouseUtils(provider=self._provider).create_global_view(self.getExecutedSql())
            clickhouse_2_csv(global_view_name, csv_file_abs_path)
        elif self.engine == OlapEngineType.STARROCKS:
            new_df = self.materializedView(is_temp=True, is_physical_table=True)
            starrocks_2_csv(new_df.getTableName(), csv_file_abs_path)

    def toSparkDf(self, session):
        """
        ClickHouse table >> spark dataframe.

        Parameters
        ----------
        :param session: The spark session.

        Example
        ----------
        ::

            import fast_causal_inference
            
            allinsql_provider = fast_causal_inference.FCIProvider("all_in_sql_guest")
            spark = allinsql_provider.buildSparkSession(group_id=GROUP_ID, gaia_id=GAIA_ID)

            df = allinsql_provider.readClickHouse('test_data_small')
            spark_df = df.toSparkDf(spark)

            print(spark_df.count())
        """

        if self.engine == OlapEngineType.CLICKHOUSE:
            return clickhouse_2_dataframe(
                session, self.data.source.clickhouse.table_name
            )
        elif self.engine == OlapEngineType.STARROCKS:
            return starrocks_2_dataframe(session, self.data.source.starrocks.table_name)
        else:
            raise Exception(f"Olap engine `{self.engine}` not supported.")

    def toClickHouse(self, clickhouse_table_name, is_drop_table=False):
        """
        Convert the ClickHouse dataframe into a table within ClickHouse to allow for permanent reuse of the dataframe by querying this table

        Example
        ----------
        
        >>> df.getTableName() # 得到df对应的临时表
        # df_table_20240610220812_600001 
        
        >>> df.toClickHouse('permanent_table') # 导出指定的表名称
        >>> # 下次可以避免再导入，直接读取“permanent_table”表做分析
        >>> import fast_causal_inference
        >>> allinsql_provider = fast_causal_inference.FCIProvider("all_in_sql_guest")
        >>> df.readClickHouse('permanent_table')

        """
        if is_drop_table:
            ClickHouseUtils.clickhouse_drop_view(clickhouse_table_name, self._provider)
        clickhouse_table_name = clickhouse_table_name
        self.materializedView(is_physical_table=True, table_name=clickhouse_table_name)

    def toStarRocks(self, starrocks_table_name):
        """
        StarRocks table >> StarRocks table.

        Example
        ----------

        >>> df.toStarRocks("new_table")
        """
        self.materializedView(is_physical_table=True, table_name=starrocks_table_name)

    def toOlap(self, table_name):
        """
        dataframe >> Olap table.

        Example
        ----------

        >>> df.toOlap("new_table")
        """
        self.materializedView(is_physical_table=True, table_name=table_name)


def readClickHouse(table_name, provider=FCIProvider("global")):
    df = DataFrame(provider=provider)
    df.data.source.clickhouse.table_name = table_name
    df.data.source.clickhouse.database = df.database
    df.fill_column_info()
    df._origin_table_name = table_name
    return df


def readStarRocks(table_name):
    
    df = DataFrame(olap_engine=OlapEngineType.STARROCKS)
    df.data.source.starrocks.table_name = table_name
    df.data.source.starrocks.database = df.database
    df.fill_column_info()
    return df


def readOlap(table_name, olap="clickhouse", provider=FCIProvider("global")):
    

    if isinstance(olap, OlapEngineType):
        olap = str(olap)
    if olap.lower() == "clickhouse":
        return readClickHouse(table_name, provider)
    elif olap.lower() == "starrocks":
        return readStarRocks(table_name)
    else:
        raise Exception(f"Unsupported olap {olap}")


def readTdw(
    session,
    db,
    table,
    group="tl",
    tdw_user=None,
    tdw_passward=None,
    priParts=None,
    str_replace="-1",
    numeric_replace=0,
    olap="clickhouse",
    provider=FCIProvider("global")
):
    get_context().set_project_conf(provider.project_conf)
    from pytoolkit import TDWSQLProvider

    tdw = TDWSQLProvider(
        session, group=group, db=db, user=tdw_user, passwd=tdw_passward
    )
    df_new = tdw.table(tblName=table, priParts=priParts)
    df_new = AisTools.preprocess_na(df_new, str_replace, numeric_replace)
    table_name = DataFrame.createTableName()
    if olap.lower() == "clickhouse":
        dataframe_2_clickhouse(dataframe=df_new, clickhouse_table_name=table_name)
        return readClickHouse(table_name, provider)
    elif olap.lower() == "starrocks":
        dataframe_2_starrocks(dataframe=df_new, starrocks_table_name=table_name)
        return readStarRocks(table_name)
    else:
        raise Exception(f"Olap engine `{olap}` not supported.")


def readSparkDf(session, dataframe, str_replace="-1", numeric_replace=0, olap="clickhouse", provider=FCIProvider("global")):
    get_context().set_project_conf(provider.project_conf)
    get_context().spark_session = session
    dataframe = AisTools.preprocess_na(dataframe, str_replace, numeric_replace)
    table_name = DataFrame.createTableName()
    if olap.lower() == "clickhouse":
        dataframe_2_clickhouse(dataframe=dataframe, clickhouse_table_name=table_name)
        return readClickHouse(table_name)
    elif olap.lower() == "starrocks":
        dataframe_2_starrocks(dataframe=dataframe, starrocks_table_name=table_name)
        return readStarRocks(table_name)
    else:
        raise Exception(f"Olap engine `{olap}` not supported.")


def readCsv(csv_file_abs_path, olap="clickhouse", provider=FCIProvider("global")):
    df = DataFrame()
    table_name = DataFrame.createTableName()
    if olap.lower() == "clickhouse":
        csv_2_clickhouse(
            csv_file_abs_path=csv_file_abs_path, clickhouse_table_name=table_name
        )
        return readClickHouse(table_name)
    elif olap.lower() == "starrocks":
        csv_2_starrocks(
            csv_file_abs_path=csv_file_abs_path, starrocks_table_name=table_name
        )
        return readStarRocks(table_name)
    else:
        raise Exception(f"Olap engine `{olap}` not supported.")

