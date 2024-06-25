from fast_causal_inference.common.context import AllInSqlGlobalContext
from fast_causal_inference.common import get_context
import os
import yaml


class FCIProvider:
    def __init__(self, database="global"):
        self._ctx = None

        self.DEFAULT_SPARK_CONFS = {}
        current_file_path = os.path.abspath(__file__)
        # 将当前文件的路径中的__init__.py替换为conf/envs.yaml
        init_conf_file_path = current_file_path.replace("dataframe/provider.py", "conf/envs.yaml")

        with open(init_conf_file_path, "r") as file:
            env_vars = yaml.safe_load(file)

            for key, value in env_vars.items():
                if key == "default-spark-confs":
                    self.DEFAULT_SPARK_CONFS = value
                    continue
                if key not in ('RAINBOW_URL', 'RAINBOW_GROUP', 'RAINBOW_ENV', 'mirrors'):
                    continue
                if not value:
                    raise Exception(
                        f"value of env `{key}` is empty, please check conf/envs.yaml."
                    )
                os.environ[key] = value

        self.database = database

        self._ctx = AllInSqlGlobalContext()
        self._ctx.set_project_conf_from_yaml('.jupyter/conf.yaml')
        self.context.logger.debug(vars(self))


    def get_context(self):
        if self._ctx is None:
            from fast_causal_inference.common import get_context
            self._ctx = get_context()
        return self._ctx

    def buildSparkSession(self, **kwargs):
        """
        Build a spark session with the specified configurations.

        Parameters
        ----------

        :param driver_cores (int, optional): The number of cores for the driver. Defaults to 4.
        :param driver_memory (str, optional): The memory for the driver. Defaults to "8g".
        :param executor_cores (int, optional): The number of cores for the executor. Defaults to 2.
        :param executor_memory (str, optional): The memory for the executor. Defaults to "10g".
        :param **kwargs: Additional keyword arguments to pass to the Spark session builder.

        Returns
        ----------

        SparkSession: A configured Spark session.

        Example
        ----------
        ::


            import fast_causal_inference
            import os
            allinsql_provider = fast_causal_inference.FCIProvider("all_in_sql_guest")
            spark = allinsql_provider.buildSparkSession(
            )
        """
        for key, value in self.DEFAULT_SPARK_CONFS.items():
            if not value:
                raise Exception(
                    f"value of env `{key}` is empty, please check conf/envs.yaml."
                )
            # self.context.logger.info(f"set enviroment variable `{key}` with `{value}`")
            os.environ[key] = value
        self.context.build_spark_session(**kwargs)
        return self.context.spark_session
    
    @property
    def context(self):
        return self.get_context()

    def get_project_conf(self):
        return self.get_context()._project_conf

    @property
    def project_conf(self):
        return self.get_project_conf()

    def _set_global_context(self):
        get_context().set_project_conf(self.get_project_conf())
        if self.context._spark_session:
            get_context().spark_session = self.context.spark_session

    def readClickHouse(self, table_name):
        """
        Read data from a ClickHouse table into a DataFrame.

        >>> import fast_causal_inference
        >>> allinsql_provider = fast_causal_inference.FCIProvider('all_in_sql_guest')
        >>> df = allinsql_provider.readClickHouse('test_data_small')
        """
        from fast_causal_inference.dataframe.dataframe import readClickHouse

        self._set_global_context()
        return readClickHouse(table_name, self)

    def readStarRocks(self, table_name):
        """
        Read data from a StarRocks table into a DataFrame.

        >>> allinsql_provider = fast_causal_inference.FCIProvider('all_in_sql_guest')
        >>> df = allinsql_provider.readStarRocks("test_data_small")
        """
        from fast_causal_inference.dataframe.dataframe import readStarRocks

        self._set_global_context()
        return readStarRocks(table_name)

    def readOlap(self, table_name, olap="clickhouse"):
        """
        Read data from a olap table into a DataFrame.

        >>> import fast_causal_inference
        >>> allinsql_provider = fast_causal_inference.FCIProvider('all_in_sql_guest')
        >>> df = allinsql_provider.readOlap("test_data_small", olap="starrocks")
        >>> df = allinsql_provider.readOlap("test_data_small", olap="clickhouse")
        """
        from fast_causal_inference.dataframe.dataframe import readOlap

        self._set_global_context()
        return readOlap(table_name, olap, self)
    
    def readSparkDf(self, session, dataframe, str_replace="-1", numeric_replace=0, olap="clickhouse"):
        """

        Parameters
        ----------

        :param session: The spark session.
        :param dataframe: The spark dataframe.
        :param str_replace: str, optional. The value to fill in na for string columns; `-1` by default.
        :param numeric_replace: int|float, optional, The value to fill in na for numeric columns; `-1` by default.
        :param olap: str optional. The type of Olap engine to save data. `clickhouse` or `starrocks`.
        :param provider: FCIProvider optional. The FCIProvider. 

        Example
        -------

        .. code-block:: python

            import fast_causal_inference
            allinsql_provider = fast_causal_inference.FCIProvider("all_in_sql_guest")
            from pytoolkit import TDWSQLProvider
            print(spark_df.count())
            df_ch = allinsql_provider.readSparkDf(spark, spark_df)

                
        """
        from fast_causal_inference.dataframe.dataframe import readSparkDf

        self._set_global_context()
        return readSparkDf(session, dataframe, olap=olap, str_replace=str_replace, numeric_replace=numeric_replace, provider=self)
    
    def readCsv(self, csv_file_abs_path, olap="clickhouse"):
        """
        Read data from a CSV file into a DataFrame.

        >>> import fast_causal_inference
        >>> allinsql_provider = fast_causal_inference.FCIProvider('all_in_sql_guest')
        >>> df = allinsql_provider.readCsv("test_data_small.csv")
        >>> df = allinsql_provider.readCsv("test_data_small.csv", olap="starrocks")
        """
        from fast_causal_inference.dataframe.dataframe import readCsv

        self._set_global_context()
        return readCsv(csv_file_abs_path, olap=olap, provider=self)

