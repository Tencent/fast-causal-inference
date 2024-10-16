.. allinsql documentation master file, created by
   sphinx-quickstart on Sun Mar 17 22:07:11 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Fast-Causal-Inference's documentation!
====================================


Quickstart
-----------------------
``fast-causal-inference`` 是一个分布式因果推断工具包，专为处理大规模数据集设计。它是由腾讯团队开发的，旨在解决业务同学在分析亿级别数据时面临的挑战。
- **分布式计算**：``fast-causal-inference`` 可以在分布式环境中运行，使其能够处理大规模数据集。它能够在不到一秒钟的时间内处理高达6亿级别的数据进行t检验。此外，它的uplift模型可以兼容数百个维度，使其成为处理复杂、高维度数据的选择。
- **多种因果推断方法**：``fast-causal-inference`` 提供了多种因果推断方法，包括实验分析方法（ `t检验 <https://tencent.github.io/fast-causal-inference/inference.html#dataframe.statistics.ttest_2samp>`_ , `SRM检验 <https://tencent.github.io/fast-causal-inference/inference.html#dataframe.statistics.srm>`_ , `非参检验 <https://tencent.github.io/fast-causal-inference/inference.html#dataframe.statistics.mann_whitney_utest>`_ , `重抽样方法 <https://tencent.github.io/fast-causal-inference/inference.html#dataframe.statistics.boot_strap>`_ , `permutation检验 <https://tencent.github.io/fast-causal-inference/inference.html#dataframe.statistics.permutation>`_ 等），观测性数据分析能力（ `Match匹配方法 <https://tencent.github.io/fast-causal-inference/inference.html#match>`_ , `Uplift模型 <https://tencent.github.io/fast-causal-inference/inference.html#uplift>`_ , `DID双重差分法 <https://tencent.github.io/fast-causal-inference/inference.html#dataframe.regression.DID>`_ , `IV工具变量法 <https://tencent.github.io/fast-causal-inference/inference.html#dataframe.regression.IV>`_ 等）

- **基础的统计分析和数据处理工具**： ``fast-causal-inference`` 提供了 `统计量和描述性分析 <https://tencent.github.io/fast-causal-inference/inference.html#regression>`_ ， `回归方法 <https://tencent.github.io/fast-causal-inference/inference.html#regression>`_ 以及 基础的数据处理工具，如 `特征工程 <https://tencent.github.io/fast-causal-inference/dataframe.html#features>`_ 。

Install
-----------------------
!pip3 install --verbose -U fast-causal-inference

Uninstall
-----------------------
!pip3 install --verbose -U fast-causal-inference

Set DataBase
-----------------------
用户需要使用FCIProvider接口登录数据库(DB)，并设置相关的Spark Session。我们的系统将不同的业务数据隔离在各自的数据库中，以保证数据的安全性和独立性。在使用数据时，系统会根据业务维度进行权限验证。

Set SparkSession
-----------------------

1. 申请spark session
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import os
    from pyspark.sql import SparkSession
    import fast_causal_inference
    allinsql_provider = fast_causal_inference.FCIProvider("olap_database")

    JARS = ",".join([
        "starrocks-spark-connector-3.1.2.jar",  # spark 读写 SR 依赖的jar包
        "mysql-connector-j-8.0.31.jar",         # spark 获取 SR 元信息依赖的jar包
    ])
    spark = (
        SparkSession.builder.enableHiveSupport()
        .config('spark.driver.memory', '8g')
        .config("spark.yarn.queue", "")
        .config('spark.executor.cores', 4)
        .config('spark.executor.memory', '8g')
        .config('spark.jars', JARS)
        .getOrCreate()
    )

    # 注意：必须在申请spark session之后，才能导入pyspark相关的包，不能提前导入
    import pyspark.sql.functions as f
    import pyspark.sql.types as t


2. 用spark session读取thive/iceberg数据
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    spark_df = spark.sql('select * from hive_database.hive_table')
    print(spark_df.count())


3. 把pyspark dataframe转换成fast-causal-inference dataframe
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    df_ch = allinsql_provider.readSparkDf(spark, spark_df)  # 把sparkDf导入到fast-causal-inference的dataframe中



.. toctree::
   :maxdepth: 2

   input-output

.. toctree::
   :maxdepth: 2

   dataframe


.. toctree::
   :maxdepth: 2

   inference
   


