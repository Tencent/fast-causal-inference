import os
from fast_causal_inference.common import get_context


def build_spark_session(
    group_id,
    gaia_id,
    cmk=None,
    driver_cores=4,
    driver_memory="8g",
    executor_cores=2,
    executor_memory="10g",
    **kwargs,
):
    if "host" in os.environ and os.environ["host"] and "taiji" in os.environ["host"]:
        import taiji_ide

        if cmk:
            taiji_ide.set_cmk(cmk)
        taiji_ide.set_spark(version="3.1.2", gaia_id=gaia_id)

    from pyspark.sql import SparkSession

    os.environ['GROUP_ID'] = str(group_id)
    os.environ['GAIA_ID'] = str(gaia_id)

    spark_builder = (
        SparkSession.builder.enableHiveSupport()
        .config("hive.metastore.uris", os.environ.get("metastore"))
        .config(
            "spark.sql.catalog.iceberg_catalog", "org.apache.iceberg.spark.SparkCatalog"
        )
        .config("spark.sql.catalog.iceberg_catalog.type", "hive")
        .config(
            "spark.sql.extensions",
            "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions",
        )
        .config("spark.yarn.queue", "{}-offline".format(group_id))
        .config("spark.yarn.dist.archives", os.environ.get("ARCHIVES"))
        .config("spark.driver.cores", driver_cores)
        .config("spark.driver.memory", driver_memory)
        .config("spark.driver.maxResultSize", "0")
        .config("spark.blacklist.enabled", "true")
        .config("spark.executor.cores", executor_cores)
        .config("spark.executor.memory", executor_memory)
        .config("spark.speculation", "false")  # 建议关闭推测执行，避免数据重复写入
        .config("spark.dynamicAllocation.enabled", "true")
        .config("spark.dynamicAllocation.maxExecutors", 200)
        .config("spark.dynamicAllocation.minExecutors", 5)
        .config("spark.network.timeout", "300s")
        .config("spark.sql.broadcastTimeout", "800")
        .config("spark.hadoop.dfs.ha.zkfc.nn.http.timeout.ms", "66000")
        .config("spark.hadoop.dfs.client.readslow-blacklist", "true")
        .config("spark.hadoop.dfs.client.readslow-cost-max", "100")
        .config("spark.hadoop.dfs.client.readslow-count-max", "5")
    )
    # .config("spark.task.maxFailures", 0)
    # .config("spark.excludeOnFailure.enabled", "false")
    for spark_key, spark_value in kwargs.items():
        spark_key_config = spark_key.replace("_", ".")
        spark_builder = spark_builder.config(spark_key_config, spark_value)
    spark = spark_builder.getOrCreate()
    sc = spark.sparkContext.getOrCreate()
    # get_context().logger.info(f"Spark Ui Web Url: {sc.uiWebUrl}")
    return spark
