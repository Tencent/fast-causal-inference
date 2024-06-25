__all__ = [
    "dataframe_2_clickhouse",
    "dataframe_2_starrocks",
    "clickhouse_2_dataframe",
    "starrocks_2_dataframe",
    "csv_2_clickhouse",
    "clickhouse_2_csv",
    "starrocks_2_csv",
    "clickhouse_2_pandas",
    "starrocks_2_pandas"
]

from time import perf_counter as _perf_counter
from fast_causal_inference.common import get_context


def dataframe_2_clickhouse(
    dataframe,
    clickhouse_table_name,
    clickhouse_partition_column=None,
    clickhouse_primary_column=None,
    is_auto_create=True,
    num_partitions=10,
    batch_size=400000,
):
    """
    dataframe write multi partition write everyone node, as distribute clickhouse table dataframe对象的数据会分片并行导入
    各个CK local节点，
    组成CK分布式表
    如果CK目标表已经存在则写入数据为Append模式
    :param dataframe:  sparkSession dataframe
    :param clickhouse_table_name:  clickhouse表名
    :param clickhouse_partition_column: 自动创建表时clickhouse_partition_column指定CK表的分区，可不填该参数则不分区
    :param clickhouse_primary_column:  自动创建表时clickhouse_primary_column指定CK表的主键列，可不填该参数则无主键，同时指定导入时候按照dataframe中某一列分bucket，该列相同值一定会落入同一个节点的本地表上，可不填该参数则row随机写入CK节点
    :param is_auto_create:   True 表示识别schema自动创建表
    :param num_partitions:  并行分片数, 默认10, 用户无需修改
    :return:
    """
    get_context().logger.info("running, please wait")
    start = _perf_counter()
    from fast_causal_inference.util.tdw import TDWUtils

    TDWUtils.datafame_2_clickhouse_distribute(
        dataframe,
        clickhouse_table_name,
        clickhouse_partition_column,
        clickhouse_primary_column,
        is_auto_create=is_auto_create,
        num_partitions=num_partitions,
        batch_size=batch_size,
    )
    end = _perf_counter()
    get_context().logger.info("done" + "time cost: %s Seconds" % (end - start))


def dataframe_2_starrocks(
    dataframe,
    starrocks_table_name,
    starrocks_partition_column=None,
    starrocks_primary_column=None,
    is_auto_create=True,
    batch_size=100000,
):
    """
    dataframe write to starrocks through `stream load`.
    :param dataframe:  sparkSession dataframe
    :param starrocks_table_name:  starrocks表名
    :param starrocks_partition_column: 自动创建表时starrocks_partition_column指定SR表的分区，可不填该参数则不分区
    :param starrocks_primary_column:  自动创建表时starrocks_primary_column指定SR表的主键列，可不填该参数则无主键，同时指定导入时候按照dataframe中某一列分bucket，该列相同值一定会落入同一个节点的本地表上，可不填该参数则row随机写入CK节点
    :param is_auto_create:   True 表示识别schema自动创建表
    :param num_partitions:  并行分片数, 默认10, 用户无需修改
    :return:
    """
    get_context().logger.info("running, please wait")
    start = _perf_counter()
    from fast_causal_inference.util.tdw import TDWUtils

    TDWUtils.datafame_2_starrocks(
        dataframe,
        starrocks_table_name,
        starrocks_partition_column,
        starrocks_primary_column,
        is_auto_create=is_auto_create,
        batch_size=batch_size,
    )
    end = _perf_counter()
    get_context().logger.info("done" + "time cost: %s Seconds" % (end - start))


# select from distribute table
def clickhouse_2_dataframe(spark_session, clickhouse_table_name):
    """
    clickhouse 出仓到 spark dataframe对象  并行分片数, 默认10, 用户无需修改
    数据从CK分布式表会并行分片读入dataframe
    :param spark_session:  sparkSession
    :param clickhouse_table_name:  clickhouse表名
    :return:
    """
    from fast_causal_inference.util.clickhouse_utils import ClickHouseUtils

    return ClickHouseUtils.clickhouse_2_dataframe(spark_session, clickhouse_table_name)


# select from distribute table
def starrocks_2_dataframe(spark_session, starrocks_table_name):
    """
    starrocks 出仓到 spark dataframe对象  并行分片数, 默认10, 用户无需修改
    数据从SR表会通过 starrocks-spark-connnector 读入dataframe
    :param spark_session:  sparkSession
    :param starrocks_table_name:  starrocks表名
    :return:
    """
    from fast_causal_inference.util import StarRocksUtils

    return StarRocksUtils.starrocks_2_dataframe(spark_session, starrocks_table_name)


def csv_2_clickhouse(
    csv_file_abs_path, clickhouse_table_name, columns_dict=None, is_auto_create=True
):
    """
    csv入仓到clickhouse
    :param csv_file_abs_path: csv文件绝对路径
    :param clickhouse_table_name:  clickhouse表名
    :param columns_dict:  columns_dict表示需要读入表中的某些列名 及 其对应的python字段类型
    for example: {"uin": int, "numbera": float, "numberb": str}
    :param is_auto_create:   True 表示识别schema自动创建表
    :return:
    """
    get_context().logger.info("running, please wait")
    start = _perf_counter()
    from fast_causal_inference.util import ClickHouseUtils

    clickhouse_utils = ClickHouseUtils()
    clickhouse_utils.csv_2_clickhouse(
        csv_file_abs_path,
        clickhouse_table_name,
        columns_dict,
        is_auto_create=is_auto_create,
    )
    end = _perf_counter()
    get_context().logger.info("done" + "time cost: %s Seconds" % (end - start))


def csv_2_starrocks(
    csv_file_abs_path, starrocks_table_name, columns_dict=None, is_auto_create=True
):
    """
    csv入仓到starrocks
    :param csv_file_abs_path: csv文件绝对路径
    :param starrocks_table_name:  starrocks表名
    :param columns_dict:  columns_dict表示需要读入表中的某些列名 及 其对应的python字段类型
    for example: {"uin": int, "numbera": float, "numberb": str}
    :param is_auto_create:   True 表示识别schema自动创建表
    :return:
    """
    get_context().logger.info("running, please wait")
    start = _perf_counter()
    from fast_causal_inference.util import StarRocksUtils

    starrocks_utils = StarRocksUtils()
    starrocks_utils.csv_2_starrocks(
        csv_file_abs_path,
        starrocks_table_name,
        columns_dict,
        is_auto_create=is_auto_create,
    )
    end = _perf_counter()
    get_context().logger.info("done" + "time cost: %s Seconds" % (end - start))


def clickhouse_2_csv(clickhouse_table_name, csv_file_abs_path):
    """
    clickhouse出仓到csv
    :param clickhouse_table_name: clickhouse表名
    :param csv_file_abs_path:  csv文件绝对路径
    :return:
    """
    get_context().logger.info("running, please wait")
    start = _perf_counter()
    from fast_causal_inference.util.clickhouse_utils import ClickHouseUtils

    clickhouse_utils = ClickHouseUtils()
    clickhouse_utils.clickhouse_2_csv(clickhouse_table_name, csv_file_abs_path)
    end = _perf_counter()
    get_context().logger.info("done" + "time cost: %s Seconds" % (end - start))


def starrocks_2_csv(starrocks_table_name, csv_file_abs_path):
    """
    starrocks出仓到csv
    :param starrocks_table_name: starrocks表名
    :param csv_file_abs_path:  csv文件绝对路径
    :return:
    """
    get_context().logger.info("running, please wait")
    start = _perf_counter()
    from fast_causal_inference.util.starrocks_utils import StarRocksUtils

    starrocks_utils = StarRocksUtils()
    starrocks_utils.starrocks_2_csv(starrocks_table_name, csv_file_abs_path)
    end = _perf_counter()
    get_context().logger.info("done" + "time cost: %s Seconds" % (end - start))
    

def clickhouse_2_pandas(execute_sql, do_log=True):
    """
    clickhouse出仓到pandas
    :param clickhouse_table_name: clickhouse表名
    :return:
    """
    if do_log:
        get_context().logger.info("running, please wait")
    start = _perf_counter()
    from fast_causal_inference.util.clickhouse_utils import ClickHouseUtils

    clickhouse_utils = ClickHouseUtils()
    result = clickhouse_utils.clickhouse_2_pandas(execute_sql)
    end = _perf_counter()
    if do_log:
        get_context().logger.info("done" + "time cost: %s Seconds" % (end - start))
    return result


def starrocks_2_pandas(starrocks_table_name, do_log=True):
    """
    starrocks出仓到pandas
    :param starrocks_table_name: starrocks表名
    :return:
    """
    if do_log:
        get_context().logger.info("running, please wait")
    start = _perf_counter()
    from fast_causal_inference.util.starrocks_utils import StarRocksUtils

    sr = StarRocksUtils()
    result = sr.starrocks_2_pandas(starrocks_table_name)
    end = _perf_counter()
    if do_log:
        get_context().logger.info("done" + "time cost: %s Seconds" % (end - start))
    return result
