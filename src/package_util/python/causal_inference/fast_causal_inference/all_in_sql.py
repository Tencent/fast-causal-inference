"""
all in sql utils
"""
from time import perf_counter as _perf_counter
import time
import requests
from .util.utils import *
from .util.spark import get_spark_session

def tdw_2_clickhouse(tdw_database_name, tdw_table_name, clickhouse_table_name, spark_session, cmk=None,
                     tdw_partition_list=None, clickhouse_partition_column=None, clickhouse_primary_column=None,
                     is_auto_create=True):
    """
    tdw入仓到clickhouse
    支持tdw分区表/非分区表, 会进行数据分片多机并行导入各个CK local节点，组成CK分布式表
    如果CK目标表已经存在则写入数据为Append模式
    :param tdw_database_name:  tdw数据库名
    :param tdw_table_name:     tdw表名
    :param clickhouse_table_name:  clickhouse表名
    :param spark_session:   sparkSession
    :param cmk:     用户本人的tianqiong CMK信息
    :param tdw_partition_list:    tdw分区列表
    :param clickhouse_partition_column:   自动创建表时clickhouse_partition_column指定CK表的分区，可不填该参数则不分区
    :param clickhouse_primary_column:     自动创建表时clickhouse_primary_column指定CK表的主键列，可不填该参数则无主键
    :param is_auto_create:   True 表示识别schema自动创建表
    :return:
    """
    print("running, please wait")
    start = _perf_counter()
    from .databus.tdw import TDWUtils
    TDWUtils.tdw_2_clickhouse_distribute(tdw_database_name=tdw_database_name, tdw_table_name=tdw_table_name,
                                         clickhouse_table_name=clickhouse_table_name, spark=spark_session, cmk=cmk,
                                         tdw_partition_list=tdw_partition_list,
                                         clickhouse_partition_column=clickhouse_partition_column,
                                         clickhouse_primary_column=clickhouse_primary_column,
                                         is_auto_create=is_auto_create)
    end = _perf_counter()
    print("done" + 'time cost: %s Seconds' % (end - start))


def clickhouse_2_tdw(clickhouse_table_name, tdw_database_name, tdw_table_name, spark_session, cmk=None,
                     is_auto_create=True):
    """
    clickhouse出仓到tdw, 会在各个CK local节点多机并行导出到Tdw表
    :param clickhouse_table_name:   clickhouse表名
    :param tdw_database_name:   tdw数据库名
    :param tdw_table_name:   tdw表名
    :param spark_session:   sparkSession
    :param cmk:     用户本人的tianqiong CMK信息
    :param is_auto_create:   True 表示识别schema自动创建表
    :return:
    """
    print("running, please wait")
    from .databus.clickhouse import ClickHouseUtils
    start = _perf_counter()
    ClickHouseUtils.clickhouse_2_tdw(clickhouse_table_name, tdw_database_name, tdw_table_name, spark_session, cmk,
                                     is_auto_create)
    end = _perf_counter()
    print("done" + 'time cost: %s Seconds' % (end - start))


# multi partition write only one node
def __dataframe_2_clickhouse_one(dataframe, clickhouse_table_name, clickhouse_partition_column=None,
                                 clickhouse_primary_column=None, is_auto_create=True, num_partitions=5,
                                 batch_size=100000):
    print("running, please wait")
    start = _perf_counter()
    from .databus.tdw import TDWUtils
    TDWUtils.datafame_2_clickhouse(dataframe, clickhouse_table_name, clickhouse_partition_column,
                                   clickhouse_primary_column, is_auto_create=is_auto_create,
                                   num_partitions=num_partitions, batch_size=batch_size)
    end = _perf_counter()
    print("done" + 'time cost: %s Seconds' % (end - start))


def dataframe_2_clickhouse(dataframe, clickhouse_table_name, clickhouse_partition_column=None,
                           clickhouse_primary_column=None, is_auto_create=True, num_partitions=10):
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
    print("running, please wait")
    start = _perf_counter()
    from .databus.tdw import TDWUtils
    TDWUtils.datafame_2_clickhouse_distribute(dataframe, clickhouse_table_name, clickhouse_partition_column,
                                              clickhouse_primary_column, is_auto_create=is_auto_create,
                                              num_partitions=num_partitions, batch_size=400000)
    end = _perf_counter()
    print("done" + 'time cost: %s Seconds' % (end - start))


# select from distribute table
def clickhouse_2_dataframe(spark_session, clickhouse_table_name, num_partitions=10):
    """
    clickhouse 出仓到 spark dataframe对象
    数据从CK分布式表会并行分片读入dataframe
    :param spark_session:  sparkSession
    :param clickhouse_table_name:  clickhouse表名
    :param num_partitions: 并行分片数, 默认10, 用户无需修改
    :return:
    """
    from .databus.clickhouse import ClickHouseUtils
    return ClickHouseUtils.clickhouse_2_dataframe(spark_session, clickhouse_table_name, partition_num=num_partitions)


# select from everyone node
def __clickhouse_2_dataframe_distribute(spark, clickhouse_table_name, clickhouse_database_name=None, partition_num=10):
    from .databus.clickhouse import ClickHouseUtils
    return ClickHouseUtils.clickhouse_2_dataframe_distribute(spark, clickhouse_table_name, partition_num,
                                                             clickhouse_database_name)


def csv_2_clickhouse(csv_file_abs_path, clickhouse_table_name, columns_dict=None, is_auto_create=True):
    """
    csv入仓到clickhouse
    :param csv_file_abs_path: csv文件绝对路径
    :param clickhouse_table_name:  clickhouse表名
    :param columns_dict:  columns_dict表示需要读入表中的某些列名 及 其对应的python字段类型
    for example: {"uin": int, "numbera": float, "numberb": str}
    :param is_auto_create:   True 表示识别schema自动创建表
    :return:
    """
    print("running, please wait")
    start = _perf_counter()
    from .databus.clickhouse import ClickHouseUtils
    clickhouse_utils = ClickHouseUtils()
    clickhouse_utils.csv_2_clickhouse(csv_file_abs_path, clickhouse_table_name, columns_dict,
                                      is_auto_create=is_auto_create)
    end = _perf_counter()
    print("done" + 'time cost: %s Seconds' % (end - start))


def clickhouse_2_csv(clickhouse_table_name, csv_file_abs_path):
    """
    clickhouse出仓到csv
    :param clickhouse_table_name: clickhouse表名
    :param csv_file_abs_path:  csv文件绝对路径
    :return:
    """
    print("running, please wait")
    start = _perf_counter()
    from .databus.clickhouse import ClickHouseUtils
    clickhouse_utils = ClickHouseUtils()
    clickhouse_utils.clickhouse_2_csv(clickhouse_table_name, csv_file_abs_path)
    end = _perf_counter()
    print("done" + 'time cost: %s Seconds' % (end - start))


def clickhouse_create_view(clickhouse_view_name, sql_statement, sql_table_name=None, sql_where=None, sql_group_by=None,
                           sql_limit=None, primary_column="tuple()", is_force_materialize=False, is_sql_complete=False, is_use_local=True):
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
    print("running, please wait")
    start = _perf_counter()
    from .databus.clickhouse import ClickHouseUtils
    ClickHouseUtils.create_view(clickhouse_view_name, sql_statement, sql_table_name, sql_where, sql_group_by,
                                sql_limit=sql_limit, primary_column=primary_column,
                                is_force_materialize=is_force_materialize, is_sql_complete=is_sql_complete, is_use_local=is_use_local)
    end = _perf_counter()
    print("done" + 'time cost: %s Seconds' % (end - start))


def clickhouse_drop_view(clickhouse_view_name):
    """
    在集群删除视图/物理表
    :param clickhouse_view_name:  视图或物理表名称
    :return:
    """
    print("running, please wait")
    start = _perf_counter()
    from .databus.clickhouse import ClickHouseUtils
    ClickHouseUtils.drop_view(clickhouse_view_name)
    end = _perf_counter()
    print("done" + 'time cost: %s Seconds' % (end - start))


def clickhouse_drop_partition(clickhouse_view_name, clickhouse_partition_name):
    """
    在集群删除分区
    :param clickhouse_view_name:  物理表名称
    :param clickhouse_partition_name:  分区名称
    :return:
    """
    print("running, please wait")
    start = _perf_counter()
    from .databus.clickhouse import ClickHouseUtils
    ClickHouseUtils.drop_partition(clickhouse_view_name, clickhouse_partition_name)
    end = _perf_counter()
    print("done" + 'time cost: %s Seconds' % (end - start))


def create_sql_instance():
    """
    创建数据分析sql实例
    :return: AllInSqlConn Instance
    """
    return AllInSqlConn()


def create():
    """
    创建数据分析sql实例
    :return: AllInSqlConn Instance
    """
    return create_sql_instance()


class AllInSqlConn:
    def __init__(self, device_id=None, db_name=None):
        datasource = dict()
        from . import PROJECT_CONF
        for cell in PROJECT_CONF['datasource']:
            datasource[cell['device_id']] = cell
        if not device_id:
            device_id = list(datasource.keys())[0]
        if not db_name:
            db_name = datasource[device_id]['clickhouse_database']
        self.device_id = device_id
        self.db_name = db_name

    def execute(self, sql, retry_times=1, is_calcite_parse=False):
        from . import PROJECT_CONF
        from . import logger
        url = PROJECT_CONF["sqlgateway"]["url"] + PROJECT_CONF["sqlgateway"]["path"]
        sql = sql.replace("\n", " ")
        sql = sql.replace("\"", "\\\"")
        json_body = '{"rawSql":"' + str(sql) + '", "creator":"' + get_user() + '", "deviceId":"' + str(
            self.device_id) + '","database":"' + self.db_name + '", "isDataframeOutput": true'
        if is_calcite_parse:
            json_body += ', "isCalciteParse": true}'
        else:
            json_body += '}'
        while retry_times > 0:
            try:
                logger.debug("url= " + url + ",data= " + json_body)
                resp = requests.post(url, data=json_body.encode("utf-8"), headers={'Content-Type': 'application/json', 'Accept': 'application/json'})
                logger.debug("response=" + resp.text)
                # result = resp.text.replace("\\t", "\t").replace("\\n", "\n").replace("\\r", "\r").replace("\\'", "'")
                if resp.json()["status"] == 0:
                    if is_calcite_parse:
                        return str(resp.json()["data"]["executeSql"])
                    else:
                        return str(resp.json()["data"]["result"])
                else:
                    if "message" in resp.json().keys():
                        return "error message:" + resp.json()["message"] + ", error code" + str(resp.json()["status"])
                    elif "error" in resp.json().keys():
                        return "error message:" + resp.json()["error"] + ", error code" + str(resp.json()["status"])
                    else:
                        return "error message none" + ", error code" + resp.json()["status"]
            except Exception as e:
                time.sleep(1)
                retry_times -= 1
                if retry_times == 0:
                    return str(e)

    def sql(self, sql, is_calcite_parse=False, is_dataframe=True):
        res = self.execute(sql, is_calcite_parse=is_calcite_parse)
        from . import logger
        if (res.find("Exception") != -1) or "error message" in res:
            logger.info("execute content result:" + str(res))
        else:
            logger.debug("execute content result:" + str(res))
        if (res.find("Exception") != -1) or "error message" in res or is_calcite_parse or not is_dataframe:
            return res
        else:
            return output_dataframe(res)
