"""
Fast Causal Inference
"""

__all__ = [
    "set_tenant",
    "set_config",
    "set_default",
    "readTdw",
    "readStarRocks",
    "readClickHouse",
    "readSparkDf",
    "readCsv",
    "get_context",
    "create",
    "create_sql_instance",
    "clickhouse_create_view",
    "clickhouse_create_view_v2",
    "clickhouse_drop_view",
    "clickhouse_drop_partition",
    "clickhouse_2_csv",
    "csv_2_clickhouse",
    "clickhouse_2_tdw",
    "dataframe_2_clickhouse",
    "clickhouse_2_dataframe",
    "tdw_2_clickhouse",
    "FCIProvider",
]

from .dataframe import (
    readClickHouse,
    readTdw,
    readStarRocks,
    readSparkDf,
    readCsv,
)
import importlib

from .dataframe.provider import FCIProvider


get_context = importlib.import_module("fast_causal_inference.common").get_context


def set_config(conf_path):
    get_context().set_project_conf_from_yaml(conf_path)

def create():
    import fast_causal_inference.util as fci_util

    return fci_util.create()


def create_sql_instance():
    import fast_causal_inference.util as fci_util

    return fci_util.create_sql_instance()


def clickhouse_create_view(*args, **kwargs):
    from fast_causal_inference.util import ClickHouseUtils

    return ClickHouseUtils.clickhouse_create_view(*args, **kwargs)


def clickhouse_create_view_v2(*args, **kwargs):
    from fast_causal_inference.util import ClickHouseUtils

    return ClickHouseUtils.clickhouse_create_view_v2(*args, **kwargs)


def clickhouse_drop_view(*args, **kwargs):
    from fast_causal_inference.util import ClickHouseUtils

    return ClickHouseUtils.clickhouse_drop_view(*args, **kwargs)


def clickhouse_drop_partition(*args, **kwargs):
    from fast_causal_inference.util import ClickHouseUtils

    return ClickHouseUtils.clickhouse_drop_partition(*args, **kwargs)


def clickhouse_2_csv(*args, **kwargs):
    from fast_causal_inference.util import ClickHouseUtils

    return ClickHouseUtils.clickhouse_2_csv(*args, **kwargs)


def csv_2_clickhouse(*args, **kwargs):
    from fast_causal_inference.util import ClickHouseUtils

    return ClickHouseUtils.csv_2_clickhouse(*args, **kwargs)


def clickhouse_2_tdw(*args, **kwargs):
    from fast_causal_inference.util import data_transformer

    return data_transformer.clickhouse_2_tdw(*args, **kwargs)


def dataframe_2_clickhouse(*args, **kwargs):
    from fast_causal_inference.util import data_transformer

    return data_transformer.dataframe_2_clickhouse(*args, **kwargs)


def clickhouse_2_dataframe(*args, **kwargs):
    from fast_causal_inference.util import data_transformer

    return data_transformer.clickhouse_2_dataframe(*args, **kwargs)


def tdw_2_clickhouse(*args, **kwargs):
    from fast_causal_inference.util import data_transformer

    return data_transformer.tdw_2_clickhouse(*args, **kwargs)
