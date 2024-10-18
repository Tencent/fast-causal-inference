"""
This module is used to store the context of the fast_causal_inference package, including the logger instance, spark
session, etc.
这是整个fast_causal_inference包的全局上下文，包括logger实例，spark session，logger，模块配置等。
如果你想访问某个变量，请先 from fast_causal_inference.common import get_context，然后 get_context()拿到全局上下文，再访问对应变量。
"""

__all__ = ["get_context"]
__author__ = "cooperxiong"

from fast_causal_inference.common.fci_logger import get_logger


class AllInSqlGlobalContext:
    def __init__(self):
        self._logger = get_logger()
        self._project_conf = None
        self._spark_session = None

    @property
    def logger(self):
        return self._logger

    @property
    def project_conf(self):
        if self._project_conf is None:
            raise ValueError("Project configuration is not set yet.")
        return self._project_conf

    @property
    def spark_session(self):
        if self._spark_session is None:
            raise ValueError(
                "Spark session is not set yet, you can use `FCIProvider.acquire_spark_session` to acquire a spark session."
            )
        return self._spark_session

    @spark_session.setter
    def spark_session(self, spark_session_):
        self._spark_session = spark_session_

    def build_spark_session(
        self,
        group_id,
        gaia_id,
        cmk=None,
        driver_cores=4,
        driver_memory="8g",
        executor_cores=2,
        executor_memory="10g",
        **kwargs,
    ):
        from fast_causal_inference.util.spark import build_spark_session

        # if self._spark_session is not None:
        #     sc = self._spark_session.sparkContext.getOrCreate()
        #     get_context().logger.info(
        #         f"Spark session already exists, using {sc.uiWebUrl}"
        #     )
        #     return self._spark_session

        self._spark_session = build_spark_session(
            group_id,
            gaia_id,
            cmk,
            driver_cores,
            driver_memory,
            executor_cores,
            executor_memory,
            **kwargs,
        )
        return self._spark_session

    def set_project_conf(self, project_conf):
        self._project_conf = project_conf

    def set_project_conf_from_yaml(self, conf_path):
        if not conf_path:
            raise Exception("please input tenant_conf")

        import sys
        if conf_path not in sys.path:
            sys.path.append(conf_path)

        with open(conf_path, "r") as f:
            import yaml
            from yaml.loader import SafeLoader

            self._project_conf = yaml.load(f, Loader=SafeLoader)


class AllInSqlGlobalCtxMgr:
    _ctx = AllInSqlGlobalContext()

    @classmethod
    def get_context(cls):
        return cls._ctx


def get_context() -> AllInSqlGlobalContext:
    return AllInSqlGlobalCtxMgr.get_context()
