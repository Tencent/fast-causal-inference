__all__ = [
    "SqlGateWayConn",
    "create",
    "create_sql_instance",
]

from fast_causal_inference.common import get_context
from fast_causal_inference.util.rtx import get_user
from fast_causal_inference.util.formatter import output_dataframe
import time
import requests


class SqlGateWayConn:
    def __init__(self, device_id=None, db_name=None, olap="clickhouse"):
        datasource = dict()
        PROJECT_CONF = get_context().project_conf

        if olap.lower() not in ["clickhouse", "starrocks"]:
            raise Exception(f"Unsupported olap `{olap}`")
        self.engine_type = olap.lower()
        
        d = None
        for cell in PROJECT_CONF["datasource"]:
            datasource[cell["device_id"]] = cell
        for device in datasource:
            if datasource[device].get(olap.lower() + "_database") is not None:
                d = device
                database = datasource[device][olap.lower() + "_database"]
                break
        if device_id is not None:
            d = device_id
        if d is None:
            raise Exception(f"Unable to get any device of engine({olap}).")
        if db_name:
            database = db_name
        self.device_id = d
        self.db_name = database

    @staticmethod
    def create_default_conn(olap="clickhouse"):
        return SqlGateWayConn(olap=olap)

    def execute(self, sql, retry_times=1, is_calcite_parse=False):
        PROJECT_CONF = get_context().project_conf
        logger = get_context().logger

        url = PROJECT_CONF["sqlgateway"]["url"] + PROJECT_CONF["sqlgateway"]["path"]
        sql = sql.replace("\n", " ")
        sql = sql.replace('"', '\\"')
        json_body = (
            '{"rawSql":"'
            + str(sql)
            + '", "creator":"'
            + get_user()
            + '", "deviceId":"'
            + str(self.device_id)
            + '","database":"'
            + self.db_name
            + '", "isDataframeOutput": true'
            + f', "engineType": "{self.engine_type}"'
        )
        if is_calcite_parse:
            json_body += ', "isCalciteParse": true}'
        else:
            json_body += "}"
        while retry_times > 0:
            try:
                logger.debug("url= " + url + ",data= " + json_body)
                resp = requests.post(
                    url,
                    data=json_body.encode("utf-8"),
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                    },
                )
                logger.debug("response=" + resp.text)
                # result = resp.text.replace("\\t", "\t").replace("\\n", "\n").replace("\\r", "\r").replace("\\'", "'")
                if resp.json()["status"] == 0:
                    if is_calcite_parse:
                        return str(resp.json()["data"]["executeSql"])
                    else:
                        return str(resp.json()["data"]["result"])
                else:
                    if "message" in resp.json().keys():
                        return (
                            "error message:"
                            + resp.json()["message"]
                            + ", error code"
                            + str(resp.json()["status"])
                        )
                    elif "error" in resp.json().keys():
                        return (
                            "error message:"
                            + resp.json()["error"]
                            + ", error code"
                            + str(resp.json()["status"])
                        )
                    else:
                        return (
                            "error message none"
                            + ", error code"
                            + resp.json()["status"]
                        )
            except Exception as e:
                time.sleep(1)
                retry_times -= 1
                if retry_times == 0:
                    return str(e)

    def sql(self, sql, is_calcite_parse=False, is_dataframe=True, connect_timeout_retry=5):
        for i in range(connect_timeout_retry):
            res = self.execute(sql, is_calcite_parse=is_calcite_parse)
            if res.find("connect timed out") != -1:
                get_context().logger.info(
                    "connect timed out, retry " + str(i) + " times"
                )
                time.sleep(1)
                continue
            if res.find("Exception") != -1 or "error message" in res:
                get_context().logger.info("execute content result:" + str(res))
            else:
                get_context().logger.debug("execute content result:" + str(res))
            if (
                (res.find("Exception") != -1)
                or "error message" in res
                or is_calcite_parse
                or not is_dataframe
            ):
                return res
            else:
                return output_dataframe(res)


def create(olap="clickhouse"):
    return SqlGateWayConn.create_default_conn(olap=olap)


def create_sql_instance(olap="clickhouse"):
    return create(olap=olap)
