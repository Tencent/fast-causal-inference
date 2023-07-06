import os

import requests

from .ols import *
from .ttest import *
from .. import RAINBOW_CONF


def get_db_name(sql):
    return "default"
    sql = sql.lower()
    pos = sql.find("from")
    if pos == -1:
        return "default"
    sql = sql[pos + 4:]
    pos = sql.find(".")
    if pos == -1:
        return "default"
    sql = sql[:pos]
    sql = sql.strip()
    return sql


def get_user():
    if str(os.environ.get("JUPYTERHUB_USER")) != "None":
        return str(os.environ.get("JUPYTERHUB_USER"))
    elif str(os.environ.get("USER")) != "None":
        return str(os.environ.get("USER"))
    else:
        return "default_user"


def sql_forward(sql):
    retry_times = 3
    url = RAINBOW_CONF["sqlgateway"]["url"] + RAINBOW_CONF["sqlgateway"]["path"]
    json_body = '{"sql":"' + str(sql) + '", "creator":"' + get_user() + '","database":"' + get_db_name(
        sql) + '", "src":"python"}'
    while retry_times > 0:
        try:
            resp = requests.post(url, data=json_body.encode("utf-8"))
            if resp.json()["status"] == 0:
                return resp.json()["forward_sql"]
            else:
                return resp.json()["msg"]
        except Exception as e:
            retry_times -= 1
            if retry_times == 0:
                return str(e)


class AllInSqlConn:
    CH_PROXY_URL = RAINBOW_CONF["chproxy"]["proxy_url"] + "/?database=" + RAINBOW_CONF["all"][
        "ch_database"] + "&user=" + RAINBOW_CONF["chproxy"]["user"] + "&password=" + RAINBOW_CONF["chproxy"][
                       "password"] + "&cluster_name=" + RAINBOW_CONF["all"]["ch_cluster_name"]

    def __init__(self, use_sql_forward=True):
        self.use_sql_forward = use_sql_forward

    def get_return_type(self, result):
        # if contain Exception
        if str(result).find("Exception") != -1:
            return "not defined"
        if str(result)[0:8] == "estimate" and str(result)[8:].find("estimate") == -1:
            return "ttest"
        elif str(result)[0:5] == "Call:":
            return "ols"
        return "not defined"

    def format_sql_result(self, text):
        result = []
        for line in text.split('\n'):
            if line:
                result.append(line)
        return result

    def exchange_sql(self, sql):
        # if is select sql and not limit, add limit 1000
        if sql[0:6].lower().find("select") != -1 and sql.find("limit") == -1:
            sql += " limit 1000"
        return sql

    # use QueryServer
    def execute(self, sql):
        sql = self.exchange_sql(sql)
        json_body = str(sql)
        try:
            resp = requests.post(AllInSqlConn.CH_PROXY_URL, data=json_body.encode('utf-8'))
        except Exception as e:
            return str(e)
        result = resp.text.replace("\\t", "\t").replace("\\n", "\n").replace("\\r", "\r").replace("\\'", "'")
        return result

    # with sql forward and format all_in_sql result
    def sql(self, sql, use_output_format=True):
        is_recursive_forcasting = False
        if sql.find("recursiveForcasting") != -1:
            is_recursive_forcasting = True

        if (self.use_sql_forward):
            sql = sql_forward(sql)
        if sql == "not perimitted":
            return "No database permissions"

        res = self.execute(sql)
        if (res.find("Exception") != -1):
            return res
        if (len(res) == 0):
            return "success"
        return_type = self.get_return_type(res)
        if return_type == "ttest":
            return Ttest(res)
        elif return_type == "ols":
            return Ols(res)
        else:
            res = self.format_sql_result(res)
            if use_output_format == False:
                return res
            filter = ['estimate', 'stderr', 't-statistic', 'p-value', 'lower', 'upper']
            final_res = []
            tmp = []
            if is_recursive_forcasting == True:
                final_res.append(tuple(['predict_index', 'estimate', 'stderr', 't-statistic', 'p-value', 'lower', 'upper']))
            for i in range(len(res)):
                over = True
                for j in res[i].split('\t'):
                    for k in j.split(' '):
                        x = k.strip()
                        if x in filter:
                            over = False
                        elif len(x) != 0:
                            tmp.append(x)
                if over == False:
                    continue
                final_res.append(tuple([int(x) if x.isdigit() else float(x) if x.replace('.', '', 1).isdigit() or x[
                    0] == '-' and x[1:].replace('.', '', 1).isdigit() else x for x in tmp]))
                tmp = []
            return final_res
