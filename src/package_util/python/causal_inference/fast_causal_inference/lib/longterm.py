from ..all_in_sql import *
from .ols import *
from .. import clickhouse_create_view

# from fast_causal_inference.lib.all_in_sql_conn import *
# from fast_causal_inference.lib.ols import *
# from fast_causal_inference.all_in_sql import *

import concurrent.futures
import time
import threading


class LongTerm:
    def __init__(self, sql, table, sample_num, bs_num):
        if bs_num > 100:
            self.result = "bs_num must less than 100!"
            return
        self.table = table
        self.sql_instance = AllInSqlConn(use_sql_forward=False)
        self.res = []
        threads = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=bs_num) as executor:
            tasks = []
            for i in range(bs_num):
                tasks.append(executor.submit(self.call_func, sql, sample_num))
            concurrent.futures.wait(tasks)

    def call_func(self, sql, sample_num):
        bs_param = self.sql_instance.sql("select DistributedNodeRowNumber() from " + self.table)
        bs_param_str = ''
        for x in bs_param:
            for y in x:
                bs_param_str += y
        sql = sql.strip()
        sql = sql[:-1] + ',bs_param=' + "'{PH}'"
        sql += ",sample_num=" + str(sample_num)
        sql += ",bs_num=1) from " + self.table

        self.forward_sql = sql_forward(sql)
        self.forward_sql = self.forward_sql.replace('{PH}', bs_param_str).replace('Lasso', '"Lasso"').replace(
            'OlsState', 'Ols')
        self.forward_sql += " settings max_threads = 1"
        self.result = []
        res = self.sql_instance.sql(self.forward_sql, False)
        self.result.append(self.format_output(res))

    def format_output(self, res):
        if isinstance(res, str) == True and res.find("Code") != -1:
            return [res]
        result = []
        i = 1
        res = res.split('\n')
        while i < len(res):
            raw = res[i]
            ttest_each_bs = []
            for k in raw.split(' '):
                try:
                    num = float(k)
                    ttest_each_bs.append(num)
                except ValueError:
                    continue
                if len(ttest_each_bs) >= 6:
                    break
            if len(ttest_each_bs) > 0:
                result.append(tuple(ttest_each_bs))
            i += 2
        return result

    def summary(self):
        return str(self.result)

    def __str__(self):
        return str(self.result)

    def get_result(self):
        return self.result
