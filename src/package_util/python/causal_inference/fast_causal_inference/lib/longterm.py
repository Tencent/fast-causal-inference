from .all_in_sql_conn import *
from .ols import *
from .. import clickhouse_create_view

#from fast_causal_inference.lib.all_in_sql_conn import *
#from fast_causal_inference.lib.ols import *
#from fast_causal_inference.all_in_sql import *

import concurrent.futures
import time
import threading

class LongTerm:
    def __init__(self, sql, table, sample_num, bs_num, cluster):
        self.sql_instance = AllInSqlConn(use_sql_forward = False)
        self.exe_table = table + "_longterm"
        num = self.sql_instance.sql("select count() from " + self.exe_table)
        if isinstance(num, list) == False:
            res = sql_instance.sql("create table " + self.exe_table + " on cluster " + cluster + " engine = MergeTree() order by order_key as select *, rand() as order_key from " + table)
            num = sql_instance.sql("select count() from " + self.exe_table)
        num = num[0][0]
        num2 = self.sql_instance.sql("select count() from " + table)
        num2 = num2[0][0]
        if num == num2:
            print("data check succ")
        else:
            self.sql_instance.sql("drop table " + self.exe_table + " on cluster " + cluster)
            print("data check error, please retry")
        self.res = []
        threads = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=bs_num) as executor:
            tasks = []
            for i in range(bs_num):
                tasks.append(executor.submit(self.call_func, sql, sample_num))
            concurrent.futures.wait(tasks)

    
    def call_func(self, sql, sample_num):
        bs_param = self.sql_instance.sql("select DistributedNodeRowNumber() from " + self.exe_table)
        bs_param_str = ''
        for x in bs_param:
            for y in x:
                bs_param_str += y

        sql = sql.strip()
        sql = sql[:-1] + ',bs_param=' + "'{PH}'"
        sql += ",sample_num=" + str(sample_num)
        sql += ",bs_num=1) from " + self.exe_table
        
        self.forward_sql = sql_forward(sql)
        self.forward_sql = self.forward_sql.replace('{PH}', bs_param_str).replace('Lasso', '"Lasso"')
        self.forward_sql += " settings max_threads = 1"

        for i in range(100):
            res = self.sql_instance.sql(self.forward_sql, use_output_format = False)
            if isinstance(res, str) == True and res.find("not input") != -1:
                continue
            else:
                break
        
        self.res.append(self.format_output(res))
    
    def format_output(self, res):
        result = []
        i = 1
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
            result.append(tuple(ttest_each_bs))
            i += 2
        return result

    def summary(self):
        return str(self.res)

    def __str__(self):
        return str(self.res)

    def get_result(self):
        return self.res

