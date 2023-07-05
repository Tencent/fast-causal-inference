from .all_in_sql_conn import *
from .ols import *
from .. import clickhouse_create_view

#from fast_causal_inference.lib.all_in_sql_conn import *
#from fast_causal_inference.lib.ols import *
#from fast_causal_inference.all_in_sql import *

import time

class LongTerm:
    def __init__(self, sql, table, sample_num, bs_num, cluster='mmdcchsvrnewtest'): # 临时版本，需要cluster参数, 后续UDF更新后会去掉
        sql_instance = AllInSqlConn(use_sql_forward = False)
        self.exe_table = table + "_longterm"
        num = sql_instance.sql("select count() from " + self.exe_table)
        if isinstance(num, list) == False:
            res = sql_instance.sql("create table " + self.exe_table + " on cluster " + cluster + " engine = MergeTree() order by order_key as select *, rand() as order_key from " + table)
            num = sql_instance.sql("select count() from " + self.exe_table)
        num = num[0][0]
        num2 = sql_instance.sql("select count() from " + table)
        num2 = num2[0][0]
        if num == num2:
            print("data check succ")
        else:
            sql_instance.sql("drop table " + self.exe_table + " on cluster " + cluster)
            print("data check error, please retry")

        bs_param = sql_instance.sql("select DistributedNodeRowNumber() from " + self.exe_table)
        bs_param_str = ''
        for x in bs_param:
            for y in x:
                bs_param_str += y

        sql = sql.strip()
        sql = sql[:-1] + ',bs_param=' + "'{PH}'"
        sql += ",sample_num=" + str(sample_num)
        sql += ",bs_num=" + str(bs_num) + ') from ' + self.exe_table
        self.forward_sql = sql_forward(sql)
        self.forward_sql = self.forward_sql.replace('{PH}', bs_param_str).replace('Lasso', '"Lasso"')

        for i in range(1000):
            res = sql_instance.sql(self.forward_sql)
            if isinstance(res, str) == True and res.find("not input") != -1:
                continue
            else:
                break
        self.res = res

    def summary(self):
        return str(self.res)

    def __str__(self):
        return str(self.res)

    def get_result(self):
        return self.res
