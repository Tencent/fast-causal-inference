import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.metrics import roc_auc_score

from fast_causal_inference.util import SqlGateWayConn, ClickHouseUtils
from fast_causal_inference.lib.causaltree import check_table, check_columns


def dot(X, beta):
    sql = "+".join([f"{X[i]}*({beta[i]})" for i in range(len(X))])
    return sql


def auc(Y, table, prob_col="prob"):
    sql_instance = SqlGateWayConn.create_default_conn()
    data = sql_instance.sql(
        f"select {Y},{prob_col} from (select *,rand()/pow(2,32) as rand from  {table}) where rand<0.2 limit 2000"
    )
    data = data.astype(float)
    auc = roc_auc_score(data[Y], data[prob_col])
    print(auc)


class Logistic:
    def __init__(self, tol=1e-12, iter=500):
        self.tol = tol
        self.iter = iter
        self.beta = []
        self.__sql_instance = SqlGateWayConn.create_default_conn()

    def __params_input_check(self):
        sql_instance = self.__sql_instance

        if self.Y == "":
            print("missing Y. You should check out the input.")
            raise ValueError
        if self.X == "":
            print("missing X. You should check out the input.")
            raise ValueError
        if self.table == "":
            print("missing table. You should check out the input.")
            raise ValueError
        else:
            sql_instance.sql(f"select {self.Y},{self.X} from {self.table}")

    def __table_variables_check(self):
        variables = [self.Y] + self.X
        check_table(table=self.table)
        check_columns(table=self.table, cols=variables, cols_nume=variables)

    def intitial(self):
        # 初始化
        # 计算截距项，使用对数几率函数将平均响应转换为对数几率
        def avg(table, x):
            sql_instance = SqlGateWayConn.create_default_conn()
            avg = float(
                sql_instance.sql(f"select avg({x}) as avg from {self.table}")["avg"][0]
            )
            return avg

        intercept = logit(avg(self.table, self.Y))
        # 初始化 beta，第一项为截距，其余项为0
        self.beta = np.concatenate(([intercept], np.zeros(len(self.X))))

    def IRLS_ch(self):
        # print(self.beta)

        sql_instance = self.__sql_instance

        X_ = ["c"] + self.X
        sql = f"""
            SELECT 
                        MatrixMultiplication(false,true)({",".join(X_)},z,s_sqrt) as XX_XY
            FROM (
                    select 
                        1 as c,{",".join(self.X)},{dot(X_,self.beta)} as eta,
                        1 / (1 + exp(-eta)) as mu,
                        mu * (1 - mu) as s,
                        sqrt(abs(s)) as s_sqrt,
                        eta + ({self.Y} - mu) / s as z
                    from {self.table}
            where s != 0
            
            SETTINGS max_parser_depth = 5000
                )"""
        XX_XY = (
            sql_instance.sql(sql)["XX_XY"][0]
            .replace("[", "")
            .replace("]", "")
            .replace(" ", "")
            .split(",")
        )
        XX_XY = np.array(XX_XY).reshape(len(X_) + 1, -1).astype(float)
        k = XX_XY.shape[0]
        XX = XX_XY[: k - 1, : k - 1]
        XY = XX_XY[k - 1, : k - 1]
        self.beta = np.linalg.solve(XX, XY)
        self.XX = XX
        self.XY = XY
        self.XX_XY = XX_XY
        # print('XX_XY','*'*100)
        # print(XX_XY)

        sql = f"""
        SELECT 
            sum(logpmf) as logpmf   

        FROM (
                select 
                    1 as c,{dot(X_,self.beta)} as eta,
                    1 / (1 + exp(eta*(-1))) as p,
                    {self.Y} * log(p) + (1 - {self.Y}) * log(1 - p) AS logpmf
                from {self.table}   
            )
        where p>0 and p <1
            SETTINGS max_parser_depth = 5000
            """
        logpmf = float(sql_instance.sql(sql)["logpmf"][0])
        return logpmf

    def fit(self, Y, X, table):
        self.table = table
        self.X = X
        self.Y = Y

        self.__params_input_check()
        self.__table_variables_check()
        self.intitial()
        ll_old = 0
        currtol = 1  # 当前的容差值
        it = 0  # 迭代次数
        ll = 0  # 对数似然

        # 当容差大于阈值且迭代次数小于最大迭代次数时，继续迭代
        while currtol / (abs(ll_old) + 0.1) > self.tol and it < self.iter:
            it += 1
            ll_old = ll  # 保存上一次的对数似然值

            ll = self.IRLS_ch()

            # 计算新旧对数似然值之间的差，作为新的容差
            currtol = abs(ll - ll_old)
            print("iter", it, "log-likelihood: ", ll)

        # 返回结果，包括 beta、迭代次数、最后的容差、最后的对数似然值和权重
        print("The results have converged and the calculation has been stopped")
        X_ = ["intercept"] + self.X
        self.result = pd.DataFrame(zip(X_, self.beta), columns=["x", "beta"])

    def summary(self):
        print(self.result)

    def predict(self, table_input, table_output, prob_col="prob"):
        self.beta = [round(i, 6) for i in self.beta]
        sql_str = f""" 
                    SELECT            
                       1 as c, exp({dot(['c']+self.X,self.beta)}) / (1 + exp({dot(['c']+self.X,self.beta)})) AS {prob_col}, *
                
                    FROM
                          {table_input}
                  """
        print(sql_str)
        ClickHouseUtils.clickhouse_drop_view(table_output)
        ClickHouseUtils.clickhouse_drop_view(table_output + "_local")
        ClickHouseUtils.clickhouse_create_view_v2(
            table_output, sql_str, is_physical_table=True
        )

    def get_auc(self, table, Y, prob_col="prob"):
        return auc(Y, table, prob_col)
