import time
from functools import partial
from typing import List, Dict

import numpy as np
import pandas as pd
import scipy
from sklearn.metrics import roc_auc_score

import fast_causal_inference
from fast_causal_inference.dataframe.functions import (
    register_fn,
    define_args,
    FnArg,
    DfFunction,
    aggregrate,
    OlapEngineType,
    DfContext,
)
from fast_causal_inference.dataframe.df_base import df_2_table, table_2_df
from functools import partial

import numpy as np
import scipy
import fast_causal_inference
from fast_causal_inference.lib.causaltree import check_table, check_columns
from fast_causal_inference.util import SqlGateWayConn, ClickHouseUtils


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
    return auc


class Logistic:
    """
    This class implements a Logistic Regression model.

    Parameters
    ----------

    tol : float
        The tolerance for stopping criteria.
    iter : int
        The maximum number of iterations.

    Example
    -------

    .. code-block:: python

        import fast_causal_inference
        from fast_causal_inference.dataframe.regression import Logistic
        table = 'test_data_small'
        df = fast_causal_inference.readClickHouse(table)
        X = ['x1', 'x2', 'x3', 'x4', 'x5', 'x_long_tail1', 'x_long_tail2']
        Y = 't_ob'

        logit = Logistic(tol=1e-6, iter=500)
        logit.fit(Y, X, df)
        logit.summary()
        # Output:
        #                  x      beta
        #     0     intercept  0.083472
        #     1            x1  0.957999
        #     2            x2  0.217600
        #     3            x3  0.534323
        #     4            x4 -0.006258
        #     5            x5 -0.020528
        #     6  x_long_tail1 -0.036267
        #     7  x_long_tail2  0.000232

        # predict
        df_predict = logit.predict(df)
        df_predict.select('prob').show()
        #                             prob
        # 0      0.549151214991665
        # 1     0.8876947633647565
        # 2    0.10790926234343089
        # 3      0.791206731095578
        # 4     0.7341882818925854
        # ..                   ...
        # 195  0.21966953201618872
        # 196   0.5813872122369445
        # 197   0.5766490178541132
        # 198   0.5210472623083635
        # 199  0.35841097345616885

        logit.get_auc(df=df_predict,Y=Y,prob_col="prob")
        # 0.7587271750805586
    """

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
            pass

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
        # 初始化 beta，第一项为截距，其余项为0
        # intercept = scipy.special.logit(avg(self.table, self.Y))
        # self.beta = np.concatenate(([intercept], np.zeros(len(self.X))))
        self.beta = np.full(len(self.X)+1,0.001)

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
#         print('XX:',XX)
#         print('XY:',XY)
        
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

    def fit(self, Y, X, df):
        self.table = df.getTableName()
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
        return self.result

    def predict(self, df, prob_col="prob"):
        from fast_causal_inference.dataframe.dataframe import readClickHouse

        table_input = df.getTableName()
        self.beta = [round(i, 6) for i in self.beta]
        
        table_output = f"logsitic_tmp_{int(time.time())}"
        fast_causal_inference.clickhouse_drop_view(table_output)
        fast_causal_inference.clickhouse_drop_view(table_output + "_local")
        
        fast_causal_inference.clickhouse_create_view(
            clickhouse_view_name=table_output,
            sql_statement=f"""
                 1 as c, exp({dot(['c']+self.X,self.beta)}) / (1 + exp({dot(['c']+self.X,self.beta)})) AS {prob_col}, *
          """,
            sql_table_name=table_input,
            is_force_materialize=True
        )
        
        return readClickHouse(table_output)

    def get_auc(self, df, Y, prob_col="prob"):
        return auc(Y, df.getTableName(), prob_col)



class MachineLearning:
    def __init__(self, model_name):
        self.model_name = model_name

    def fit_impl(self, func, expr, df):
        new_df = df.materializedView()
        self.model = new_df.agg(func(expr=expr).alias("machine_learning_model"))
        if df.engine == OlapEngineType.CLICKHOUSE:
            self.model_sql = (
                "with ("
                + self.model.getExecutedSql().replace(
                    self.model_name, self.model_name + "State"
                )
                + ") as machine_learning_model"
            )
        elif df.engine == OlapEngineType.STARROCKS:
            self.model_sql = (
                "with machine_learning_model_tbl as ("
                + self.model.getExecutedSql().replace(
                    self.model_name.lower(), self.model_name.lower() + "_train"
                ).replace(",'" + expr.replace(' ', '').replace("~", ",").replace("+", ",") + "'", '')
                + ")"
            )
        else:
            raise Exception(f"Unsupported olap `{df.engine}`.")

    def effect_impl(self, expr, df, effect_name):
        expr = expr.replace("+", ",")
        if self.model_sql is None:
            raise Exception("model is not fitted yet")
        new_df = df
        if df.engine == OlapEngineType.CLICKHOUSE:
            new_df = new_df.withColumn(
                effect_name, "evalMLMethod({},{})".format("machine_learning_model", expr)
            )
        elif df.engine == OlapEngineType.STARROCKS:
            new_df = new_df.withColumn(
                effect_name, "eval_ml_method({},[{}])".format("machine_learning_model_tbl.machine_learning_model", expr)
            )
            new_df.data.source.starrocks.table_name += ", machine_learning_model_tbl"
        else:
            raise Exception(f"Unsupported olap `{df.engine}`.")
        new_df._set_cte(self.model_sql)
        new_df = new_df.materializedView(is_physical_table=True)
        return new_df

    def summary(self):
        self.model.show()


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="ols")
@register_fn(engine=OlapEngineType.STARROCKS, name="ols")
@define_args(
    FnArg(name="expr", is_param=True),
    FnArg(name="use_bias", default="True", is_param=True),
)
@aggregrate
class AggOLSDfFunction(DfFunction):
    def sql_impl_default(
        self,
        ctx: DfContext,
        fn_args: List[FnArg],
        fn_params: List[FnArg],
        arg_dict: Dict,
    ) -> str:
        expr_arg: FnArg = arg_dict["expr"]
        use_bias_arg: FnArg = arg_dict["use_bias"]
        expr = expr_arg.sql(ctx)
        use_bias = use_bias_arg.sql(ctx)
        sql = self.fn_name(ctx) + f"({expr}, {use_bias})"
        return sql


"""
Example:
import fast_causal_inference.dataframe.regression as Regression
df.agg(Regression.ols('y~x2+x1+x3')).show()
df.ols('y~x1+x2+x3', True).show()
"""


def ols(expr=None, use_bias=True):
    return DfFnColWrapper(AggOLSDfFunction(), {"expr": expr, "use_bias": use_bias}, [])


class Ols(MachineLearning):
    """
    This function is for an Ordinary Least Squares (OLS) model calculated using Matrix Multiplication. The fit method is used to train the model using a specified regression formula and dataset. The effect method is used to make predictions based on the trained model, the regression formula, and a new dataset. The predicted results are stored in a column with a specified name in the DataFrame.

    Parameters:
    use_bias: bool, default=True, whether to use an intercept

    Methods:
        fit(expr, df): Train the model
            expr : str, regression formula
            df : DataFrame, dataset
        effect(expr, df, effect_name): Predict
            expr : str, regression formula
            df : DataFrame, dataset
            effect_name : str, column name for the prediction result, default is 'effect'
        summary(): Display the summary of the model

    Example
    -------

    .. code-block:: python

        import fast_causal_inference
        import fast_causal_inference.dataframe.regression as Regression
        table = 'test_data_small'
        df = fast_causal_inference.readClickHouse(table)
        model = Regression.Ols(True)
        model.fit('y~x1+x2+t_ob', df)
        model.summary()
        effect_df = model.effect('x1+x2+x3', df)
        effect_df.show()
        # Call:
        #   lm( formula = y ~ x1 + x2 + t_ob )

        # Coefficients:
        # .               Estimate    Std. Error  t value     Pr(>|t|)    
        # (Intercept)     11.511121   0.198929    57.865567   0.000000    
        # x1              1.719414    0.129866    13.239884   0.000000    
        # x2              2.052233    0.060800    33.753934   0.000000    
        # t_ob            1.394370    0.265055    5.260684    0.000000    

        # Residual standard error: 11.890735 on 9996 degrees of freedom
        # Multiple R-squared: 0.132555, Adjusted R-squared: 0.132295
        # F-statistic: 509.167914 on 3 and 9996 DF,  p-value: 0.000000



        #                                        id            x1            x2  \
        # 0    50dbff92-0e99-49c3-812f-2a65f17f520f   -.803850404   1.561695102   
        # 1    a7049ddb-9022-4ecf-8223-942c3f21fd0b     .82455722   -.095066668   
        # 2    6d504691-b71d-4264-83de-39bc662dff4e    .975138411    .237623743   
        # 3    c098d0ae-c9ae-4622-83f8-3a011f0e416a     .42667456     .57384993   
        # 4    cecd70bb-33e0-4401-849a-56b74469a54b   1.721194032  -4.227066101   
        # ..                                    ...           ...           ...   
        # 195  fb1f392d-7280-4d7d-a717-6803c8c6dfb6   -.015682112   1.145413777   
        # 196  32b2d482-afd9-4e9a-a723-785632dca284  -1.076187849   3.156201929   
        # 197  a0950a09-b060-4974-a7a1-ce6c10a955fc   -.061784784   1.653462859   
        # 198  737d315b-d86c-4040-a7b2-79afd03760f9     -.3858315   -.428380518   
        # 199  a47e7c05-4d50-4821-a801-038fc4d99ed8   -.173463897   1.169004732   

        #               x3           x4           x5 x_long_tail1 x_long_tail2 x_cat1  \
        # 0     .335643191   .252417981   .461031926   .573269927   .004126571      A   
        # 1     .425597547  4.965615365  2.473710557   .008362166   .188719888      A   
        # 2    2.138533269  2.941316363   1.76753946   .195095212   .038055998      A   
        # 3    9.417911389  1.698327168  1.171077429   .589245253   .173969038      D   
        # 4     .070605806  5.275816894   2.01594999   .010935887   .057313867      D   
        # ..           ...          ...          ...          ...          ...    ...   
        # 195  2.350821578   .124625107   .878188616   .362945955   .662389064      D   
        # 196   .121511627   .214719672  4.836191783   2.95175992  2.598434239      E   
        # 197  1.892911074  3.022031194  3.242927457   .048108021   .755433176      A   
        # 198   .925096628  3.643185963  4.691846645   .056882169   .564901713      E   
        # 199   .171662364  1.970967488  2.382641976   .187394679   .378980135      D   

        #     treatment t_ob             y          y_ob numerator_pre     numerator  \
        # 0           0    0   6.045351799     .69386093   3.921805702   6.045351799   
        # 1           1    0   11.31694439   3.094807368   2.369767229   11.31694439   
        # 2           1    1   22.78502239   1.954622918   8.473373029   22.78502239   
        # 3           0    1   32.00966866   2.545508352   31.47257041   32.00966866   
        # 4           0    0  -7.450356197   2.321110403  -8.701238237  -7.450356197   
        # ..        ...  ...           ...           ...           ...           ...   
        # 195         0    1   8.399368024   -.200178401   10.82530486   8.399368024   
        # 196         0    0   10.35573834   -.741978015   10.17899899   10.35573834   
        # 197         1    0   25.26195413   -.457014837   9.811443968   25.26195413   
        # 198         1    1   10.28529304    .442979076   2.629481075   10.28529304   
        # 199         1    0   12.46995844  -1.268767159   3.077989132   12.46995844   

        #     denominator_pre  denominator      weight        day_        effect  
        # 0       2.295974349   .522501408  .528460071  2023-11-22  13.801943468  
        # 1       4.795001425  9.205336272  .973275925  2023-11-22  13.327217563  
        # 2       5.201623767   8.81335379  .495188548  2023-11-22  16.657354015  
        # 3       4.246789161  5.704188561  .301483964  2023-11-22  26.554481368  
        # 4       6.311670804  5.780272219  .960892091  2023-11-22   5.894090149  
        # ..              ...          ...         ...         ...           ...  
        # 195     2.007430247  2.490694717  .777676162  2023-11-22  17.112729269  
        # 196      13.6036156  12.03186531  .622044071  2023-11-22  16.307404123  
        # 197     7.813151878  7.422506574  .405256309  2023-11-22  17.437598142  
        # 198     10.35406506  11.72363904  .249501394  2023-11-22  11.258507714  
        # 199      5.56374576  7.894552568  .405447064  2023-11-22  13.851296301  

        # [200 rows x 20 columns]
    """


    def __init__(self, use_bias=True):
        super().__init__("Ols")
        self.use_bias = use_bias

    def fit(self, expr, df):
        func = partial(ols, use_bias=self.use_bias)
        super().fit_impl(func, expr, df)

    def effect(self, expr, df, effect_name="effect"):
        return super().effect_impl(expr, df, effect_name)


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="wls")
@register_fn(engine=OlapEngineType.STARROCKS, name="wls")
@define_args(
    FnArg(name="expr", is_param=True),
    FnArg(name="use_bias", default="True", is_param=True),
    FnArg(name="weight"),
)
@aggregrate
class AggWLSDfFunction(DfFunction):
    def sql_impl_default(
        self,
        ctx: DfContext,
        fn_args: List[FnArg],
        fn_params: List[FnArg],
        arg_dict: Dict,
    ) -> str:
        expr_arg: FnArg = arg_dict["expr"]
        use_bias_arg: FnArg = arg_dict["use_bias"]
        weight_arg: FnArg = arg_dict["weight"]
        expr = expr_arg.sql(ctx)
        use_bias = use_bias_arg.sql(ctx)
        weight = weight_arg.sql(ctx)
        sql = self.fn_name(ctx) + f"({expr}, {weight}, {use_bias})"
        return sql


"""
Parameters:
    expr : str, 回归公式
    weight : str, 权重列名
    use_bias : bool, default=True, 是否使用截距

Example
-------

.. code-block:: python

    import fast_causal_inference.dataframe.regression as Regression
    df.agg(Regression.wls('y~x2+x1+x3', use_bias=True)).debug()
    df.wls('y~x1+x2+x3', weight='0.5').show()
"""


def wls(expr=None, weight="1", use_bias=True):
    return DfFnColWrapper(
        AggWLSDfFunction(), {"expr": expr, "use_bias": use_bias}, [weight]
    )


class Wls(MachineLearning):
    """
    This function is for a Weighted Least Squares (WLS) model. The fit method is used to train the model using a specified regression formula and dataset. The effect method is used to make predictions based on the trained model, the regression formula, and a new dataset. The predicted results are stored in a column with a specified name in the DataFrame. The weight parameter specifies the column name for weights in the DataFrame.

    Parameters:
        weight : str, column name for weights
        use_bias : bool, default=True, whether to use an intercept

    Methods:
        fit(expr, df): Train the model
            expr : str, regression formula
            df : DataFrame, dataset
        effect(expr, df, effect_name): Predict
            expr : str, regression formula
            df : DataFrame, dataset
            effect_name : str, column name for the prediction result, default is 'effect'
        summary(): Display the summary of the model

    Example
    -------

    .. code-block:: python

        import fast_causal_inference.dataframe.regression as Regression
        model = Regression.Wls(weight='1', use_bias=True)
        model.fit('y~x1+x2+x3', df)
        model.summary()
        effect_df = model.effect('x1+x2+x3', df)
        effect_df.show()

    """

    def __init__(self, weight="1", use_bias=True):
        super().__init__("Wls")
        self.weight = weight
        self.use_bias = use_bias

    def fit(self, expr, df):
        func = partial(wls, weight=self.weight, use_bias=self.use_bias)
        super().fit_impl(func, expr, df)

    def effect(self, expr, df, effect_name="effect"):
        return super().effect_impl(expr, df, effect_name)


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="stochasticLogisticRegression")
@define_args(
    FnArg(name="learning_rate", is_param=True, default=0.00001),
    FnArg(name="l1", is_param=True, default=0.1),
    FnArg(name="batch_size", is_param=True, default=15),
    FnArg(name="method", is_param=True, default="SGD"),
    FnArg(name="expr"),
)
@aggregrate
class AggStochasticLogisticRegressionDfFunction(DfFunction):
    def sql_impl_clickhouse(
        self,
        ctx: DfContext,
        fn_args: List[FnArg],
        fn_params: List[FnArg],
        arg_dict: Dict,
    ) -> str:
        learning_rate = arg_dict["learning_rate"].sql(ctx)
        l1 = arg_dict["l1"].sql(ctx)
        batch_size = arg_dict["batch_size"].sql(ctx)
        method = f"'{arg_dict['method'].sql(ctx)}'"
        expr = arg_dict["expr"].sql(ctx)
        sql = (
            self.fn_name(ctx)
            + f"({learning_rate}, {l1}, {batch_size}, {method})({expr.replace('+', ',').replace('~', ',')})"
        )
        return sql


"""
Parameters:
    expr : str, 回归公式
    learning_rate : float, default=0.00001, 学习率
    l1 : float, default=0.1, L1正则化系数
    batch_size : int, default=15, 批量大小
    method : str, default='SGD', 优化方法

Example
-------

.. code-block:: python

    import fast_causal_inference.dataframe.regression as Regression
    df.stochastic_logistic_regression('y~x1+x2+x3', learning_rate=0.00001, l1=0.1, batch_size=15, method='Lasso').show()
    df.agg(Regression.stochastic_logistic_regression('y~x1+x2+x3', learning_rate=0.00001, l1=0.1, batch_size=15, method='SGD')).show()

"""


def stochastic_logistic_regression(
    expr, learning_rate=0.00001, l1=0.1, batch_size=15, method="SGD"
):
    return DfFnColWrapper(
        AggStochasticLogisticRegressionDfFunction(),
        {
            "learning_rate": learning_rate,
            "l1": l1,
            "batch_size": batch_size,
            "method": method,
        },
        [expr],
    )


class StochasticLogisticRegression(MachineLearning):
    """
    This function is for a Stochastic Logistic Regression model. The fit method is used to train the model using a specified regression formula and a dataset. The effect method is used to make predictions based on the trained model, the regression formula, and a new dataset. The predicted results are stored in a column with a specified name in the DataFrame. The learning_rate, l1, batch_size, and method parameters are used to control the learning rate, L1 regularization coefficient, batch size, and optimization method respectively.

    Parameters:
        learning_rate : float, default=0.00001, learning rate
        l1 : float, default=0.1, L1 regularization coefficient
        batch_size : int, default=15, batch size
        method : str, default='SGD', optimization method

    Methods:
        fit(expr, df): Train the model
            expr : str, regression formula
            df : DataFrame, dataset
        effect(expr, df, effect_name): Predict
            expr : str, regression formula
            df : DataFrame, dataset
            effect_name : str, column name for the prediction result, default is 'effect'

    Example
    -------

    .. code-block:: python

        import fast_causal_inference.dataframe.regression as Regression
        model = Regression.StochasticLogisticRegression(learning_rate=0.00001, l1=0.1, batch_size=15, method='SGD')
        model.fit('y~x1+x2+x3', df)
        effect_df = model.effect('x1+x2+x3', df)
        effect_df.show()

    """

    def __init__(self, learning_rate=0.00001, l1=0.1, batch_size=15, method="SGD"):
        super().__init__("stochasticLogisticRegression")
        self.learning_rate = learning_rate
        self.l1 = l1
        self.batch_size = batch_size
        self.method = method

    def fit(self, expr, df):
        func = partial(
            stochastic_logistic_regression,
            learning_rate=self.learning_rate,
            l1=self.l1,
            batch_size=self.batch_size,
            method=self.method,
        )
        super().fit_impl(func, expr, df)

    def effect(self, expr, df, effect_name="effect"):
        return super().effect_impl(expr, df, effect_name)


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="stochasticLinearRegression")
@define_args(
    FnArg(name="learning_rate", is_param=True, default=0.00001),
    FnArg(name="l1", is_param=True, default=0.1),
    FnArg(name="batch_size", is_param=True, default=15),
    FnArg(name="method", is_param=True, default="SGD"),
    FnArg(name="expr"),
)
@aggregrate
class AggStochasticLinearRegressionDfFunction(DfFunction):
    def sql_impl_clickhouse(
        self,
        ctx: DfContext,
        fn_args: List[FnArg],
        fn_params: List[FnArg],
        arg_dict: Dict,
    ) -> str:
        learning_rate = arg_dict["learning_rate"].sql(ctx)
        l1 = arg_dict["l1"].sql(ctx)
        batch_size = arg_dict["batch_size"].sql(ctx)
        method = f"'{arg_dict['method'].sql(ctx)}'"
        expr = arg_dict["expr"].sql(ctx)
        sql = (
            self.fn_name(ctx)
            + f"({learning_rate}, {l1}, {batch_size}, {method})({expr.replace('+', ',').replace('~', ',')})"
        )
        return sql


def stochastic_linear_regression(
    expr, learning_rate=0.00001, l1=0.1, batch_size=15, method="SGD"
):
    return DfFnColWrapper(
        AggStochasticLinearRegressionDfFunction(),
        {
            "learning_rate": learning_rate,
            "l1": l1,
            "batch_size": batch_size,
            "method": method,
        },
        [expr],
    )


class StochasticLinearRegression(MachineLearning):
    """
    This function is for a Stochastic Linear Regression model. The fit method is used to train the model using a specified regression formula and a dataset. The effect method is used to make predictions based on the trained model, the regression formula, and a new dataset. The predicted results are stored in a column with a specified name in the DataFrame. The learning_rate, l1, batch_size, and method parameters are used to control the learning rate, L1 regularization coefficient, batch size, and optimization method respectively.

    Parameters:
        learning_rate : float, default=0.00001, learning rate
        l1 : float, default=0.1, L1 regularization coefficient
        batch_size : int, default=15, batch size
        method : str, default='SGD', optimization method

    Methods:
        fit(expr, df): Train the model
            expr : str, regression formula
            df : DataFrame, dataset
        effect(expr, df, effect_name): Predict
            expr : str, regression formula
            df : DataFrame, dataset
            effect_name : str, column name for the prediction result, default is 'effect'

    Example
    -------

    .. code-block:: python

        import fast_causal_inference.dataframe.regression as Regression
        model = Regression.StochasticLinearRegression(learning_rate=0.00001, l1=0.1, batch_size=15, method='SGD')
        model.fit('y~x1+x2+x3', df)
        effect_df = model.effect('x1+x2+x3', df)
        effect_df.show()

    """

    def __init__(self, learning_rate=0.00001, l1=0.1, batch_size=15, method="SGD"):
        super().__init__("stochasticLinearRegression")
        self.learning_rate = learning_rate
        self.l1 = l1
        self.batch_size = batch_size
        self.method = method

    def fit(self, expr, df):
        func = partial(
            stochastic_linear_regression,
            learning_rate=self.learning_rate,
            l1=self.l1,
            batch_size=self.batch_size,
            method=self.method,
        )
        super().fit_impl(func, expr, df)

    def effect(self, expr, df, effect_name="effect"):
        return super().effect_impl(expr, df, effect_name)


from typing import List, Dict
from fast_causal_inference.dataframe.functions import (
    DfFnColWrapper,
    register_fn,
    define_args,
    FnArg,
    DfFunction,
    aggregrate,
    OlapEngineType,
    DfContext,
)


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="did")
@register_fn(engine=OlapEngineType.STARROCKS, name="did")
@define_args(
    FnArg(name="Y"),
    FnArg(name="treatment"),
    FnArg(name="time"),
    FnArg(name="X", default="", is_variadic=True),
)
@aggregrate
class AggDIDDfFunction(DfFunction):
    def sql_impl_default(
        self,
        ctx: DfContext,
        fn_args: List[FnArg],
        fn_params: List[FnArg],
        arg_dict: Dict,
    ) -> str:
        Y = arg_dict["Y"].sql(ctx)
        treatment = arg_dict["treatment"].sql(ctx)
        time = arg_dict["time"].sql(ctx)
        X = arg_dict["X"]
        X_sql = ""
        for x in X.column:
            if x.sql(ctx) != "":
                X_sql += ", " + x.sql(ctx)
        sql = self.fn_name(ctx) + f"({Y}, {treatment}, {time}{X_sql})"
        return sql


def did(Y, treatment, time, *X):
    return DfFnColWrapper(AggDIDDfFunction(), {}, [Y, treatment, time, *X])


class DID:
    """
    Parameters
    ----------

        :Y: Column name, refers to the outcome of interest, a numerical variable.
        :treatment: Column name, a Boolean variable, can only take values 0 or 1, where 1 represents the experimental group.
        :time: Column name, a Boolean variable, represents the time factor. time = 0 represents before the strategy takes effect, time = 1 represents after the strategy takes effect.
        :(Optional parameter) X: Some covariates before the experiment, which can be used to reduce variance. Written in the form of ['x1', 'x2', 'x3'] they must be numerical variables.

    Example
    ----------

    .. code-block:: python

        import fast_causal_inference.dataframe.regression as Regression
        model = Regression.DID()
        model.fit(df=df,Y='y',treatment='treatment',time='t_ob',X=['x1','x2'])
        model.summary()
        # Call:
        # lm( formula = y ~ treatment + t_ob + treatment*t_ob + x1 + x2 )

        # Coefficients:
        # .               Estimate    Std. Error  t value     Pr(>|t|)
        # (Intercept)     4.461905    0.213302    20.918288   0.000000
        # treatment       13.902920   0.291365    47.716586   0.000000
        # t_ob            0.416831    0.280176    1.487748    0.136849
        # treatment*t_ob  1.812698    0.376476    4.814905    0.000001
        # x1              1.769065    0.100727    17.562939   0.000000
        # x2              2.020569    0.047162    42.842817   0.000000

        # Residual standard error: 9.222100 on 9994 degrees of freedom
        # Multiple R-squared: 0.478329, Adjusted R-squared: 0.478068
        # F-statistic: 1832.730042 on 5 and 9994 DF,  p-value: 0.000000

        # other ways
        import fast_causal_inference.dataframe.regression as Regression
        df.did('y', 'treatment', 't_ob',['x1','x2','x3']).show()
        df.agg(Regression.did('y', 'treatment', 't_ob',['x1','x2','x3'])).show()

    """

    def __init__(self):
        pass

    def fit(self, df, Y, treatment, time, X=[]):
        self.result = df.did(Y, treatment, time, *X)

    def summary(self):
        self.result.show()


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="ivregression")
@register_fn(engine=OlapEngineType.STARROCKS, name="ivregression")
@define_args(FnArg(name="formula", is_param=True))
@aggregrate
class AggIvregressionDfFunction(DfFunction):
    def sql_impl_default(
        self,
        ctx: DfContext,
        fn_args: List[FnArg],
        fn_params: List[FnArg],
        arg_dict: Dict,
    ) -> str:
        formula = arg_dict["formula"].sql(ctx)
        sql = self.fn_name(ctx) + f"({formula})"
        return sql


"""
parameters：
formula：回归的表达式，和R的语法类似。可以有多个内生变量的方程，但是要求IV个数需要大于内生变量个数，否则会有共线性问题。
"""


def iv_regression(formula):
    return DfFnColWrapper(AggIvregressionDfFunction(), {"formula": formula}, [])


class IV:
    """
    Instrumental Variable (IV) estimator class.
    Instrumental variables (IV) is a method used in statistics, econometrics, epidemiology, and related disciplines to estimate causal relationships when controlled experiments are not feasible or when a treatment is not successfully delivered to every unit in a randomized experiment.
    The idea behind IV is to use a variable, known as an instrument, that is correlated with the endogenous explanatory variables (the variables that are correlated with the error term), but uncorrelated with the error term itself. This allows us to isolate the variation in the explanatory variable that is purely due to the instrument and thus uncorrelated with the error term, which can then be used to estimate the causal effect of the explanatory variable on the dependent variable.

    Here is an example:

    .. math::

       t_{ob} = treatment + X_1 + X_2

    .. math::

       Y = \hat{t}_{ob} + X_1 + X_2

    - :math:`X_1` and :math:`X_2` are independent variables or predictors.
    - :math:`t_{ob}` is the dependent variable that you are trying to explain or predict.
    - :math:`treatment` is an independent variable representing some intervention or condition that you believe affects :math:`t_{ob}`.
    - :math:`Y` is the dependent variable that you are trying to explain or predict.
    - :math:`\hat{t}_{ob}` is the predicted value of :math:`t_{ob}` from the first equation.

    We first regress :math:`X_3` on the treatment and the other exogenous variables :math:`X_1` and :math:`X_2` to get the predicted values :math:`\hat{t}_{ob}`. Then, we replace :math:`t_{ob}` with :math:`\hat{t}_{ob}` in the second equation and estimate the parameters. This gives us the causal effect of :math:`t_{ob}` on :math:`Y`, purged of the endogeneity problem.

    :Methods:
    - fit: Fits the model with the given formula.
    - summary: Displays the summary of the model fit.

    Example
    ----------
    
    .. code-block:: python

        import fast_causal_inference.dataframe.regression as Regression
        model = Regression.IV()
        model.fit(df,formula='y~(t_ob~treatment)+x1+x2')
        model.summary()

        df.iv_regression('y~(t_ob~treatment)+x1+x2').show()
        df.agg(Regression.iv_regression('y~(t_ob~treatment)+x1+x2')).show()
    """

    def __init__(self):
        """
        Initialize the IV estimator class.
        """
        pass

    def fit(self, df, formula):
        """
        Fits the model with the given formula.

        :param formula: str, the formula to fit the model.
        :type formula: str, required
        """
        self.result = df.iv_regression(formula)

    def summary(self):
        """
        Displays the summary of the model fit.
        """
        self.result.show()
