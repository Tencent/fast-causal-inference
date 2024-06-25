from fast_causal_inference.util import SqlGateWayConn, ClickHouseUtils


import concurrent.futures
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class LongTermBase:
    def __init__(self, sql, table, sample_num, bs_num, cluster):
        self.sql_instance = SqlGateWayConn.create_default_conn()
        self.exe_table = table
        self.res = []
        threads = []
        self.call_func(sql, sample_num)
        with concurrent.futures.ThreadPoolExecutor(max_workers=bs_num) as executor:
            tasks = []
            for i in range(bs_num - 1):
                tasks.append(executor.submit(self.call_func, sql, sample_num))
            concurrent.futures.wait(tasks)

    def call_func(self, sql, sample_num):
        bs_param = self.sql_instance.sql(
            "select DistributedNodeRowNumber() from " + self.exe_table
        ).values[0][0]
        bs_param_str = ""
        for x in bs_param:
            for y in x:
                bs_param_str += y

        sql = sql.strip()
        sql = sql[:-1] + "," + "'{PH}'"
        sql += "," + str(sample_num)
        sql += ",1) from " + self.exe_table

        self.forward_sql = self.sql_instance.sql(sql, is_calcite_parse=True)
        #print('forward', self.forward_sql)
        self.forward_sql = self.forward_sql.replace("{PH}", bs_param_str).replace(
            "Lasso", '"Lasso"'
        )
        self.forward_sql += " settings max_threads = 1"
        self.forward_sql = (
            self.forward_sql.replace("sample_num =", "")
            .replace("bs_num =", "")
            .replace("bs_param = ", "")
        )

        for i in range(100):
            clickhosue_utils = ClickHouseUtils()
            res = clickhosue_utils.execute(self.forward_sql)
            #print(self.forward_sql)
            #print('res',res)
            if isinstance(res, str) == True and res.find("not input") != -1:
                continue
            else:
                break
        self.res.append(self.format_output(res))

    def format_output(self, res):
        if isinstance(res, str) == True and res.find("Code") != -1:
            return [res]
        result = []

        res = res[0]
        res = str(res).split("\\n")

        i = 2
        while i < len(res):
            raw = res[i]
            ttest_each_bs = []
            for k in raw.split(" "):
                try:
                    num = float(k)
                    ttest_each_bs.append(num)
                except ValueError:
                    continue
                if len(ttest_each_bs) >= 8:
                    break
            result.append(tuple(ttest_each_bs))
            i += 3
        return result

    def summary(self):
        return str(self.res)

    def __str__(self):
        print(
            "mean0       mean1       estimate    stderr      t-statistic p-value     lower       upper"
        )
        return str(self.res)

    def get_result(self):
        return self.res


def ttest(result):
    result = np.array(result)
    mean = result[1, 1] - result[0, 1]
    std = np.sqrt(
        result[1, 2] / result[1, 3] + result[0, 2] / result[0, 3]
    ) 
    return [mean, std, mean - 1.96 * std, mean + 1.96 * std]


def longterm_plot(df):
    sns.set()
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
    count = 0
    # l = len(days)

    effect = df[["t", "mean_pred", "std_pred", "lower_pred", "upper_pred"]]
    effect.columns = ["t", "mean", "stderr", "lower", "upper"]
    effect = effect.dropna()
    ground_truth_ = df[["t", "estimate", "stderr", "lower", "upper"]]
    ground_truth_.columns = ["t", "mean", "stderr", "lower", "upper"]

    axs.plot(
        ground_truth_["t"],
        ground_truth_["mean"],
        "--",
        color="#636e72",
        label="true effect",
    )
    axs.fill_between(
        ground_truth_["t"],
        ground_truth_["lower"],
        ground_truth_["upper"],
        alpha=0.2,
        color="#636e72",
    )
    axs.plot(
        effect["t"],
        effect["mean"],
        "-o",
        markersize=2,
        color="#e74c3c",
        label="estimation",
    )
    axs.fill_between(
        effect["t"], effect["lower"], effect["upper"], alpha=0.2, color="#e74c3c"
    )
    axs.legend(loc="lower right", prop=dict(size=10))

    axs.set_ylabel("Average Treatment Effect")
    axs.set_xlabel("Time")

    fig.tight_layout(rect=[0, 0.03, 1, 0.99])
    plt.show()

    
class LongTerm:
    '''
    This class is used to estimate the long-term treatment effect of a given treatment on a given outcome and surrogates.

    
    Parameters
    ----------
    
        df : DataFrame
            The DataFrame containing the data.
        surrogates : list
            The list of surrogates. Each surrogate is a list of strings, where the first element is the outcome and the rest are the surrogates.
        train_Ts : str
            The training time period. Default is "2~1".
        predict_Ts : str
            The prediction time period. Default is "3-5".
        key_metric : int
            The index of the outcome in the surrogates. Default is 1.
        treatment : str
            The name of the treatment column. Default is "treatment".
        model : str
            The model to use. Default is "Ols(True)".
        bs_num : int
            The number of bootstrap samples. Default is 100.

    Examples
    --------
    
    .. python code::

        import fast_causal_inference
        from fast_causal_inference.dataframe.longterm import LongTerm

        surrogates = [['Y_t1', 'm0_t1', 'm1_t1', 'm2_t1', 'm3_t1'],
        ['Y_t2', 'm0_t2', 'm1_t2', 'm2_t2', 'm3_t2'],
        ['Y_t3', 'm0_t3', 'm1_t3', 'm2_t3', 'm3_t3'],
        ['Y_t4', 'm0_t4', 'm1_t4', 'm2_t4', 'm3_t4'],
        ['Y_t5', 'm0_t5', 'm1_t5', 'm2_t5', 'm3_t5'],
        ['Y_t6', 'm0_t6', 'm1_t6', 'm2_t6', 'm3_t6'],
        ['Y_t7', 'm0_t7', 'm1_t7', 'm2_t7', 'm3_t7'],
        ['Y_t8', 'm0_t8', 'm1_t8', 'm2_t8', 'm3_t8'],
        ['Y_t9', 'm0_t9', 'm1_t9', 'm2_t9', 'm3_t9'],
        ['Y_t10', 'm0_t10', 'm1_t10', 'm2_t10', 'm3_t10']]

        df = fast_causal_inference.readClickHouse('test_data_small_longterm')
        model = LongTerm(df,
                        surrogates,# surrogate 名称
                        train_Ts='2~1', #训练的时间段
                        predict_Ts='3-6', #预测的时间段
                        key_metric=1, # 关心的Y是第几个变量，比如这里关心的是Y，是第一个变量
                        treatment='treatment', #实验组/对照组
                        model='Ols(True)', # 选择的模型
                        bs_num=100)
        model.fit()
        model.plot()
        model.get_results()
        #    t  estimate    stderr     lower     upper  mean_pred  std_pred  lower_pred  \
        # 0  1 -1.540014  0.027064 -1.593059 -1.486968        NaN       NaN         NaN   
        # 1  2 -0.652995  0.014609 -0.681629 -0.624362        NaN       NaN         NaN   
        # 2  3 -0.436955  0.015226 -0.466798 -0.407112  -0.437204  0.021734   -0.482200   
        # 3  4 -0.312327  0.017019 -0.345684 -0.278969  -0.313110  0.027234   -0.361011   
        # 4  5 -0.263152  0.019328 -0.301035 -0.225269  -0.260078  0.028940   -0.314888   
        # 5  6 -0.250987  0.021776 -0.293667 -0.208307  -0.245519  0.035277   -0.311426   

        # upper_pred  
        # 0         NaN  
        # 1         NaN  
        # 2   -0.406035  
        # 3   -0.262586  
        # 4   -0.208558  
        # 5   -0.183373  
    
    '''
    def __init__(self, df, surrogates, train_Ts="2~1", predict_Ts="3-5", key_metric=1, treatment="groupid", model="Ols(False)", bs_num=100):
        self.df = df
        self.surrogates = surrogates
        self.train_Ts = train_Ts
        self.predict_Ts = predict_Ts
        self.key_metric = key_metric
        self.treatment = treatment
        self.model = model
        self.bs_num = bs_num
        self.sql_instance = SqlGateWayConn.create_default_conn()
        self.table = df.getTableName()
        self.results = None

    def fit(self):
        transposed = list(zip(*np.array(self.surrogates).T))
        x_list_string = str(transposed).replace("'", "")
        # print(x_list_string)

        cnt = int(self.sql_instance.sql(f"select count(*) from {self.table}").values[0][0])
        string = f"""select recursiveForcasting({x_list_string},{self.train_Ts},{self.predict_Ts}, S{self.key_metric}, {self.treatment}, \'{self.model}\') """
        print(string)
        x = LongTermBase(
            string, table=self.table, sample_num=cnt//2, bs_num=self.bs_num, cluster="allinsql"
        )

        x_ = np.array(x.get_result()).T[2, :, :]
        result = []
        for m in x_:
            tmp = m
            std = np.std(tmp)
            mean = np.mean(tmp)
            upper = np.quantile(tmp, 0.975)
            lower = np.quantile(tmp, 0.025)
            result.append([mean, std, lower, upper])

        pred_df = pd.DataFrame(
            result, columns=["mean_pred", "std_pred", "lower_pred", "upper_pred"]
        )
        predict_Ts = [int(i) for i in self.predict_Ts.split("-")]
        T = len(self.surrogates)
        pred_df["t"] = list(range(predict_Ts[0], predict_Ts[1] + 1))

        key_metric_list = [f"{s[self.key_metric-1]}" for s in self.surrogates]
        print('key_metric_list:', key_metric_list)
        results = []
        cols = ["t", "estimate", "stderr", "lower", "upper"]
        for i in range(T):
            string = f"""select {self.treatment},avg({key_metric_list[i]}),varSamp({key_metric_list[i]}),count(*)  
            from {self.table} group by {self.treatment} order by {self.treatment}"""
            result = self.sql_instance.sql(string).astype(float).values
            result = [i + 1] + ttest(result)
            results.append(result)
        self.results = pd.DataFrame(results, columns=cols)
        self.results = pd.merge(self.results, pred_df, on=["t"], how="outer")

    def plot(self):
        if self.results is not None:
            longterm_plot(self.results)
        else:
            print("Please call the fit method before plotting.")

    def get_results(self):
        return self.results
    
    
