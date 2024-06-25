from fast_causal_inference.util import SqlGateWayConn, ClickHouseUtils


import concurrent.futures
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class LongTerm:
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
            i += 2
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
    )  # * np.sqrt(3507712.0/95885423.0)
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


def Long_term_result(
    table,
    surrogates,
    T,
    train_Ts="2~1",
    predict_Ts="3-5",
    key_metric=1,
    treatment="groupid",
    model="Ols(False)",
    bs_num=100,
):
    x_list_string = (
        "["
        + ", ".join(
            ["(" + f"{i}, ".join(surrogates) + f"{i})" for i in range(1, T + 1)]
        )
        + "]"
    )
    sql_instance = SqlGateWayConn.create_default_conn()
    cnt = int(sql_instance.sql(f"select count(*) from {table}").values[0][0])
    string = f"""select recursiveForcasting({x_list_string},{train_Ts},{predict_Ts}, S{key_metric}, {treatment}, \'{model}\') """

    x = LongTerm(
        string, table=table, sample_num=cnt, bs_num=bs_num, cluster="mmdcchfishertest"
    )

    x_ = np.array(x.get_result()).T[0, :, :]
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
    predict_Ts = [int(i) for i in predict_Ts.split("-")]
    pred_df["t"] = list(range(predict_Ts[0], predict_Ts[1] + 1))

    results = []
    cols = ["t", "estimate", "stderr", "lower", "upper"]
    key_metric = surrogates[key_metric - 1]
    key_metric_list = [f"{key_metric}{i}" for i in range(1, T + 1)]
    for i in range(T):
        key_metric = key_metric_list[i]
        string = f"""select {treatment},avg({key_metric}),varSamp({key_metric}),count(*)  
        from {table} group by {treatment} order by {treatment}"""
        result = sql_instance.sql(string).astype(float).values
        result = [i + 1] + ttest(result)
        results.append(result)
    df = pd.DataFrame(results, columns=cols)
    df = pd.merge(df, pred_df, on=["t"], how="left")
    longterm_plot(df)
    return df
