import random
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pyspark.sql.functions as f
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType

from fast_causal_inference.util import SqlGateWayConn, ClickHouseUtils

warnings.filterwarnings("ignore")


def ROC_curve(table, label, P):
    sql_instance = SqlGateWayConn.create_default_conn()
    # 计算 ROC curve
    quantiles = list(np.linspace(0, 1, 1001))
    quantiles = [str(i) for i in quantiles]
    thresholds_str = (
        sql_instance.sql(
            f"select quantiles({','.join(quantiles)})({P}) as res from {table}"
        )
        .values[0][0]
        .replace("[", "")
        .replace("]", "")
        .split(",")
    )
    thresholds_str = list(set(thresholds_str))

    # 使用一次SQL查询计算TPR和FPR
    query = f"""
    SELECT
        threshold,
        SUM(CASE WHEN {label} = 1 AND {P} >= threshold THEN 1 ELSE 0 END) as TP,
        SUM(CASE WHEN {label} = 0 AND {P} >= threshold THEN 1 ELSE 0 END) as FP,
        SUM(CASE WHEN {label} = 0 AND {P} < threshold THEN 1 ELSE 0 END) as TN,
        SUM(CASE WHEN {label} = 1 AND {P} < threshold THEN 1 ELSE 0 END) as FN
    FROM {table}
    ARRAY JOIN [{','.join(thresholds_str)}] as threshold
    GROUP BY threshold
    order by threshold
    """
    results = (
        sql_instance.sql(query)[["threshold", "TP", "FP", "TN", "FN"]]
        .astype(float)
        .values
    )
    roc_data = []
    for result in results:
        threshold, tp, fp, tn, fn = result
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        roc_data.append((fpr, tpr))

    # 绘制ROC曲线
    roc_data.sort()
    fpr, tpr = zip(*roc_data)
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Roc Curve")
    plt.show()


def LogisticRegression_spark(
    spark,
    table,
    label,
    X,
    categorical_columns,
    maxIter=40,
    regParam=0,
    elasticNetParam=0.8,
    repartition_num=400,
):
    def random_func(n):
        return random.randint(0, n)

    data = ClickHouseUtils.clickhouse_2_dataframe(spark, table)
    data = data.replace("", "fill_unknown")
    random_udf = udf(random_func, IntegerType())
    data = data.withColumn("bucket_id", random_udf(f.lit(repartition_num))).repartition(
        repartition_num, "bucket_id"
    )
    data = data.cache()
    indexers = [
        StringIndexer(inputCol=column, outputCol=column + "_index")
        for column in categorical_columns
    ]
    encoders = [
        OneHotEncoder(inputCol=column + "_index", outputCol=column + "_onehot")
        for column in categorical_columns
    ]
    assembler = VectorAssembler(
        inputCols=X + [column + "_onehot" for column in categorical_columns],
        outputCol="features",
    )
    label_indexer = StringIndexer(inputCol=label, outputCol="label")
    lr = LogisticRegression(
        maxIter=maxIter, regParam=regParam, elasticNetParam=elasticNetParam
    )
    pipeline = Pipeline(stages=indexers + encoders + [assembler, label_indexer, lr])
    model = pipeline.fit(data)
    predictions = model.transform(data)
    evaluator = BinaryClassificationEvaluator(
        rawPredictionCol="rawPrediction", labelCol="label", metricName="areaUnderROC"
    )

    # 计算AUC
    auc = evaluator.evaluate(predictions)

    # 输出AUC
    print("*" * 30)
    print("AUC: ", auc)
    print("*" * 30)

    # 定义一个提取DenseVector第一个元素的UDF
    def get_first_element(vector):
        return float(vector[0])

    # 注册UDF并指定返回类型
    get_first_element_udf = udf(get_first_element, returnType="float")
    # 使用withColumn添加新列
    predictions = predictions.withColumn(
        "score", get_first_element_udf(predictions["probability"])
    )

    predictions = predictions[data.columns + ["score"]]
    predictions = predictions.drop("day_").drop("id")
    table_output = f"{table}_{int(time.time())}"
    ClickHouseUtils.dataframe_2_clickhouse(
        dataframe=predictions, clickhouse_table_name=table_output, is_auto_create=True
    )
    print("*" * 30)
    print("table_output:", table_output)
    print("*" * 30)

    return table_output
