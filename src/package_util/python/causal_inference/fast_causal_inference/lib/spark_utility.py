from typing import List, Tuple, Any, List
from functools import reduce
from distutils.version import LooseVersion
import numpy as np
import pandas as pd
from scipy.stats import gamma  # type: ignore
from pyspark import __version__ as V
import decimal
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql.types import DoubleType, ArrayType, FloatType
from pyspark.ml.feature import Bucketizer
from pyspark.ml import Pipeline
import pyspark.sql.functions as F
from pyspark.ml.feature import (
    MinMaxScaler,
    StandardScaler,
    StringIndexer,
    VectorAssembler,
    QuantileDiscretizer,
)

######################## simulate_randomized_trial  ###############################


def _sigmoid(x: float) -> float:
    """Helper function to apply sigmoid to a float.
    Args:
        x: a float to apply the sigmoid function to.

    Returns:
        (float): x after applying the sigmoid function.
    """
    return 1 / (1 + np.exp(-x))


def simulate_randomized_trial(
    n: int = 1000,
    p: int = 5,
    sigma: float = 1.0,
    binary_outcome: bool = False,
    add_cost_benefit: bool = False,
) -> pd.DataFrame:
    """Simulates a synthetic dataset corresponding to a randomized trial
        The version with continuous outcome and without cost/benefit columns corresponds to Setup B in Nie X. and Wager S. (2018) 'Quasi-Oracle Estimation of Heterogeneous Treatment Effects' and is aligned with the implementation in the CausalML package.
    Args:
        n (int, optional): number of observations to generate
        p (int optional): number of covariates. Should be >= 5, since treatment heterogeneity is determined based on the first 5 features.
        sigma (float): standard deviation of the error term
        binary_outcome (bool): whether the outcome should be binary or continuous
        add_cost_benefit (bool): whether to generate cost and benefit columns
    Returns:
        (pandas.DataFrame): a dataframe containing the following columns:
            - treatment
            - outcome
            - propensity
            - expected_outcome
            - actual_cate
            - benefit (only if add_cost_benefit=True)
            - cost (only if add_cost_benefit=True)
    """

    X = np.random.normal(loc=0.0, scale=1.0, size=n * p).reshape((n, -1))
    b = np.maximum(np.repeat(0.0, n), X[:, 0] + X[:, 1] + X[:, 2]) + np.maximum(
        np.repeat(0.0, n), X[:, 3] + X[:, 4]
    )
    e = np.repeat(0.5, n)
    tau = X[:, 0] + np.log1p(np.exp(X[:, 1]))

    w = np.random.binomial(1, e, size=n)

    if binary_outcome:
        y1 = b + (1 - 0.5) * tau + sigma * np.random.normal(loc=0.0, scale=1.0, size=n)
        y0 = b + (0 - 0.5) * tau + sigma * np.random.normal(loc=0.0, scale=1.0, size=n)
        y1_binary = pd.Series(_sigmoid(y1) > 0.5).astype(
            np.int32
        )  # potential outcome when w=1
        y0_binary = pd.Series(_sigmoid(y0) > 0.5).astype(
            np.int32
        )  # potential outcome when w=0

        # observed outcome
        y = y0_binary
        y[w == 1] = y1_binary  # type: ignore

        # ensure that tau is between [-1, 1]
        tau = _sigmoid(y1) - _sigmoid(y0)

    else:
        y = b + (w - 0.5) * tau + sigma * np.random.normal(loc=0.0, scale=1.0, size=n)

    data = pd.DataFrame(
        {
            "treatment": w,
            "outcome": y,
            "propensity": e,
            "expected_outcome": b,
            "actual_cate": tau,
        }
    )
    features = pd.DataFrame(
        X, columns=[f"feature_{i}" for i in range(1, X.shape[1] + 1)]
    )
    data = pd.concat([data, features], axis=1)

    if add_cost_benefit:
        data["benefit"] = gamma.rvs(3, size=n)
        data.loc[data["outcome"] == 0, "benefit"] = 0
        data["cost"] = data["benefit"] * 0.25
        data.loc[data["treatment"] == 0, "cost"] = 0

    return data


######################## get num and cat features spark df ###############################


def get_num_cat_feat(
    input_spark_df: DataFrame, exclude_list: List[str] = []
) -> Tuple[List[str], List[str]]:
    """
    desc: return cat and num features list from a spark df, a step before any encoding on cat features
    inputs:
        * input_spark_df: the input spark data frame to be checked.
        * exclude_list (list of str): the excluded column name list, which will be excluded for the categorization.
    output:
        * numeric_columns (list of str): the list of numeric column names.
        * string_columns (list of str): the list of categorical column names.
    """
    timestamp_columns = [
        item[0]
        for item in input_spark_df.dtypes
        if item[1].lower().startswith(("time", "date"))
    ]

    # categorize the remaining columns into categorical and numeric columns
    string_columns = [
        item[0]
        for item in input_spark_df.dtypes
        if item[1].lower().startswith("string")
        and item[0] not in exclude_list + timestamp_columns
    ]

    numeric_columns = [
        item[0]
        for item in input_spark_df.dtypes
        if item[1].lower().startswith(("big", "dec", "double", "int", "float"))
        and item[0] not in exclude_list + timestamp_columns
    ]

    # check whether all the columns are covered
    all_cols = timestamp_columns + string_columns + numeric_columns + exclude_list

    if len(set(all_cols)) == len(input_spark_df.columns):
        print("All columns are been covered.")
        print(f"numeric colums are {numeric_columns}")
        print(f"string colums are {string_columns}")

    elif len(set(all_cols)) < len(input_spark_df.columns):
        not_handle_list = list(set(input_spark_df.columns) - set(all_cols))
        print(f"Not all columns are covered. The columns missed out: {not_handle_list}")
    else:
        mistake_list = list(set(all_cols) - set(input_spark_df.columns))
        print(f"The columns been hardcoded wrongly: {mistake_list}")

    return numeric_columns, string_columns


######################## get cat features less than X cardinality ################################


def get_cat_feat_one_hot(
    input_spark_df: DataFrame, cat_cols: List[str], num: int
) -> List[str]:
    """
    only return categorical columns with cardinality less or equal to some threshold
    input:
        * input_spark_df: the input spark data frame to be checked.
        * cat_cols: list of categorical columns to be checked.
        * num: threshold selected. eg: 50. which means if a categorical feature takes more than 50 distinct values will be ruled out.
    output:
        * cat_cols_one_hot: list of categorical columns can be used for one hot.
    """

    cat_cols_one_hot = []

    for categoricalCol in cat_cols:
        cnt = input_spark_df.select(F.countDistinct(categoricalCol)).collect()[0][0]
        if cnt <= num:
            cat_cols_one_hot += [categoricalCol]

    return cat_cols_one_hot


########################### string indexer ###########################


def str_index_cat_cols(cat_cols: List[str]) -> List[StringIndexer]:
    """
    only to stringIndex cols (per item frequency), no encoding applied
    input:
        * cat_cols: cat cols to be string indexed
        * stages: input stages (from any previous stages)
    output:
        * stages: modified stages for spark pipeline
    """

    stages = []

    for categoricalCol in cat_cols:
        stringIndexer = StringIndexer(
            inputCol=categoricalCol,
            outputCol=categoricalCol + "_idx",
            handleInvalid="skip",
        )
        stages += [stringIndexer]

    return stages


########################### one hot encoder ###########################


def one_hot_encode_cat_cols(cat_cols: List[str]) -> List[Any]:
    """
    perform one hot encoding for cat_cols
    input:
        * cat_cols: categorical columns already str indexed
    output:
        * stages
    """

    # Category Indexing with StringIndexer, will encode to numerical according to frequency, highest frequency will be encoded to 0
    # when applying this stringIndexer onto another dataset and encounter missing encoded value, we can throw exception or setHandleInvalid(“skip”)
    # like indexer.fit(df1).setHandleInvalid("skip").transform(df2), will remove all rows unable to encode
    # no indexing applied
    # stringIndexer = StringIndexer(inputCol=categoricalCol, outputCol=categoricalCol + "Index")

    # Use OneHotEncoder to convert categorical variables into binary SparseVectors，
    # binary sparse vectors like (2,[0],[1.0]) means a vector of length 2 with 1.0 at position 0 and 0 elsewhere.
    # spark OHE will automatically drop the last category, you can force it not to drop by dropLast=False
    # it omits the final category to break the correlation between features

    # column is already indexed, with suffix _index as default
    if LooseVersion(V) < LooseVersion("3.0"):
        from pyspark.ml.feature import OneHotEncoderEstimator

        encoder = OneHotEncoderEstimator(
            inputCols=cat_cols,
            outputCols=[categoricalCol + "_enc" for categoricalCol in cat_cols],
            dropLast=False,
            # handleInvalid="keep",
        )
    else:
        from pyspark.ml.feature import OneHotEncoder

        encoder = OneHotEncoder(
            inputCols=cat_cols,
            outputCols=[categoricalCol + "_enc" for categoricalCol in cat_cols],
            dropLast=False,
            # handleInvalid="keep",
        )
    # Add stages.  These are not run here, but will run all at once later on.
    stages = [encoder]

    return stages


########################### assemble encoder ##########################


def assemble_into_features(
    num_cols: List[str],
    cat_cols: List[str],
    scale_method: str = None,
    one_hot_enc: bool = False,
    one_hot_cols: List[str] = None,
) -> List:
    """
    assemble all features into vector
    cat_cols with suffix
    num cols
    input:
        * scale_method: 0 for min max and 1 for standard
    output:
        * stages
    """

    # to combine all the feature columns into a single vector column.
    # This includes both the numeric columns and the one-hot encoded binary vector columns in our dataset.
    # Transform all features into a vector using VectorAssembler

    # empty stage holder
    stages = []

    # perform str indexing
    str_ind_stages = str_index_cat_cols(cat_cols)

    stages += str_ind_stages

    all_indexed_catcols = [a + "_idx" for a in cat_cols]

    if one_hot_enc:
        str_diff_catcols = list(set(cat_cols) - set(one_hot_cols))
        str_indexed_catcols = [a + "_idx" for a in str_diff_catcols]
        indexed_catcols = [a + "_idx" for a in one_hot_cols]

        # perform OHE encoding

        ohe_stages = one_hot_encode_cat_cols(indexed_catcols)
        stages += ohe_stages

        encoded_catcols = [a + "_enc" for a in indexed_catcols]
        assemblerInputs = num_cols + str_indexed_catcols + encoded_catcols

    else:
        assemblerInputs = num_cols + all_indexed_catcols

    # VectorAssembler only applied to numerical or transformed categorical columns
    assembler = VectorAssembler(
        inputCols=assemblerInputs, outputCol="mid_features", handleInvalid="skip"
    )
    stages += [assembler]

    # then we apply scaling on the vectorized features, 2 additional params are:
    # withStd: True by default. Scales the data to unit standard deviation.
    # withMean: False by default. Centers the data with mean before scaling.
    if scale_method is not None:
        assert scale_method in ["standard", "minmax"]
        if scale_method == "standard":
            scaler = StandardScaler(
                inputCol="mid_features", outputCol="features", withMean=True
            )
        else:
            scaler = MinMaxScaler(
                min=0, max=1, inputCol="mid_features", outputCol="features"
            )

        stages += [scaler]

    return stages


########################### quantile discretizer ###########################


def quantile_num_cols(
    num_cols: List[str], num_cols_bins: int = 2, relative_error: float = 0.001
) -> List[QuantileDiscretizer]:
    """
    quantile discrete numeric cols
    input:
        * num_cols: num cols to be quantile discreted
        * stages: input stages (from any previous stages)
    output:
        * stages: modified stages for spark pipeline
    """

    stages = []

    for numericCol in num_cols:
        quantileDiscretizer = QuantileDiscretizer(
            inputCol=numericCol,
            outputCol=numericCol + "_bin",
            numBuckets=num_cols_bins,
            relativeError=relative_error,
            handleInvalid="skip",
        )
        stages += [quantileDiscretizer]

    return stages


########################### generate test data ###########################


def _sigmoid(x: float) -> float:
    """Helper function to apply sigmoid to a float.
    Args:
        x: a float to apply the sigmoid function to.

    Returns:
        (float): x after applying the sigmoid function.
    """
    return 1 / (1 + np.exp(-x))


def simulate_randomized_trial(
    n: int = 1000,
    p: int = 5,
    sigma: float = 1.0,
    binary_outcome: bool = False,
    add_cost_benefit: bool = False,
) -> pd.DataFrame:
    """Simulates a synthetic dataset corresponding to a randomized trial
        The version with continuous outcome and without cost/benefit columns corresponds to Setup B in Nie X. and Wager S. (2018) 'Quasi-Oracle Estimation of Heterogeneous Treatment Effects' and is aligned with the implementation in the CausalML package.
    Args:
        n (int, optional): number of observations to generate
        p (int optional): number of covariates. Should be >= 5, since treatment heterogeneity is determined based on the first 5 features.
        sigma (float): standard deviation of the error term
        binary_outcome (bool): whether the outcome should be binary or continuous
        add_cost_benefit (bool): whether to generate cost and benefit columns
    Returns:
        (pandas.DataFrame): a dataframe containing the following columns:
            - treatment
            - outcome
            - propensity
            - expected_outcome
            - actual_cate
            - benefit (only if add_cost_benefit=True)
            - cost (only if add_cost_benefit=True)
    """

    X = np.random.normal(loc=0.0, scale=1.0, size=n * p).reshape((n, -1))
    b = np.maximum(np.repeat(0.0, n), X[:, 0] + X[:, 1] + X[:, 2]) + np.maximum(
        np.repeat(0.0, n), X[:, 3] + X[:, 4]
    )
    e = np.repeat(0.5, n)
    tau = X[:, 0] + np.log1p(np.exp(X[:, 1]))

    w = np.random.binomial(1, e, size=n)

    if binary_outcome:
        y1 = b + (1 - 0.5) * tau + sigma * np.random.normal(loc=0.0, scale=1.0, size=n)
        y0 = b + (0 - 0.5) * tau + sigma * np.random.normal(loc=0.0, scale=1.0, size=n)
        y1_binary = pd.Series(_sigmoid(y1) > 0.5).astype(
            np.int32
        )  # potential outcome when w=1
        y0_binary = pd.Series(_sigmoid(y0) > 0.5).astype(
            np.int32
        )  # potential outcome when w=0

        # observed outcome
        y = y0_binary
        y[w == 1] = y1_binary  # type: ignore

        # ensure that tau is between [-1, 1]
        tau = _sigmoid(y1) - _sigmoid(y0)

    else:
        y = b + (w - 0.5) * tau + sigma * np.random.normal(loc=0.0, scale=1.0, size=n)

    data = pd.DataFrame(
        {
            "treatment": w,
            "outcome": y,
            "propensity": e,
            "expected_outcome": b,
            "actual_cate": tau,
        }
    )
    features = pd.DataFrame(
        X, columns=[f"feature_{i}" for i in range(1, X.shape[1] + 1)]
    )
    data = pd.concat([data, features], axis=1)

    if add_cost_benefit:
        data["benefit"] = gamma.rvs(3, size=n)
        data.loc[data["outcome"] == 0, "benefit"] = 0
        data["cost"] = data["benefit"] * 0.25
        data.loc[data["treatment"] == 0, "cost"] = 0

    return data


########################### smd for numeric features ###########################
def get_num_smd(
    matched_df: DataFrame, treatment: str, num_fea_list: List[str]
) -> DataFrame:
    """
    desc: returns a dataframe of smd given numeric feature list
    inputs:
        * sparkdf: input spark dataframe (Please read dataframe from cache or hdfs !!!)
        * treatment: treatment column name
        * num_fea_list: numeric features which need to be test
    output:
        * dataframe: returns a spark dataframe with smd of numeric variables
    """

    # get smd of numeric variables (including string indexed columns)
    pre_num_smd_list = []  # List[Dataframe]
    for fea in num_fea_list:
        pre_num_smd_df = (
            matched_df.select(treatment, fea)
            .groupby(treatment)
            .agg(
                F.round(F.variance(fea), 4).alias("var"),
                F.round(F.mean(fea), 4).alias("mean"),
            )
            .withColumn("variable", F.lit(fea))
        )
        pre_num_smd_list.append(pre_num_smd_df)
        pre_num_smd_df = reduce(DataFrame.unionAll, pre_num_smd_list)

    # split into treatment and control dataframe, return cat_smd_df
    df_num_1 = pre_num_smd_df.filter(F.col(treatment) == 1.0).select(
        "variable", F.col("mean").alias("mean_1"), F.col("var").alias("var_1")
    )

    df_num_0 = pre_num_smd_df.filter(F.col(treatment) == 0.0).select(
        "variable", F.col("mean").alias("mean_0"), F.col("var").alias("var_0")
    )

    smd_df = (
        df_num_1.join(df_num_0, on="variable")
        .withColumn(
            "smd",
            F.round(
                (F.col("mean_1") - F.col("mean_0"))
                / F.sqrt(0.5 * (F.col("var_1") + F.col("var_0"))),
                4,
            ),
        )
        .select("variable", "smd")
    )

    return smd_df


def feature_process(
    sparkdf,
    label: str,
    exclude_list: [],
    scale_method: str = "minmax",
    one_hot_enc: bool = False,
    one_hot_threshold: int = 30,
):
    """
    desc: returns a spark dataframe with propensity added as a column
    inputs:
        * sparkdf: the data frame contains features and label.
        * label: column name consider as label.
        * exclude_list: list of columns not processed by feature process.
        * scale_method: defaul is "minmax".
        * one_hot_enc: whether to use one hot.
        * one_hot_threshold: if one_hot_enc set to true, specify a threshold for column with certain cardinality will be included.
    output:
        * feature_df(dataframe): dataframe which is ready for spark model.
    """

    # ! label has to be binary 0 or 1

    numeric_columns, string_columns = get_num_cat_feat(
        input_spark_df=sparkdf, exclude_list=exclude_list
    )
    assert (
        label in numeric_columns
    ), "ERROR: label must be included and has to be numeric and binary!"

    numeric_columns.remove(label)

    for col in numeric_columns:
        sparkdf = sparkdf.withColumn(col, sparkdf[col].cast(DoubleType()))

    sparkdf = sparkdf.na.fill(0.00)
    sparkdf = sparkdf.na.fill("NA")

    if one_hot_enc:
        threshold = one_hot_threshold if one_hot_threshold is not None else 30
        one_hot_col = get_cat_feat_one_hot(
            input_spark_df=sparkdf, cat_cols=string_columns, num=threshold
        )
        stages = assemble_into_features(
            num_cols=numeric_columns,
            cat_cols=string_columns,
            scale_method=scale_method,
            one_hot_enc=True,
            one_hot_cols=one_hot_col,
        )
    else:
        stages = assemble_into_features(
            num_cols=numeric_columns,
            cat_cols=string_columns,
            scale_method=scale_method,
        )

    pipeline = Pipeline(stages=stages)

    pipe = pipeline.fit(sparkdf)
    feature_df = pipe.transform(sparkdf)

    return pipe, feature_df
