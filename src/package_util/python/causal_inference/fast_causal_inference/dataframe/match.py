import fast_causal_inference.lib.tools as ais_tools
from fast_causal_inference.dataframe import *
import fast_causal_inference.dataframe.regression as Regression
from fast_causal_inference.dataframe.dataframe import readClickHouse, DataFrame
from fast_causal_inference.util import ClickHouseUtils, StarRocksUtils, SqlGateWayConn
import seaborn as sns
from matplotlib import rcParams
from fast_causal_inference.dataframe.df_base import OlapEngineType
import numpy as np
import pandas as pd


class CaliperMatching:
    """
    This class implements the Caliper Matching method for causal inference.

    Parameters
    ----------

    caliper : float, default=0.2
        The caliper width for matching. 
        
    Methods
    -------

    fit(dataframe, treatment, score, exacts=[], alias = 'matching_index'):
        Apply the Caliper Matching method to the input dataframe.

    Example
    -------

    .. code-block:: python

        import fast_causal_inference
        import fast_causal_inference.dataframe.match as Match
        df = fast_causal_inference.readClickHouse('test_data_small')
        model = Match.CaliperMatching(0.5)
        tmp = model.fit(df, treatment='treatment', score='weight', exacts=['x_cat1'])
        match_df = tmp.filter("matching_index!=0") # filter out the unmatched records
    >>> print('sample size Before match: ')
    >>> df.count().show()
    >>> print('sample size After match: ')
    >>> match_df.count().show()
    sample size Before match:
    10000
    sample size After match:
    9652
    >>> import fast_causal_inference.dataframe.match as Match
    >>> d1 = Match.smd(df, 'treatment', ['x1','x2'])
    >>> print(d1)
         Control  Treatment       SMD
    x1 -0.012658  -0.023996 -0.011482
    x2  0.005631   0.037718  0.016156
    >>> import fast_causal_inference.dataframe.match as Match
    >>> d2 = Match.smd(match_df, 'treatment', ['x1','x2'])
    >>> print(d2)
         Control  Treatment       SMD
    x1 -0.015521  -0.025225 -0.009821
    x2  0.004834   0.039698  0.017551

    >>> Match.matching_plot(df_score,'treatment','prob')
    >>> Match.matching_plot(match_df,'treatment','prob')
    """

    def __init__(self, caliper=0.2, k=1):
        self.caliper = caliper
        self.k = k

    def fit(self, dataframe, treatment, score, exacts=[], left_treatment = 0, alias="matching_index"):
        """
        Apply the Caliper Matching method to the input dataframe.

        Parameters
        ----------

        dataframe : DataFrame
            The input dataframe.
        treatment : str
            The treatment column name.
        score : str
            The propensity score column name.
        exacts : list, default=''
            The column names for exact matching, ['x_cat1'].
        alias : str, default='matching_index'
            The alias for the matching index column in the output dataframe.

        Returns
        -------

        DataFrame
            The output dataframe with an additional column for the matching index.
        """
        if dataframe.engine == OlapEngineType.STARROCKS:
            return self.fit_impl_sr(dataframe, treatment, score, exacts, alias)
        new_table_name = DataFrame.createTableName()
        view_df = dataframe.materializedView(is_temp=True)
        sql = f"""
        select *, toInt64(0) as {alias} from {view_df.getTableName()} limit 0
        """
        ClickHouseUtils.clickhouse_create_view_v2(
            table_name=new_table_name,
            select_statement=sql,
            origin_table_name=view_df.getTableName(),
            is_physical_table=True,
        )

        clickhouse_utils = ClickHouseUtils()
        physical_table_name = DataFrame.createTableName(is_temp=True)
        physical_sql = f"create table {clickhouse_utils.DEFAULT_DATABASE}.{physical_table_name} engine = MergeTree() order by tuple() as select * from {view_df.getTableName()}"
        clickhouse_utils.execute(physical_sql)
        
        exacts = [f"concat('pre{i}-' , toString({elem}))" for i, elem in enumerate(exacts, start=1)]
        exacts = ','.join(exacts)
        if exacts != "":
            exacts = "," + exacts
        if left_treatment == 1:
            treatment = f"1-{treatment}"
        sql = f""" insert into {new_table_name}
        with (select CaliperMatchingInfo({self.k})({treatment}, {score}, {self.caliper}{exacts}) from {physical_table_name}) as matching_info 
        select *, CaliperMatching(matching_info, {treatment}, {score}, {self.caliper}{exacts}) as {alias} from {physical_table_name}
        """
        clickhouse_utils.execute(sql)
        clickhouse_utils.clickhouse_drop_view(physical_table_name)
        df = readClickHouse(new_table_name)
        flush_sql = f"SYSTEM FLUSH DISTRIBUTED {new_table_name}"
        clickhouse_utils.execute(flush_sql)
        return df

    def fit_impl_sr(self, dataframe, treatment, score, exacts=[], alias="matching_index"):
        new_table_name = DataFrame.createTableName()
        view_df = dataframe.materializedView(is_temp=True)
        sql = f"""
        select *, cast(0 as bigint) as {alias} from {view_df.getTableName()} limit 0
        """
        StarRocksUtils().create_view(
            view_name=new_table_name,
            sql_statement=sql,
            is_table=True
        )
        dataframe.materializedView(is_physical_table=True)

        physical_df = view_df.materializedView(
            is_physical_table=True, is_distributed_create=False, is_temp=True
        )
        exacts = '+'.join(exacts)
        if exacts != "":
            exacts = ",[" + exacts.replace("+", ",") + "]"
        sql = f""" 
        insert into {new_table_name}
        with matching_info_tbl as (select caliper_matching_info({treatment}, {score}, {self.caliper}{exacts}) as matching_info from {physical_df.getTableName()})
        select {",".join(physical_df.columns)}, caliper_matching(matching_info, {treatment}, {score}, {self.caliper}{exacts}) as {alias} from {physical_df.getTableName()}, matching_info_tbl
        """
        sr = StarRocksUtils()
        sr.execute(sql)
        return readStarRocks(new_table_name)



def smd(df, T, cols):
    """
    Calculate the Standardized Mean Difference (SMD) for the input dataframe.

    Parameters
    ----------

    df : DataFrame
        The input dataframe.
    T : str
        The treatment column name.
    cols : str
        The column names to calculate the SMD, separated by '+'.

    Returns
    -------

    DataFrame
        The output dataframe with the SMD results.

    Example
    -------

    >>> import fast_causal_inference.dataframe.match as Match
    >>> d2 = Match.smd(match_df, 'treatment', ['x1','x2'])
    >>> print(d2)
         Control  Treatment       SMD
    x1 -0.015521  -0.025225 -0.009821
    x2  0.004834   0.039698  0.017551
    

    """
    df = df.materializedView()
    if df.engine == OlapEngineType.STARROCKS:
        return _smd_impl_sr(df, T, cols)
    new_df = df.materializedView(is_temp=True)
    pandas_result = ais_tools.SMD(new_df.getTableName(), T, cols)
    return pandas_result


def _smd_impl_sr(df, T, cols):
    new_df = df.materializedView(is_temp=True)
    table = new_df.getTableName()
    sql_instance = df.sql_conn
    types = df.dtypes
    numerical_cols = []
    for col in cols:
        if df.is_numeric_type(types[col]):
            numerical_cols.append(col)
    string = ",".join(
        [f"avg({i}) as {i}_avg,var_samp({i}) as {i}_std" for i in numerical_cols]
    )
    res = sql_instance.sql(
        f"select {T} as T, {string} from {table} group by {T} order by {T} "
    )
    res = np.array(res)[:, 1:].T.reshape(-1, 4)
    res = pd.DataFrame(
        res, columns=["Control", "Treatment", "Control_var", "Treatment_var"]
    )
    res = res.astype(float)
    res["SMD"] = (res["Treatment"] - res["Control"]) / np.sqrt(
        0.5 * (res["Control_var"] + res["Treatment_var"])
    )
    res = res[["Control", "Treatment", "SMD"]]
    res.index = numerical_cols
    res = res.sort_values("SMD")
    return res


def matching_plot(
    df,
    T,
    col,
    xlim=(0, 1),
    figsize=(8, 8),
    xlabel="",
    ylabel="density",
    legend=["Control", "Treatment"],
):
    """This function plots the overlaid distribution of col in df over
    treat and control group.

    Parameters
    ----------

    table: str
        The name of the table to query from.

    T : str
        The name of the treatment indicator column in the table.

    col : str
        The name of the column that corresponds to the variable to plot.

    xlim : tuple, optional
        The tuple of xlim of the plot. (0,1) by default.

    figsize : tuple, optional
        The size of the histogram; (8,8) by default.

    xlabel : str, optional
        The name of xlabel; col by default.

    ylabel : str, optional
        The name of ylabel; `density` by default.

    legend : iterable, optional
        The legend; `Control` and  `Treatment` by default.

    Yields
    ------

    An overlaied histogram

    >>> import fast_causal_inference.dataframe.match as Match
    >>> Match.matching_plot(df,'treatment','x1')
    """
    df = df.materializedView()
    table = df.getTableName()
    sql_instance = SqlGateWayConn.create_default_conn()
    x1 = sql_instance.sql(
        f"select {col} from {table} where {T}=1 order by rand() limit 10000"
    )
    x0 = sql_instance.sql(
        f"select {col} from {table} where {T}=0 order by rand() limit 10000"
    )
    rcParams["figure.figsize"] = figsize[0], figsize[1]
    ax = sns.distplot(x0)
    sns.distplot(x1)
    ax.set_xlim(xlim[0], xlim[1])
    if len(xlabel) == 0:
        ax.set_xlabel(col)
    else:
        ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(legend)
    del x1, x0



class PrognosticScoreMatching:
    """
    This class implements the Prognostic Score Matching method for causal inference.


    Methods 
    -------
    
    fit_prog_score(outcome_name, prog_input_covs, train_model_data=None,model_name = "")
        Fit the propensity score given the inputs. 

    match(outcome_name, model_name="", prog_input_covs=[], training_data=None, testing_data=None, effect_df_filtered_cols=[],caliper_size=10, exact_cols=[])
        Perform matching and returns the matched data.
    """

    def __init__(self, treat_data, control_data):
        """
        Parameters
        ----------
        treat_data : DataFrame  
            The DataFrame with treated data only
        control_data : DataFrame
            The DataFrame with control data only
        """
        self.treat_data = treat_data
        self.control_data = control_data 
        self.prog_model = {} 
        self.prog_model_covs = {} 
        self.prog_model_outcome_name = {} 
        
    def fit_prog_score(self, outcome_name, prog_input_covs, train_model_data=None,model_name = ""):
        """
        Fit the prognostic score model given the inputs. The default version uses 
        a simple OLS for predicting the baseline prognostic score. 

        Parameters 
        ----------

        outcome_name : str 
            The column name of the outcome variable. 
        
        prog_input_covs: list 
            The list of covariates used for train the prognostic score model. 

        train_model_data : DataFrame, optional 
            The data used for train the model. Default is None, which will use the 
            train_data passed in when constructing the object. 
        
        model_name : str, optional 
            The name of the model. Default is "", where the new model is named 'modelX' where 
            X is the number of the maximal number of models the current matching object has. 
    
        """
        if train_model_data is None:
            train_model_data = self.control_data
        prog_model = Regression.Ols()
        prog_model.fit(f"{outcome_name}~{'+'.join(prog_input_covs)}", train_model_data)
        if len(model_name) == 0:
            model_name = 'model_progscore_' + str(len(self.prog_model.keys()) + 1) 
        self.prog_model[model_name] = prog_model 
        self.prog_model_covs[model_name] = prog_input_covs 
        self.prog_model_outcome_name[model_name] = "effect_" + model_name 
    
    def match(self,outcome_name, model_name="", prog_input_covs=[], training_data=None, testing_data=None, effect_df_filtered_cols=[],caliper_size=10, exact_cols=[]): 
        
        """
        Perform the prognostic score matching.

        Parameters 
        ----------

        outcome_name : str 
            The column name of the outcome variable. 
        
        model_name: str, optional  
            The name of the model to used for predicting the prognostic score for the testing data. 
            If the name of the model is not pre-trained, a new model will be trained based on the input covariates. 
            
        prog_input_covs: list, optional 
            The list of covariates used for train the prognostic score model. 
            If the model 

        train_model_data : DataFrame, optional 
            The data used for train the model. Default is None, which will use the 
            train_data passed in when constructing the object. 
        
        testing_data : DataFrame, optional 
            The data used for matching. Note that the testing data should not contain the data used for training the prognostic score model. 
            Otherwise, the inference can be problematic. None by default, where the testing data passed in when creating the object 
            is used for conducting the inference. 

        effect_df_filtered_cols : list, optional 
            The columns necessary for output matched data. Empty by default, where all the columns are kept after matching.
            We suggest users to specify columns that are necessary for outcome analysis after matching to reduce uncessary 
            space for large-scale computation. 
        
        caliper_size : numeric, optional
            The size of caliper for matching. 10 by default 
        
        exact_cols : list, optional 
            The columns for exact matching. Empty by default, where no exact conntrain is enforced for matching.    
        """
        
        ## TODO @shichaohan     
        ## 1. optimize for caliper size 
        if (training_data is None) and (testing_data is None):
            ## Do train test split 
            control_train, control_test = self.control_data.split(0.5)
            control_train = control_train.drop("if_test")
            control_test = control_test.drop("if_test")
            training_data = control_train 
            testing_data = (self.treat_data.withColumn("t_indicator", Fn.lit(1))).union(control_test.withColumn("t_indicator", Fn.lit(0)))            
            
        ## Step 1: Fit prognostic score model if necessary 
        if not model_name in list(self.prog_model.keys()):
            self.fit_prog_score(outcome_name, prog_input_covs, model_name)
        prog_score_model = self.prog_model[model_name]
        
        ## Step 2: Get the effect 
        prog_input_covs = self.prog_model_covs[model_name]
        effect_col = self.prog_model_outcome_name[model_name]
        effect_df = prog_score_model.effect('+'.join(prog_input_covs), testing_data, effect_name=effect_col)
        if len(effect_df_filtered_cols) == 0:
            effect_df_filtered= effect_df 
        else: 
            
            effect_df_filtered = effect_df.select(effect_df_filtered_cols)
        
        ## Step 3: Matching
        model_match = Match.CaliperMatching(caliper_size)
        if len(exact_cols) > 0:
            matched_result = model_match.fit(effect_df_filtered, treatment='t_indicator', score=effect_col, exacts=exact_cols)
        else: 
            matched_result = model_match.fit(effect_df_filtered, treatment='t_indicator', score=effect_col)
        
        return matched_result 
            
        
        

