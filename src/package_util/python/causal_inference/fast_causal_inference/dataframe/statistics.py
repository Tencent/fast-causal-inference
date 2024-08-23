from fast_causal_inference.dataframe.regression import df_2_table
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
from fast_causal_inference.dataframe.df_base import (
    df_2_table,
)
from fast_causal_inference.util import create_sql_instance


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="DeltaMethod")
@register_fn(engine=OlapEngineType.STARROCKS, name="DeltaMethod")
@define_args(
    FnArg(name="expr", is_param=True), FnArg(name="std", default="True", is_param=True)
)
@aggregrate
class AggDeltaMethodDfFunction(DfFunction):
    # @classmethod
    # def _extract_cols_from_expr(cls, expr):
    #     matches = re.findall(r'avg\((.*?)\)', expr)
    #     unique_matches = list(set(matches))
    #     encoded_matches = [(match, f'X{i + 1}') for i, match in enumerate(unique_matches)]
    #     result = expr
    #     for key, value in encoded_matches:
    #         result = result.replace(f'avg({key})', f'avg({value})')
    #     return result, tuple(col for col, _ in encoded_matches)

    def sql_impl_default(
        self,
        ctx: DfContext,
        fn_args: List[FnArg],
        fn_params: List[FnArg],
        arg_dict: Dict,
    ) -> str:
        expr_arg: FnArg = arg_dict["expr"]
        std_arg: FnArg = arg_dict["std"]
        expr = expr_arg.sql(ctx)
        std = std_arg.sql(ctx)
        sql = self.fn_name(ctx) + f"({expr}, {std})"
        return sql


def delta_method(expr=None, std=False):
    """
    Compute the delta method on the given expression.

    :param expr: Form like f (avg(x1), avg(x2), ...) , f is the complex function expression, x1 and x2 are column names, the columns involved here must be numeric.
    :type expr: str, optional
    :param std: Whether to return standard deviation, default is True.
    :type std: bool, optional

    :return: DataFrame contains the following columns: var or std computed by delta_method.
    :rtype: DataFrame

    Example
    ----------

    .. code-block:: python

        import fast_causal_inference
        allinsql_provider = fast_causal_inference.FCIProvider(database="all_in_sql_guest")
        # 这是一个模拟的实验明细表！可以用该表来模拟实验分析
        df = allinsql_provider.readClickHouse(f'expt_detail_test_appid')

    >>> # 看实验组的指标均值的方差（人均指标）
    >>> df.filter("groupname='B1'").delta_method('avg(numerator)',std=False)
    0.03447617786500999


    >>> # 看每个实验组的指标均值的方差（人均指标）
    >>> df.groupBy('groupname').delta_method('avg(numerator)',std=False)
    groupname Deltamethod('x1', true)(numerator)
    0        A1                         .118752394
    1        B1                         .185677618
    2        A2                         .118817924

    >>> # 看每个实验组的指标均值的标准差（比例指标）
    >>> df.groupBy('groupname').delta_method('avg(numerator)/avg(denominator)',std=True)
    groupname Deltamethod('x1/x2', true)(numerator, denominator)
    0        A1                                         .022480953
    1        B1                                         .029102234
    2        A2                                         .022300609

    """
    return DfFnColWrapper(AggDeltaMethodDfFunction(), {"expr": expr, "std": std}, [])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="ttest_1samp")
@register_fn(engine=OlapEngineType.STARROCKS, name="ttest_1samp")
@define_args(
    FnArg(name="Y", is_param=True),
    FnArg(name="alternative", default="two-sided", is_param=True),
    FnArg(name="mu", default="0", is_param=True),
    FnArg(name="X", default="", is_param=True),
)
@aggregrate
class AggTTest1SampDfFunction(DfFunction):
    def sql_impl_default(
        self,
        ctx: DfContext,
        fn_args: List[FnArg],
        fn_params: List[FnArg],
        arg_dict: Dict,
    ) -> str:
        Y = arg_dict["Y"].sql(ctx)
        alternative = arg_dict["alternative"].sql(ctx)
        mu = arg_dict["mu"].sql(ctx)
        X = arg_dict["X"].sql(ctx)
        x_str = "" if not X else f", {X}"
        sql = self.fn_name(ctx) + f"({Y}, {alternative}, {mu}{x_str})"
        return sql

def ttest_1samp(
    Y,
    alternative="two-sided",
    mu=0,
    X="",
):
    """
    This function is used to calculate the t-test for the mean of one group of scores. It returns the calculated t-statistic and the two-tailed p-value.

    :param Y: str, form like f (avg(x1), avg(x2), ...), f is the complex function expression, x1 and x2 are column names, the columns involved here must be numeric.
    :type Y: str, required
    :param alternative: str, use 'two-sided' for two-tailed test, 'greater' for one-tailed test in the positive direction, and 'less' for one-tailed test in the negative direction.
    :type alternative: str, optional
    :param mu: the mean of the null hypothesis.
    :type mu: float, optional
    :param X: str, an expression used as continuous covariates for CUPED variance reduction. It follows the regression approach and can be a simple form like 'avg(x1)/avg(x2)','avg(x3)','avg(x1)/avg(x2)+avg(x3)'.
    :type X: str, optional
    :return: a testResult object that contains p-value, statistic, stderr and confidence interval. 
    :rtype: testResult

    Example:
    ----------------
    .. code-block:: python

        import fast_causal_inference
        allinsql_provider = fast_causal_inference.FCIProvider(database="all_in_sql_guest")
        # 这是一个模拟的实验明细表！可以用该表来模拟实验分析
        df = allinsql_provider.readClickHouse(f'expt_detail_test_appid')

    >>> # 均值指标检验，检验实验组均值是否=0
    >>> df.filter('groupid=12343').ttest_1samp('avg(numerator)', alternative = 'two-sided',mu=0)
    estimate    stderr t-statistic   p-value      lower      upper
    0  19.737005  0.185678  106.297168  0.000000  19.372997  20.101013

    >>> # 比率指标检验，检验实验组指标是否=0
    >>> df.filter('groupid=12343').ttest_1samp('avg(numerator)/avg(denominator)', alternative = 'two-sided',mu=0)
    estimate    stderr t-statistic   p-value     lower     upper
    0  2.485402  0.029102   85.402433  0.000000  2.428349  2.542454

    >>> # 下钻分析，groupBy首次命中日期，去看不同天命中实验的用户的策略效果是否有所不同
    >>> df.groupBy('first_hit_ds').filter('groupid=12343').ttest_1samp('avg(numerator)', alternative = 'two-sided',mu=0)
    first_hit_ds   estimate    stderr t-statistic   p-value      lower      upper
    0     20240102   7.296926  0.063260  115.347550  0.000000   7.172769   7.421083
    1     20240103   1.137118  0.194346    5.851008  0.000000   0.749203   1.525034
    2     20240101  22.723438  0.201318  112.873450  0.000000  22.328747  23.118130
    """
    return DfFnColWrapper(AggTTest1SampDfFunction(),{"Y": Y, "alternative": f"'{alternative}'", "mu": mu, "X": X},[])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="ttest_2samp")
@register_fn(engine=OlapEngineType.STARROCKS, name="ttest_2samp")
@define_args(
    FnArg(name="Y", is_param=True),
    FnArg(name="alternative", default="two-sided", is_param=True),
    FnArg(name="X", default="", is_param=True),
    FnArg(name="pse", default="", is_param=True),
    FnArg(name="index"),
)
@aggregrate
class AggTTest2SampDfFunction(DfFunction):
    def sql_impl_default(
        self,
        ctx: DfContext,
        fn_args: List[FnArg],
        fn_params: List[FnArg],
        arg_dict: Dict,
    ) -> str:
        Y = arg_dict["Y"].sql(ctx)
        alternative = arg_dict["alternative"].sql(ctx)
        index = arg_dict["index"].sql(ctx)
        X = arg_dict["X"].sql(ctx)
        pse = arg_dict["pse"].sql(ctx)
        x_str = "" if not X else f", {X}"
        x_str = x_str if not pse else f", pse = {pse}"
        sql = self.fn_name(ctx) + f"({Y}, {index}, {alternative}{x_str})"
        return sql


def ttest_2samp(Y, index, alternative="two-sided", X="", pse=""):
    """
            This function is used to calculate the t-test for the means of two independent samples of scores. It returns the calculated t-statistic and the two-tailed p-value.

            :param Y: str, form like f (avg(x1), avg(x2), ...), f is the complex function expression, x1 and x2 are column names, the columns involved here must be numeric.
            :type Y: str, required
            :param index: str, the treatment variable.
            :type index: str, required
            :param alternative: str, use 'two-sided' for two-tailed test, 'greater' for one-tailed test in the positive direction, and 'less' for one-tailed test in the negative direction.
            :type alternative: str, optional
            :param X: str, an expression used as continuous covariates for CUPED variance reduction. It follows the regression approach and can be a simple form like 'avg(x1)/avg(x2)','avg(x3)','avg(x1)/avg(x2)+avg(x3)'.
            :type X: str, optional
            :param pse: str, an expression used as discrete covariates for post-stratification variance reduction. It involves grouping by a covariate, calculating variances separately, and then weighting them. It can be any complex function form, such as 'x_cat1'.
            :type pse: str, optional
            :return: a testResult object that contains p-value, statistic, stderr and confidence interval. 
            :rtype: testResult

            Example:
            ----------------

            .. code-block:: python
            
                import fast_causal_inference
                allinsql_provider = fast_causal_inference.FCIProvider(database="all_in_sql_guest")
                # 这是一个模拟的实验明细表！可以用该表来模拟实验分析
                df = allinsql_provider.readClickHouse(f'expt_detail_test_appid')

            >>> # 均值指标检验
            >>> df.ttest_2samp('avg(numerator)', 'if(groupid=12343,1,0)', alternative = 'two-sided')
                mean0      mean1   estimate    stderr t-statistic   p-value      lower      upper
            0  4.700087  19.737005  15.036918  0.203794   73.784922  0.000000  14.637441  15.436395


            >>> # 比例指标检验，用历史数据做CUPED方差削减
            >>> df.ttest_2samp('avg(numerator)/avg(denominator)', 'if(groupid=12343,1,0)', alternative = 'two-sided', X = 'avg(numerator_pre)/avg(denominator_pre)')
                mean0     mean1  estimate    stderr t-statistic   p-value     lower     upper
            0  0.793732  2.486118  1.692386  0.026685   63.419925  0.000000  1.640077  1.744694


            >>> # 比例指标检验，用首次命中日期做后分层（PSE）方差削减
            >>> df.ttest_2samp('avg(numerator)/avg(denominator)', 'if(groupid=12343,1,0)', alternative = 'two-sided', pse = 'first_hit_ds')
                mean0     mean1  estimate    stderr t-statistic   p-value     lower     upper
            0  1.432746  1.792036  0.359289  0.038393    9.358290  0.000000  0.284032  0.434547


            >>> # 下钻分析，groupBy首次命中日期，去看不同天命中实验的用户的策略效果是否有所不同
            >>> df.groupBy('first_hit_ds').ttest_2samp('avg(numerator)', 'if(groupid=12343,1,0)', alternative = 'two-sided')
            first_hit_ds      mean0      mean1  estimate    stderr t-statistic   p-value     lower     upper
            0     20240102   6.170508   7.296926  1.126418  0.076667   14.692392  0.000000  0.976093  1.276742
            1     20240103  -0.925249   1.137118  2.062368  0.205304   10.045430  0.000000  1.659735  2.465000
            2     20240101  13.679315  22.723438  9.044123  0.231238   39.111779  0.000000  8.590796  9.497451
    """    
    return DfFnColWrapper(
        AggTTest2SampDfFunction(),
        {"Y": Y, "alternative": f"'{alternative}'", "X": X, "pse": pse},
        [index],
    )


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="xexpt_ttest_2samp")
@register_fn(engine=OlapEngineType.STARROCKS, name="xexpt_ttest_2samp")
@define_args(
    FnArg(name="numerator"),
    FnArg(name="denominator"),
    FnArg(name="index"),
    FnArg(name="uin"),
    FnArg(name="metric_type", default="avg", is_param=True),
    FnArg(name="group_buckets", default="[1,1]", is_param=True),
    FnArg(name="alpha", default="0.05", is_param=True),
    FnArg(name="MDE", default="0.005", is_param=True),
    FnArg(name="power", default="0.8", is_param=True),
    FnArg(name="X", default="", is_param=True),
)
@aggregrate
class AggXexptTTest2SampDfFunction(DfFunction):
    def sql_impl_default(
        self,
        ctx: DfContext,
        fn_args: List[FnArg],
        fn_params: List[FnArg],
        arg_dict: Dict,
    ) -> str:
        numerator = arg_dict["numerator"].sql(ctx)
        denominator = arg_dict["denominator"].sql(ctx)
        index = arg_dict["index"].sql(ctx)
        metric_type = arg_dict["metric_type"].sql(ctx)
        group_buckets = arg_dict["group_buckets"].sql(ctx)
        alpha = arg_dict["alpha"].sql(ctx)
        MDE = arg_dict["MDE"].sql(ctx)
        power = arg_dict["power"].sql(ctx)
        X = arg_dict["X"].sql(ctx)
        uin = arg_dict["uin"].sql(ctx)

        if metric_type == "avg":
            group_buckets = ""
            metric_type = ""
        else:
            group_buckets = "," + group_buckets
            metric_type = ",'" + metric_type + "'"

        if X != "":
            X = "," + X

        sql = (
            self.fn_name(ctx)
            + "("
            + numerator
            + ","
            + denominator
            + ","
            + index
            + ","
            + uin
            + metric_type
            + group_buckets
            + ","
            + str(alpha)
            + ","
            + str(MDE)
            + ","
            + str(power)
            + X
            + ")"
        )
        return sql


def xexpt_ttest_2samp(
    numerator,
    denominator,
    index,
    uin,
    metric_type="avg",
    group_buckets=[1,1],
    alpha=0.05,
    MDE=0.005,
    power=0.8,
    X="",
):  
    """
        微信实验平台计算两个独立样本得分均值的t检验的函数说明，它返回计算出的t统计量和双尾p值，统计效力（power）,建议样本量，最小检验差异（MDE）


        参数
        ----------

            numerator : str
                列名，分子，可以使用SQL表达式，该列必须是数值型。

            denominator : str
                列名，分母，可以使用SQL表达式，该列必须是数值型。

            index : str
                列名。该列名用于区分对照组和实验组。
                该列应仅包含两个不同的值，这些值将根据它们的字符顺序用于确定对照组和实验组。
                例如，如果可能的值是0或1，那么0将被视为对照组。 比如"if(groupid=12343,1,0)"
                同样，如果值是'A'和'B'，那么'A'将被视为对照组。 比如"if(groupid=12343,'B','A')"

            uin : str
                列名，用于分桶样本，可以包含SQL表达式，数据类型为int64。如果需要随机分桶，可以使用'rand()'函数。

            metric_type : str, 可选
                'avg'用于均值指标/比例指标的检验，avg(num)/avg(demo)，默认为'avg'。
                'sum'用于SUM指标的检验，此时分母可以省略或为1

            group_buckets : list, 可选
                每个组的流量桶数，仅当metric_type='sum'时有效，是用来做SUM指标检验的
                默认为[1,1]，对应的是实验组：对照组的流量比例

            alpha : float, 可选
                显著性水平，默认为0.05。

            MDE : float, 可选
                最小测试差异，默认为0.005。

            power : float, 可选
                统计功效，默认为0.8。

            X : str, 可选
                用作CUPED方差减少的连续协变量的表达式。
                它遵循回归方法，可以是像'avg(x1)/avg(x2)'，
                'avg(x3)'，'avg(x1)/avg(x2)+avg(x3)'这样的简单形式。


        Returns
        -------

        DataFrame
            一个包含以下列的DataFrame：
            groupname : str
                组的名称。
            numerator : float
                该组分子求和
            denominator : float, 可选
                该组分母求和。仅当metric_type='avg'时出现。
            numerator_pre : float, 可选
                实验前该组分子求和。仅当metric_type='avg'时出现。
            denominator_pre : float, 可选
                实验前该组分母求和。仅当metric_type='avg'时出现。
            mean : float, 可选
                该组度量的平均值。仅当metric_type='avg'时出现。
            std_samp : float, 可选
                该组度量的样本标准差。仅当metric_type='avg'时出现。
            ratio : float, 可选
                分组桶的比率。仅当metric_type='sum'时出现。
            diff_relative : float
                两组之间的相对差异。
            95%_relative_CI : tuple
                相对差异的95%置信区间。
            p-value : float
                计算出的p值。
            t-statistic : float
                计算出的t统计量。
            power : float
                测试的统计功效，即在原假设为假时正确拒绝原假设的概率。功效是基于提供的`mde`（最小可检测效应）、`std_samp`（度量的标准差）和样本大小计算的。
            recommend_samples : int
                推荐的样本大小，以达到检测指定`mde`（最小可检测效应）的所需功效，给定`std_samp`（度量的标准差）。
            mde : float
                设计测试以检测的最小可检测效应大小，基于输入的`power`水平、`std_samp`（度量的标准差）和样本大小计算得出。

        Example:
        ----------

        .. code-block:: python
 
            import fast_causal_inference
            allinsql_provider = fast_causal_inference.FCIProvider(database="all_in_sql_guest")
            # 这是一个模拟的实验明细表！可以用该表来模拟实验分析
            df = allinsql_provider.readClickHouse(f'expt_detail_test_appid')

        >>> # SUM指标检验， metric_type = 'sum', group_buckets需要填写实验组和对照组的流量之比
        >>> df.xexpt_ttest_2samp('numerator', 'denominator', "if(groupid=12343,'B','A')", uin = 'rand()', metric_type = 'sum', group_buckets=[1,1]).show()
        groupname ratio      numerator diff_relative            95%_relative_CI   p-value t-statistic     power recommend_samples       mde
        0         A     1   23058.627723                                                                                                     
        1         B     1  100540.303112   336.020323%  [321.350645%,350.690001%]  0.000000   44.899925  0.050511          23944279  0.209664

        >>> # SUM指标检验 并做CUPED， metric_type = 'sum', group_buckets需要填写实验组和对照组的流量之比
        >>> df.xexpt_ttest_2samp('numerator', 'denominator', "if(groupid=12343,1,0)", uin = 'rand()', metric_type = 'sum', group_buckets=[1,1], X = 'avg(numerator_pre)').show()
        groupname ratio      numerator diff_relative            95%_relative_CI   p-value t-statistic     power recommend_samples       mde
        0         A     1   23058.627723                                                                                                     
        1         B     1  100540.303112   181.046828%  [174.102340%,187.991317%]  0.000000   51.103577  0.052285          13279441  0.099253

        >>> # 均值指标或者比例指标的检验，metric_type = 'avg',
        >>> df.xexpt_ttest_2samp('numerator', 'denominator', "if(groupid=12343,'B','A')", uin = 'rand()', metric_type = 'avg').show()
        groupname   denominator      numerator      mean  std_samp diff_relative            95%_relative_CI   p-value t-statistic      diff  \
        0         A  29023.233157   23058.627723  0.794489  2.879344                                                                            
        1         B  40452.337656  100540.303112  2.485402  5.612429   212.830364%  [204.781181%,220.879546%]  0.000000   51.830152  1.690913   
                        95%_CI     power recommend_samples       mde  
        0                                                             
        1  [1.626963,1.754863]  0.051700          21414774  0.115042  

        >>> # 均值指标检验，因为均值指标分母是1，因此'denominator'可以用'1'代替
        >>> df.xexpt_ttest_2samp('numerator', '1', "if(groupid=12343,'B','A')", uin = 'rand()', metric_type = 'avg').show()
        groupname   denominator      numerator      mean  std_samp diff_relative            95%_relative_CI   p-value t-statistic      diff  \
        0         A  29023.233157   23058.627723  0.794489  2.879344                                                                            
        1         B  40452.337656  100540.303112  2.485402  5.612429   212.830364%  [204.781181%,220.879546%]  0.000000   51.830152  1.690913   
                        95%_CI     power recommend_samples       mde  
        0                                                             
        1  [1.626963,1.754863]  0.051700          21414774  0.115042  

        >>> # 均值指标或者比例指标的检验，CUPED，metric_type = 'avg'
        >>> df.xexpt_ttest_2samp('numerator', 'denominator', "if(groupid=12343,'B','A')", uin = 'rand()', metric_type = 'avg', X = 'avg(numerator_pre)/avg(denominator_pre)').show()
        groupname   denominator      numerator denominator_pre numerator_pre      mean  std_samp diff_relative            95%_relative_CI   p-value  \
        0         A  29023.233157   23058.627723    29131.831739  21903.112431  0.793621  1.257317                                                      
        1         B  40452.337656  100540.303112    30776.559777  23096.875608  2.486223  4.701808   213.275723%  [207.220762%,219.330684%]  0.000000   

        t-statistic      diff               95%_CI     power recommend_samples       mde  
        0                                                                                   
        1   69.044764  1.692601  [1.644548,1.740655]  0.053007          12118047  0.086540  

        >>> # 也可以用df.agg(S.xexpt_ttest_2samp()的形式
        >>> import fast_causal_inference.dataframe.statistics as S
        >>> df.agg(S.xexpt_ttest_2samp('numerator', 'denominator',  "if(groupid=12343,'B','A')", uin = 'rand()', metric_type = 'avg', alpha = 0.05, MDE = 0.005, power = 0.8, X = 'avg(numerator_pre)')).show()
        groupname   denominator      numerator      mean  std_samp diff_relative            95%_relative_CI   p-value t-statistic      diff  \
        0         A  29023.233157   23058.627723  0.980173  2.316314                                                                            
        1         B  40452.337656  100540.303112  2.661489  5.929225   171.532540%  [165.040170%,178.024910%]  0.000000   51.789763  1.681316   
                        95%_CI     power recommend_samples       mde  
        0                                                             
        1  [1.617679,1.744952]  0.052615          13932096  0.092791  

        >>> # 下钻分析，groupBy首次命中日期，去看不同天命中实验的用户的策略效果是否有所不同
        >>> df.groupBy('first_hit_ds').xexpt_ttest_2samp('numerator', 'denominator', "if(groupid=12343,'B','A')", uin = 'rand()', metric_type = 'avg').show()
        first_hit_ds groupname   denominator     numerator       mean  std_samp diff_relative              95%_relative_CI   p-value t-statistic  \
        0     20240102         A  12692.309491  13013.602024   1.025314  1.756287                                                                    
        1     20240102         B   5270.784857   6501.560937   1.233509  2.151984    20.305499%      [13.900858%,26.710141%]  0.000000    6.216446   
        2     20240103         A  11217.541056  -1787.581977  -0.159356  1.248860                                                                    
        3     20240103         B    195.396019     77.324041   0.395730  1.328970  -348.330767%  [-230.430534%,-466.231000%]  0.000000    5.794129   
        4     20240101         A    5113.38261  11832.607676   2.314047  3.443390                                                                    
        5     20240101         B  34986.156779  93961.418134   2.685674  5.632780    16.059621%      [11.247994%,20.871248%]  0.000000    6.543296   
            diff               95%_CI     power recommend_samples        mde  
        0                                                                        
        1  0.208195  [0.142527,0.273863]  0.052688           1765574   0.091511  
        2                                                                        
        3  0.555086  [0.367205,0.742967]  0.050008          22171277  -1.684254  
        4                                                                        
        5  0.371627  [0.260284,0.482970]  0.054768           6616698   0.068761  


    """
    return DfFnColWrapper(
        AggXexptTTest2SampDfFunction(),
        {
            "metric_type": metric_type,
            "group_buckets": str(group_buckets),
            "alpha": alpha,
            "MDE": MDE,
            "power": power,
            "X": X,
        },
        [numerator, denominator, index, uin],
    )


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="SRM")
@register_fn(engine=OlapEngineType.STARROCKS, name="srm")
@define_args(
    FnArg(name="x"),
    FnArg(name="groupby"),
    FnArg(name="ratio", default="[1,1]", is_param=True),
)
@aggregrate
class AggSRMDfFunction(DfFunction):
    def sql_impl_default(
        self,
        ctx: DfContext,
        fn_args: List[FnArg],
        fn_params: List[FnArg],
        arg_dict: Dict,
    ) -> str:
        x = arg_dict["x"].sql(ctx)
        groupby = arg_dict["groupby"].sql(ctx)
        ratio = arg_dict["ratio"].sql(ctx)
        sql = self.fn_name(ctx) + f"({x + ',' + groupby + ',' + ratio})"
        return sql


def srm(x, groupby, ratio="[1,1]"):
    """
    perform srm test


    :param x: column name, the numerator of the metric, can use SQL expression, the column must be numeric.
        If you are concerned about whether the sum of x1 meets expectations, you should fill in x1, then it will calculate sum(x1);
        If you are concerned about whether the sample size meets expectations, you should fill in 1, then it will calculate sum(1).
    :type x: str, required
    :param groupby: column name, representing the field for aggregation grouping, can support Integer/String.
    :type groupby: str, required
    :param ratio: list. The expected traffic ratio, needs to be filled in according to the order of the groupby field. Each value must be >0. For example, [1,1,2] represents the expected ratio is 1:1:2.
    :type ratio: list, required

    :return: DataFrame contains the following columns:
    groupname: the name of the group.
    f_obs: the observed traffic.
    ratio: the expected traffic ratio.
    chisquare: the calculated chi-square.
    p-value: the calculated p-value.

    Example:
    ----------------

    .. code-block:: python

        import fast_causal_inference.dataframe.statistics as S
    >>> df.srm('x1', 'treatment', '[1,2]').show()
            groupname   f_obs       ratio       chisquare   p-value
    0           23058.627723 1.000000    48571.698643 0.000000
    1           1.0054e+05  1.000000

    >>> df.agg(S.srm('x1', 'treatment', '[1,2]')).show()
            groupname   f_obs       ratio       chisquare   p-value
    0           23058.627723 1.000000    48571.698643 0.000000
    1           1.0054e+05  1.000000
    """
    return DfFnColWrapper(AggSRMDfFunction(), {"ratio": ratio}, [x, groupby])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="mannWhitneyUTest")
@register_fn(engine=OlapEngineType.STARROCKS, name="mann_whitney_u_test")
@define_args(
    FnArg(name="alternative", is_param=True, default="two-sided"),
    FnArg(name="continuity_correction", is_param=True, default=1),
    FnArg(name="sample_data"),
    FnArg(name="sample_index"),
)
@aggregrate
class AggMannWhitneyUTestDfFunction(DfFunction):
    def sql_impl_clickhouse(
        self,
        ctx: DfContext,
        fn_args: List[FnArg],
        fn_params: List[FnArg],
        arg_dict: Dict,
    ) -> str:
        alternative = f"'{arg_dict['alternative'].sql(ctx)}'"
        continuity_correction = arg_dict["continuity_correction"].sql(ctx)
        sample_data = arg_dict["sample_data"].sql(ctx)
        sample_index = arg_dict["sample_index"].sql(ctx)
        sql = (
            self.fn_name(ctx)
            + f"({alternative}, {continuity_correction})({sample_data}, {sample_index})"
        )
        return sql

    def sql_impl_starrocks(
        self,
        ctx: DfContext,
        fn_args: List[FnArg],
        fn_params: List[FnArg],
        arg_dict: Dict,
    ) -> str:
        alternative = f"'{arg_dict['alternative'].sql(ctx)}'"
        continuity_correction = arg_dict["continuity_correction"].sql(ctx)
        sample_data = arg_dict["sample_data"].sql(ctx)
        sample_index = arg_dict["sample_index"].sql(ctx)
        sql = (
            self.fn_name(ctx)
            + f"({sample_data}, {sample_index}, {alternative}, {continuity_correction})"
        )
        return sql


def mann_whitney_utest(
    sample_data, sample_index, alternative="two-sided", continuity_correction=1
):
    """
    This function is used to calculate the Mann-Whitney U test. It returns the calculated U-statistic and the two-tailed p-value.

    :param sample_data: column name, the numerator of the metric, can use SQL expression, the column must be numeric.
    :type sample_data: str, required
    :param sample_index: column name, the index to represent the control group and the experimental group, 1 for the experimental group and 0 for the control group.
    :type sample_index: str, required
    :param alternative:
        'two-sided': the default value, two-sided test.
        'greater': one-tailed test in the positive direction.
        'less': one-tailed test in the negative direction.
    :type alternative: str, optional
    :param continuous_correction: bool, default 1, whether to apply continuity correction.
    :type continuous_correction: bool, optional

    :return: Tuple with two elements:
    U-statistic: Float64.
    p-value: Float64.

    Example:
    ----------------

    .. code-block:: python

        import fast_causal_inference
        allinsql_provider = fast_causal_inference.FCIProvider(database="all_in_sql_guest")
        # 这是一个模拟的实验明细表！可以用该表来模拟实验分析

        df = allinsql_provider.readClickHouse(f'expt_detail_test_appid')

    >>> # 返回的结果是 [U统计量,P值]
    >>> df.mann_whitney_utest('numerator', 'if(groupid=12343,1,0)').show()
    [2349578.0, 0.0]
    >>> df.agg(S.mann_whitney_utest('numerator', 'if(groupid=12343,1,0)')).show()
    [2349578.0, 0.0]

    """
    return DfFnColWrapper(
        AggMannWhitneyUTestDfFunction(),
        {"alternative": alternative, "continuity_correction": continuity_correction},
        [sample_data, sample_index],
    )


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="kolmogorovSmirnovTest")
@define_args(FnArg(name="sample_data"), FnArg(name="sample_index"))
@aggregrate
class AggKolmogorovSmirnovTestDfFunction(DfFunction):
    pass


def kolmogorov_smirnov_test(sample_data, sample_index):
    """
    This function is used to calculate the Kolmogorov-Smirnov test for goodness of fit. It returns the calculated statistic and the two-tailed p-value.


    :param sample_data: Sample data. Integer, Float or Decimal.
    :type sample_data: int, float or decimal, required
    :param sample_index: Sample index. Integer.
    :type sample_index: int, required

    :return: Tuple with two elements:
    calculated statistic: Float64.
    calculated p-value: Float64.


    Example:
    ----------------

    .. code-block:: python

        import fast_causal_inference
        import fast_causal_inference.dataframe.statistics as S
        allinsql_provider = fast_causal_inference.FCIProvider(database="all_in_sql_guest")
        df = allinsql_provider.readClickHouse(f'test_data_small')

    >>> df.kolmogorov_smirnov_test('y', 'treatment').show()
    [0.6382961593945475, 0.0]
    >>> df.agg(S.kolmogorov_smirnov_test('y', 'treatment')).show()
    [0.6382961593945475, 0.0]

    """
    return DfFnColWrapper(
        AggKolmogorovSmirnovTestDfFunction(), {}, [sample_data, sample_index]
    )


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="studentTTest")
@define_args(FnArg(name="sample_data"), FnArg(name="sample_index"))
@aggregrate
class AggStudentTTestDfFunction(DfFunction):
    pass


def student_ttest(sample_data, sample_index):
    """
    This function is used to calculate the t-test for the mean of one group of scores. It returns the calculated t-statistic and the two-tailed p-value.

    :param sample_data: column name, the numerator of the metric, can use sql expression, the column must be numeric
    :type sample_data: str, required
    :param sample_index: column name, the index to represent the control group and the experimental group, 1 for the experimental group and 0 for the control group
    :type sample_index: str, required

    :return: Tuple with two elements:
    calculated statistic: Float64.
    calculated p-value: Float64.

    Example
    ----------------

    .. code-block:: python

        import fast_causal_inference
        import fast_causal_inference.dataframe.statistics as S
        allinsql_provider = fast_causal_inference.FCIProvider(database="all_in_sql_guest")
        df = allinsql_provider.readClickHouse(f'test_data_small')

    >>> df.student_ttest('y', 'treatment').show()
    [-72.8602591880598, 0.0]

    >>> df.agg(S.student_ttest('y', 'treatment')).show()
    [-72.8602591880598, 0.0]

    """
    return DfFnColWrapper(AggStudentTTestDfFunction(), {}, [sample_data, sample_index])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="welchTTest")
@define_args(FnArg(name="sample_data"), FnArg(name="sample_index"))
@aggregrate
class AggWelchTTestDfFunction(DfFunction):
    pass


def welch_ttest(sample_data, sample_index):
    """
    This function is used to calculate welch's t-test for the mean of two independent samples of scores. It returns the calculated t-statistic and the two-tailed p-value.

    :param sample_data: column name, the numerator of the metric, can use sql expression, the column must be numeric
    :type sample_data: str, required
    :param sample_index: column name, the index to represent the control group and the experimental group, 1 for the experimental group and 0 for the control group
    :type sample_index: str, required

    :return: Tuple with two elements:
    calculated statistic: Float64.
    calculated p-value: Float64.

    Example
    ----------------

    .. code-block:: python

        import fast_causal_inference
        import fast_causal_inference.dataframe.statistics as S
        allinsql_provider = fast_causal_inference.FCIProvider(database="all_in_sql_guest")
        df = allinsql_provider.readClickHouse(f'test_data_small')

    >>> df.welch_ttest('y', 'treatment').show()
    [-73.78492246858345, 0.0]
    >>> df.agg(S.welch_ttest('y', 'treatment')).show()
    [-73.78492246858345, 0.0]

    """
    return DfFnColWrapper(AggWelchTTestDfFunction(), {}, [sample_data, sample_index])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="meanZTest")
@define_args(
    FnArg(name="sample_data"),
    FnArg(name="sample_index"),
    FnArg(name="population_variance_x", is_param=True),
    FnArg(name="population_variance_y", is_param=True),
    FnArg(name="confidence_level", is_param=True),
)
@aggregrate
class AggMeanZTestDfFunction(DfFunction):
    pass


def mean_z_test(
    sample_data,
    sample_index,
    population_variance_x,
    population_variance_y,
    confidence_level,
):
    """
    This function is used to calculate the z-test for the mean of two independent samples of scores. It returns the calculated z-statistic and the two-tailed p-value.

    :param sample_data: column name, the numerator of the metric, can use sql expression, the column must be numeric
    :type sample_data: str, required
    :param sample_index: column name, the index to represent the control group and the experimental group, 1 for the experimental group and 0 for the control group
    :type sample_index: str, required
    :param population_variance_x: Variance for control group.
    :type population_variance_x: Float, required
    :param population_variance_y: Variance for experimental group.
    :type population_variance_y: Float, required
    :param confidence_level: Confidence level in order to calculate confidence intervals.
    :type confidence_level: Float, required

    :return:

    Example
    ----------------

    .. code-block:: python

        import fast_causal_inference
        import fast_causal_inference.dataframe.statistics as S
        allinsql_provider = fast_causal_inference.FCIProvider(database="all_in_sql_guest")
        df = allinsql_provider.readClickHouse(f'test_data_small')
        df.mean_z_test('y', 'treatment', 0.9, 0.9, 0.95).show()
        df.agg(S.mean_z_test('y', 'treatment', 0.9, 0.9, 0.95)).show()


    """
    return DfFnColWrapper(
        AggMeanZTestDfFunction(),
        {
            "population_variance_x": population_variance_x,
            "population_variance_y": population_variance_y,
            "confidence_level": confidence_level,
        },
        [sample_data, sample_index],
    )


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="bootStrap")
@define_args(FnArg(name="func"), FnArg(name="resample_frac"), FnArg(name="n_resamples"))
@aggregrate
class BootStrapDfFunction(DfFunction):
    def sql_impl_clickhouse(
        self,
        ctx: DfContext,
        fn_args: List[FnArg],
        fn_params: List[FnArg],
        arg_dict: Dict,
    ) -> str:
        func = arg_dict['func'].sql(ctx)
        resample_frac = float(arg_dict['resample_frac'].sql(ctx))
        if not (resample_frac > 0 and resample_frac <= 1):
            raise Exception(f"resample_frac({resample_frac}) must be between (0, 1].")
        n_resamples = arg_dict['n_resamples'].sql(ctx)
        df = ctx.dataframe
        count = df.count()
        n_samples = max(1, int(resample_frac * count))
        return f"{self.fn_name(ctx)}({func}, {n_samples}, {n_resamples})"


def boot_strap(func, resample_frac=1, n_resamples=100):
    """
    Compute a two-sided bootstrap confidence interval of a statistic.
    boot_strap sample_num samples from data and compute the func.

    :param func: function to apply.
    :type func: str, required
    :param resample_frac: ratio of samples.
    :type resample_frac: float, not required
    :param n_resamples: number of bootstrap samples.
    :type n_resamples: int, not required

    :return: list of calculated statistics.
    :type return: Array(Float64).

    Example
    ----------------

    .. code-block:: python

        import fast_causal_inference
        import fast_causal_inference.dataframe.statistics as S
        allinsql_provider = fast_causal_inference.FCIProvider(database="all_in_sql_guest")
        df = allinsql_provider.readClickHouse(f'test_data_small')
        df.boot_strap(func='avg(x1)', resample_frac=1, n_resamples=100).show()
        df.agg(S.boot_strap(func="ttest_1samp(avg(x1), 'two-sided',0)", resample_frac=0.5, n_resamples=100)). show()
        df.agg(S.boot_strap(func="ttest_2samp(avg(x1), treatment, 'two-sided')", resample_frac=1, n_resamples=100)). show()

    """
    return DfFnColWrapper(
        BootStrapDfFunction(),
        {},
        ["'" + func.replace("'", "@") + "'", resample_frac, n_resamples],
    )


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="Permutation")
@define_args(
    FnArg(name="func"),
    FnArg(name="permutation_num"),
    FnArg(name="mde", default=""),
    FnArg(name="mde_type", default=""),
)
@aggregrate
class PermutationDfFunction(DfFunction):
    pass


def permutation(func, permutation_num, mde="0", mde_type="1"):
    """
    :param func: function to apply.
    :type func: str, required
    :param permutation_num: number of permutations.
    :type permutation_num: int, required
    :param col: columns to apply function to.
    :type col: int, float or decimal, required

    :return: list of calculated statistics.
    :type return: Array(Float64).

    Example
    ----------------

    .. code-block:: python

        import fast_causal_inference
        import fast_causal_inference.dataframe.statistics as S
        allinsql_provider = fast_causal_inference.FCIProvider(database="all_in_sql_guest")
        df = allinsql_provider.readClickHouse(f'test_data_small')
        df.permutation('mannWhitneyUTest', 3, 'x1')
        df.agg(S.permutation('mannWhitneyUTest', 3, 'x1')).show()

    """
    return DfFnColWrapper(
        PermutationDfFunction(),
        {},
        ["'" + func.replace("'", "@") + "'", permutation_num, mde, mde_type],
    )


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="MatrixMultiplication")
@register_fn(engine=OlapEngineType.STARROCKS, name="matrix_multiplication")
@define_args(
    FnArg(name="std", default="False", is_param=True),
    FnArg(name="invert", default="False", is_param=True),
    FnArg(name="col", is_variadic=True),
)
@aggregrate
class AggMatrixMultiplicationDfFunction(DfFunction):
    def sql_impl_clickhouse(
        self,
        ctx: DfContext,
        fn_args: List[FnArg],
        fn_params: List[FnArg],
        arg_dict: Dict,
    ) -> str:
        col = ", ".join(map(lambda x: x.sql(ctx), arg_dict["col"].column))
        std = arg_dict["std"].sql(ctx)
        invert = arg_dict["invert"].sql(ctx)
        sql = self.fn_name(ctx) + f"({std}, {invert})" + f"({col})"
        return sql

    def sql_impl_starrocks(
        self,
        ctx: DfContext,
        fn_args: List[FnArg],
        fn_params: List[FnArg],
        arg_dict: Dict,
    ) -> str:
        col = ", ".join(map(lambda x: x.sql(ctx), arg_dict["col"].column))
        std = arg_dict["std"].sql(ctx)
        invert = arg_dict["invert"].sql(ctx)
        sql = self.fn_name(ctx) + f"([{col}], {std}, {invert})"
        return sql


def matrix_multiplication(*col, std=False, invert=False):
    """
    :param col: columns to apply function to.
    :type col: int, float or decimal, required
    :param std: whether to return standard deviation.
    :type std: bool, required
    :param invert: whether to invert the matrix.
    :type invert: bool, required

    :return: list of calculated statistics.
    :type return: Array(Float64).

    Example
    ----------------

    .. code-block:: python

        import fast_causal_inference
        import fast_causal_inference.dataframe.statistics as S
        allinsql_provider = fast_causal_inference.FCIProvider(database="all_in_sql_guest")
        df = allinsql_provider.readClickHouse(f'test_data_small')
        df.matrix_multiplication('x1', 'x2', std = False, invert = False).show()
        df.agg(S.matrix_multiplication('x1', 'x2', std = False, invert = False)).show()
        df.agg(S.matrix_multiplication('x1', 'x2', std = True, invert = True)).show()
    """

    return DfFnColWrapper(
        AggMatrixMultiplicationDfFunction(), {"std": std, "invert": invert}, col
    )




import scipy.stats as stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns
from matplotlib import rcParams
import warnings

def ttest_1samp_py(result):
    # Compute the ATE
    ATE = np.mean(result)

    # Compute standard deviation
    std = np.std(result)

    # Compute t-value
    t_value = ATE / std

    # Compute p-value
    p_value = (1 - stats.t.cdf(abs(t_value), len(result) - 1)) * 2

    # Compute 95% confidence interval
    confidence_interval = [ATE - 1.96 * std, ATE + 1.96 * std]

    # Return the results
    return {
        "ATE": ATE,
        "stddev": std,
        "p_value": p_value,
        "95% confidence_interval": confidence_interval,
    }

def ATE_estimator_base(table, Y, T, B=500):
    sql_instance = create_sql_instance()
    n = int(sql_instance.sql(f"select count(*) as cnt from {table}")["cnt"][0])
    res = (
        sql_instance.sql(
            f"""WITH (
      SELECT DistributedNodeRowNumber(1)(0)
      FROM {table}
    ) AS pa
    SELECT
      BootStrapMulti('sum:1;sum:1;sum:1;sum:1',  {n}, {B}, pa)(
      {Y}*{T},{T},{Y}*(1-{T}),(1-{T})) as res
    FROM 
    {table}
    ;
    """
        )["res"][0]
        .replace("]", "")
        .replace(" ", "")
        .split("[")
    )

    # Process the query results
    res = [i.split(",") for i in res if i != ""]
    res = np.array([[float(j) for j in i if j != ""] for i in res])
    return res

def IPW_estimator_base(table, Y, T, P, B=500):
    sql_instance = create_sql_instance()
    n = int(sql_instance.sql(f"select count(*) as cnt from {table}")["cnt"][0])
    res = (
        sql_instance.sql(
            f"""WITH (
      SELECT DistributedNodeRowNumber(1)(0)
      FROM {table}
    ) AS pa
    SELECT
      BootStrapMulti('sum:1;sum:1;sum:1;sum:1',  {n}, {B}, pa)(
      {Y}*{T}/({P}+0.01), {T}/({P}+0.01), {Y}*(1-{T})/(1-{P}+0.01), (1-{T})/(1-{P}+0.01)) as res
    FROM 
    {table}
    ;
    """
        )["res"][0]
        .replace("]", "")
        .replace(" ", "")
        .split("[")
    )

    # Process the query results
    res = [i.split(",") for i in res if i != ""]
    res = np.array([[float(j) for j in i if j != ""] for i in res])
    return res

def IPWestimator(df, Y, T, P, B=500):
    """
    Estimate the Average Treatment Effect (ATE) using Inverse Probability of Treatment Weighting (IPTW).

    :param table: the name of the input data table.
    :type table: str, required
    :param Y: the column name of the outcome variable.
    :type Y: str, required
    :param T: the column name of the treatment variable.
    :type T: str, required
    :param P: the column name of the propensity score.
    :type P: str, required
    :param B: the number of bootstrap samples, default is 500.
    :type B: int, optional

    :return: dict, containing the following key-value pairs:
    'ATE': Average Treatment Effect.
    'stddev': Standard deviation.
    'p_value': p-value.
    '95% confidence_interval': 95% confidence interval.

    Example
    ----------

    .. code-block:: python

        import fast_causal_inference
        allinsql_provider = fast_causal_inference.FCIProvider(database="all_in_sql_guest")
        df = allinsql_provider.readClickHouse(f'test_data_small')

        Y = 'numerator'
        T = 'treatment'
        P = 'weight'
        import fast_causal_inference.dataframe.statistics as S
        S.IPWestimator(df,Y,T,P,B=500)
    """
    table = df_2_table(df)
    # Create SQL instance
    sql_instance = create_sql_instance()

    # Get the number of rows in the table
    
    table0 = f'{table}  where {T} = 0'
    table1 = f'{table}  where {T} = 1'

    res1 = IPW_estimator_base(table1, Y, T, P, B)
    res0 = IPW_estimator_base(table0, Y, T, P, B)
    
    # Calculate IPTW estimates
    result = res1[0, :] / res1[1, :] - res0[2, :] / res0[3, :]
    ATE = np.mean(result)

    ttest_res = ttest_1samp_py(result)
    return ttest_res


def ATEestimator(df, Y, T, B=500):
    """
    Estimate the Average Treatment Effect (ATE) using a simple difference in means approach.

    :param table: the name of the input data table.
    :type table: str, required
    :param Y: the column name of the outcome variable.
    :type Y: str, required
    :param T: the column name of the treatment variable.
    :type T: str, required
    :param B: the number of bootstrap samples, default is 500.
    :type B: int, optional

    Return
    ----------  

    dict, containing the following key-value pairs:
    'ATE': Average Treatment Effect.
    'stddev': Standard deviation.
    'p_value': p-value.
    '95% confidence_interval': 95% confidence interval.

    Example
    ----------

    .. code-block:: python
    
        import fast_causal_inference
        allinsql_provider = fast_causal_inference.FCIProvider(database="all_in_sql_guest")
        df = allinsql_provider.readClickHouse(f'test_data_small')

        Y = 'numerator'
        T = 'treatment'
        import fast_causal_inference.dataframe.statistics as S
        S.ATEestimator(df,Y,T,B=500)

    """
    table = df_2_table(df)
    # Create SQL instance
    sql_instance = create_sql_instance()
 
    table1 = f'{table}  where {T} = 1'
    table0 = f'{table}  where {T} = 0'

    res1 = ATE_estimator_base(table1, Y, T, B)
    res0 = ATE_estimator_base(table0, Y, T, B)
    
    # Calculate IPTW estimates
    result = res1[0, :] / res1[1, :] - res0[2, :] / res0[3, :]
    ATE = np.mean(result)

    ttest_res = ttest_1samp_py(result)
    
    return ttest_res

    
