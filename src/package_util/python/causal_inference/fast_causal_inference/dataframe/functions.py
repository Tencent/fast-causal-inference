import copy
from typing import List, Dict

from fast_causal_inference.dataframe import ais_dataframe_pb2 as DfPb
from fast_causal_inference.dataframe.df_base import (
    DfColumnInternalNode,
    DfColumnLeafNode,
    aggregrate,
    register_fn,
    OlapEngineType,
    define_args,
    DfFunction,
    DfColumnNode,
    FnArg,
    DfContext,
)


class DfFnColWrapper:
    def __init__(self, fn: DfFunction, params: Dict, columns: List[DfColumnNode]):
        self._fn = fn
        self._params = copy.deepcopy(params)
        self._columns = copy.deepcopy(columns)

    @property
    def fn(self):
        return self._fn

    @property
    def params(self):
        return self._params

    @property
    def columns(self):
        return self._columns

    def has_agg_func(self):
        if self._fn.is_agg_func():
            return True
        for col in self._columns:
            if isinstance(col, DfFnColWrapper) and col.has_agg_func():
                return True
        return False

    def alias(self, alias_):
        self._fn.alias = alias_
        return self

    def __add__(self, rhs):
        return add(self, rhs)

    def __radd__(self, lhs):
        return add(lhs, self)

    def __sub__(self, rhs):
        return subtract(self, rhs)

    def __rsub__(self, lhs):
        return subtract(lhs, self)

    def __mul__(self, rhs):
        return multiply(self, rhs)

    def __rmul__(self, lhs):
        return multiply(lhs, self)

    def __truediv__(self, rhs):
        return divide(self, rhs)

    def __rtruediv__(self, lhs):
        return divide(lhs, self)

    def __mod__(self, rhs):
        return modulo(self, rhs)

    def __rmod__(self, lhs):
        return modulo(lhs, self)

    def __eq__(self, rhs):
        return eq(self, rhs)

    def __ne__(self, rhs):
        return ne(self, rhs)

    def __lt__(self, rhs):
        return lt(self, rhs)

    def __le__(self, rhs):
        return le(self, rhs)

    def __gt__(self, rhs):
        return gt(self, rhs)

    def __ge__(self, rhs):
        return ge(self, rhs)


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="")
@register_fn(engine=OlapEngineType.STARROCKS, name="")
@define_args(FnArg(name="x"), FnArg(name="y"))
class AddDfFunction(DfFunction):
    def sql_impl_default(
        self,
        ctx: DfContext,
        fn_args: List[FnArg],
        fn_params: List[FnArg],
        arg_dict: Dict,
    ) -> str:
        return "(" + arg_dict["x"].sql(ctx) + " + " + arg_dict["y"].sql(ctx) + ")"


def add(x, y):
    return DfFnColWrapper(AddDfFunction(), {}, [x, y])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="")
@register_fn(engine=OlapEngineType.STARROCKS, name="")
@define_args(FnArg(name="x"), FnArg(name="y"))
class SubtractDfFunction(DfFunction):
    def sql_impl_default(
        self,
        ctx: DfContext,
        fn_args: List[FnArg],
        fn_params: List[FnArg],
        arg_dict: Dict,
    ) -> str:
        return "(" + arg_dict["x"].sql(ctx) + " - " + arg_dict["y"].sql(ctx) + ")"


def subtract(x, y):
    return DfFnColWrapper(SubtractDfFunction(), {}, [x, y])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="")
@register_fn(engine=OlapEngineType.STARROCKS, name="")
@define_args(FnArg(name="x"), FnArg(name="y"))
class MultiplyDfFunction(DfFunction):
    def sql_impl_default(
        self,
        ctx: DfContext,
        fn_args: List[FnArg],
        fn_params: List[FnArg],
        arg_dict: Dict,
    ) -> str:
        return "(" + arg_dict["x"].sql(ctx) + " * " + arg_dict["y"].sql(ctx) + ")"


def multiply(x, y):
    return DfFnColWrapper(MultiplyDfFunction(), {}, [x, y])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="")
@register_fn(engine=OlapEngineType.STARROCKS, name="")
@define_args(FnArg(name="x"), FnArg(name="y"))
class DivideDfFunction(DfFunction):
    def sql_impl_default(
        self,
        ctx: DfContext,
        fn_args: List[FnArg],
        fn_params: List[FnArg],
        arg_dict: Dict,
    ) -> str:
        return "(" + arg_dict["x"].sql(ctx) + " / " + arg_dict["y"].sql(ctx) + ")"


def divide(x, y):
    return DfFnColWrapper(DivideDfFunction(), {}, [x, y])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="")
@register_fn(engine=OlapEngineType.STARROCKS, name="")
@define_args(FnArg(name="x"), FnArg(name="y"))
class ModuloDfFunction(DfFunction):
    def sql_impl_default(
        self,
        ctx: DfContext,
        fn_args: List[FnArg],
        fn_params: List[FnArg],
        arg_dict: Dict,
    ) -> str:
        return "(" + arg_dict["x"].sql(ctx) + " % " + arg_dict["y"].sql(ctx) + ")"


def modulo(x, y):
    return DfFnColWrapper(ModuloDfFunction(), {}, [x, y])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="")
@register_fn(engine=OlapEngineType.STARROCKS, name="")
@define_args(FnArg(name="self"))
class SelfRefDfFunction(DfFunction):
    def sql_impl_default(
        self,
        ctx: DfContext,
        fn_args: List[FnArg],
        fn_params: List[FnArg],
        arg_dict: Dict,
    ) -> str:
        return "(" + arg_dict["self"].sql(ctx) + ")"


def col(*cols):
    if len(cols) != 1:
        raise Exception(f"number of columns({len(cols)}) is not 1.")
    return DfFnColWrapper(SelfRefDfFunction(), {}, cols)

@register_fn(engine=OlapEngineType.CLICKHOUSE, name="")
@register_fn(engine=OlapEngineType.STARROCKS, name="")
@define_args(FnArg(name="col"))
class LitDfFunction(DfFunction):
    def sql_impl_default(
        self,
        ctx: DfContext,
        fn_args: List[FnArg],
        fn_params: List[FnArg],
        arg_dict: Dict,
    ) -> str:
        col = arg_dict["col"].column
        if isinstance(col, DfColumnNode):
            return arg_dict["col"].sql(ctx)
        else:
            raise Exception(
                "Logical Error: col can only be DfColumnLeafNode|DfColumnInternalNode."
            )


def lit(col):
    """
    lit is used to create a constant column.

    >>> import fast_causal_inference.dataframe.functions as Fn
    >>> df_new = df.withColumn('constant', Fn.lit(1))

    """
    if isinstance(col, str):
        col = "'" + col + "'"
    return DfFnColWrapper(LitDfFunction(), {}, [col])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="any")
@register_fn(engine=OlapEngineType.STARROCKS, name="any")
@define_args(FnArg(name="col"))
@aggregrate
class AggAnyDfFunction(DfFunction):
    pass


def any(col):
    """
    any is used to aggregate a column with any value.
    """
    return DfFnColWrapper(AggAnyDfFunction(), {}, [col])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="stddevPop")
@register_fn(engine=OlapEngineType.STARROCKS, name="stddev_pop")
@define_args(FnArg(name="col"))
@aggregrate
class AggStddevPopDfFunction(DfFunction):
    pass


def stddevPop(col):
    """
    stddevPop is used to calculate the population standard deviation of a column.

     Example:
    ----------

    .. code-block:: python

        import fast_causal_inference.dataframe.functions as Fn
        df = fast_causal_inference.readClickHouse('test_data_small')
        df.stddevPop('numerator').show()
        df.groupBy('treatment').stddevPop('numerator').show()
        df.groupBy('treatment').agg(Fn.stddevPop('numerator').alias('numerator')).show()
        df.groupBy('treatment').agg({'numerator':'stddevPop', 'numerator_pre':'stddevPop'}).show()

    """
    return DfFnColWrapper(AggStddevPopDfFunction(), {}, [col])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="stddevSamp")
@register_fn(engine=OlapEngineType.STARROCKS, name="stddev_samp")
@define_args(FnArg(name="col"))
@aggregrate
class AggStddevSampDfFunction(DfFunction):
    pass


def stddevSamp(col):
    """
    stddevSamp is used to calculate the sample standard deviation of a column.

    Example:
    ----------

    .. code-block:: python

        import fast_causal_inference.dataframe.functions as Fn
        df = fast_causal_inference.readClickHouse('test_data_small')
        df.stddevSamp('numerator').show()
        df.groupBy('treatment').stddevSamp('numerator').show()
        df.groupBy('treatment').agg(Fn.stddevSamp('numerator').alias('numerator')).show()
        df.groupBy('treatment').agg({'numerator':'stddevSamp', 'numerator_pre':'stddevSamp'}).show()
    """
    return DfFnColWrapper(AggStddevSampDfFunction(), {}, [col])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="varPop")
@register_fn(engine=OlapEngineType.STARROCKS, name="var_pop")
@define_args(FnArg(name="col"))
@aggregrate
class AggVarPopDfFunction(DfFunction):
    pass


def varPop(col):
    """
    varPop is used to calculate the population variance of a column.

    Example:
    ----------

    .. code-block:: python

        import fast_causal_inference.dataframe.functions as Fn
        df = fast_causal_inference.readClickHouse('test_data_small')
        df.varPop('numerator').show()
        df.groupBy('treatment').agg(Fn.varPop('numerator').alias('numerator')).show()
        df.groupBy('treatment').varPop('numerator').show()
        df.groupBy('treatment').agg({'numerator':'varPop', 'numerator_pre':'varPop'}).show()
    """
    return DfFnColWrapper(AggVarPopDfFunction(), {}, [col])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="varSamp")
@register_fn(engine=OlapEngineType.STARROCKS, name="var_samp")
@define_args(FnArg(name="col"))
@aggregrate
class AggVarSampDfFunction(DfFunction):
    pass


def varSamp(col):
    """
    varSamp is used to calculate the sample variance of a column.

    Example:
    ----------

    .. code-block:: python

        import fast_causal_inference.dataframe.functions as Fn
        df = fast_causal_inference.readClickHouse('test_data_small')
        df.varSamp('numerator').show()
        df.groupBy('treatment').agg(Fn.varSamp('numerator').alias('numerator')).show()
        df.groupBy('treatment').varSamp('numerator').show()
        df.groupBy('treatment').agg({'numerator':'varSamp', 'numerator_pre':'varSamp'}).show()

    """
    return DfFnColWrapper(AggVarSampDfFunction(), {}, [col])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="corr")
@register_fn(engine=OlapEngineType.STARROCKS, name="corr")
@define_args(FnArg(name="x"), FnArg(name="y"))
@aggregrate
class AggCorrDfFunction(DfFunction):
    pass


def corr(x, y):
    """
    corr is used to calculate the correlation between two columns.

    Example:
    ----------

    .. code-block:: python

        import fast_causal_inference.dataframe.functions as Fn
        df = fast_causal_inference.readClickHouse('test_data_small')
        df.corr('numerator', 'numerator_pre').show()
        df.groupBy('treatment').agg(Fn.corr('numerator', 'numerator_pre').alias('numerator')).show()
        df.groupBy('treatment').corr('numerator', 'numerator_pre').show()


    """
    return DfFnColWrapper(AggCorrDfFunction(), {}, [x, y])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="count")
@register_fn(engine=OlapEngineType.STARROCKS, name="count")
@define_args(FnArg(name="expr", is_param=True))
@aggregrate
class AggCountDfFunction(DfFunction):
    pass


def count(*, expr="*"):
    """
    count is used to count the number of rows.

    Example:
    ----------

    .. code-block:: python

        import fast_causal_inference.dataframe.functions as Fn
        df = fast_causal_inference.readClickHouse('test_data_small')
        df.count().show()
        df.groupBy('treatment').count().show()
        df.groupBy('treatment').agg(Fn.count().alias('numerator')).show()
    """
    return DfFnColWrapper(AggCountDfFunction(), {"expr": expr}, [])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="max")
@register_fn(engine=OlapEngineType.STARROCKS, name="max")
@define_args(FnArg(name="lhs"), FnArg(name="rhs"))
class MaxDfFunction(DfFunction):
    pass


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="max")
@register_fn(engine=OlapEngineType.STARROCKS, name="max")
@define_args(FnArg(name="x"))
@aggregrate
class AggMaxDfFunction(DfFunction):
    pass


def max(*cols):
    """
    max is used to calculate the maximum value of a column.

    Example:
    ----------

    .. code-block:: python

        import fast_causal_inference.dataframe.functions as Fn
        df = fast_causal_inference.readClickHouse('test_data_small')
        df.max('numerator').show()
        df.groupBy('treatment').max('numerator').show()
        df.groupBy('treatment').agg(Fn.max('numerator').alias('numerator')).show()
    """
    if len(cols) == 1:
        return DfFnColWrapper(AggMaxDfFunction(), {}, cols)
    if len(cols) == 2:
        return DfFnColWrapper(MaxDfFunction(), {}, cols)
    raise Exception(f"number of columns({len(cols)}) is neither 1 nor 2.")


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="min")
@register_fn(engine=OlapEngineType.STARROCKS, name="min")
@define_args(FnArg(name="lhs"), FnArg(name="rhs"))
class MinDfFunction(DfFunction):
    pass


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="min")
@register_fn(engine=OlapEngineType.STARROCKS, name="min")
@define_args(FnArg(name="x"))
@aggregrate
class AggMinDfFunction(DfFunction):
    pass


def min(*cols):
    """
    min is used to calculate the minimum value of a column.

    Example:
    ----------

    .. code-block:: python

        import fast_causal_inference.dataframe.functions as Fn
        df = fast_causal_inference.readClickHouse('test_data_small')
        df.min('numerator').show()
        df.groupBy('treatment').min('numerator').show()
        df.groupBy('treatment').agg(Fn.min('numerator').alias('numerator')).show()
    """
    if len(cols) == 1:
        return DfFnColWrapper(AggMinDfFunction(), {}, cols)
    if len(cols) == 2:
        return DfFnColWrapper(MinDfFunction(), {}, cols)
    raise Exception(f"number of columns({len(cols)}) is neither 1 nor 2.")


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="avg")
@register_fn(engine=OlapEngineType.STARROCKS, name="avg")
@define_args(FnArg(name="x"))
@aggregrate
class AggAvgDfFunction(DfFunction):
    pass


def avg(col1):
    """
    avg is used to calculate the average value of a column.

    Example:
    ----------

    .. code-block:: python

        import fast_causal_inference.dataframe.functions as Fn
        df = fast_causal_inference.readClickHouse('test_data_small')
        df.avg('numerator').show()
        df.groupBy('treatment').avg('numerator').show()
        df.groupBy('treatment').agg(Fn.avg('numerator').alias('numerator')).show()

    """
    return DfFnColWrapper(AggAvgDfFunction(), {}, [col1])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="sum")
@register_fn(engine=OlapEngineType.STARROCKS, name="sum")
@define_args(FnArg(name="x"))
@aggregrate
class AggSumDfFunction(DfFunction):
    pass


def sum(col1):
    """
    sum is used to calculate the sum of a column.

    Example:
    ----------

    .. code-block:: python

        import fast_causal_inference.dataframe.functions as Fn
        df = fast_causal_inference.readClickHouse('test_data_small')
        df.sum('numerator').show()
        df.groupBy('treatment').sum('numerator').show()
        df.groupBy('treatment').agg(Fn.sum('numerator').alias('numerator')).show()
    """
    return DfFnColWrapper(AggSumDfFunction(), {}, [col1])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="mean")
@register_fn(engine=OlapEngineType.STARROCKS, name="mean")
@define_args(FnArg(name="x"))
@aggregrate
class AggMeanDfFunction(DfFunction):
    pass


def mean(col1):
    """
    mean is used to calculate the mean of a column.

    Example:
    ----------

    .. code-block:: python

        import fast_causal_inference.dataframe.functions as Fn
        df = fast_causal_inference.readClickHouse('test_data_small')
        df.mean('numerator').show()
        df.groupBy('treatment').mean('numerator').show()
        df.groupBy('treatment').agg(Fn.mean('numerator').alias('numerator')).show()
    """
    return DfFnColWrapper(AggMeanDfFunction(), {}, [col1])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="")
@register_fn(engine=OlapEngineType.STARROCKS, name="")
@define_args(FnArg(name="level", is_param=True), FnArg(name="exact", is_param=True), FnArg(name="x"))
@aggregrate
class AggQuantileDfFunction(DfFunction):
    def sql_impl_starrocks(self, ctx: DfContext, fn_args: List[FnArg], fn_params: List[FnArg], arg_dict: Dict) -> str:
        x = arg_dict['x'].sql(ctx)
        level = arg_dict['level'].sql(ctx)
        exact = arg_dict["exact"].sql(ctx)
        fn_name = ""
        if exact.lower() == "true":
            fn_name = "percentile_disc"
        elif exact.lower() == "false":
            fn_name = "percentile_approx"
        else:
            raise Exception(f"invalid arg for `exact`: {exact}")
        return f'{fn_name}({x}, {level})'

    def sql_impl_clickhouse(self, ctx: DfContext, fn_args: List[FnArg], fn_params: List[FnArg], arg_dict: Dict) -> str:
        x = arg_dict['x'].sql(ctx)
        level = arg_dict['level'].sql(ctx)
        exact = arg_dict["exact"].sql(ctx)
        fn_name = ""
        if exact.lower() == "true":
            fn_name = "quantileExact"
        elif exact.lower() == "false":
            fn_name = "quantile"
        else:
            raise Exception(f"invalid arg for `exact`: {exact}")
        return f'{fn_name}({level})({x})'


def quantile(x, *, level, exact=False):
    """
    quantile is used to calculate the quantile of a column.

    Example:
    ----------
    
    .. code-block:: python

        import fast_causal_inference.dataframe.functions as Fn
        df = fast_causal_inference.readClickHouse('test_data_small')
        df.quantile('numerator', level=0.5).show()
        df.quantile('numerator', level=0.5, exact=True).show()
        df.groupBy('treatment').quantile('numerator', level=0.5).show()
        df.groupBy('treatment').quantile('numerator', level=0.5, exact=True).show()
        df.groupBy('treatment').agg(Fn.quantile('numerator', level=0.5).alias('numerator')).show()
    """
    return DfFnColWrapper(AggQuantileDfFunction(), {"level": level, "exact": exact}, [x])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="")
@register_fn(engine=OlapEngineType.STARROCKS, name="")
@define_args(FnArg(name="exact", is_param=True), FnArg(name="x"), FnArg(name="levels", is_variadic=True))
@aggregrate
class AggQuantilesDfFunction(DfFunction):
    def sql_impl_starrocks(self, ctx: DfContext, fn_args: List[FnArg], fn_params: List[FnArg], arg_dict: Dict) -> str:
        x = arg_dict['x'].sql(ctx)
        levels = arg_dict['levels']
        exact = arg_dict["exact"].sql(ctx)
        result = []
        fn_name = ""
        if exact.lower() == "true":
            fn_name = "percentile_disc"
        elif exact.lower() == "false":
            fn_name = "percentile_approx"
        else:
            raise Exception(f"invalid arg for `exact`: {exact}")
        for level in levels.column:
            result.append(f"{fn_name}({x}, {level.sql(ctx)})")
        return f'[{", ".join(result)}]'

    def sql_impl_clickhouse(self, ctx: DfContext, fn_args: List[FnArg], fn_params: List[FnArg], arg_dict: Dict) -> str:
        x = arg_dict['x'].sql(ctx)
        levels = arg_dict['levels']
        param = ", ".join(level.sql(ctx) for level in levels.column)
        exact = arg_dict["exact"].sql(ctx)
        fn_name = ""
        if exact.lower() == "true":
            fn_name = "quantilesExact"
        elif exact.lower() == "false":
            fn_name = "quantiles"
        else:
            raise Exception(f"invalid arg for `exact`: {exact}")
        return f"{fn_name}({param})({x})"


def quantiles(x, *level, exact=False):
    """
    quantiles is used to calculate the quantiles of a column.

    Example:
    ----------

    .. code-block:: python

        import fast_causal_inference.dataframe.functions as Fn
        df = fast_causal_inference.readClickHouse('test_data_small')
        df.quantiles("numerator", 0.25, 0.5, 0.75, 0.99).show()
        df.quantiles("numerator", 0.25, 0.5, 0.75, 0.99, exact=True).show()
        df.groupBy('treatment').quantiles("numerator", 0.25, 0.5, 0.75, 0.99, exact=True).show()
        df.groupBy('treatment').agg(Fn.quantiles("numerator", 0.25, 0.5, 0.75, 0.99, exact=True).alias('quantiles')).show()
    """
    return DfFnColWrapper(AggQuantilesDfFunction(), {"exact": exact}, [x, *level])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="covarPop")
@define_args(FnArg(name="x"), FnArg(name="y"))
@aggregrate
class AggCovarPopDfFunction(DfFunction):
    pass


def covarPop(x, y):
    """
    covarPop is used to calculate the population covariance between two columns.

    Example:
    ----------

    .. code-block:: python

        import fast_causal_inference.dataframe.functions as Fn
        df = fast_causal_inference.readClickHouse('test_data_small')
        df.covarPop('numerator', 'numerator_pre').show()
        df.groupBy('treatment').agg(Fn.covarPop('numerator', 'numerator_pre').alias('numerator')).show()
        df.groupBy('treatment').covarPop('numerator', 'numerator_pre').show()

    """
    return DfFnColWrapper(AggCovarPopDfFunction(), {}, [x, y])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="covarSamp")
@define_args(FnArg(name="x"), FnArg(name="y"))
@aggregrate
class AggCovarSampDfFunction(DfFunction):
    pass


def covarSamp(x, y):
    """
    covarSamp is used to calculate the sample covariance between two columns.

    Example:
    ----------

    .. code-block:: python

        import fast_causal_inference.dataframe.functions as Fn
        df = fast_causal_inference.readClickHouse('test_data_small')
        df.covarSamp('numerator', 'numerator_pre').show()
        df.groupBy('treatment').agg(Fn.covarSamp('numerator', 'numerator_pre').alias('numerator')).show()
        df.groupBy('treatment').covarSamp('numerator', 'numerator_pre').show()
    """
    return DfFnColWrapper(AggCovarSampDfFunction(), {}, [x, y])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="anyLast")
@define_args(FnArg(name="x"), FnArg(name="y"))
@aggregrate
class AggAnyLastDfFunction(DfFunction):
    pass


def anyLast(x, y):
    return DfFnColWrapper(AggAnyLastDfFunction(), {}, [x, y])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="anyMin")
@define_args(FnArg(name="x"), FnArg(name="y"))
@aggregrate
class AggAnyMinDfFunction(DfFunction):
    pass


def anyMin(x, y):
    # """
    # anyMin is used to calculate the minimum value of two columns.

    # >>> import fast_causal_inference.dataframe.functions as Fn
    # >>> df.anyMin('x1', 'x2').show()
    # """
    return DfFnColWrapper(AggAnyMinDfFunction(), {}, [x, y])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="anyMax")
@define_args(FnArg(name="x"), FnArg(name="y"))
@aggregrate
class AggAnyMaxDfFunction(DfFunction):
    pass


def anyMax(x, y):
    # """
    # anyMax is used to calculate the maximum value of two columns.

    # >>> import fast_causal_inference.dataframe.functions as Fn
    # >>> df_new = df.withColumn('new_column', Fn.anyMax('x1', 'x2'))
    # """
    return DfFnColWrapper(AggAnyMaxDfFunction(), {}, [x, y])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="sqrt")
@register_fn(engine=OlapEngineType.STARROCKS, name="sqrt")
@define_args(FnArg(name="x"))
class SqrtDfFunction(DfFunction):
    pass


def sqrt(x):
    """
    sqrt is used to calculate the square root of a column.

    >>> import fast_causal_inference.dataframe.functions as Fn
    >>> df_new = df.withColumn('new_column', Fn.sqrt('weight'))
    >>> df_new.avg('new_column').show()

    """
    return DfFnColWrapper(SqrtDfFunction(), {}, [x])


"""
If somehow we need to bypass Calcite, you need to implement the function `sql_impl_{engine}`
"""


# basic functions
@register_fn(engine=OlapEngineType.CLICKHOUSE, name="abs")
@register_fn(engine=OlapEngineType.STARROCKS, name="abs")
@define_args(FnArg(name="col"))
class AbsDfFunction(DfFunction):
    pass


def abs(col):
    """
    abs is used to calculate the absolute value of a column.

    >>> import fast_causal_inference.dataframe.functions as Fn
    >>> df_new = df.withColumn('new_column', Fn.abs('weight'))
    >>> df_new.avg('new_column').show()
    """
    return DfFnColWrapper(AbsDfFunction(), {}, [col])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="mod")
@register_fn(engine=OlapEngineType.STARROCKS, name="mod")
@define_args(FnArg(name="x"), FnArg(name="y"))
class ModDfFunction(DfFunction):
    pass


def mod(x, y):
    """
    mod is used to calculate the modulo of column x by y.

    >>> import fast_causal_inference.dataframe.functions as Fn
    >>> df_new = df.withColumn('new_column', Fn.mod('weight', 2))
    >>> df_new.avg('new_column').show()
    """
    return DfFnColWrapper(ModDfFunction(), {}, [x, y])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="floor")
@register_fn(engine=OlapEngineType.STARROCKS, name="floor")
@define_args(FnArg(name="x"))
class FloorDfFunction(DfFunction):
    pass


def floor(x):
    """
    floor is used to calculate the largest integer less than or equal to the column x.

    >>> import fast_causal_inference.dataframe.functions as Fn
    >>> df_new = df.withColumn('new_column', Fn.floor('weight'))
    >>> df_new.avg('new_column').show()
    """
    return DfFnColWrapper(FloorDfFunction(), {}, [x])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="ceil")
@register_fn(engine=OlapEngineType.STARROCKS, name="ceil")
@define_args(FnArg(name="x"))
class CeilDfFunction(DfFunction):
    pass


def ceil(x):
    """
    ceil is used to calculate the smallest integer greater than or equal to the column x.

    >>> import fast_causal_inference.dataframe.functions as Fn
    >>> df_new = df.withColumn('new_column', Fn.ceil('weight'))
    >>> df_new.avg('new_column').show()
    """
    return DfFnColWrapper(CeilDfFunction(), {}, [x])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="rand")
@define_args()
class RandDfFunction(DfFunction):
    pass


def rand():
    """
    rand is used to generate a random number.

    >>> import fast_causal_inference.dataframe.functions as Fn
    >>> df_new = df.withColumn('new_column', Fn.rand())
    >>> df_new.avg('new_column').show()
    """
    return DfFnColWrapper(RandDfFunction(), {}, [])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="rand")
@register_fn(engine=OlapEngineType.STARROCKS, name="rand")
@define_args()
class RandCanonicalDfFunction(DfFunction):
    def sql_impl_starrocks(
        self,
        ctx: DfContext,
        fn_args: List[FnArg],
        fn_params: List[FnArg],
        arg_dict: Dict,
    ) -> str:
        return self.fn_name(ctx) + "()"

    def sql_impl_clickhouse(
        self,
        ctx: DfContext,
        fn_args: List[FnArg],
        fn_params: List[FnArg],
        arg_dict: Dict,
    ) -> str:
        return f"({self.fn_name(ctx)}() / pow(2,32))"

    
def rand_cannonical():
    """
    rand_cannonical is used to generate a random float64 number in [0,1].

    >>> import fast_causal_inference.dataframe.functions as Fn
    >>> df_new = df.withColumn('new_column', Fn.rand())
    >>> df_new.avg('new_column').show()
    """
    return DfFnColWrapper(RandCanonicalDfFunction(), {}, [])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="pow")
@register_fn(engine=OlapEngineType.STARROCKS, name="pow")
@define_args(FnArg(name="x"), FnArg(name="y"))
class PowDfFunction(DfFunction):
    pass


def pow(x, y):
    """
    pow is used to calculate the column x raised to the power y.

    >>> import fast_causal_inference.dataframe.functions as Fn
    >>> df_new = df.withColumn('new_column', Fn.pow('weight', 2))
    >>> df_new.avg('new_column').show()
    """
    return DfFnColWrapper(PowDfFunction(), {}, [x, y])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="power")
@register_fn(engine=OlapEngineType.STARROCKS, name="power")
@define_args(FnArg(name="x"), FnArg(name="y"))
class PowerDfFunction(DfFunction):
    pass


def power(x, y):
    """
    power is used to calculate the column x raised to the power y.

    >>> import fast_causal_inference.dataframe.functions as Fn
    >>> df_new = df.withColumn('new_column', Fn.power('weight', 2))
    >>> df_new.avg('new_column').show()
    """
    return DfFnColWrapper(PowerDfFunction(), {}, [x, y])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="exp")
@register_fn(engine=OlapEngineType.STARROCKS, name="exp")
@define_args(FnArg(name="x"))
class ExpDfFunction(DfFunction):
    pass


def exp(x):
    """
    exp is used to calculate e raised to the power of column x.

    >>> import fast_causal_inference.dataframe.functions as Fn
    >>> df_new = df.withColumn('new_column', Fn.exp('weight'))
    >>> df_new.avg('new_column').show()
    """
    return DfFnColWrapper(ExpDfFunction(), {}, [x])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="log")
@register_fn(engine=OlapEngineType.STARROCKS, name="log")
@define_args(FnArg(name="base"), FnArg(name="x"))
class LogDfFunction(DfFunction):
    pass


def log(base, x):
    """
    log is used to calculate the natural logarithm of column x.

    >>> import fast_causal_inference.dataframe.functions as Fn
    >>> df_new = df.withColumn('new_column', Fn.log('weight'))
    >>> df_new.avg('new_column').show()
    """
    return DfFnColWrapper(LogDfFunction(), {}, [base, x])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="ln")
@register_fn(engine=OlapEngineType.STARROCKS, name="ln")
@define_args(FnArg(name="x"))
class LnDfFunction(DfFunction):
    pass


def ln(x):
    """
    ln is used to calculate the natural logarithm of column x.

    >>> import fast_causal_inference.dataframe.functions as Fn
    >>> df_new = df.withColumn('new_column', Fn.ln('weight'))
    >>> df_new.avg('new_column').show()
    """
    return DfFnColWrapper(LnDfFunction(), {}, [x])


def log(x):
    """
    log is used to calculate the natural logarithm of column x.

    >>> import fast_causal_inference.dataframe.functions as Fn
    >>> df_new = df.withColumn('new_column', Fn.log('weight'))
    >>> df_new.avg('new_column').show()
    """
    return DfFnColWrapper(LnDfFunction(), {}, [x])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="exp2")
@define_args(FnArg(name="x"))
class Exp2DfFunction(DfFunction):
    pass


def exp2(x):
    """
    exp2 is used to calculate 2 raised to the power of column x.

    >>> import fast_causal_inference.dataframe.functions as Fn
    >>> df_new = df.withColumn('new_column', Fn.exp2('weight'))
    >>> df_new.avg('new_column').show()
    """
    return DfFnColWrapper(Exp2DfFunction(), {}, [x])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="log2")
@define_args(FnArg(name="x"))
class Log2DfFunction(DfFunction):
    pass


def log2(x):
    """
    log2 is used to calculate the base 2 logarithm of column x.

    >>> import fast_causal_inference.dataframe.functions as Fn
    >>> df_new = df.withColumn('new_column', Fn.log2('weight'))
    >>> df_new.avg('new_column').show()
    """
    return DfFnColWrapper(Log2DfFunction(), {}, [x])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="murmurHash3_64")
@define_args(FnArg(name="x"))
class MurmurHash3_64DfFunction(DfFunction):
    pass


def murmur_hash3_64(x):
    """
    murmur_hash3_64 is used to calculate the murmur3 hash of column x.

    >>> import fast_causal_inference.dataframe.functions as Fn
    >>> df_new = df.withColumn('new_column', Fn.murmur_hash3_64('weight'))
    >>> df_new.avg('new_column').show()
    """
    return DfFnColWrapper(MurmurHash3_64DfFunction(), {}, [x])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="murmurHash3_32")
@register_fn(engine=OlapEngineType.STARROCKS, name="murmur_hash3_32")
@define_args(FnArg(name="x"))
class MurmurHash3_32DfFunction(DfFunction):
    pass


def murmur_hash3_32(x):
    """
    murmur_hash3_32 is used to calculate the murmur3 hash of column x.

    >>> import fast_causal_inference.dataframe.functions as Fn
    >>> df_new = df.withColumn('new_column', Fn.murmur_hash3_32('weight'))
    >>> df_new.avg('new_column').show()
    """
    return DfFnColWrapper(MurmurHash3_32DfFunction(), {}, [x])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="isNull")
@register_fn(engine=OlapEngineType.STARROCKS, name="isnull")
@define_args(FnArg(name="x"))
class IsNullDfFunction(DfFunction):
    pass


def isnull(x):
    """
    isnull is used to check if column x is null.

    >>> import fast_causal_inference.dataframe.functions as Fn
    >>> df_new = df.withColumn('new_column', Fn.isnull('weight'))
    >>> df_new.avg('new_column').show()
    """
    return DfFnColWrapper(IsNullDfFunction(), {}, [x])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="isNotNull")
@register_fn(engine=OlapEngineType.STARROCKS, name="isnotnull")
@define_args(FnArg(name="x"))
class IsNotNullDfFunction(DfFunction):
    pass


def isnotnull(x):
    """
    isnotnull is used to check if column x is not null.

    >>> import fast_causal_inference.dataframe.functions as Fn
    >>> df_new = df.withColumn('new_column', Fn.isnotnull('weight'))
    >>> df_new.avg('new_column').show()
    """
    return DfFnColWrapper(IsNotNullDfFunction(), {}, [x])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="round")
@register_fn(engine=OlapEngineType.STARROCKS, name="round")
@define_args(FnArg(name="x"), FnArg(name="n"))
class RoundDfFunction(DfFunction):
    def sql_impl_default(
        self,
        ctx: DfContext,
        fn_args: List[FnArg],
        fn_params: List[FnArg],
        arg_dict: Dict,
    ) -> str:
        x = arg_dict["x"].sql(ctx)
        n = arg_dict["n"].sql(ctx)
        if n == "":
            return self.fn_name(ctx) + "(" + x + ")"
        return self.fn_name(ctx) + "(" + x + ", " + n + ")"


def round(x, n=""):
    """
    round is used to round column x to y decimal places.

    >>> import fast_causal_inference.dataframe.functions as Fn
    >>> df_new = df.withColumn('new_column', Fn.round('weight', 2))
    >>> df_new.avg('new_column').show()
    """
    return DfFnColWrapper(RoundDfFunction(), {}, [x, n])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="cbrt")
@define_args(FnArg(name="x"))
class CBRTDfFunction(DfFunction):
    pass


def cbrt(x):
    """
    cbrt is used to calculate the cube root of column x.

    >>> import fast_causal_inference.dataframe.functions as Fn
    >>> df_new = df.withColumn('new_column', Fn.cbrt('weight'))
    >>> df_new.avg('new_column').show()
    """
    return DfFnColWrapper(CBRTDfFunction(), {}, [x])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="erf")
@define_args(FnArg(name="x"))
class ERFDfFunction(DfFunction):
    pass


def erf(x):
    """
    erf is used to calculate the error function of column x.

    >>> import fast_causal_inference.dataframe.functions as Fn
    >>> df_new = df.withColumn('new_column', Fn.erf('weight'))
    >>> df_new.avg('new_column').show()
    """
    return DfFnColWrapper(ERFDfFunction(), {}, [x])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="erfc")
@define_args(FnArg(name="x"))
class ERFCDfFunction(DfFunction):
    pass


def erfc(x):
    """
    erfc is used to calculate the complementary error function of column x.

    >>> import fast_causal_inference.dataframe.functions as Fn
    >>> df_new = df.withColumn('new_column', Fn.erfc('weight'))
    >>> df_new.avg('new_column').show()
    """
    return DfFnColWrapper(ERFCDfFunction(), {}, [x])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="lgamma")
@define_args(FnArg(name="x"))
class LGammaDfFunction(DfFunction):
    pass


def lgamma(x):
    """
    lgamma is used to calculate the log gamma function of column x.

    >>> import fast_causal_inference.dataframe.functions as Fn
    >>> df_new = df.withColumn('new_column', Fn.lgamma('weight'))
    >>> df_new.avg('new_column').show()
    """
    return DfFnColWrapper(LGammaDfFunction(), {}, [x])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="tgamma")
@define_args(FnArg(name="x"))
class TGammaDfFunction(DfFunction):
    pass


def tgamma(x):
    """
    tgamma is used to calculate the gamma function of column x.

    >>> import fast_causal_inference.dataframe.functions as Fn
    >>> df_new = df.withColumn('new_column', Fn.tgamma('weight'))
    >>> df_new.avg('new_column').show()
    """
    return DfFnColWrapper(TGammaDfFunction(), {}, [x])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="sin")
@register_fn(engine=OlapEngineType.STARROCKS, name="sin")
@define_args(FnArg(name="x"))
class SinDfFunction(DfFunction):
    pass


def sin(x):
    """
    sin is used to calculate the sine of column x.

    >>> import fast_causal_inference.dataframe.functions as Fn
    >>> df_new = df.withColumn('new_column', Fn.sin('weight'))
    >>> df_new.avg('new_column').show()
    """
    return DfFnColWrapper(SinDfFunction(), {}, [x])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="cos")
@register_fn(engine=OlapEngineType.STARROCKS, name="cos")
@define_args(FnArg(name="x"))
class CosDfFunction(DfFunction):
    pass


def cos(x):
    """
    cos is used to calculate the cosine of column x.

    >>> import fast_causal_inference.dataframe.functions as Fn
    >>> df_new = df.withColumn('new_column', Fn.cos('weight'))
    >>> df_new.avg('new_column').show()
    """
    return DfFnColWrapper(CosDfFunction(), {}, [x])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="tan")
@register_fn(engine=OlapEngineType.STARROCKS, name="tan")
@define_args(FnArg(name="x"))
class TanDfFunction(DfFunction):
    pass


def tan(x):
    """
    tan is used to calculate the tangent of column x.

    >>> import fast_causal_inference.dataframe.functions as Fn
    >>> df_new = df.withColumn('new_column', Fn.tan('weight'))
    >>> df_new.avg('new_column').show()
    """
    return DfFnColWrapper(TanDfFunction(), {}, [x])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="asin")
@register_fn(engine=OlapEngineType.STARROCKS, name="asin")
@define_args(FnArg(name="x"))
class AsinDfFunction(DfFunction):
    pass


def asin(x):
    """
    asin is used to calculate the arcsine of column x.

    >>> import fast_causal_inference.dataframe.functions as Fn
    >>> df_new = df.withColumn('new_column', Fn.asin('weight'))
    >>> df_new.avg('new_column').show()
    """
    return DfFnColWrapper(AsinDfFunction(), {}, [x])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="acos")
@register_fn(engine=OlapEngineType.STARROCKS, name="acos")
@define_args(FnArg(name="x"))
class AcosDfFunction(DfFunction):
    pass


def acos(x):
    """
    acos is used to calculate the arccosine of column x.

    >>> import fast_causal_inference.dataframe.functions as Fn
    >>> df_new = df.withColumn('new_column', Fn.acos('weight'))
    >>> df_new.avg('new_column').show()
    """
    return DfFnColWrapper(AcosDfFunction(), {}, [x])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="atan")
@register_fn(engine=OlapEngineType.STARROCKS, name="atan")
@define_args(FnArg(name="x"))
class AtanDfFunction(DfFunction):
    pass


def atan(x):
    """
    atan is used to calculate the arctangent of column x.

    >>> import fast_causal_inference.dataframe.functions as Fn
    >>> df_new = df.withColumn('new_column', Fn.atan('weight'))
    >>> df_new.avg('new_column').show()
    """
    return DfFnColWrapper(AtanDfFunction(), {}, [x])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="exp10")
@define_args(FnArg(name="x"))
class Exp10DfFunction(DfFunction):
    pass


def exp10(x):
    """
    exp10 is used to calculate 10 raised to the power of column x.

    >>> import fast_causal_inference.dataframe.functions as Fn
    >>> df_new = df.withColumn('new_column', Fn.exp10('weight'))
    >>> df_new.avg('new_column').show()
    """
    return DfFnColWrapper(Exp10DfFunction(), {}, [x])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="log10")
@define_args(FnArg(name="x"))
class Log10DfFunction(DfFunction):
    pass


def log10(x):
    """
    log10 is used to calculate the base 10 logarithm of column x.

    >>> import fast_causal_inference.dataframe.functions as Fn
    >>> df_new = df.withColumn('new_column', Fn.log10('weight'))
    >>> df_new.avg('new_column').show()
    """
    return DfFnColWrapper(Log10DfFunction(), {}, [x])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="intExp2")
@define_args(FnArg(name="x"))
class IntExp2DfFunction(DfFunction):
    pass


def intExp2(x):
    """
    intExp2 is used to calculate 2 raised to the power of column x, and the result is an integer.

    >>> import fast_causal_inference.dataframe.functions as Fn
    >>> df_new = df.withColumn('new_column', Fn.intExp2('weight'))
    >>> df_new.avg('new_column').show()
    """
    return DfFnColWrapper(IntExp2DfFunction(), {}, [x])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="intExp10")
@define_args(FnArg(name="x"))
class IntExp10DfFunction(DfFunction):
    pass


def intExp10(x):
    """
    intExp10 is used to calculate 10 raised to the power of column x, and the result is an integer.

    >>> import fast_causal_inference.dataframe.functions as Fn
    >>> df_new = df.withColumn('new_column', Fn.intExp10('weight'))
    >>> df_new.avg('new_column').show()
    """
    return DfFnColWrapper(IntExp10DfFunction(), {}, [x])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="gcd")
@define_args(FnArg(name="x"), FnArg(name="y"))
class GCDDfFunction(DfFunction):
    pass


def gcd(x, y):
    """
    gcd is used to calculate the greatest common divisor of column x and y.

    >>> import fast_causal_inference.dataframe.functions as Fn
    >>> df_new = df.withColumn('new_column', Fn.gcd('weight', 'height'))
    >>> df_new.avg('new_column').show()
    """
    return DfFnColWrapper(GCDDfFunction(), {}, [x, y])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="lcm")
@define_args(FnArg(name="x"), FnArg(name="y"))
class LCMDfFunction(DfFunction):
    pass


def lcm(x, y):
    """
    lcm is used to calculate the least common multiple of column x and y.

    >>> import fast_causal_inference.dataframe.functions as Fn
    >>> df_new = df.withColumn('new_column', Fn.lcm('weight', 'height'))
    >>> df_new.avg('new_column').show()
    """
    return DfFnColWrapper(LCMDfFunction(), {}, [x, y])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="If")
@register_fn(engine=OlapEngineType.STARROCKS, name="if")
@define_args(FnArg(name="cond"), FnArg(name="x"), FnArg(name="y"))
class IfDfFunction(DfFunction):
    pass


def If(cond, x, y):
    """
    If is used to create a new column based on the condition x. If x is true, y is returned, otherwise z is returned.

    >>> import fast_causal_inference.dataframe.functions as Fn
    >>> df_new = df.withColumn('new_column', Fn.If(df['weight'] > 0.5, '>0.5', '<0.5'))
    >>> df_new = df.withColumn('new_column', Fn.If('weight>0.5', 1, 0))
    >>> df_new.show()
    """
    return DfFnColWrapper(IfDfFunction(), {}, [cond, x, y])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="e")
@register_fn(engine=OlapEngineType.STARROCKS, name="e")
@define_args()
class ConstantEDfFunction(DfFunction):
    pass


def e():
    """
    e is used to get the mathematical constant e.

    >>> import fast_causal_inference.dataframe.functions as Fn
    >>> df_new = df.withColumn('new_column', Fn.e())
    >>> df_new.avg('new_column').show()
    """
    return DfFnColWrapper(ConstantEDfFunction(), {}, [])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="pi")
@register_fn(engine=OlapEngineType.STARROCKS, name="pi")
@define_args()
class ConstantPiDfFunction(DfFunction):
    pass


def pi():
    """
    pi is used to get the mathematical constant pi.

    >>> import fast_causal_inference.dataframe.functions as Fn
    >>> df_new = df.withColumn('new_column', Fn.pi())
    >>> df_new.avg('new_column').show()
    """
    return DfFnColWrapper(ConstantPiDfFunction(), {}, [])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="L1Norm")
@define_args(FnArg(name="x"))
class L1NormDfFunction(DfFunction):
    pass


def L1Norm(x):
    """
    L1Norm is used to calculate the L1 norm of column x.

    >>> import fast_causal_inference.dataframe.functions as Fn
    >>> df_new = df.withColumn('new_column', Fn.L1Norm('weight'))
    >>> df_new.avg('new_column').show()
    """
    return DfFnColWrapper(L1NormDfFunction(), {}, [x])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="L2Norm")
@define_args(FnArg(name="x"))
class L2NormDfFunction(DfFunction):
    pass


def L2Norm(x):
    """
    L2Norm is used to calculate the L2 norm of column x.

    >>> import fast_causal_inference.dataframe.functions as Fn
    >>> df_new = df.withColumn('new_column', Fn.L2Norm('weight'))
    >>> df_new.avg('new_column').show()
    """
    return DfFnColWrapper(L2NormDfFunction(), {}, [x])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="LinfNorm")
@define_args(FnArg(name="x"))
class LinfNormDfFunction(DfFunction):
    pass


def LinfNorm(x):
    """
    LinfNorm is used to calculate the L-infinity norm of column x.

    >>> import fast_causal_inference.dataframe.functions as Fn
    >>> df_new = df.withColumn('new_column', Fn.LinfNorm('weight'))
    >>> df_new.avg('new_column').show()
    """
    return DfFnColWrapper(LinfNormDfFunction(), {}, [x])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="LpNorm")
@define_args(FnArg(name="x"))
class LpNormDfFunction(DfFunction):
    pass


def LpNorm(x):
    """
    LpNorm is used to calculate the Lp norm of column x.

    >>> import fast_causal_inference.dataframe.functions as Fn
    >>> df_new = df.withColumn('new_column', Fn.LpNorm('weight', 2))
    >>> df_new.avg('new_column').show()
    """
    return DfFnColWrapper(LpNormDfFunction(), {}, [x])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="L1Distance")
@define_args(FnArg(name="x"), FnArg(name="y"))
class L1DistanceDfFunction(DfFunction):
    pass


def L1Distance(x, y):
    """
    L1Distance is used to calculate the L1 distance between column x and y.

    >>> import fast_causal_inference.dataframe.functions as Fn
    >>> df_new = df.withColumn('new_column', Fn.L1Distance('weight', 'height'))
    >>> df_new.avg('new_column').show()
    """
    return DfFnColWrapper(L1DistanceDfFunction(), {}, [x, y])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="L2Distance")
@define_args(FnArg(name="x"), FnArg(name="y"))
class L2DistanceDfFunction(DfFunction):
    pass


def L2Distance(x, y):
    """
    L2Distance is used to calculate the L2 distance between column x and y.

    >>> import fast_causal_inference.dataframe.functions as Fn
    >>> df_new = df.withColumn('new_column', Fn.L2Distance('weight', 'height'))
    >>> df_new.avg('new_column').show()
    """
    return DfFnColWrapper(L2DistanceDfFunction(), {}, [x, y])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="L2SquaredDistance")
@define_args(FnArg(name="x"), FnArg(name="y"))
class L2SquaredDistanceDfFunction(DfFunction):
    pass


def L2SquaredDistance(x, y):
    """
    L2SquaredDistance is used to calculate the squared L2 distance between column x and y.

    >>> import fast_causal_inference.dataframe.functions as Fn
    >>> df_new = df.withColumn('new_column', Fn.L2SquaredDistance('weight', 'height'))
    >>> df_new.avg('new_column').show()
    """
    return DfFnColWrapper(L2SquaredDistanceDfFunction(), {}, [x, y])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="LinfDistance")
@define_args(FnArg(name="x"), FnArg(name="y"))
class LinfDistanceDfFunction(DfFunction):
    pass


def LinfDistance(x, y):
    """
    LinfDistance is used to calculate the L-infinity distance between column x and y.

    >>> import fast_causal_inference.dataframe.functions as Fn
    >>> df_new = df.withColumn('new_column', Fn.LinfDistance('weight', 'height'))
    >>> df_new.avg('new_column').show()
    """
    return DfFnColWrapper(LinfDistanceDfFunction(), {}, [x, y])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="LpDistance")
@define_args(FnArg(name="x"), FnArg(name="y"))
class LpDistanceDfFunction(DfFunction):
    pass


def LpDistance(x, y):
    """
    LpDistance is used to calculate the Lp distance between column x and y.

    >>> import fast_causal_inference.dataframe.functions as Fn
    >>> df_new = df.withColumn('new_column', Fn.LpDistance('weight', 'height', 2))
    >>> df_new.avg('new_column').show()
    """
    return DfFnColWrapper(LpDistanceDfFunction(), {}, [x, y])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="L1Normalize")
@define_args(FnArg(name="x"))
class L1NormalizeDfFunction(DfFunction):
    pass


def L1Normalize(x):
    """
    L1Normalize is used to normalize column x using L1 norm.

    >>> import fast_causal_inference.dataframe.functions as Fn
    >>> df_new = df.withColumn('new_column', Fn.L1Normalize('weight'))
    >>> df_new.avg('new_column').show()
    """
    return DfFnColWrapper(L1NormalizeDfFunction(), {}, [x])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="L2Normalize")
@define_args(FnArg(name="x"))
class L2NormalizeDfFunction(DfFunction):
    pass


def L2Normalize(x):
    """
    L2Normalize is used to normalize column x using L2 norm.

    >>> import fast_causal_inference.dataframe.functions as Fn
    >>> df_new = df.withColumn('new_column', Fn.L2Normalize('weight'))
    >>> df_new.avg('new_column').show()
    """
    return DfFnColWrapper(L2NormalizeDfFunction(), {}, [x])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="LinfNormalize")
@define_args(FnArg(name="x"))
class LinfNormalizeDfFunction(DfFunction):
    pass


def LinfNormalize(x):
    """
    LinfNormalize is used to normalize column x using L-infinity norm.

    >>> import fast_causal_inference.dataframe.functions as Fn
    >>> df_new = df.withColumn('new_column', Fn.LinfNormalize('weight'))
    >>> df_new.avg('new_column').show()
    """
    return DfFnColWrapper(LinfNormalizeDfFunction(), {}, [x])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="LpNormalize")
@define_args(FnArg(name="x"))
class LpNormalizeDfFunction(DfFunction):
    pass


def LpNormalize(x):
    """
    LpNormalize is used to normalize column x using Lp norm.

    >>> import fast_causal_inference.dataframe.functions as Fn
    >>> df_new = df.withColumn('new_column', Fn.LpNormalize('weight', 2))
    >>> df_new.avg('new_column').show()
    """
    return DfFnColWrapper(LpNormalizeDfFunction(), {}, [x])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="cosineDistance")
@define_args(FnArg(name="x"), FnArg(name="y"))
class cosineDistanceDfFunction(DfFunction):
    pass


def cosineDistance(x, y):
    """
    cosineDistance is used to calculate the cosine distance between column x and y.

    >>> import fast_causal_inference.dataframe.functions as Fn
    >>> df_new = df.withColumn('new_column', Fn.cosineDistance('weight', 'height'))
    >>> df_new.avg('new_column').show()
    """
    return DfFnColWrapper(cosineDistanceDfFunction(), {}, [x, y])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="cosineSimilarity")
@define_args(FnArg(name="x"), FnArg(name="y"))
class cosineSimilarityDfFunction(DfFunction):
    pass


def cosineSimilarity(x, y):
    """
    cosineSimilarity is used to calculate the cosine similarity between column x and y.

    >>> import fast_causal_inference.dataframe.functions as Fn
    >>> df_new = df.withColumn('new_column', Fn.cosineSimilarity('weight', 'height'))
    >>> df_new.avg('new_column').show()
    """
    return DfFnColWrapper(cosineSimilarityDfFunction(), {}, [x, y])


def desc(column):
    """
    desc is used to sort column x in descending order.

    >>> import fast_causal_inference.dataframe.functions as Fn
    >>> df_new = df.orderBy(Fn.desc('weight'))
    >>> df_new.show()
    """
    order = DfPb.Order()
    order.column.name = column
    order.desc = True
    return order


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="")
@register_fn(engine=OlapEngineType.STARROCKS, name="")
@define_args(FnArg(name="x"), FnArg(name="y"))
class LessThanDfFunction(DfFunction):
    def sql_impl_default(
        self,
        ctx: DfContext,
        fn_args: List[FnArg],
        fn_params: List[FnArg],
        arg_dict: Dict,
    ) -> str:
        return "(" + arg_dict["x"].sql(ctx) + " < " + arg_dict["y"].sql(ctx) + ")"


def lt(x, y):
    return DfFnColWrapper(LessThanDfFunction(), {}, [x, y])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="")
@register_fn(engine=OlapEngineType.STARROCKS, name="")
@define_args(FnArg(name="x"), FnArg(name="y"))
class LessEqualsDfFunction(DfFunction):
    def sql_impl_default(
        self,
        ctx: DfContext,
        fn_args: List[FnArg],
        fn_params: List[FnArg],
        arg_dict: Dict,
    ) -> str:
        return "(" + arg_dict["x"].sql(ctx) + " <= " + arg_dict["y"].sql(ctx) + ")"


def le(x, y):
    return DfFnColWrapper(LessEqualsDfFunction(), {}, [x, y])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="")
@register_fn(engine=OlapEngineType.STARROCKS, name="")
@define_args(FnArg(name="x"), FnArg(name="y"))
class GreaterThanDfFunction(DfFunction):
    def sql_impl_default(
        self,
        ctx: DfContext,
        fn_args: List[FnArg],
        fn_params: List[FnArg],
        arg_dict: Dict,
    ) -> str:
        return "(" + arg_dict["x"].sql(ctx) + " > " + arg_dict["y"].sql(ctx) + ")"


def gt(x, y):
    return DfFnColWrapper(GreaterThanDfFunction(), {}, [x, y])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="")
@register_fn(engine=OlapEngineType.STARROCKS, name="")
@define_args(FnArg(name="x"), FnArg(name="y"))
class GraterEqualsDfFunction(DfFunction):
    def sql_impl_default(
        self,
        ctx: DfContext,
        fn_args: List[FnArg],
        fn_params: List[FnArg],
        arg_dict: Dict,
    ) -> str:
        return "(" + arg_dict["x"].sql(ctx) + " >= " + arg_dict["y"].sql(ctx) + ")"


def ge(x, y):
    return DfFnColWrapper(GraterEqualsDfFunction(), {}, [x, y])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="")
@register_fn(engine=OlapEngineType.STARROCKS, name="")
@define_args(FnArg(name="x"), FnArg(name="y"))
class EqualsDfFunction(DfFunction):
    def sql_impl_default(
        self,
        ctx: DfContext,
        fn_args: List[FnArg],
        fn_params: List[FnArg],
        arg_dict: Dict,
    ) -> str:
        return "(" + arg_dict["x"].sql(ctx) + " = " + arg_dict["y"].sql(ctx) + ")"


def eq(x, y):
    return DfFnColWrapper(EqualsDfFunction(), {}, [x, y])


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="")
@register_fn(engine=OlapEngineType.STARROCKS, name="")
@define_args(FnArg(name="x"), FnArg(name="y"))
class NotEqualsDfFunction(DfFunction):
    def sql_impl_default(
        self,
        ctx: DfContext,
        fn_args: List[FnArg],
        fn_params: List[FnArg],
        arg_dict: Dict,
    ) -> str:
        return "(" + arg_dict["x"].sql(ctx) + " != " + arg_dict["y"].sql(ctx) + ")"


def ne(x, y):
    return DfFnColWrapper(NotEqualsDfFunction(), {}, [x, y])


