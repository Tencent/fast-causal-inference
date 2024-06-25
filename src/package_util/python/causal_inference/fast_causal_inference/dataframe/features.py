from lib2to3.pgen2.token import STAR
from fast_causal_inference.dataframe.dataframe import readOlap
import fast_causal_inference.lib.tools as ais_tools
from fast_causal_inference.dataframe import readClickHouse
from fast_causal_inference.dataframe.functions import (
    DfFnColWrapper,
    register_fn,
    define_args,
    FnArg,
    DfFunction,
    OlapEngineType,
)
import re
from pypinyin import lazy_pinyin
import time

class OneHotEncoder:
    """

    This class implements the OneHotEncoder method for causal inference.

    Parameters
    ----------

    cols : list, default=None
        The columns to be one-hot encoded.

    Methods
    -------
    
    fit(dataframe):
        Apply the OneHotEncoder method to the input dataframe.

    Example
    -------

    .. code-block:: python

        import fast_causal_inference
        import fast_causal_inference.dataframe.features as Features
        df = fast_causal_inference.readClickHouse('test_data_small')
        one_hot = Features.OneHotEncoder()
        df_new = one_hot.fit(df, cols=['x_cat1'])
        df_new.printSchema()
    """

    def __init__(self):
        pass

    def fit(self, df, cols):
        from fast_causal_inference.util import ClickHouseUtils, StarRocksUtils

        new_df = df.materializedView(is_temp=True)
        sql_instance = df.sql_conn

        def add_suffix_to_duplicates(lst):
            seen = {}
            for i, item in enumerate(lst):
                if item not in seen:
                    seen[item] = 0
                else:
                    seen[item] += 1
                    lst[i] = lst[i] + "_" + str(seen[item])
            return lst
        
        table = new_df.getTableName()

        string_list = []
        cols = list(set(cols))
        values_dict = {}
        for col in cols:
            data = sql_instance.sql(
                f"select {col},count(*) as cnt from {table} group by {col} order by {col}"
            )
            values = list(data[col])[1:]
            column_values = [
                f"{col}_value_{re.sub(r'[^0-9a-zA-Z_]', '_', '_'.join(lazy_pinyin(i)))}"
                for i in values
            ]
            column_values = add_suffix_to_duplicates(column_values)
            values_dict.update(dict(zip(column_values, values)))
            string_list += [
                f"if({col}='{values_dict[i]}',1,0) as {i}" for i in column_values
            ]
        table_new = f"{table}_{int(time.time())}"
        if new_df.engine == OlapEngineType.CLICKHOUSE:
            ClickHouseUtils().clickhouse_create_view(
                clickhouse_view_name=table_new,
                sql_statement=f'*,{",".join(string_list)}',
                sql_table_name=table,
                is_force_materialize=True,
            )
        elif new_df.engine == OlapEngineType.STARROCKS:
            StarRocksUtils().create_view(
                view_name=table_new,
                sql_statement=f'select *,{",".join(string_list)} from {table}',
                is_table=True
            )
        else:
            raise Exception(f"Unsupported olap `{new_df.engine}`.")
        # print("get new table: ", table_new)
        # desc_string = ";".join(
        #     [
        #         f"ALTER TABLE {table_new}  COMMENT column {col} '{values_dict[col]}' "
        #         for col in values_dict
        #     ]
        # )
        # sql_instance.sql(desc_string)
        return readOlap(table_new, olap=new_df.engine)


from fast_causal_inference.dataframe.functions import (
    DfFnColWrapper,
    register_fn,
    define_args,
    FnArg,
    DfFunction,
    OlapEngineType,
)


@register_fn(engine=OlapEngineType.CLICKHOUSE, name="cutbins")
@register_fn(engine=OlapEngineType.STARROCKS, name="cutbins")
@define_args(
    FnArg(name="column"), FnArg(name="bins"), FnArg(name="if_string", default="True")
)
class CutbinsDfFunction(DfFunction):
    pass


def cut_bins(column, bins, if_string=True):
    bins_str = ""
    if isinstance(bins, str):
        bins_str = bins
    elif isinstance(bins, list):
        bins_str = "[" + ",".join([str(x) for x in bins]) + "]"
    else:
        raise ValueError(f"bins({bins}) must be a str or a list")
    return DfFnColWrapper(CutbinsDfFunction(), {}, [column, bins_str, if_string])



class Bucketizer:
    """
        This function applies the bucketizing transformation to the specified columns of the input dataframe.


        Parameters

        :param df: The input dataframe to be transformed.
        :type df: DataFrame
        :param inputCols: A list of column names in the dataframe to be bucketized.
        :type inputCols: list
        :param splitsArray: A list of lists, where each inner list contains the split points for bucketizing the corresponding column in inputCols.
        :type splitsArray: list
        :param quantilesArray: A list of quantiles for bucketizing the corresponding column in inputCols.
        :type quantilesArray: list
        :param outputCols: A list of output column names after bucketizing. If not provided, '_buckets' will be appended to the original column names.
        :type outputCols: list, optional
        :param if_string: A flag indicating whether the bin values should be treated as strings. Default is True.
        :type if_string: bool, optional
        :return: The transformed dataframe with bucketized columns.
        :rtype: DataFrame

        Example
        -------
        
        .. code-block:: python

            >>> import fast_causal_inference
            >>> import fast_causal_inference.dataframe.features as Features
            >>> df = fast_causal_inference.readClickHouse('test_data_small')
            
            # bucketize the columns 'x1' and 'x2' with the specified split points, and return string values, like [1,3)
            >>> bucketizer = Features.Bucketizer()
            >>> df_new = bucketizer.fit(df=df,inputCols=['x1','x2'],splitsArray=[[1,3],[0,2]],if_string=True)
            >>> df_new.select('x1','x2','x1_buckets','x2_buckets').head(5).show()
                                x1            x2 x1_buckets x2_buckets
            0  -0.131301907  -3.152383354          1          0
            1  -0.966931088  -0.427920835          1          0
            2   1.257744217  -2.050358546      [1,3)          0
            3  -0.777228042  -2.621604715          1          0
            4  -0.669571385   0.606404768          1      [0,2)

            # bucketize the columns 'x1' and 'x2' with the specified split points, and return int values, like 1
            >>> bucketizer = Features.Bucketizer()
            >>> df_new = bucketizer.fit(df=df,inputCols=['x1','x2'],splitsArray=[[1,3],[0,2]],if_string=False)
            >>> df_new.select('x1','x2','x1_buckets','x2_buckets').head(5).show()
                        x1            x2 x1_buckets x2_buckets
            0  -0.131301907  -3.152383354          1          1
            1  -0.966931088  -0.427920835          1          1
            2   1.257744217  -2.050358546          2          1
            3  -0.777228042  -2.621604715          1          1
            4  -0.669571385   0.606404768          1          2

            # bucketize the columns 'x1' and 'x2' with the same quantile split points, and return string values, like [1,3)
            >>> bucketizer = Features.Bucketizer()
            >>> df_new = bucketizer.fit(df=df,inputCols=['x1','x2'],quantilesArray=[0.25,0.5,0.75],if_string=False)
            >>> df_new.select('x1','x2','x1_buckets','x2_buckets').head(5).show()
                             x1            x2                         x1_buckets  \
                0   0.412755786  -0.540878472  [-0.6984664715000001,0.636346717)   
                1   0.947846754   4.296402092                    >=-0.0356790885   
                2  -2.115056015  -0.564252981                -0.6984664715000001   
                3  -0.117271254   3.065135957  [-0.6984664715000001,0.636346717)   
                4   0.115932199   1.410948606  [-0.6984664715000001,0.636346717)   

                        x2_buckets  
                0     1.36825694575  
                1  >=-1.29057637775  
                2     1.36825694575  
                3  >=-1.29057637775  
                4  >=-1.29057637775


            # bucketize the columns 'x1' and 'x2' with the different quantile split points, and return string values, like [1,3)
            >>> bucketizer = Features.Bucketizer()
            >>> df_new = bucketizer.fit(df=df,inputCols=['x1','x2'],quantilesArray=[[0.5,0.8],[0.1,0.3]],if_string=False)
            >>> df_new.select('x1','x2','x1_buckets','x2_buckets').head(5).show()
                            x1            x2                        x1_buckets  \
                0   0.412755786  -0.540878472  [-0.01859157,0.8138407770000001)   
                1   0.947846754   4.296402092              >=0.8138407770000001   
                2  -2.115056015  -0.564252981                       -0.01859157   
                3  -0.117271254   3.065135957                       -0.01859157   
                4   0.115932199   1.410948606  [-0.01859157,0.8138407770000001)   

                    x2_buckets  
                0  >=-1.009238587  
                1  >=-1.009238587  
                2  >=-1.009238587  
                3  >=-1.009238587  
                4  >=-1.009238587  
        """

    def __init__(self):
        pass

    def fit(self, df, inputCols, splitsArray=None, quantilesArray=None, outputCols=[], if_string=True):
        df = df.materializedView(is_temp=True)
        import fast_causal_inference.dataframe.functions as Fn
        dtypes = df.dtypes
        for col in inputCols:
            if dtypes[col] not in ('double', 'bigint', 'int', 'float'):
                raise ValueError(f"{col} is not a numeric column")

        if splitsArray is not None:
            if len(splitsArray) != len(inputCols):
                raise ValueError("splitsArray length must equal to inputCols length")
            else:
                splitsArray_dict = {inputCols[i]: splitsArray[i] for i in range(len(inputCols))}

        elif quantilesArray is not None:
            if len(quantilesArray) != len(inputCols):
                raise ValueError("quantilesArray length must equal to inputCols length")
            else:
                quantilesArray_dict = {inputCols[i]: quantilesArray[i] for i in range(len(inputCols))}
                # Check if all quantiles are between 0 and 1
                for quantiles in quantilesArray_dict.values():
                    if not all(0 < q < 1 for q in quantiles):
                        raise ValueError("quantilesArray must be in the range of 0 to 1")
                # Calculate splits using the quantiles
                splitsArray_dict = {}
                for col in inputCols:
                    quantiles = quantilesArray_dict[col]
                    splits = df.agg(Fn.quantiles(col, *quantiles).alias(col))
                    splitsArray_dict[col] = splits
                    # splitsArray_dict[col] = list(set([float(i) for i in splits[col][0].replace("[", "").replace("]", "").split(",")]))

        else:
            quantilesArray = [0.25, 0.5, 0.75, 0.9, 0.95]
            splitsArray_dict = {}
            for col in inputCols:
                splits = df.agg(Fn.quantiles(col, *quantilesArray).alias(col))
                splitsArray_dict[col] = splits

        if len(outputCols) == 0:
            outputCols = [i + "_buckets" for i in inputCols]
        for i in range(len(inputCols)):
            col = inputCols[i]
            df = df.withColumn(
                outputCols[i], cut_bins(col, splitsArray_dict[col], if_string)
            )
        return df
