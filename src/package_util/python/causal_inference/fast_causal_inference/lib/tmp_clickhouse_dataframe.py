# -*- coding: utf-8 -*-
# Copyright 2023 Tencent Inc.  All rights reserved.
# Author: shichaohan@tencent.com

import time

import mmh3
import numpy as np

from fast_causal_inference.util import to_pandas, ClickHouseUtils


class TmpClickhouseDataFrame:
    """
    A Clikhouse Dataframe object that allows
    for easy python manipulation(temporarily).
    Table creation and manipulation are primarily
    implemented by Clickhouse SQL.

    Attributes
    ----------

    ch_df_name : str
        The name of the current (possibly uninstantiated)
        table in Clickhouse.

    original_df_name : str
        The name of the instantiated Clickhouse table.
        Usually the current data table is created by
        querying from this original data table.


    is_written : boolean
        An indicator of whether the current dataframe
        object is written to ClickHouse.

    select_sentence : list
        The select statement for query for the current table.
        Once created becomes empty. For lazy evaluation.

    select_sentence_from_parent : list
        The select statement from the parent table.
        Once created becomes empty. For lazy evaluation.

    where_sentence : list
        Where condition.


    groupby_sentence : str
        Group by columns.

    ais_instance : object
        An all in sql instance used for sql query.

    """

    ch_df_name = ""
    original_df_name = ""
    is_written = False
    select_sentence = []
    select_sentence_from_parent = []
    where_sentence = []
    groupby_sentence = ""
    ais_instance = None

    def __init__(
        self,
        ais_instance,
        df_name="",
        original_ch_df="",
        select_sentence=[],
        groupby_sentence="",
        where_sentence=[],
        select_sentence_from_parent=[],
    ):
        self.select_sentence = select_sentence
        self.where_sentence = where_sentence
        self.groupby_sentence = groupby_sentence
        self.ais_instance = ais_instance
        self.select_sentence_from_parent = select_sentence_from_parent

        if df_name == original_ch_df:
            self.is_written = True

        if df_name == "":
            instance_add = (
                self.ais_instance.__repr__().split(" ")[-1].replace(">", "")[2:]
            )
            rand_num = str(
                np.abs(mmh3.hash(instance_add, seed=np.random.randint(100000000)))
            )
            self.ch_df_name = "t_" + instance_add + str(int(time.time())) + rand_num
        else:
            self.ch_df_name = df_name
        self.select_sentence += ["*"]
        self.original_df_name = original_ch_df

    def filter(self, filter_cond):
        """
        Filter dataframe.

        Parameters
        ----------

        filter_cond : str
            The condition for filtering

        Returns
        -------
            new_df_obj: object
                Returns a new dataframe object.

        Examples
        --------
        >>> df = TmpClickhouseDataFrame(sql_instance,ch_df_name,original_ch_df = ch_df_name)
        >>> df_nonzero = df.filter("numerator>0")
        """
        new_df_obj = self.make_a_copy()
        new_df_obj.where_sentence += [filter_cond]
        return new_df_obj

    def write_to_ch(
        self, verbose=False, is_physical_table=True, is_distributed_create=False
    ):
        """
        Write dataframe to clickhouse.

        Parameters
        ----------

        verbose : boolean, optional
            Whether to print debug strings while executing; False by default.

        Examples
        --------
        >>> df = TmpClickhouseDataFrame(sql_instance,ch_df_name,original_ch_df = ch_df_name)
        >>> df_nonzero = df.filter("numerator>0")
        >>> df_nonzero.write_to_ch()
        """
        if len(self.ch_df_name) < 1:
            print("Please check: Initial table is not instantiated. ")
        statm = f"""select {','.join(list(set(self.select_sentence + self.select_sentence_from_parent)))} from {self.original_df_name} 
        where {" and ".join(list(set(self.where_sentence)))} """

        if len(self.groupby_sentence) > 0:
            statm += f""" goupr by {self.groupby_sentence}"""
        if not self.is_written:
            if verbose:
                print("executing ", statm)
            ClickHouseUtils.clickhouse_create_view_v2(
                self.ch_df_name,
                statm,
                is_physical_table=True,
                is_distributed_create=False,
            )
            self.is_written = True
            self.original_df_name = self.ch_df_name
            self.select_sentence = ["*"]
            self.select_from_parent = ["*"]

    def __repr__(self):
        return self.ch_df_name

    def make_a_copy(self):
        """
        Copy the current table.

        Returns
        -------
            new_df_obj: object
                Returns a new dataframe object.

        Examples
        --------
        >>> df = TmpClickhouseDataFrame(sql_instance,ch_df_name,original_ch_df = ch_df_name)
        >>> df_copy = df.make_a_copy()
        """

        new_df_obj = TmpClickhouseDataFrame(
            self.ais_instance,
            original_ch_df=self.original_df_name,
            select_sentence=[],
            select_sentence_from_parent=self.select_sentence
            + self.select_sentence_from_parent,
            groupby_sentence=self.groupby_sentence,
            where_sentence=self.where_sentence.copy(),
        )
        return new_df_obj

    def withColumn(self, new_col_name, new_column_statement):
        """
        Add a column to the table.

        Parameters
        ----------

        new_col_name : str
            The name of the new column.


        new_col_statement : str
            The creation statement of the new column.


        Returns
        -------
            new_df_obj: object
                Returns a new dataframe object.

        Examples
        --------
        >>> df = TmpClickhouseDataFrame(sql_instance,ch_df_name,original_ch_df = ch_df_name)
        >>> df = df.withColumn('t_indicaotr', 'groupid = 1234')
        """
        new_df = self.make_a_copy()
        new_df.select_sentence += [f"""({new_column_statement}) as {new_col_name} """]
        return new_df

    def ttest_2samp(
        self,
        outcome,
        index,
        alternative="two-sided",
        cuped="",
        pse="",
        where_cond="",
        verbose=False,
    ):
        """
        Performs two-sample t-test.

        Parameters
        ----------

        outcome : str
            The expression for outcome. avg(column) etc.


        index : str
            The column name for binary treatment indicator.

        alternative : str, optional
            One of 'tow-sided','less','greater'; 'two-sided' by default.

        cuped : str, optional
            The expression for CUPED variance reduction.

        pse : str, optional
            The expression for PSE variance reduction.

        where_cond : str, optional
            The condition for additional filtering in ttest 2samp.

        verbose : boolean, optional
            Whether to


        Returns
        -------
            result : object
                Returns a pandas dataframe containing the testing result.

        Examples
        --------
        >>> df = TmpClickhouseDataFrame(sql_instance,ch_df_name,original_ch_df = ch_df_name)
        >>> df = df.withColumn('t_indicaotr', 'groupid = 1234')
        >>> df.ttest_2samp("avg(numerator)", "t_indicator", where_cond="metric_id=1234",verbose=True)
        """

        if not self.is_written:
            self.write_to_ch()

        cuped_statm = ""
        pse_statm = ""
        if len(cuped) > 0:
            cuped_statm += "," + cuped
        if len(pse) > 0:
            pse_statm += ",pse=" + pse

        statm = f"""select ttest_2samp({outcome}, {index}, '{alternative}' {cuped_statm} {pse_statm} ) as ttest_result
        from {self.ch_df_name} """
        if len(where_cond) > 0:
            statm += " where " + where_cond

        if verbose:
            print("start exectuing ", statm)
        ttest_result = self.ais_instance.sql(statm)
        return to_pandas(ttest_result["ttest_result"])

    def show(self):
        if not self.is_written:
            self.write_to_ch()

        return self.ais_instance.sql("select * from " + self.ch_df_name)

    def mannWhitneyUTest(self, outcome, index, alternative="two-sided", where_cond=""):
        """
        Performs two-sample Mann-Whitney U Test.

        Parameters
        ----------

        outcome : str
            The expression for outcome column.


        index : str
            The column name for binary treatment indicator.

        alternative : str, optional
            One of 'tow-sided','less','greater'; 'two-sided' by default.

        where_cond : str, optional
            The condition for additional filtering in ttest 2samp.

        Returns
        -------
            result : object
                Returns a pandas dataframe containing the testing result.

        Examples
        --------
        >>> df = TmpClickhouseDataFrame(sql_instance,ch_df_name,original_ch_df = ch_df_name)
        >>> df = df.withColumn('t_indicaotr', 'groupid = 1234')
        >>> df.mannWhitneyUTest("numerator", "t_indicator", where_cond="metric_id=1234")
        """

        if not self.is_written:
            self.write_to_ch()
        statm = f"""select mannWhitneyUTest('{alternative}')({outcome}, {index}) as utest_result
        from {self.ch_df_name} """
        if len(where_cond) > 0:
            statm += " where " + where_cond
        utest_result = self.ais_instance.sql(statm)
        return utest_result["utest_result"]
