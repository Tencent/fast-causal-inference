import time
import math
from scipy.stats import norm
import json
import sys
import os

import pandas as pd
import numpy as np
import seaborn as sns
import time
from graphviz import Digraph
import textwrap
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import fdrcorrection
import pickle
import warnings
from .. import create_sql_instance, clickhouse_create_view, clickhouse_drop_view
#from fast_causal_inference import create_sql_instance, clickhouse_create_view, clickhouse_drop_view

class CausalForest:
    
    def __init__(self, depth = 10, min_node_size = -1, mtry = 3, num_trees = 10, sample_fraction = 0.7, weight_index = ''):
        self.depth = depth
        self.min_node_size = min_node_size
        self.mtry = mtry
        self.num_trees = num_trees
        self.sample_fraction = sample_fraction
        self.weight_index = weight_index
    
    def fit(self, y, t, x, table):
        self.table = table
        self.y = y
        self.t = t
        self.x = x
        self.ts = current_time_ms = int(time.time() * 1000)
        print(self.ts)
        # create model table
        self.model_table = 'model_' + table
        print(self.model_table)
        
        self.sql_instance = create_sql_instance()
        count = self.sql_instance.sql("select count() as cnt from " + table)
        if isinstance(count, str):
            print(count)
            return
        self.table_count = count['cnt'][0]
        if self.min_node_size == -1:
            self.min_node_size = int(max(int(self.table_count) / 2048, 1))
        self.config = f"""select '{{"weight_index":2, "outcome_index":0, "treatment_index":1, "min_node_size":{self.min_node_size}, "sample_fraction":{self.sample_fraction}, "mtry":{self.mtry}, "num_trees":{self.num_trees}}}' as model, rand() as ver"""
        clickhouse_drop_view(clickhouse_view_name = self.model_table)
        clickhouse_create_view(clickhouse_view_name = self.model_table, sql_statement = self.config, is_sql_complete = True, sql_table_name=table, primary_column = 'ver', is_force_materialize = False)
        if self.weight_index == '':
            self.weight_index = '1 / ' + str(self.table_count)
        self.xs = x.replace('+', ',')
            
        self.init_sql = f"""

        insert into {self.model_table} (model, ver)  
        WITH
        (select max(ver) from {self.model_table}) as ver0,
            (
                SELECT model
                FROM {self.model_table}
                WHERE ver = ver0 limit 1
            ) AS model
        SELECT CausalForest(model)({y}, {t}, {self.weight_index}, {self.xs}), ver0 + 1
        FROM {self.table}

        """
        res = self.sql_instance.sql(self.init_sql)
        
        self.train_sql = f"""
        
        insert into {self.model_table} (model, ver)  
        WITH
        (select max(ver) from {self.model_table}) as ver0,
            (
                SELECT model
                FROM {self.model_table}
                WHERE ver = ver0 limit 1
            ) AS model,
        (
        SELECT CausalForest(model)({y}, {t}, {self.weight_index}, {self.xs})
        FROM {table}
        ) as calcnumerdenom,
        (
        SELECT CausalForest(calcnumerdenom)({y}, {t}, {self.weight_index}, {self.xs})
        FROM {table}
        ) as split_pre
        SELECT CausalForest(split_pre)({y}, {t}, {self.weight_index}, {self.xs}), ver0 + 1
        FROM {table}

        """
        
        for i in range(self.depth):
            print('depth: ' + str(i+1))
            res = self.sql_instance.sql(self.train_sql)
            
    def effect(self, output_table, input_table = ''):
        if input_table != '':
            self.table = input_table
        self.output_table = output_table
        clickhouse_drop_view(clickhouse_view_name = output_table)
        self.predict_sql = f"""
            insert into {self.output_table} 
            WITH
                             (
                                 SELECT max(ver)
                                 FROM {self.model_table}
                             ) AS ver0,
                             (
                                 SELECT model
                                 FROM {self.model_table}
                                 WHERE ver = ver0 limit 1
                             ) AS model,
                         (SELECT CausalForestPredictState(model)(number) FROM numbers(0)) as predict_model
                         select *, evalMLMethod(predict_model, {self.t}, {self.weight_index}, {self.xs}) as effect
            FROM {self.table} 
        """
        
        self.create_table = f"""
        select *, 0.0 as effect from {self.table} limit 0
        """
        clickhouse_drop_view(clickhouse_view_name = self.output_table)
        clickhouse_create_view(clickhouse_view_name = self.output_table, sql_statement = self.create_table, is_sql_complete = True, sql_table_name=self.table, primary_column = 'effect', is_use_local=True)
        self.sql_instance.sql(self.predict_sql)
        print("succ")
        

