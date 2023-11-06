import scipy.stats as stats
from .. import create_sql_instance, clickhouse_create_view,clickhouse_drop_view
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns
from matplotlib import rcParams
import warnings

def IPWestimator(table,Y,T,P,n=100000,B=500):
    sql_instance = create_sql_instance()
    result = sql_instance.sql(f"""
    with (select DistributedNodeRowNumber() from {table}) as pa 
    select BootStrap('avg', {n}, {B}, pa)({Y}*{T}/({P}+0.001) + {Y}*(1-{T})/(1-{P}+0.001)) as ATE from {table};
    """).values[0][0].replace('[','').replace(']','').split(',')
    result = np.array([float(i) for i in result])
    ATE = np.mean(result)
    std = np.std(result)
    t_value = ATE/std
    p_value = (1 - stats.t.cdf(abs(t_value), n-1)) * 2
    confidence_interval = [ATE-1.96*std,ATE+1.96*std]
    return {'ATE':ATE,'stddev':std,'p_value':p_value,'confidence_interval':confidence_interval}

def ATEestimator(table,Y,T,n=100000,B=500):
    sql_instance = create_sql_instance()
    result = sql_instance.sql(f"""
    with (select DistributedNodeRowNumber() from {table}) as pa 
    select BootStrap('avg', {n}, {B}, pa)({Y}*{T}-{Y}*(1-{T})) as ATE from {table};
    """).values[0][0].replace('[','').replace(']','').split(',')
    result = np.array([float(i) for i in result])
    ATE = np.mean(result)
    std = np.std(result)
    t_value = ATE/std
    p_value = (1 - stats.t.cdf(abs(t_value), n-1)) * 2
    confidence_interval = [ATE-1.96*std,ATE+1.96*std]
    return {'ATE':ATE,'stddev':std,'p_value':p_value,'confidence_interval':confidence_interval}