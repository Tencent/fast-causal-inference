from .. import create_sql_instance, clickhouse_create_view,clickhouse_drop_view
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns
from matplotlib import rcParams
import warnings

from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae
from sklearn.metrics import roc_auc_score, auc, roc_curve, precision_recall_curve
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.model_selection import train_test_split
from matplotlib import rcParams
import warnings
warnings.filterwarnings('ignore')



def check_table(table):
    sql_instance = create_sql_instance()
    x = sql_instance.sql(f"select count(*) as cnt from {table} ")
    if "Code: 60" in x:
        print(x)
        raise ValueError
    elif int(x['cnt'][0])==0:
        print("There's no data in the table")
        raise ValueError
    else:
        return 1

def get_columns(table):
    # get all columns from table
    sql_instance = create_sql_instance()
    tmp = sql_instance.sql(f"desc {table} ")
    col_list = list(tmp['name'])
    colType_list = list(tmp['type'])
    cols_type = dict(zip(col_list,colType_list))
    return cols_type

def check_columns(table, cols):
    # check if cols exsits in table and return numerical_cols and string_cols
    sql_instance = create_sql_instance()
    cols_type = get_columns(table)
    cols_o = list(cols_type.keys())

    # check exist
    other_variables = set(cols) - set(cols_o)
    if len(other_variables) != 0:
        print(f"variable {other_variables} can't be find in the table {table}")
        raise ValueError

    string_cols = []
    numerical_cols = []
    for col in cols:
        if cols_type[col] not in ['UInt8', 'UInt16', 'UInt32', 'UInt64', 'UInt128', 'UInt256',
                              'Int8', 'Int16', 'Int32', 'Int64', 'Int128', 'Int256', 'Float32', 'Float64']:
            print(f"The type of {col} is not numeric")
            string_cols.append(col)
        else:
            numerical_cols.append(col)
    return numerical_cols,string_cols

def check_column(table, col):
    sql_instance = create_sql_instance()
    cols_type = get_columns(table)

    if col not in cols_type.keys():
        print(f"There is no column named {col} in the table")
        raise("ValueError")
        return -1
    if cols_type[col] not in ['UInt8', 'UInt16', 'UInt32', 'UInt64', 'UInt128', 'UInt256',
                              'Int8', 'Int16', 'Int32', 'Int64', 'Int128', 'Int256', 'Float32', 'Float64']:
        print(f"The type of {col} is not numeric")
        return 0
    else:
        return 1
    
        
    
def matching_plot(table,T,col):
    sql_instance = create_sql_instance()
    check_table(table)
    x1 = sql_instance.sql(f"select {col} from {table} where {T}=1 limit 10000")
    x0 = sql_instance.sql(f"select {col} from {table} where {T}=0 limit 10000")
    rcParams['figure.figsize'] = 8,8
    ax = sns.distplot(x0)
    sns.distplot(x1)
    ax.set_xlim(0, 1)
    ax.set_xlabel(col)
    ax.set_ylabel('density')
    ax.legend(['Control', 'Treatment'])
    del x1,x0
    
    

def SMD(table,T,cols):
    cols = list(set(cols))
    sql_instance = create_sql_instance()
    check_table(table)
    numerical_cols,string_cols = check_columns(table, cols)
    string = ','.join([f'avg({i}) as {i}_avg,varSamp({i}) as {i}_std' for i in numerical_cols])
    res = sql_instance.sql(f"select {T} as T, {string} from {table} group by {T} order by {T} ")
    res = np.array(res)[:,1:].T.reshape(-1,4)
    res = pd.DataFrame(res,columns=['Control','Treatment','Control_var','Treatment_var'])
    res = res.astype(float)
    res['SMD'] = (res['Treatment'] - res['Control'])/np.sqrt(0.5*(res['Control_var']+res['Treatment_var']))
    res = res[['Control','Treatment','SMD']]
    res.index = numerical_cols
    res = res.sort_values("SMD")
    return res

    
    
def data_split(table,test_size=0.5):
    sql_instance = create_sql_instance()
    # table check
    check_table(table)

    table_tmp = f'{table}_{int(time.time())}'
    table_train = f'{table_tmp}_train'
    table_test = f'{table_tmp}_test'
    
    clickhouse_create_view(
    clickhouse_view_name=table_tmp,
    sql_statement=f"""*,if(rand()/pow(2,32)<{test_size},1,0) as if_test""", 
    sql_table_name = table, 
    primary_column="if_test",
    is_force_materialize=True)
    
    clickhouse_create_view(
    clickhouse_view_name=table_train,
    sql_statement="""*""", 
    sql_table_name = table_tmp, 
    sql_where = """ if_test=0""", 
    primary_column="if_test",
    is_force_materialize=True)
    
    clickhouse_create_view(
    clickhouse_view_name=table_test,
    sql_statement="""*""", 
    sql_table_name = table_tmp, 
    sql_where = """ if_test=1 """,
    primary_column="if_test",
    is_force_materialize=True)
    
    clickhouse_drop_view(clickhouse_view_name=table_tmp) 
    print("table_train:",table_train)
    print("table_test:",table_test)
    
    return table_train,table_test
    
def describe(table,cols='*'):
    sql_instance = create_sql_instance()
    check_table(table)
    if cols == '*':
        cols = list(get_columns(table).keys())
    numerical_cols_all,string_cols = check_columns(table, cols)
    k = len(numerical_cols_all)
    res_all = pd.DataFrame([],columns = ['count','avg','std','min','quantile_0.25','quantile_0.5',
                                               'quantile_0.75','quantile_0.90','quantile_0.99','max'])
    for i in range(k):
        numerical_cols = numerical_cols_all[i*10:min((i+1)*10,k)]
        if len(numerical_cols)==0:
            break
        sql_list = [f"""count({numerical_cols[i]}) as cnt_{i},
                    avg({numerical_cols[i]}) as x{i}_avg, 
                    stddevSamp({numerical_cols[i]}) as x{i}_std,
                    min({numerical_cols[i]}) as x{i}_min,                   
                    quantile(0.25)({numerical_cols[i]}) as x{i}_25_quantile,
                    quantile(0.50)({numerical_cols[i]}) as x{i}_50_quantile,
                    quantile(0.75)({numerical_cols[i]}) as x{i}_75_quantile,
                    quantile(0.90)({numerical_cols[i]}) as x{i}_90_quantile,
                    quantile(0.99)({numerical_cols[i]}) as x{i}_99_quantile,
                    max({numerical_cols[i]}) as x{i}_max""" for i in range(len(numerical_cols))]
        result = sql_instance.sql(f'''select {','.join(sql_list)}  from {table}''').astype(float).values[0]
        res2 = pd.DataFrame(result.reshape(-1,10),columns=['count','avg','std','min','quantile_0.25','quantile_0.5',
                                               'quantile_0.75','quantile_0.90','quantile_0.99','max'])
        # res = pd.concat([res1,res2],axis=1)
        res = res2
        res.index = numerical_cols
        res_all = pd.concat([res_all,res])
    return res_all

    
    

plt.rcParams['font.sans-serif']=['SimHei']  # 用来正常显示中文标签  
plt.rcParams['axes.unicode_minus']=False  # 用来正常显示负号

def histplot(table,col,bin_num=50):
    sql_instance = create_sql_instance()
    sns.set()
    check_table(table)
    res = check_column(table,col)
    if res==0:
        df = sql_instance.sql(f"""
            select
                {col} as bins,count(*) as count
            from
                {table}
            group by {col}
            order by {col} 
        """)
        
        bins = df['bins']
        result = df['count']
        result = [float(i) for i in result]
        plt.bar(bins, result, width=1, edgecolor='black')
        plt.title(f'{col}', fontsize=16)
        plt.xlabel('Bar Chart')
        plt.ylabel(f'Count')
        plt.show()
    else:
        result2 = list(sql_instance.sql(f"""
        select {col}
        from(
            select
                {col},rand() as rand
            from
                {table}
        ) as a
        order by rand
        limit 100000
        """).values)
        result2 = [float(i[0]) for i in result2]
        # 使用sns在第二个子图上绘制KDE图
        sns.histplot(result2,kde=True,bins=bin_num)
        plt.title(col, fontsize=16)
        plt.xlabel('Histogram')
        plt.show()
        del result2        
        
    # return df

def boxplot(table,col):
    sql_instance = create_sql_instance()
    check_table(table)
    if check_column(table,col)<=0:
        raise("ValueError")
        return
    else:
        res = list(sql_instance.sql(f"""
            select
                min({col}) as numerator_min,         
                max({col}) as numerator_max,
                quantile(0.25)({col}) as numerator_25_quantile,
                quantile(0.50)({col}) as numerator_50_quantile,
                quantile(0.75)({col}) as numerator_75_quantile   
            from
                {table}
        """).values[0])
        res = [float(i) for i in res]

        # 计算五个统计量
        minimum = res[0]
        maximum = res[1]
        q1 = res[2]
        median = res[3]
        q3 = res[4]

        # 计算箱子的高度和位置
        box_height = q3 - q1
        box_position = median

        # 计算须的位置和长度
        whisker_left = np.max([q1 - 1.5 * box_height, minimum])
        whisker_right = np.min([q3 + 1.5 * box_height, maximum])
        whisker_length = whisker_right - whisker_left


        # 计算异常值
        bins = np.linspace(0,1,101)
        outliers = list(sql_instance.sql(f"""
        select
            {','.join([f'quantile({i})({col})' for i in bins])}
        from
            (select {col}
            from {table}
            where ({col} < {whisker_left}) or ({col} > {whisker_right}) )""").values[0])
        outliers = [float(i) for i in outliers]

        # 打印结果
        print("min:", minimum)
        print("25_quantile:", q1)
        print("50_quantile:", median)
        print("75_quantile:", q3)
        print("max:", maximum)
        # print("Box height:", box_height)
        # print("Box position:", box_position)
        # print("Whisker left:", whisker_left)
        # print("Whisker right:", whisker_right)
        # print("Whisker length:", whisker_length)
        # print("Outliers:", outliers)

        # 绘制箱线图
        sns.set()
        fig, ax = plt.subplots()

        # 绘制箱子
        rect = plt.Rectangle((box_position - 0.25, q1), 0.5, box_height, fill=False, edgecolor='#4C72B0', linewidth=1.5)
        ax.add_patch(rect)

        # 绘制中位数
        plt.plot([box_position - 0.25, box_position + 0.25], [median, median], color='#4C72B0', linewidth=1.5)

        # 绘制须
        plt.plot([box_position, box_position], [whisker_left, q1], color='#4C72B0', linewidth=1.5)
        plt.plot([box_position, box_position], [whisker_right, q3], color='#4C72B0', linewidth=1.5)

        # 绘制异常值
        plt.scatter([box_position] * len(outliers), outliers, marker='o', color='#4C72B0', alpha=0.7, s=10)

        # 设置坐标轴标签
        plt.xticks([box_position], ['Boxplot'])
        plt.tick_params(labelsize=12)

        plt.title(col, fontsize=16)
        plt.ylabel('value', fontsize=12)

        # 自适应调整盒子的宽度
        plt.xlim(box_position - 0.5, box_position + 0.5)

        # 显示图形
        plt.show()



