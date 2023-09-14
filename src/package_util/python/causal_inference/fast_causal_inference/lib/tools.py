from fast_causal_inference import create_sql_instance, clickhouse_create_view,clickhouse_drop_view
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

def check_table(table):
    sql_instance = create_sql_instance()
    x = sql_instance.sql(f"select count(*) as cnt from {table} ")
    if "Code: 60" in x:
        print(x)
        raise ValueError
        return -1
    elif int(x['cnt'][0])==0:
        print("There's no data in the table")
        raise ValueError
        return  0
    else:
        return 1

def check_column(table, col):
    sql_instance = create_sql_instance()
    x = sql_instance.sql(f"desc {table} ")
    cols_type = dict(zip(x['name'],x['type']))
    col_list = list(cols_type.keys())
        
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
    bucket_column="if_test",
    is_force_materialize=True)
    
    clickhouse_create_view(
    clickhouse_view_name=table_train,
    sql_statement="""*""", 
    sql_table_name = table_tmp, 
    sql_where = """ if_test=0""", 
    bucket_column="if_test",
    is_force_materialize=True)
    
    clickhouse_create_view(
    clickhouse_view_name=table_test,
    sql_statement="""*""", 
    sql_table_name = table_tmp, 
    sql_where = """ if_test=1 """,
    bucket_column="if_test",
    is_force_materialize=True)
    
    clickhouse_drop_view(clickhouse_view_name=table_tmp) 
    print("table_train:",table_train)
    print("table_test:",table_test)
    
    return table_train,table_test
    
def describe(table,X):
    sql_instance = create_sql_instance()
    check_table(table)
    results = []
    cols = X.split('+')
    for col in cols:
        if check_column(table, col)<=0:
            result = [0,0,0,0,0,0,0,0,0]
        else:
            result = list(sql_instance.sql(f"""
                select
                    count({col}) as cnt,
                    stddevSamp({col}) as numerator_std,
                    min({col}) as numerator_min,         
                    quantileExact(0.25)({col}) as numerator_25_quantile,
                    quantileExact(0.50)({col}) as numerator_50_quantile,
                    quantileExact(0.75)({col}) as numerator_75_quantile,
                    quantileExact(0.90)({col}) as numerator_90_quantile,
                    quantileExact(0.99)({col}) as numerator_99_quantile,
                    max({col}) as numerator_max
                from
                    {table}
            """).values[0])
            result = [float(i) for i in result]
            results.append(result)
    results = pd.DataFrame(np.array(results),columns=['count','std','min','quantile_0.25','quantile_0.5',
                                           'quantile_0.75','quantile_0.90','quantile_0.99','max'],index=cols)
    return results

    
    

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
                quantileExact(0.25)({col}) as numerator_25_quantile,
                quantileExact(0.50)({col}) as numerator_50_quantile,
                quantileExact(0.75)({col}) as numerator_75_quantile   
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
            {','.join([f'quantileExact({i})({col})' for i in bins])}
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