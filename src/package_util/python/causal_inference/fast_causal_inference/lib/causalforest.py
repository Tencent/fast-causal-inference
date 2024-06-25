import time

from fast_causal_inference.util import SqlGateWayConn, ClickHouseUtils


class CausalForest:
    def __init__(
        self,
        depth=10,
        min_node_size=-1,
        mtry=3,
        num_trees=10,
        sample_fraction=0.7,
        weight_index="",
        honesty=False,
        honesty_fraction=0.5,
        quantile_num=50,
    ):
        self.depth = depth
        self.min_node_size = min_node_size
        self.mtry = mtry
        self.num_trees = num_trees
        self.sample_fraction = sample_fraction
        self.weight_index = weight_index
        self.honesty = 0
        if honesty == True:
            self.honesty = 1
        self.honesty_fraction = honesty_fraction
        self.quantile_num = quantile_num
        self.quantile_num = max(1, min(100, self.quantile_num))

    def fit(self, y, t, x, table):
        self.table = table
        self.origin_table = table
        self.y = y
        self.t = t
        self.x = x
        self.ts = current_time_ms = int(time.time() * 1000)

        # create model table
        self.model_table = "model_" + table + str(self.ts)

        sql_instance = SqlGateWayConn.create_default_conn()
        count = sql_instance.sql("select count() as cnt from " + table)
        if isinstance(count, str):
            print(count)
            return
        self.table_count = count["cnt"][0]

        self.mtry = min(30, self.mtry)
        self.num_trees = min(200, self.num_trees)
        calc_min_node_size = int(max(int(self.table_count) / 128, 1))
        self.min_node_size = max(self.min_node_size, calc_min_node_size)
        # insert into {self.model_table}
        self.config = f""" select '{{"max_centroids":1024,"max_unmerged":2048,"honesty":{self.honesty},"honesty_fraction":{self.honesty_fraction}, "quantile_size":{self.quantile_num}, "weight_index":2, "outcome_index":0, "treatment_index":1, "min_node_size":{self.min_node_size}, "sample_fraction":{self.sample_fraction}, "mtry":{self.mtry}, "num_trees":{self.num_trees}}}' as model, {self.ts} as ver"""
        ClickHouseUtils.clickhouse_drop_view(clickhouse_view_name=self.model_table)
        ClickHouseUtils.clickhouse_drop_view(clickhouse_view_name=self.model_table)
        ClickHouseUtils.clickhouse_create_view(
            clickhouse_view_name=self.model_table,
            sql_statement=self.config,
            is_sql_complete=True,
            sql_table_name=table,
            primary_column="ver",
            is_force_materialize=False,
        )
        sql_instance.sql(self.config)
        if self.weight_index == "":
            self.weight_index = "1 / " + str(self.table_count)
        self.xs = x.replace("+", ",")

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
        res = sql_instance.sql(self.init_sql)

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
            print("deep " + str(i + 1) + " train over")
            res = sql_instance.execute(self.train_sql)
            if isinstance(res, str) == True and res.find("train over") != -1:
                print("----------训练结束----------")
                break

    def effect(self, output_table, input_table="", xs=""):
        if xs != "":
            self.xs = xs
        if input_table != "":
            self.table = input_table
        self.output_table = output_table
        ClickHouseUtils.clickhouse_drop_view(clickhouse_view_name=output_table)
        self.predict_sql = f"""
            WITH
                             (
                                 SELECT max(ver)
                                 FROM {self.model_table}
                             ) AS ver0,
                             (
                                 SELECT model
                                 FROM {self.model_table}
                                 WHERE ver = ver0 limit 1
                             ) AS pure,
                         (SELECT CausalForestPredict(pure)({self.y}, 0, {self.weight_index}, {self.xs}) FROM {self.origin_table}) as model,
                         (SELECT CausalForestPredictState(model)(number) FROM numbers(0)) as predict_model
                         select *, evalMLMethod(predict_model, 0, {self.weight_index}, {self.xs}) as effect
            FROM {self.table} 
        """

        ClickHouseUtils.clickhouse_drop_view(clickhouse_view_name=self.output_table)
        ClickHouseUtils.clickhouse_drop_view(clickhouse_view_name=self.output_table)
        ClickHouseUtils.clickhouse_create_view(
            clickhouse_view_name=self.output_table,
            sql_statement=self.predict_sql,
            is_sql_complete=True,
            sql_table_name=self.model_table,
            primary_column="effect",
            is_use_local=False,
        )
        print("succ")
