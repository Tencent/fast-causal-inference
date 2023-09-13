from ..all_in_sql import *
from .ols import *
from .. import clickhouse_create_view


def PolynomialFeatures(x, k, type='k'):
    res = []
    effect_ph = []
    marginal_effect_ph = []
    for i in range(1, k + 1):
        res.append('pow(' + str(x) + ',' + str(i) + ')')
        effect_ph.append('pow(' + '@PH' + ',' + str(i) + ')')
        marginal_effect_ph.append('pow(' + '@PH' + str(i) + ',' + str(i) + ')')

    return [res, effect_ph, marginal_effect_ph]


class LinearDML:
    def __init__(self, Y, T, X, W='', model_y='Ols', model_t='Ols', fit_cate_intercept=True, discrete_treatment=True,
                 categories=[0, 1], cv=3, table='', treatment_featurizer='', debug=False):
        if model_y != '' and model_y[0] == 'o':
            model_y = model_y[0].upper() + model_y[1:]
        if model_t != '' and model_t[0] == 'o':
            model_t = model_t[0].upper() + model_t[1:]

        self.treatment_featurizer = ''
        self.effect_ph = ''
        self.marginal_effect_ph = ''
        if treatment_featurizer != '':
            self.treatment_featurizer = treatment_featurizer[0]
            self.effect_ph = treatment_featurizer[1]
            self.marginal_effect_ph = treatment_featurizer[2]
        self.Y = Y
        self.T = T
        self.debug = debug
        self.X = X
        self.table = table
        self.sql_instance = AllInSqlConn()
        self.dml_sql = self.get_dml_sql(table, Y, T, X, W, model_y, model_t, cv, self.treatment_featurizer)
        self.ols = self.sql_instance.sql(self.dml_sql)

        if self.debug == True:
            print("dml_sql: ", self.dml_sql)
            print("forward_sql: ", self.forward_sql)

        if isinstance(self.ols, Ols) == True:
            self.success = True
        else:
            self.success = False

    def get_dml_sql(self, table, Y, T, X, W, model_y, model_t, cv, treatment_featurizer):
        sql = "select linearDML("
        sql += Y + "," + T + "," + X + ",";
        if W.strip() != '':
            sql += W + ",";
        sql += "model_y=" + model_y + "," + "model_t=" + model_t + "," + "cv=" + str(cv)
        if treatment_featurizer != '':
            sql += ",treatment_featurizer=" + "+".join(treatment_featurizer)
        sql += " ) from " + table
        print(sql)
        return sql;

    def __str__(self):
        return str(self.ols)

    def summary(self):
        if (self.debug):
            print("success: ", self.success)
        if self.success == False:
            return str(self.ols)
        return self.ols.get_dml_summary()

    def exchange_dml_sql(self, sql, use_interval=False):
        pos = sql.find("final_model")
        if pos == -1:
            raise Exception("Logical Error: final_model not found in sql")
        sql = sql[0:pos + len("final_model")]
        pos = sql.rfind("Ols")
        if pos == -1:
            raise Exception("Logical Error: Ols not found in sql")
        if use_interval == True:
            sql = sql[0:pos] + "OlsIntervalState" + sql[pos + len("Ols"):]
        else:
            sql = sql[0:pos] + "OlsState" + sql[pos + len("Ols"):]
        return sql

    def effect(self, X='', T0=0, T1=1, table_output='', table_predict=''):
        if table_predict == '':
            table_predict = self.table
        if self.success == False:
            return str(self.ols)
        if X == '':
            X = self.X
        sql = self.exchange_dml_sql(self.forward_sql) + "\n"
        if self.treatment_featurizer != '':
            tmp_eval = 'evalMLMethod(final_model'
            for x in X.split('+'):
                for y in self.effect_ph:
                    tmp_eval += ',' + str(x) + '*' + str(y);
            for y in self.effect_ph:
                tmp_eval += ',' + str(y)
            tmp_eval += ')'
            sql += 'select ' + tmp_eval.replace('@PH', str(T1)) + ' - ' + tmp_eval.replace('@PH',
                                                                                           str(T0)) + ' as predict from ' + table_predict
            if table_output == '':
                sql += ' limit 100'
        else:
            X = X.replace('+', ',')
            X1 = X.split(',')
            X1 = [x + " as " + x for x in X1]
            if table_output == '':
                sql += 'select evalMLMethod(final_model, ' + X + ', ' + str(
                    T1 - T0) + ') as predict from ' + table_predict + ' limit 100'
            else:
                sql += 'select '
                if table_output == '':
                    sql += self.Y + ' as Y,' + self.T + " as T,"
                sql += X + ', evalMLMethod(final_model, ' + X + ', ' + str(
                    T1 - T0) + ') as predict from ' + table_predict
        if self.debug:
            print("effect sql: ", sql)
        if table_output != '':
            if X.count("+") >= 30 or X.count(",") >= 30:
                print("The number of x exceeds the limit 30")
            sql = "create table " + table_output + " on cluster allinsql engine = MergeTree() order by predict as " + sql
        tmp_sql = AllInSqlConn(use_sql_forward=False)
        t = tmp_sql.sql(sql)
        return t

    def ate(self, X='', T0=0, T1=1):
        if self.success == False:
            return str(self.ols)
        if X == '':
            X = self.X
        sql = self.exchange_dml_sql(self.forward_sql) + "\n"
        X = X.replace('+', ',')
        sql += 'select  avg(evalMLMethod(final_model, ' + X + ', ' + str(T1 - T0) + ')) from ' + self.table
        sql += " limit 100"
        if self.debug:
            print("ate sql: ", sql)
        tmp_sql = AllInSqlConn(use_sql_forward=False)
        t = tmp_sql.sql(sql)
        return t

    def effect_interval(self, X='', T0=0, T1=1, alpha=0.05, table_output=''):
        if self.success == False:
            return str(self.ols)
        if X == '':
            X = self.X
        sql = self.exchange_dml_sql(self.forward_sql, use_interval=True) + "\n"
        X = X.replace('+', ',')
        X1 = X.split(',')
        X1 = [x + " as " + x for x in X1]
        if table_output == '':
            sql += 'select evalMLMethod(final_model,\'confidence\',' + str(1 - alpha) + ', ' + X + ', ' + str(
                T1 - T0) + ') from ' + self.table + ' limit 100'
        else:
            sql += 'select ' + ", ".join(X1) + ', evalMLMethod(final_model,\'confidence\',' + str(
                1 - alpha) + ', ' + X + ', ' + str(T1 - T0) + ') as predict from ' + self.table

        if table_output != '':
            clickhouse_create_view(clickhouse_view_name=table_output, sql_statement=sql, bucket_column='predict',
                                   is_force_materialize=True, use_sql_forward=False, is_sql_complete=True,
                                   sql_table_name=self.table)
            return
        if self.debug:
            print("effect interval sql: ", sql)
        tmp_sql = AllInSqlConn(use_sql_forward=False)
        t = tmp_sql.sql(sql)
        return t

    def ate_interval(self, X='', T0=0, T1=1, alpha=0.05):
        if self.success == False:
            return str(self.ols)
        if X == '':
            X = self.X
        sql = self.exchange_dml_sql(self.forward_sql, use_interval=True) + "\n"
        X = X.replace('+', ',')
        Xs = X.split(',')
        for i in range(len(Xs)):
            Xs[i] = 'avg(' + Xs[i] + ')'
        X = ','.join(Xs)
        sql += 'select evalMLMethod(final_model,\'confidence\',' + str(1 - alpha) + ', ' + X + ', ' + str(
            T1 - T0) + ') from ' + self.table
        sql += " limit 100"
        if self.debug:
            print("ate interval sql: ", sql)
        tmp_sql = AllInSqlConn(use_sql_forward=False)
        t = tmp_sql.sql(sql)
        if str(t).find('DB::Exception') != -1:
            return t
        s = "mean_point\tci_mean_lower\tci_mean_upper\t\n"
        t = t[0][0]
        t = t[1:-1]
        t = t.split(',')
        for i in range(len(t)):
            s += str(round(float(t[i]), 10)) + '\t'
        return s

    def get_sql(self, X):
        sql = self.exchange_dml_sql(self.forward_sql) + "\n"
        tmp_eval = 'evalMLMethod(final_model'
        for x in X.split('+'):
            for y in self.marginal_effect_ph:
                tmp_eval += ',' + str(x) + '*' + str(y);
        for y in self.marginal_effect_ph:
            tmp_eval += ',' + str(y)
        tmp_eval += ')'

        evals = []
        for i in range(0, len(self.marginal_effect_ph) + 1):
            evals.append(tmp_eval)

        for i in range(1, len(self.marginal_effect_ph) + 1):
            for j in range(0, len(evals)):
                if i != j:
                    evals[j] = evals[j].replace('@PH' + str(i), '0')
                else:
                    evals[j] = evals[j].replace('@PH' + str(i), '1')

        sql_const = sql + ' select '
        sql_effect = sql_const
        sql_ate = sql_const + ' avg( '
        for i in range(1, len(self.marginal_effect_ph) + 1):
            sql_const += evals[i] + ' - ' + evals[0] + ' as predict' + str(i) + ','
            sql_effect += evals[i] + ' - ' + evals[0] + ' +'
            sql_ate += evals[i] + ' - ' + evals[0] + ' +'

        sql_const = sql_const[:-1] + ' from ' + self.table
        sql_effect = sql_effect[:-1] + ' as predict from ' + self.table
        sql_ate = sql_ate[:-1] + ') as predict from ' + self.table

        return [sql_const, sql_effect, sql_ate]

    def const_marginal_effect(self, X='', table_output=''):
        if self.success == False:
            return str(self.ols)
        if X == '':
            X = self.X
        if self.marginal_effect_ph == '':
            return "Error: treatment featurizer is empty!"
        sql = self.get_sql(X)[0]
        if table_output == '':
            sql += ' limit 100'
        if table_output != '':
            clickhouse_create_view(clickhouse_view_name=table_output, sql_statement=sql, bucket_column='predict1',
                                   is_force_materialize=True, use_sql_forward=False, is_sql_complete=True,
                                   sql_table_name=self.table)
            return
        if self.debug:
            print("effect sql: ", sql)
        tmp_sql = AllInSqlConn(use_sql_forward=False)
        t = tmp_sql.sql(sql)
        return t

    def marginal_effect(self, X='', table_output=''):
        if self.success == False:
            return str(self.ols)
        if X == '':
            X = self.X
        if self.marginal_effect_ph == '':
            return "Error: treatment featurizer is empty!"
        sql = self.get_sql(X)[1]
        if table_output == '':
            sql += ' limit 100'

        if table_output != '':
            if sql.count("+") >= 30 or sql.count(",") >= 30:
                print("The number of x exceeds the limit 40")
            clickhouse_create_view(clickhouse_view_name=table_output, sql_statement=sql, bucket_column='predict',
                                   is_force_materialize=True, use_sql_forward=False, is_sql_complete=True,
                                   sql_table_name=self.table)
        tmp_sql = AllInSqlConn(use_sql_forward=False)
        t = tmp_sql.sql(sql)
        return t

    def marginal_ate(self, X='', table_output=''):
        if self.success == False:
            return str(self.ols)
        if X == '':
            X = self.X
        if self.marginal_effect_ph == '':
            return "Error: treatment featurizer is empty!"
        sql = self.get_sql(X)[2]
        if table_output == '':
            sql += ' limit 100'
        if table_output != '':
            clickhouse_create_view(clickhouse_view_name=table_output, sql_statement=sql, bucket_column='predict',
                                   is_force_materialize=True, use_sql_forward=False, is_sql_complete=True,
                                   sql_table_name=self.table)
            return
        if self.debug:
            print("effect sql: ", sql)
        tmp_sql = AllInSqlConn(use_sql_forward=False)
        t = tmp_sql.sql(sql)
        return t


class NonParamDML:
    def __init__(self, Y, T, X, W='', model_y='Ols', model_t='Ols', fit_cate_intercept=True, discrete_treatment=True,
                 categories=[0, 1], cv=3, table='', debug=False):
        self.debug = debug
        self.X = X
        self.table = table
        self.sql_instance = AllInSqlConn()
        self.dml_sql = self.get_dml_sql(table, Y, T, X, W, model_y, model_t, cv)
        self.forward_sql = sql_forward(self.dml_sql)
        self.ols = self.sql_instance.sql(self.dml_sql)
        if self.debug == True:
            print("dml_sql: ", self.dml_sql)
            print("forward_sql: ", self.forward_sql)

        if isinstance(self.ols, Ols) == True:
            self.success = True
        else:
            self.success = False

    def get_dml_sql(self, table, Y, T, X, W, model_y, model_t, cv):
        sql = "select nonParamDML("
        sql += Y + "," + T + "," + X + ",";
        if W.strip() != '':
            sql += W + ",";
        sql += "model_y=" + model_y + "," + "model_t=" + model_t + "," + "cv=" + str(cv) + ")"
        sql += " from " + table
        return sql;

    def __str__(self):
        return str(self.ols)

    def summary(self):
        if (self.debug):
            print("success: ", self.success)
        if self.success == False:
            return str(self.ols)
        return self.ols.get_dml_summary()

    def exchange_dml_sql(self, sql, use_interval=False):
        pos = sql.find("final_model")
        if pos == -1:
            raise Exception("Logical Error: final_model not found in sql")
        sql = sql[0:pos + len("final_model")]
        pos = sql.rfind("Ols")
        if pos == -1:
            raise Exception("Logical Error: Ols not found in sql")
        if use_interval == True:
            sql = sql[0:pos] + "OlsIntervalState" + sql[pos + len("Ols"):]
        else:
            sql = sql[0:pos] + "OlsState" + sql[pos + len("Ols"):]
        return sql
