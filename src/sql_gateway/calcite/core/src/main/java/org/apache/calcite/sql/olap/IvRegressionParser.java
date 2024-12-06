/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to you under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.calcite.sql.olap;

import org.apache.calcite.sql.*;
import org.apache.calcite.sql.parser.SqlParserPos;

import java.util.ArrayList;
import java.util.List;

public class IvRegressionParser extends SqlCallCausal {
  private ArrayList<String> xs;
  String y;
  String ds;
  String iv;

  static String with_template_1 = "\n" +
      "    ( \n" +
      "        SELECT OlsState(true)(@D1, @IV1@X) \n" +
      "        FROM @TBL \n" +
      "    ) AS model1, \n" +
      "    ( \n" +
      "        SELECT OlsState(true)(@Y, evalMLMethod(model1, @IV1@X)@X) \n" +
      "        FROM @TBL \n" +
      "    ) AS model_final, \n" +
      "    ( \n" +
      "      select  \n" +
      "      MatrixMultiplication(true, false)(1, evalMLMethod(model1, @IV1@X)@X) from @TBL \n" +
      "    ) as xx_inverse, \n" +
      "    ( \n" +
      "      select  \n" +
      "      MatrixMultiplication(false, true)(1, evalMLMethod(model1, @IV1@X)@X, ABS(@Y - " +
      "evalMLMethod(model_final, @D1@X))) from @TBL \n" +
      "    )  as xx_weighted ";

  static String with_template_starrocks = "\n" +
      "    model1_tbl AS ( \n" +
      "        SELECT ols_train(@D1,[@IV1@X],true) as model1\n" +
      "        FROM @TBL \n" +
      "    ), \n" +
      "    model_final_tbl AS ( \n" +
      "        SELECT ols_train(@Y,[eval_ml_method(model1, [@IV1@X])@X],true) as model_final\n" +
      "        FROM @TBL,model1_tbl \n" +
      "    ), \n" +
      "    xx_inverse_tbl AS ( \n" +
      "      select  \n" +
      "      matrix_multiplication([1,eval_ml_method(model1,[@IV1@X])@X],true,false) as xx_inverse from @TBL,model1_tbl \n" +
      "    ), \n" +
      "    xx_weighted_tbl AS ( \n" +
      "      select  \n" +
      "      matrix_multiplication([1,eval_ml_method(model1,[@IV1@X])@X,ABS(@Y-" +
      "eval_ml_method(model_final,[@D1@X]))],false,true) as xx_weighted from @TBL,model1_tbl,model_final_tbl \n" +
      "    )";

  static String func_template_1 = "Ols('@ORIGIN_Y,predict(@PREDICT_PARAM) @X',  true, false, toString(xx_inverse), toString(xx_weighted))(@Y,evalMLMethod(model1, @IV1@X)@X)";

  static String func_template_starrocks = "ols(@Y,[eval_ml_method(model1,[@IV1@X])@X],true,'@ORIGIN_Y,predict(@PREDICT_PARAM) @X',xx_inverse,xx_weighted)";

  public IvRegressionParser(SqlParserPos pos, String y, String iv, String ds, ArrayList<String> xs, EngineType engineType) {
    super(pos, engineType);
    this.y = y;
    this.iv = iv;
    this.ds = ds;
    this.xs = xs;
    this.causal_function_name = "ivregression";
  }

  @Override public SqlOperator getOperator() {
    return null;
  }

  @Override public void unparseClickHouse(SqlWriter writer, int leftPrec, int rightPrec) {
    // parse : Y ~ (D1 ~ IV1) + (D2 ~ IV2) + ...

    this.ds = "toFloat64(" + this.ds + ")";
    this.iv = "toFloat64(" + this.iv + ")";
    this.y = "toFloat64(" + this.y + ")";

    String ph_xs = "";
    for (int i = 0; i < xs.size(); i++)
      ph_xs += "," + xs.get(i).replaceAll("\\+", ",");


    String with = with_template_1;
    with = with.replaceAll("@Y", this.y);
    with = with.replaceAll("@D1", this.ds);
    with = with.replaceAll("@IV1", this.iv);
    with = with.replaceAll("@X", ph_xs);
    withs.add(with);

    String func = func_template_1;
    func = func.replaceAll("@Y", this.y);
    func = func.replaceAll("@D1", this.ds);
    func = func.replaceAll("@IV1", this.iv);
    func = func.replaceAll("@X", ph_xs);
    String PREDICT_PARAM = String.format(this.iv + ph_xs).replaceAll(",", "+");
    func = func.replaceAll("@PREDICT_PARAM", PREDICT_PARAM.replaceAll("toFloat64", ""));
    func = func.replaceAll("@ORIGIN_Y", this.y.substring(0, this.y.length() - 1).replaceAll("toFloat64\\(", ""));
    writer.print(func);
  }

  @Override public void unparseStarRocks(SqlWriter writer, int leftPrec, int rightPrec) {
    // parse : Y ~ (D1 ~ IV1) + (D2 ~ IV2) + ...

    this.replace_table = "@TBL, model1_tbl, model_final_tbl, xx_inverse_tbl, xx_weighted_tbl";

    this.ds = String.format("cast(%s as double)", this.ds);
    StringBuilder ph_xs = new StringBuilder();
    for (String x : xs)
      ph_xs.append(",").append(x.replaceAll("\\+", ","));
    String X = ph_xs.toString();
    String PREDICT_PARAM = String.format("(" + this.iv + ")" + ph_xs).replaceAll(",", "+");
    this.iv = String.format("cast(%s as double)", this.iv);
    this.y = String.format("cast(%s as double)", this.y);

    String with = with_template_starrocks;
    with = with.replaceAll("@Y", this.y);
    with = with.replaceAll("@D1", this.ds);
    with = with.replaceAll("@IV1", this.iv);
    with = with.replaceAll("@X", X);
    withs.add(with);

    String func = func_template_starrocks;
    func = func.replaceAll("@Y", this.y);
    func = func.replaceAll("@D1", this.ds);
    func = func.replaceAll("@IV1", this.iv);
    func = func.replaceAll("@X", X);
    func = func.replaceAll("@PREDICT_PARAM", PREDICT_PARAM);
    func = func.replaceAll("@ORIGIN_Y", this.y.substring(0, this.y.length() - 1).replaceAll("toFloat64\\(", ""));
    writer.print(func);
  }

  @Override public List<SqlNode> getOperandList() {
    return null;
  }
}
