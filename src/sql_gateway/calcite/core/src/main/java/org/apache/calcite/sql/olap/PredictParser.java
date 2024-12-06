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

public class PredictParser extends SqlCallCausal {
  private ArrayList<String> params;

  private String model = "";


  public PredictParser(SqlParserPos pos, String model, ArrayList<String> params, EngineType engineType) {
    super(pos, engineType);
    this.model = model;
    this.causal_function_name = "predict";
    this.params = params;
  }

  @Override public SqlOperator getOperator() {
    return null;
  }

  @Override public void unparseClickHouse(SqlWriter writer, int leftPrec, int rightPrec) {
    withs.add("(select " + model.replaceAll("olsState", "OlsState").replaceAll("\\+",",").replaceAll("~", ",") + " from @TBL) as model");
    writer.print("evalMLMethod(model");
    for (int i = 0; i < params.size(); ++i) {
      writer.print("," + params.get(i));
    }
    writer.print(")");
  }

  @Override public void unparseStarRocks(SqlWriter writer, int leftPrec, int rightPrec) {
    model = model.replaceAll("olsState", "ols_train").replaceAll("\\+",",").replaceAll("~", ",");
    String[] cols = model.substring(model.indexOf('(') + 1, model.indexOf(")")).split(",");
    String y = cols[0];
    ArrayList<String> X = new ArrayList<String>();
    String useBias = "true";

    for (int i = 1; i < cols.length; ++i) {
      if (i == cols.length - 1 && (cols[i].equals("TRUE") || cols[i].equals("true") || cols[i].equals("FALSE") || cols[i].equals("false"))) {
        useBias = cols[i];
      } else {
        X.add(cols[i]);
      }
    }
    withs.add(String.format("__eval_ml_tmp_tbl__ as (select ols_train(%s,[%s],%s) as model from @TBL)", y, String.join(",", X), useBias));
    this.replace_table = "@TBL, __eval_ml_tmp_tbl__";
    writer.print("eval_ml_method(model,");
    writer.print(String.format("[%s]", String.join(",", params)));
    writer.print(")");
  }

  static Boolean isBooleanStr(String str) {
    if (str.equals("TRUE") || str.equals("true")) {
      return true;
    }
    if (str.equals("FALSE") || str.equals("false")) {
      return false;
    }
    return null;
  }

  @Override public List<SqlNode> getOperandList() {
    return null;
  }
}
