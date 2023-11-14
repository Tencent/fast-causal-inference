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

  public PredictParser(SqlParserPos pos) {
    super(pos);
  }

  public PredictParser(SqlParserPos pos, String model, ArrayList<String> params) {
    super(pos);
    this.model = model;
    this.causal_function_name = "predict";
    this.params = params;
  }

  @Override public SqlOperator getOperator() {
    return null;
  }

  @Override public void unparse(SqlWriter writer, int leftPrec, int rightPrec) {
    withs.add("(select " + model.replaceAll("olsState", "OlsState").replaceAll("\\+",",").replaceAll("~", ",") + " from @TBL) as model");
    writer.print("evalMLMethod(model");
    for (int i = 0; i < params.size(); ++i) {
      writer.print("," + params.get(i));
    }
    writer.print(")");
  }

  @Override public List<SqlNode> getOperandList() {
    return null;
  }
}
