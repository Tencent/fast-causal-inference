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

public class OlsParser extends SqlCallCausal {
  private ArrayList<String> args;
  private ArrayList<SqlNode> params;

  private boolean use_state;


  public OlsParser(SqlParserPos pos, ArrayList<String> args, ArrayList<SqlNode> params, boolean use_state, EngineType engineType) {
    super(pos, engineType);
    this.args = args;
    this.params = params;
    this.causal_function_name = "ols";
    this.use_state = use_state;
    if (args.get(args.size() - 1).equals("0")) {
      params.add(SqlLiteral.createBoolean(false, SqlParserPos.ZERO));
      args.remove(args.size() - 1);
    }
  }

  @Override public SqlOperator getOperator() {
    return null;
  }

  @Override public void unparseClickHouse(SqlWriter writer, int leftPrec, int rightPrec) {
    if (this.use_state)
      writer.print("OlsState(");
    else
      writer.print("Ols(");
    // args to String with comma
    writer.print('\'' + String.join(",", args) + "',");

    for (SqlNode param : params) {
      param.unparse(writer, leftPrec, rightPrec);
    }

    if (params.isEmpty())
      writer.print("true");
    writer.print(")(");
    for (int i = 0; i < args.size(); i++) {
      if (i != 0)
        writer.print(",");
      writer.print(args.get(i));
    }
    writer.print(")");
  }

  @Override public void unparseStarRocks(SqlWriter writer, int leftPrec, int rightPrec) {
    if (this.use_state) {
      writer.print("ols_train(");
    }
    else {
      writer.print("ols(");
    }

    ArrayList<String> cols = new ArrayList<>();
    cols.add(args.get(0));
    cols.add(String.format("[%s]", String.join(",", args.subList(1, args.size()))));
    writer.print(String.join(",", cols));

    for (SqlNode param : params) {
      writer.print(",");
      param.unparse(writer, leftPrec, rightPrec);
    }

    if (params.isEmpty()){
      writer.print(",true");
    }
    writer.print(String.format(",'%s'", String.join(",", args)));
    writer.print(")");
  }

  @Override public List<SqlNode> getOperandList() {
    return null;
  }
}
