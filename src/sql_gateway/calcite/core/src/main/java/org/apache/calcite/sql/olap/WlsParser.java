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

public class WlsParser extends SqlCallCausal {
  private ArrayList<SqlNode> args;
  private ArrayList<SqlNode> params;


  public WlsParser(SqlParserPos pos, ArrayList<SqlNode> args, ArrayList<SqlNode> params, EngineType engineType) {
    super(pos, engineType);
    this.args = args;
    this.params = params;
    this.causal_function_name = "wls";
  }

  @Override public SqlOperator getOperator() {
    return null;
  }

  @Override public void unparseClickHouse(SqlWriter writer, int leftPrec, int rightPrec) {
    writer.print("Wls(");
    args.add(params.get(0));
    if (params.size() == 2) {
      writer.print(params.get(1) + ")(");
    }
    for (int i = 0; i < args.size(); i++) {
      if (i != 0)
        writer.print(",");
      args.get(i).unparse(writer, leftPrec, rightPrec);
    }
    writer.print(")");
  }

  @Override public void unparseStarRocks(SqlWriter writer, int leftPrec, int rightPrec) {
    writer.print("wls(");
    args.get(0).unparse(writer, leftPrec, rightPrec);
    writer.print(",[");
    for (int i = 1; i < args.size(); i++) {
      if (i > 1){
        writer.print(",");
      }
      args.get(i).unparse(writer, leftPrec, rightPrec);
    }
    writer.print("]");
    for (SqlNode param : params) {
      writer.print(",");
      param.unparse(writer, leftPrec, rightPrec);
    }
    if (params.size() == 1) {
      writer.print(",true");
    }
    writer.print(")");
  }

  @Override public List<SqlNode> getOperandList() {
    return null;
  }
}
