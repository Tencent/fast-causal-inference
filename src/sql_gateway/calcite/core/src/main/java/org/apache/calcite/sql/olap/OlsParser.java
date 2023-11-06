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

  public OlsParser(SqlParserPos pos) {
    super(pos);
  }

  public OlsParser(SqlParserPos pos, ArrayList<String> args, ArrayList<SqlNode> params, boolean use_state) {
    super(pos);
    this.args = args;
    this.params = params;
    this.causal_function_name = "ols";
    this.use_state = use_state;
  }

  @Override public SqlOperator getOperator() {
    return null;
  }

  @Override public void unparse(SqlWriter writer, int leftPrec, int rightPrec) {
    if (this.use_state)
      writer.print("OlsState(");
    else
      writer.print("Ols(");
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
      //args.get(i).unparse(writer, leftPrec, rightPrec);
    }
    writer.print(")");
  }

  @Override public List<SqlNode> getOperandList() {
    return null;
  }
}
