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

public class DIDParser extends SqlCallCausal {
  private ArrayList<String> args;


  public DIDParser(SqlParserPos pos, ArrayList<String> args, EngineType engineType) {
    super(pos, engineType);
    this.args = args;
    this.causal_function_name = "did";
  }

  @Override public SqlOperator getOperator() {
    return null;
  }

  @Override public void unparseClickHouse(SqlWriter writer, int leftPrec, int rightPrec) {
    writer.print("Ols(");
    args.add(3, args.get(1) + "*" + args.get(2));
    String arguments = String.join(",", args);
    writer.print('\'' + arguments + "')");
    writer.print("(" + arguments + ")");
  }

  @Override public void unparseStarRocks(SqlWriter writer, int leftPrec, int rightPrec) {
    if (args.size() < 3) {
      throw new RuntimeException("number of args is less than three.");
    }
    String Y = args.get(0);
    ArrayList<String> X = new ArrayList<>();
    X.add(args.get(1));
    X.add(args.get(2));
    X.add(String.format("%s*%s", args.get(1), args.get(2)));
    for (int i = 3; i < args.size(); ++i) {
      X.add(args.get(i));
    }
    StringBuilder names = new StringBuilder("'" + Y);
    for (String x : X) {
      names.append(",").append(x);
    }
    names.append("'");
    writer.print(String.format("ols(%s,[%s],true,%s)", Y, String.join(",", X), names));
  }

  @Override public List<SqlNode> getOperandList() {
    return null;
  }
}
