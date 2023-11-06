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
import org.apache.calcite.util.Pair;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import static java.util.Collections.sort;

public class Ttest1SampParser extends SqlCallCausal {
  private ArrayList<String> map, map_cuped;
  private String func, func_cuped, mu, altertive;

  public Ttest1SampParser(SqlParserPos pos) {
    super(pos);
  }

  public Ttest1SampParser(SqlParserPos pos, ArrayList<String> map, String func, String altertive, String mu, ArrayList<String> map_cuped, String func_cuped) {
    super(pos);
    this.map = map;
    this.func = func;
    this.func_cuped = func_cuped;
    this.map_cuped = map_cuped;
    this.mu = mu.replaceAll("'", "").replaceAll("\"", "");
    this.altertive = altertive.replaceAll("'", "").replaceAll("\"", "");
    this.causal_function_name = "ttest_1samp";
  }

  @Override public SqlOperator getOperator() {
    return null;
  }

  public String exchangeFunc(String func, int offset) {
    int max_index = 0;
    for (int i = 1; ; i++) {
      if (!func.contains("x" + Integer.toString(i))) {
        max_index = i - 1;
        break;
      }
    }
    for (int i = max_index; i >= 0; i--) {
      func = func.replaceAll("x" + Integer.toString(i), "x" + Integer.toString(i + offset));
    }
    return func;
  }

  @Override public void unparse(SqlWriter writer, int leftPrec, int rightPrec) {
    writer.print("Ttest_1samp");
    writer.print("(\'" + func + "\',");
    writer.print("\'" + altertive + "\',");
    writer.print(mu);
    if (func_cuped.length() != 0) {
      writer.print(",\'X=" + exchangeFunc(func_cuped, map.size()) + "\'");
    }
    writer.print(")(");

    for (int i = 0; i < map.size(); i++) {
      if (i != 0) writer.print(",");
      writer.print(map.get(i).replaceAll("`", "").replaceAll(" ", ""));
    }

    for (int i = 0; i < map_cuped.size(); i++) {
      writer.print(",");
      writer.print(map_cuped.get(i).replaceAll("`", "").replaceAll(" ", ""));
    }
    writer.print(")");
  }

  @Override public List<SqlNode> getOperandList() {
    return null;
  }
}
