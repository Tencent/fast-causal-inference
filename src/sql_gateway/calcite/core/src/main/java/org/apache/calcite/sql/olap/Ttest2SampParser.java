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
import java.util.stream.Collectors;

import static java.util.Collections.sort;

public class Ttest2SampParser extends SqlCallCausal {
  private ArrayList<String> map, map_cuped;
  private String func, func_cuped, index, altertive;
  boolean edegeworth;

  private ArrayList<String> pses;


  public Ttest2SampParser(SqlParserPos pos, ArrayList<String> map, String func, String index, String altertive, ArrayList<String> map_cuped, String func_cuped, ArrayList<String> pses, boolean edegeworth, EngineType engineType) {
    super(pos, engineType);
    this.map = map;
    this.func = func;
    this.func_cuped = func_cuped;
    this.map_cuped = map_cuped;
    //this.index = index.replaceAll("'", "").replaceAll("\"", "");
    this.index = index;
    this.altertive = altertive.replaceAll("'", "").replaceAll("\"", "");
    this.causal_function_name = "ttest_2samp";
    this.pses = pses;
    this.edegeworth = edegeworth;
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

  @Override public void unparseClickHouse(SqlWriter writer, int leftPrec, int rightPrec) {
    writer.print("Ttest_2samp");
    writer.print("(\'" + func + "\',");
    writer.print("\'" + altertive + "\'");
    if (!pses.isEmpty()) {
      writer.print("," + String.valueOf(pses.size()));
    }
    if (func_cuped.length() != 0) {
      writer.print(",\'X=" + exchangeFunc(func_cuped, map.size()) + "\'");
    }
    if (edegeworth) {
      writer.print(",true");
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
    for (int i = 0; i < pses.size(); i++) {
      writer.print("," + pses.get(i));
    }
    writer.print("," + index + ")");
  }

  @Override public void unparseStarRocks(SqlWriter writer, int leftPrec, int rightPrec) {
    // TODO(xjx): add pse
    writer.print("ttest_2samp");
    writer.print("(");

    ArrayList<String> params = new ArrayList<>();

    map.replaceAll(s -> s.replaceAll("`", ""));
    map_cuped.replaceAll(s -> s.replaceAll("`", ""));
    ArrayList<String> dataColumns = new ArrayList<>();
    dataColumns.addAll(map);
    dataColumns.addAll(map_cuped);
    String data = dataColumns.stream().collect(Collectors.joining(",", "[", "]"));

    params.add("'" + func + "'");
    params.add("'" + altertive + "'");
    params.add(index);
    params.add(data);

    if (!func_cuped.isEmpty()) {
      params.add("'X=" + exchangeFunc(func_cuped, map.size()) + "'");
    }

    writer.print(String.join(",", params));
    writer.print(")");
  }

  @Override public List<SqlNode> getOperandList() {
    return null;
  }
}
