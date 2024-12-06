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
import java.util.Iterator;
import java.util.List;
import java.util.stream.Collectors;

import static java.util.Collections.sort;

public class XexptTtest2SampParser extends SqlCallCausal {
  private ArrayList<String> map_cuped;
  private String func_cuped, index, numerator, denominator, uin, ratio0, ratio1;
  private ArrayList<String> args;

  private boolean is_avg;
  public XexptTtest2SampParser(SqlParserPos pos, String numerator, String denominator, String uin, String index, String func_cuped, ArrayList<String> map_cuped, ArrayList<String> args, boolean is_avg, String ratio0, String ratio1, EngineType engineType) {
    super(pos, engineType);
    this.numerator = SqlForwardUtil.exchangIdentity(numerator);
    this.denominator = SqlForwardUtil.exchangIdentity(denominator);
    this.uin = SqlForwardUtil.exchangIdentity(uin);
    this.index = SqlForwardUtil.exchangIdentity(index);
    this.func_cuped = func_cuped;
    this.map_cuped = map_cuped;
    this.args = args;
    this.causal_function_name = "xexpt_ttest_2samp";
    this.ratio0 = ratio0;
    this.ratio1 = ratio1;
    this.is_avg = is_avg;
  }

  static private final String alpha = "0.05";
  static private final String mde = "0.005";
  static private final String power = "0.8";

  @Override public SqlOperator getOperator() {
    return null;
  }

  @Override public void unparseClickHouse(SqlWriter writer, int leftPrec, int rightPrec) {
    writer.print("Xexpt_Ttest_2samp(");

    ArrayList<String> alphaMdePower = extractAlphaMdePower();
    for (int i = 0; i < 3; ++i) {
      if (i > 0) {
        writer.print(",");
      }
      writer.print(alphaMdePower.get(i));
    }

    if (!is_avg) {
      writer.print(",'sum'");
      String ratio_str = ", " + ratio0 + "," + ratio1 + " ";
      writer.print(ratio_str);
    }
    if (func_cuped.length() != 0)
      writer.print(",'X=" + SqlForwardUtil.exchangeFunc(func_cuped, 2) + "'");
    writer.print(")(" + numerator + "," + denominator);

    for (int i = 0; i < map_cuped.size(); i++) {
      writer.print(",");
      writer.print(map_cuped.get(i).replaceAll("`", "").replaceAll(" ", ""));
    }

    writer.print("," + uin);
    writer.print("," + index + ")");
  }

  @Override public void unparseStarRocks(SqlWriter writer, int leftPrec, int rightPrec) {
    writer.print("xexpt_ttest_2samp(");

    ArrayList<String> params = new ArrayList<>();
    params.add(uin);
    params.add(index);

    ArrayList<String> dataColumns = new ArrayList<>();
    dataColumns.add(numerator);
    dataColumns.add(denominator);
    map_cuped.replaceAll(s -> s.replaceAll("`", ""));
    dataColumns.addAll(map_cuped);
    String data = dataColumns.stream().collect(Collectors.joining(",", "[", "]"));
    params.add(data);

    if (!func_cuped.isEmpty()) {
      params.add(String.format("'X=%s'", SqlForwardUtil.exchangeFunc(func_cuped, 2)));
    } else {
      params.add("'X='");
    }

    params.addAll(extractAlphaMdePower());
    if (!is_avg) {
      params.add("'sum'");
      params.add(String.format("[%s,%s]", ratio0, ratio1));
    }

    writer.print(String.join(",", params));
    writer.print(")");
  }

  private ArrayList<String> extractAlphaMdePower() {
    Iterator<String> iterator = args.iterator();
    while (iterator.hasNext()) {
      String item = iterator.next();
      if (item.toLowerCase().trim().equals("'sum'")) {
        is_avg = false;
        iterator.remove();
      } else if (item.toLowerCase().trim().equals("'avg'")) {
        is_avg = true;
        iterator.remove();
      }
    }

    ArrayList<String> arg_default = new ArrayList<>();
    arg_default.add(alpha);
    arg_default.add(mde);
    arg_default.add(power);
    for (int i = 0; i < 3; ++i) {
      if (i < args.size())
        arg_default.set(i, args.get(i));
    }
    return arg_default;
  }

  @Override public List<SqlNode> getOperandList() {
    return null;
  }
}
