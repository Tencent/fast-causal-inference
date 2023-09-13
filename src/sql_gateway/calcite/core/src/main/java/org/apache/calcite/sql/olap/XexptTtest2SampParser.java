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

public class XexptTtest2SampParser extends SqlCallCausal {
  private HashMap<String, String> map_cuped;
  private String func_cuped, index, numerator, denominator, uin ;
  private ArrayList<String> args;


  public XexptTtest2SampParser(SqlParserPos pos) {
    super(pos);
  }

  public XexptTtest2SampParser(SqlParserPos pos, String numerator, String denominator, String uin, String index, String func_cuped, HashMap<String, String> map_cuped, ArrayList<String> args) {
    super(pos);
    this.numerator = SqlForwardUtil.exchangIdentity(numerator);
    this.denominator = SqlForwardUtil.exchangIdentity(denominator);
    this.uin = SqlForwardUtil.exchangIdentity(uin);
    this.index = SqlForwardUtil.exchangIdentity(index);
    this.func_cuped = func_cuped;
    this.map_cuped = map_cuped;
    this.args = args;
    this.causal_function_name = "xexpt_ttest_2samp";
  }

  static private String alpha = "0.05";
  static private String mde = "0.005";
  static private String power = "0.08";

  @Override public SqlOperator getOperator() {
    return null;
  }

  @Override public void unparse(SqlWriter writer, int leftPrec, int rightPrec) {
    writer.print("Xexpt_Ttest_2samp(");
    ArrayList<String> arg_default = new ArrayList<>();
    arg_default.add(alpha);
    arg_default.add(mde);
    arg_default.add(power);
    for (int i = 0; i < 3; ++i) {
      if (i != 0)
        writer.print(",");
      if (i < args.size())
        writer.print(args.get(i));
      else
        writer.print(arg_default.get(i));
    }
    if (func_cuped.length() != 0)
      writer.print(",'X=" + SqlForwardUtil.exchangeFunc(func_cuped, 2) + "'");
    writer.print(")(" + numerator + "," + denominator);

    ArrayList<Pair<Integer, String>> args_cuped = new ArrayList<>();
    map_cuped.forEach((key,value) -> {
      args_cuped.add(new Pair<>(Integer.valueOf(value), key));
    });
    sort(args_cuped);
    for (int i = 0; i < args_cuped.size(); i++) {
      writer.print(",");
      writer.print(args_cuped.get(i).getValue().replaceAll("`", "").replaceAll(" ", ""));
    }

    writer.print("," + uin);
    writer.print("," + index + ")");
  }

  @Override public List<SqlNode> getOperandList() {
    return null;
  }
}
