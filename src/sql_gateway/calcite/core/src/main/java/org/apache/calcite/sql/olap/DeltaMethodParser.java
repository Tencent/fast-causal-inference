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

public class DeltaMethodParser extends SqlCallCausal {
  private HashMap<String, String> map;
  private String func;

  public DeltaMethodParser(SqlParserPos pos) {
    super(pos);
  }

  public DeltaMethodParser(SqlParserPos pos, HashMap<String, String> map, String func) {
    super(pos);
    this.map = map;
    this.func = func;
  }

  @Override public SqlOperator getOperator() {
    return null;
  }

  @Override public void unparse(SqlWriter writer, int leftPrec, int rightPrec) {
    writer.print("Deltamethod");
    writer.print("(\'");
    writer.print(func);
    writer.print("\')(");
    ArrayList<Pair<Integer, String>> args = new ArrayList<>();
    map.forEach((key,value) -> {
      args.add(new Pair<>(Integer.valueOf(value), key));
    });
    sort(args);
    for (int i = 0; i < args.size(); i++) {
      if (i != 0) writer.print(",");
      writer.print(args.get(i).getValue().replaceAll("`", ""));
    }
    writer.print(")");
    this.causal_function_name = "deltamethod";
  }

  @Override public List<SqlNode> getOperandList() {
    return null;
  }
}
