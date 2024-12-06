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
  private ArrayList<String> map;
  private String func;

  private String is_std;

  public DeltaMethodParser(SqlParserPos pos, ArrayList<String> map, String func, String is_std, EngineType engineType) {
    super(pos, engineType);
    this.map = map;
    this.func = func;
    this.is_std = is_std;
    this.causal_function_name = "deltamethod";
    this.engineType = engineType;
  }

  @Override public SqlOperator getOperator() {
    return null;
  }

  @Override  public void unparseClickHouse(SqlWriter writer, int leftPrec, int rightPrec) {
    writer.print("Deltamethod");
    writer.print("(\'");
    writer.print(func);
    writer.print("\'," + is_std + ")(");
    for (int i = 0; i < map.size(); i++) {
      if (i != 0) writer.print(",");
      writer.print(map.get(i).replaceAll("`", ""));
    }
    writer.print(")");
  }

  @Override  public void unparseStarRocks(SqlWriter writer, int leftPrec, int rightPrec) {
    writer.print("delta_method");
    writer.print("(\'");
    writer.print(func);
    writer.print("\', " + is_std + ", [");
    for (int i = 0; i < map.size(); i++) {
      if (i > 0) {
        writer.print(",");
      }
      writer.print(map.get(i).replaceAll("`", ""));
    }
    writer.print("])");
  }

  @Override public List<SqlNode> getOperandList() {
    return null;
  }
}

