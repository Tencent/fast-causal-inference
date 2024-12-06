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
package org.apache.calcite.sql;

import org.apache.calcite.sql.olap.EngineType;
import org.apache.calcite.sql.parser.SqlParserPos;

import java.util.ArrayList;
import java.util.List;

public abstract class SqlCallCausal extends SqlCall {
  public String causal_function_name = "";
  public String replace_sql = "";

  public EngineType engineType = EngineType.ClickHouse;

  public String replace_table = "";
  public ArrayList<String> withs = new ArrayList<String>();

  protected SqlCallCausal(SqlParserPos pos, EngineType engineType) {
    super(pos);
    this.engineType = engineType;
  }

  @Override public SqlOperator getOperator() {
    return null;
  }

  @Override public void unparse(SqlWriter writer, int leftPrec, int rightPrec) {
    if (engineType == EngineType.ClickHouse) {
      unparseClickHouse(writer, leftPrec, rightPrec);
    } else if (engineType == EngineType.StarRocks) {
      unparseStarRocks(writer, leftPrec, rightPrec);
    } else {
      throw new RuntimeException("Unsupported engine type: " + engineType);
    }
  }

  public void unparseClickHouse(SqlWriter writer, int leftPrec, int rightPrec) {
    throw new RuntimeException("Unsupported engine type: " + engineType);
  }

  public void unparseStarRocks(SqlWriter writer, int leftPrec, int rightPrec) {
    throw new RuntimeException("Unsupported engine type: " + engineType);
  }

  @Override public List<SqlNode> getOperandList() {
    return null;
  }
}
