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

import org.apache.calcite.avatica.util.Casing;
import org.apache.calcite.sql.*;
import org.apache.calcite.sql.dialect.MysqlSqlDialect;
import org.apache.calcite.sql.parser.SqlParseException;
import org.apache.calcite.sql.parser.SqlParser;
import org.apache.calcite.sql.parser.SqlParserPos;

import java.util.ArrayList;
import java.util.List;



public class BootStrapParser extends SqlCallCausal {
  private String bootStrapFunc;
  private String sample_num;
  private String bs_num;

  public BootStrapParser(SqlParserPos pos, String bootStrapFunc, String sample_num, String bs_num, EngineType engineType) {
    super(pos, engineType);
    this.bootStrapFunc = bootStrapFunc.trim();
    this.sample_num = sample_num;
    this.bs_num = bs_num;
  }

  @Override public SqlOperator getOperator() {
    return null;
  }

  @Override public void unparseClickHouse(SqlWriter writer, int leftPrec, int rightPrec) {
    String with = "(SELECT DistributedNodeRowNumber(0)(1) from @TBL) as bs_param";
    withs.add(with);

    /*  适配 bootStrapMulti TODO
    ArrayList<String> funcs = new ArrayList<>();
    ArrayList<String> cols = new ArrayList<>();
    String[] funcAndCols = bootStrapFunc.split(";");
    for (String col : funcAndCols) {
      // finde first '('
      int idx = col.indexOf("(");
      if (idx == -1) {
        throw new RuntimeException("Invalid bootStrapFunc: " + bootStrapFunc);
      }
      String func = col.substring(0, idx);
      String colName = col.substring(idx + 1, col.length() - 1);
      funcs.add(func);
      cols.add(colName);
    }
     */

    bootStrapFunc = bootStrapFunc.substring(1, bootStrapFunc.length() - 1).replace("@", "'");
    String parse_sql = "SELECT " + bootStrapFunc;
    SqlParser parser = SqlParser.create(parse_sql, SqlParser.Config.DEFAULT
        .withUnquotedCasing(Casing.UNCHANGED)
        .withCaseSensitive(false)
        .withQuotedCasing(Casing.UNCHANGED)
        .withEngineType(engineType));

    String unparserSql;
    try {
      SqlNode sqlNode = parser.parseStmt();
      unparserSql = sqlNode.toSqlString(MysqlSqlDialect.DEFAULT).getSql().replaceAll("`", "").replaceAll("SELECT ", "").trim();
    } catch (Exception e) {
      unparserSql = bootStrapFunc;
      //throw new RuntimeException(e);
    }

    String func = "";
    String colName = "";

    int index = unparserSql.indexOf(")(");
    if (index != -1) {
      func = unparserSql.substring(0, index + 1);
      colName = unparserSql.substring(index + 2, unparserSql.length() - 1);
    } else {
      index = unparserSql.indexOf("(");
      if (index != -1) {
        func = unparserSql.substring(0, index);
        colName = unparserSql.substring(index + 1, unparserSql.length() - 1);
      } else {
        throw new RuntimeException("Invalid bootStrapFunc: " + bootStrapFunc);
      }
    }
    func = func.replaceAll("'", "\"");

    writer.print("BootStrap('");
    writer.print(func + "'," + sample_num + "," + bs_num);
    writer.print(", bs_param)(" + colName + ")");
  }

  @Override public List<SqlNode> getOperandList() {
    return null;
  }
}
