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



public class PermutationParser extends SqlCallCausal {
  private String permutationFunc;
  private String mde;
  private String mdeType;
  private String permutationNum;

  public PermutationParser(SqlParserPos pos, String permutationFunc, String permutationNum, String mde, String mdeType, EngineType engineType) {
    super(pos, engineType);
    this.permutationFunc = permutationFunc.trim();
    this.permutationNum = permutationNum;
    this.mde = mde;
    this.mdeType = mdeType;
  }

  @Override public SqlOperator getOperator() {
    return null;
  }

  @Override public void unparseClickHouse(SqlWriter writer, int leftPrec, int rightPrec) {

    permutationFunc = permutationFunc.substring(1, permutationFunc.length() - 1).replace("@", "'");
    String parse_sql = "SELECT " + permutationFunc;
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
      throw new RuntimeException(e);
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
        throw new RuntimeException("Invalid permutationFunc: " + permutationFunc);
      }
    }
    func = func.replaceAll("'", "\"");

    writer.print("Permutation('");
    writer.print(func + "'," + permutationNum + "," + mde + "," + mdeType);
    writer.print(")(" + colName.replaceAll(",TREATMENT", "") + ")");
  }

  @Override public List<SqlNode> getOperandList() {
    return null;
  }
}
