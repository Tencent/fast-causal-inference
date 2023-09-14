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
import org.apache.calcite.sql.dialect.PrestoSqlDialect;
import org.apache.calcite.sql.parser.SqlParseException;
import org.apache.calcite.sql.parser.SqlParser;

import java.util.ArrayList;
import java.util.List;

public class SqlForward {

  String sql;
  String forwardSql;
  ArrayList<String> forwardUdfs;

  ArrayList<String> withs = new ArrayList<String>();

  String replace_sql = "";

  String replace_table = "";

  public void parseNode(SqlNode node) {
    if (!(node instanceof SqlCallCausal)) {
      return ;
    }
    SqlCallCausal causalNode = (SqlCallCausal) node;
    forwardUdfs.add(causalNode.causal_function_name);
    for (String with : causalNode.withs) {
      withs.add(with);
    }
    if (replace_sql.equals("") && !causalNode.replace_sql.equals("")) {
      replace_sql = causalNode.replace_sql;
    }
    if (replace_table.equals("") && !causalNode.replace_table.equals("")) {
      replace_table = causalNode.replace_table;
    }
  }
  // Config
  /* TODO : Distinguish between different engines: clickhouse, starrocks...*/
  public SqlForward(String sql) throws SqlParseException {
    this.sql = sql;
    forwardUdfs = new ArrayList<String>();
    SqlParser parser = SqlParser.create(sql, SqlParser.Config.DEFAULT
        .withUnquotedCasing(Casing.UNCHANGED)
        .withCaseSensitive(false)
        .withQuotedCasing(Casing.UNCHANGED));
    SqlNode sqlNode = parser.parseStmt();
    //String unparserSql = sqlNode.toSqlString(new SqlDialect(SqlDialect.EMPTY_CONTEXT)).getSql();
    String unparserSql = sqlNode.toSqlString(MysqlSqlDialect.DEFAULT).getSql();

    SqlSelect sqlSelect;
    if (sqlNode instanceof SqlOrderBy) {
      SqlOrderBy sqlOrderBy = (SqlOrderBy) sqlNode;
      sqlSelect = (SqlSelect) sqlOrderBy.query;
    }
    else
      sqlSelect = (SqlSelect)sqlNode;
    SqlIdentifier sqlIdentifier = (SqlIdentifier)sqlSelect.getFrom();
    SqlNodeList selectList = (SqlNodeList)sqlSelect.getSelectList();
    for (SqlNode node : selectList) {
      parseNode(node);
      // parser alias
      if (node instanceof  SqlBasicCall && ((SqlBasicCall) node).getOperator().toString() == "AS") {
        parseNode(((SqlBasicCall) node).getOperandList().get(0));
      }
    }
    String table_name = "";
    if (sqlIdentifier != null)
      table_name = sqlIdentifier.toString();
    String with_sql = "with ";
    for (int i = 0; i < withs.size(); ++i) {
      if (i != 0) {
        with_sql += ",\n";
      }
      with_sql += withs.get(i) + " \n";
    }
    String forwardSql = "";
    if (!with_sql.equals("with ")) {
      forwardSql += with_sql;
    }
    if (replace_sql.equals(""))
      forwardSql += unparserSql;
    else
      forwardSql += replace_sql;
    System.out.println("tablename:" + table_name);
    if (!table_name.equals(""))
      forwardSql = forwardSql.replace("@TBL", table_name);

    if (!replace_table.equals("")) {
      int lastIndex = forwardSql.lastIndexOf(table_name);
      if (lastIndex != -1) {
        forwardSql = forwardSql.substring(0, lastIndex) + replace_table + forwardSql.substring(lastIndex + table_name.length());
      }
    }

    this.forwardSql = forwardSql.replaceAll("`", "");
  }

  public String getForwardSql() {
    return forwardSql;
  }

  public ArrayList<String> getForwardUdfs() {
    return forwardUdfs;
  }

  private static List<String> extractTableAliases(SqlNode node) {
    final List<String> tables = new ArrayList<>();

    if (node.getKind().equals(SqlKind.ORDER_BY)) {
      node = ((SqlSelect) ((SqlOrderBy) node).query).getFrom();
    } else {
      node = ((SqlSelect) node).getFrom();
    }

    if (node == null) {
      return tables;
    }

    if (node.getKind().equals(SqlKind.AS)) {
      tables.add(((SqlBasicCall) node).operand(1).toString());
      return tables;
    }

    if (node.getKind().equals(SqlKind.JOIN)) {
      final SqlJoin from = (SqlJoin) node;

      if (from.getLeft().getKind().equals(SqlKind.AS)) {
        tables.add(((SqlBasicCall) from.getLeft()).operand(1).toString());
      } else {
        SqlJoin left = (SqlJoin) from.getLeft();

        while (!left.getLeft().getKind().equals(SqlKind.AS)) {
          tables.add(((SqlBasicCall) left.getRight()).operand(1).toString());
          left = (SqlJoin) left.getLeft();
        }

        tables.add(((SqlBasicCall) left.getLeft()).operand(1).toString());
        tables.add(((SqlBasicCall) left.getRight()).operand(1).toString());
      }

      tables.add(((SqlBasicCall) from.getRight()).operand(1).toString());
      return tables;
    }

    return tables;
  }
}
