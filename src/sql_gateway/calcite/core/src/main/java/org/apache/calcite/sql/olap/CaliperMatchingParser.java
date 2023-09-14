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

import java.util.List;

public class CaliperMatchingParser extends SqlCallCausal {
  private String treatment, target, split_value;

  static private String with_template = "   (\n"
  +
      "        SELECT @PHSplictNum\n"
  +
      "    ) AS VAR_WEIGHT,\n"
  +
      "     table_1 as (\n"
  +
      "        select *, \n"
  +
      "        @PHt as m_caliper_t,\n"
  +
      "        @PHv as m_caliper_v\n"
  +
      "        from @TBL\n"
  +
      "),\n"
  +
      "    table_2 AS\n"
  +
      "    (\n"
  +
      "        SELECT\n"
  +
      "            count() as m_caliper_cnt, \n"
  +
      "            min(m_caliper_num) AS m_caliper_index_min,\n"
  +
      "            max(m_caliper_num) AS m_caliper_index_max,\n"
  +
      "            sum(multiIf(m_caliper_cnt < 2, 0, m_caliper_index_min > m_caliper_index_max, " +
      "m_caliper_index_max, m_caliper_index_min)) OVER (ROWS BETWEEN UNBOUNDED PRECEDING AND 1 " +
      "PRECEDING) AS m_caliper_matching_index,\n"
  +
      "            m_caliper_g\n"
  +
      "        FROM\n"
  +
      "        (\n"
  +
      "            SELECT\n"
  +
      "                count(1) AS m_caliper_num,\n"
  +
      "                m_caliper_t,\n"
  +
      "                toUInt32(m_caliper_v / VAR_WEIGHT) AS m_caliper_g\n"
  +
      "            FROM table_1\n"
  +
      "            GROUP BY\n"
  +
      "                m_caliper_g,\n"
  +
      "                m_caliper_t \n"
  +
      "        )\n"
  +
      "        GROUP BY m_caliper_g\n"
  +
      "),\n"
  +
      "    final_table AS\n"
  +
      "    (\n"
  +
      "        SELECT\n"
  +
      "            *,\n"
  +
      "            row_number() OVER (PARTITION BY toUInt32(m_caliper_v / VAR_WEIGHT), " +
      "m_caliper_t) AS m_caliper_rn, \n"
  +
      "            if(m_caliper_rn <= if(m_caliper_cnt < 2, 0, m_caliper_index_min), " +
      "(m_caliper_rn + m_caliper_matching_index) * m_caliper_t, 0) as m_caliper_index \n"
  +
      "        FROM table_1 as m_a\n"
  +
      "        LEFT JOIN\n"
  +
      "        (\n"
  +
      "            SELECT\n"
  +
      "                m_caliper_cnt, \n"
  +
      "                m_caliper_index_min,\n"
  +
      "                m_caliper_matching_index,\n"
  +
      "                m_caliper_g\n"
  +
      "            FROM table_2\n"
  +
      "        ) AS m_b ON toUInt32(m_a.m_caliper_v / VAR_WEIGHT) = m_b.m_caliper_g\n"
  +
      "    )";

  public CaliperMatchingParser(SqlParserPos pos) {
    super(pos);
  }

  public CaliperMatchingParser(SqlParserPos pos, String treatment, String target, String split_value) {
    super(pos);
    this.treatment = SqlForwardUtil.exchangIdentity(treatment);
    this.target = SqlForwardUtil.exchangIdentity(target);
    this.split_value = SqlForwardUtil.exchangIdentity(split_value);
    this.causal_function_name = "caliperMatching";
    this.replace_table = "final_table";
  }

  @Override public SqlOperator getOperator() {
    return null;
  }

  @Override public void unparse(SqlWriter writer, int leftPrec, int rightPrec) {
    String with = with_template;
    with = with.replaceAll("@PHt", treatment);
    with = with.replaceAll("@PHv", target);
    with = with.replaceAll("@PHSplictNum", split_value);
    withs.add(with);
    writer.print("m_caliper_index");
  }

  @Override public List<SqlNode> getOperandList() {
    return null;
  }
}
