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

public class ExactMatchingParser extends SqlCallCausal {
  private String treatment;
  private ArrayList<String> labels;
  static private String with_template = "table_1 AS\n"
  +
      "    (\n"
  +
      "        SELECT\n"
  +
      "            *,\n"
  +
      "            @PH1 AS mm_t,\n"
  +
      "            @PHends\n"
  +
      "        FROM @TBL\n"
  +
      "),\n"
  +
      "    table_2 AS\n"
  +
      "    (\n"
  +
      "        SELECT\n"
  +
      "            count() as mm_cnt, \n"
  +
      "            min(mm_num) AS mm_index_min,\n"
  +
      "            max(mm_num) AS mm_index_max,\n"
  +
      "            sum(multiIf(mm_cnt < 2, 0, mm_index_min > mm_index_max, mm_index_max, " +
      "mm_index_min)) OVER (ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS " +
      "mm_matching_index,\n"
  +
      "            @PHLabels\n"
  +
      "        FROM\n"
  +
      "        (\n"
  +
      "            SELECT\n"
  +
      "                count(1) AS mm_num,\n"
  +
      "                mm_t,\n"
  +
      "                @PHLabels\n"
  +
      "            FROM table_1\n"
  +
      "            GROUP BY\n"
  +
      "                mm_t,\n"
  +
      "                @PHLabels\n"
  +
      "        )\n"
  +
      "        GROUP BY @PHLabels\n"
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
      "            row_number() OVER (PARTITION BY mm_t, @PHLabels) AS mm_rn,\n"
  +
      "            if(mm_rn <= if(mm_cnt < 2, 0, mm_index_min), (mm_rn + mm_matching_index) * " +
      "mm_t, 0) as mm_index \n"
  +
      "        FROM table_1 AS m_a\n"
  +
      "        LEFT JOIN\n"
  +
      "        (\n"
  +
      "            SELECT\n"
  +
      "                mm_cnt, \n"
  +
      "                mm_index_min,\n"
  +
      "                mm_matching_index,\n"
  +
      "                @PHLabels\n"
  +
      "            FROM table_2\n"
  +
      "        ) AS m_b ON @PHLabelEqual\n"
  +
      "    )";

  public ExactMatchingParser(SqlParserPos pos) {
    super(pos);
  }

  public ExactMatchingParser(SqlParserPos pos, String treatment, ArrayList<String> labels) {
    super(pos);
    this.treatment = SqlForwardUtil.exchangIdentity(treatment);
    this.labels = labels;
    this.causal_function_name = "exactMatching";
    this.replace_table = "final_table";
  }

  @Override public SqlOperator getOperator() {
    return null;
  }

  @Override public void unparse(SqlWriter writer, int leftPrec, int rightPrec) {
    String PHLabels = "";
    for (int i = 1; i <= labels.size(); i++) {
      PHLabels += "l" + String.valueOf(i);
      PHLabels += (i == labels.size() ? " " : ",");
      if (i == labels.size())
        PHLabels += " ";
      else PHLabels += ",";
    }

    String PHends = "";
    for (int i = 1; i <= labels.size(); i++) {
      PHends += labels.get(i-1) + " as " + "l" + String.valueOf(i) + (i == labels.size() ? " " : ",");
    }

    String PHLabelsA = "";
    String PHLabelsB = "";
    for (int i = 1; i <= labels.size(); i++) {
      PHLabelsA += "a.l" + String.valueOf(i) +  " as " + "l" + String.valueOf(i) + (i == labels.size() ? " " : ",");
      PHLabelsB += "b.l" + String.valueOf(i) +  " as " + "l" + String.valueOf(i) + (i == labels.size() ? " " : ",");
    }

    String PHLabelEqual = "";
    for (int i = 1; i <= labels.size(); i++) {
      PHLabelEqual += "m_a.l" + String.valueOf(i) + " = m_b.l" + String.valueOf(i) + (i == labels.size() ? "" : " and ");
    }

    String with = with_template;
    with = with.replaceAll("@PH1", treatment);
    with = with.replaceAll("@PHLabels", PHLabels);
    with = with.replaceAll("@PHends", PHends);
    with = with.replaceAll("@PHLabelA", PHLabelsA);
    with = with.replaceAll("@PHLabelB", PHLabelsB);
    with = with.replaceAll("@PHLabelEqual", PHLabelEqual);
    withs.add(with);
    writer.print("mm_index");
  }

  @Override public List<SqlNode> getOperandList() {
    return null;
  }
}
