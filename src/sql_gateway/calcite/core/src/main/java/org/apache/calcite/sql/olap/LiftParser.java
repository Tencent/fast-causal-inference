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

public class LiftParser extends SqlCallCausal {

  private String  with_template = "( \n"
  +
      "    select toUInt64(@PH_K) \n"
  +
      "  ) as mm_k, \n"
  +
      "  ( \n"
  +
      "  select  \n"
  +
      "  tuple( \n"
  +
      "  floor(count() / mm_k), \n"
  +
      "  count(), \n"
  +
      "  avg(@PH_T), \n"
  +
      "  avg(@PH_Y), \n"
  +
      "  mm_k - count() % mm_k \n"
  +
      "  ) from @TBL \n"
  +
      "  ) as mm_params\n";

  private String func_template_true = "select ratio, \n"
  +
      "    (sum(sum1) OVER w1 / sum(cnt1) OVER w1) - (sum(sum0) OVER w1 / sum(cnt0) OVER w1) AS " +
      "lift, \n"
  +
      "    lift * ratio as gain, \n"
  +
      "    (sum(sum1) OVER w2 / sum(cnt1) OVER w2) - (sum(sum0) OVER w2 / sum(cnt0) OVER w2) AS " +
      "ate, \n"
  +
      "    ate * ratio as ramdom_gain \n"
  +
      "FROM \n"
  +
      "( \n"
  +
      "    SELECT \n"
  +
      "        if(rn <= ((mm_params.5) * (mm_params.1)), floor((rn - 1) / (mm_params.1)), floor((" +
      "(rn - ((mm_params.1) * (mm_params.5))) - 1) / ((mm_params.1) + 1)) + (mm_params.5)) + 1 AS" +
      " gid, \n"
  +
      "        max(rn) AS max_rn, \n"
  +
      "        max_rn / (mm_params.2) AS ratio, \n"
  +
      "        sum(if(mm_t = 0, mm_y, 0)) AS sum0, \n"
  +
      "        sum(if(mm_t = 1, mm_y, 0)) AS sum1, \n"
  +
      "        countIf(mm_t = 0) AS cnt0, \n"
  +
      "        countIf(mm_t = 1) AS cnt1 \n"
  +
      "    FROM \n"
  +
      "    ( \n"
  +
      "        SELECT \n"
  +
      "            row_number() OVER (ORDER BY mm_ite DESC) AS rn, \n"
  +
      "            @PH_ITE AS mm_ite, \n"
  +
      "            @PH_Y AS mm_y, \n"
  +
      "            @PH_T AS mm_t \n"
  +
      "        FROM @TBL \n"
  +
      "        ORDER BY mm_ite DESC \n"
  +
      "    ) \n"
  +
      "    GROUP BY gid \n"
  +
      ") \n"
  +
      "WINDOW \n"
  +
      "    w1 AS (ORDER BY ratio ASC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW), \n"
  +
      "    w2 AS (PARTITION BY 0) \n"
  +
      "ORDER BY ratio ASC";

  private String func_template_false = "select ratio, \n"
  +
      "    sum(sumt) OVER w1 / (ratio * (mm_params.2)) AS t_, \n"
  +
      "    sum(sumy) OVER w1 / (ratio * (mm_params.2)) AS y_, \n"
  +
      "    ((((sum(sumty) OVER w1 - (sum(sumt) OVER w1 * y_)) - (sum(sumy) OVER w1 * t_)) + (((y_" +
      " * t_) * ratio) * (mm_params.2))) / ((sum(sumtt) OVER w1 - ((2 * sum(sumt) OVER w1) * t_))" +
      " + (((t_ * t_) * ratio) * (mm_params.2))))  AS lift, \n"
  +
      "    lift * ratio as gain, \n"
  +
      "    ((((sum(sumty) OVER w2 - (sum(sumt) OVER w2 * (mm_params.4))) - (sum(sumy) OVER w2 * " +
      "(mm_params.3))) + (((mm_params.4) * (mm_params.3)) * (mm_params.2))) / ((sum(sumtt) OVER " +
      "w2 - ((2 * sum(sumt) OVER w2) * (mm_params.3))) + (((mm_params.3) * (mm_params.3)) * " +
      "(mm_params.2))))  AS ate, \n"
  +
      "    ate * ratio as ramdom_gain \n"
  +
      "FROM \n"
  +
      "( \n"
  +
      "    SELECT \n"
  +
      "        if(rn <= ((mm_params.5) * (mm_params.1)), floor((rn - 1) / (mm_params.1)), floor((" +
      "(rn - ((mm_params.1) * (mm_params.5))) - 1) / ((mm_params.1) + 1)) + (mm_params.5)) + 1 AS" +
      " gid, \n"
  +
      "        max(rn) AS max_rn, \n"
  +
      "        max_rn / (mm_params.2) AS ratio, \n"
  +
      "        sum(mm_t) AS sumt, \n"
  +
      "        sum(mm_t * mm_t) AS sumtt, \n"
  +
      "        sum(mm_y) AS sumy, \n"
  +
      "        sum(mm_t * mm_y) AS sumty \n"
  +
      "    FROM \n"
  +
      "    ( \n"
  +
      "        SELECT \n"
  +
      "            row_number() OVER (ORDER BY mm_ite DESC) AS rn, \n"
  +
      "            @PH_ITE AS mm_ite, \n"
  +
      "            @PH_Y AS mm_y, \n"
  +
      "            @PH_T AS mm_t \n"
  +
      "        FROM @TBL \n"
  +
      "        ORDER BY mm_ite DESC \n"
  +
      "    ) \n"
  +
      "    GROUP BY gid \n"
  +
      ") \n"
  +
      "WINDOW \n"
  +
      "    w1 AS (ORDER BY ratio ASC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW), \n"
  +
      "    w2 AS (PARTITION BY 0) \n"
  +
      "ORDER BY ratio ASC";


  String ite, t, y, k, flag;

  public LiftParser(SqlParserPos pos) {
    super(pos);
  }

  public LiftParser(SqlParserPos pos, String ite, String Y, String T, String k, String flag) {
    super(pos);
    this.ite = SqlForwardUtil.exchangIdentity(ite);
    this.y = SqlForwardUtil.exchangIdentity(Y);
    this.t = SqlForwardUtil.exchangIdentity(T);
    this.k = SqlForwardUtil.exchangIdentity(k);
    this.flag = SqlForwardUtil.exchangIdentity(flag);
    with_template = with_template.replaceAll("@PH_ITE", ite);
    with_template = with_template.replaceAll("@PH_Y", this.y);
    with_template = with_template.replaceAll("@PH_T", this.t);
    with_template = with_template.replaceAll("@PH_K", k);

    func_template_true = func_template_true.replaceAll("@PH_ITE", ite);
    func_template_true = func_template_true.replaceAll("@PH_Y", this.y);
    func_template_true = func_template_true.replaceAll("@PH_T", this.t);
    func_template_true = func_template_true.replaceAll("@PH_K", k);

    func_template_false = func_template_false.replaceAll("@PH_ITE", ite);
    func_template_false = func_template_false.replaceAll("@PH_Y", this.y);
    func_template_false = func_template_false.replaceAll("@PH_T", this.t);
    func_template_false = func_template_false.replaceAll("@PH_K", k);
    this.causal_function_name = "lift";
  }

  @Override public SqlOperator getOperator() {
    return null;
  }

  @Override public void unparse(SqlWriter writer, int leftPrec, int rightPrec) {
    withs.add(with_template);
    if (flag.equals("true"))
      replace_sql = func_template_true;
    else
      replace_sql = func_template_false;
  }

  @Override public List<SqlNode> getOperandList() {
    return null;
  }
}
