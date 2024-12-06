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

  private String with_template_starrcoks = " mm_params as ( \n"
      +
      "  select  \n"
      +
      "  floor(count() / cast(@PH_K as bigint)) as m1, \n"
      +
      "  count() as m2, \n"
      +
      "  avg(@PH_T) as m3, \n"
      +
      "  avg(@PH_Y) as m4, \n"
      +
      "  cast(@PH_K as bigint) - count() % cast(@PH_K as bigint) as m5, \n"
      +
      "  0 as m6 \n"
      +
      "  from @TBL \n"
      +
      "  )\n";

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

  private String func_template_strarocks_true = "select ratio, \n"
      +
      " lift, \n"
      +
      " lift * ratio as gain, \n"
      +
      " ate, \n"
      +
      " ate * ratio as ramdom_gain \n"
      +
      " FROM \n"
      +
      " ( \n"
      +
      "    SELECT \n"
      +
      "      (sum(sum1) OVER (ORDER BY ratio ASC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) \n"
      +
      "      / sum(cnt1) OVER (ORDER BY ratio ASC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)) - \n"
      +
      "      (sum(sum0) OVER (ORDER BY ratio ASC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) / \n"
      +
      "      sum(cnt0) OVER (ORDER BY ratio ASC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)) AS lift, \n"
      +
      "      (sum(sum1) OVER (PARTITION BY mm_params.m6) / sum(cnt1) OVER (PARTITION BY mm_params.m6)) - \n"
      +
      "      (sum(sum0) OVER (PARTITION BY mm_params.m6) / sum(cnt0) OVER (PARTITION BY mm_params.m6)) AS ate, \n"
      +
      "       * \n"
      +
      "    FROM \n"
      +
      "    ("
      +
      "         SELECT \n"
      +
      "             if(rn <= ((mm_params.m5) * (mm_params.m1)), floor((rn - 1) / (mm_params.m1)), floor((" +
      "               (rn - ((mm_params.m1) * (mm_params.m5))) - 1) / ((mm_params.m1) + 1)) + (mm_params.m5)) + 1 AS" +
      "               gid, \n"
      +
      "             max(rn) AS max_rn, \n"
      +
      "             max(rn) / max(mm_params.m2) AS ratio, \n"
      +
      "             sum(if(mm_t = 0, mm_y, 0)) AS sum0, \n"
      +
      "             sum(if(mm_t = 1, mm_y, 0)) AS sum1, \n"
      +
      "             count(if(mm_t = 0, 1, NULL)) AS cnt0, \n"
      +
      "             count(if(mm_t = 1, 1, NULL)) AS cnt1 \n"
      +
      "         FROM \n"
      +
      "         ( \n"
      +
      "               SELECT \n"
      +
      "                     row_number() OVER (ORDER BY mm_ite DESC) AS rn, \n"
      +
      "                     * \n"
      +
      "               FROM \n"
      +
      "               ( \n"
      +
      "                     SELECT \n"
      +
      "                           @PH_ITE as mm_ite, \n"
      +
      "                           @PH_Y AS mm_y, \n"
      +
      "                           @PH_T AS mm_t \n"
      +
      "                     FROM @TBL \n"
      +
      "               ) tmp \n"
      +
      "               ORDER BY mm_ite DESC \n"
      +
      "          ) tmp1, mm_params \n"
      +
      "          GROUP BY gid \n"
      +
      "     ) tmp2, mm_params \n"
      +
      " ) tmp3, mm_params \n"
      +
      " ORDER BY ratio ASC";

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


  private String func_template_starrocks_false = "select ratio, \n" +
      " t_, \n"
      +
      " y_, \n"
      +
      " lift, \n"
      +
      " lift * ratio as gain, \n"
      +
      " ate, \n"
      +
      " ate * ratio as ramdom_gain \n"
      +
      " FROM \n"
      +
      " ( \n"
      +
      "     select *, \n"
      +
      "         ((((sum(sumty) OVER (ORDER BY ratio ASC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) - \n"
      +
      "         (sum(sumt) OVER (ORDER BY ratio ASC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) * y_)) - \n"
      +
      "         (sum(sumy) OVER (ORDER BY ratio ASC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) * t_)) + \n"
      +
      "         (((y_* t_) * ratio) * (mm_params.m2))) / \n"
      +
      "         ((sum(sumtt) OVER (ORDER BY ratio ASC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) - \n"
      +
      "         ((2 * sum(sumt) OVER (ORDER BY ratio ASC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)) * t_)) \n"
      +
      "         + (((t_ * t_) * ratio) * (mm_params.m2))))  AS lift, \n"
      +
      "         ((((sum(sumty) OVER (PARTITION BY mm_params.m6) - \n"
      +
      "         (sum(sumt) OVER (PARTITION BY mm_params.m6) * (mm_params.m4))) - \n"
      +
      "         (sum(sumy) OVER (PARTITION BY mm_params.m6) * \n"
      +
      "         (mm_params.m3))) + (((mm_params.m4) * (mm_params.m3)) * \n"
      +
      "         (mm_params.m2))) / ((sum(sumtt) OVER (PARTITION BY mm_params.m6) - \n"
      +
      "         ((2 * sum(sumt) OVER (PARTITION BY mm_params.m6)) * \n"
      +
      "         (mm_params.m3))) + (((mm_params.m3) * (mm_params.m3)) * (mm_params.m2))))  AS ate \n"
      +
      "         from ( \n"
      +
      "             select *, \n" +
      "             sum(sumt) OVER (ORDER BY ratio ASC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) / (ratio * (mm_params.m2)) AS t_, \n"
      +
      "             sum(sumy) OVER (ORDER BY ratio ASC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) / (ratio * (mm_params.m2)) AS y_ \n"
      +
      "              from ( \n"
      +
      "                 SELECT \n"
      +
      "                     if(rn <= ((mm_params.m5) * (mm_params.m1)), floor((rn - 1) / (mm_params.m1)), floor(((rn - ((mm_params.m1) * (mm_params.m5))) - 1) / ((mm_params.m1) + 1)) + (mm_params.m5)) + 1 AS gid, \n"
      +
      "                     max(rn) AS max_rn, \n"
      +
      "                     max(rn) / max(mm_params.m2) AS ratio, \n"
      +
      "                     sum(mm_t) AS sumt, \n"
      +
      "                     sum(mm_t * mm_t) AS sumtt, \n"
      +
      "                     sum(mm_y) AS sumy, \n"
      +
      "                     sum(mm_t * mm_y) AS sumty \n"
      +
      "                     FROM \n"
      +
      "                     ( \n"
      +
      "                         SELECT \n"
      +
      "                             row_number() OVER (ORDER BY mm_ite DESC) AS rn, \n"
      +
      "                             * \n"
      +
      "                         FROM \n"
      +
      "                         ( \n"
      +
      "                            SELECT \n"
      +
      "                                 @PH_ITE as mm_ite, \n"
      +
      "                                 @PH_Y AS mm_y, \n"
      +
      "                                 @PH_T AS mm_t \n"
      +
      "                            FROM @TBL \n"
      +
      "                         ) tmp \n"
      +
      "                         ORDER BY mm_ite DESC \n"
      +
      "                      )  tmp1, mm_params \n"
      +
      "                      GROUP BY gid \n"
      +
      "                 ) tmp2, mm_params \n"
      +
      "         ) tmp3, mm_params \n"
      +
      " ) tmp4 \n" +
      " ORDER BY ratio ASC";


  String ite, t, y, k, flag;


  public LiftParser(SqlParserPos pos, String ite, String Y, String T, String k, String flag, EngineType engineType) {
    super(pos, engineType);
    this.ite = SqlForwardUtil.exchangIdentity(ite);
    this.y = SqlForwardUtil.exchangIdentity(Y);
    this.t = SqlForwardUtil.exchangIdentity(T);
    this.k = SqlForwardUtil.exchangIdentity(k);
    this.flag = SqlForwardUtil.exchangIdentity(flag);

    if (engineType == EngineType.ClickHouse) {
      with_template = with_template.replaceAll("@PH_ITE", ite).replaceAll("@PH_Y", this.y);
      with_template = with_template.replaceAll("@PH_T", this.t).replaceAll("@PH_K", k);

      func_template_true = func_template_true.replaceAll("@PH_ITE", ite).replaceAll("@PH_Y", this.y);
      func_template_true = func_template_true.replaceAll("@PH_T", this.t).replaceAll("@PH_K", k);

      func_template_false = func_template_false.replaceAll("@PH_ITE", ite).replaceAll("@PH_Y", this.y);
      func_template_false = func_template_false.replaceAll("@PH_T", this.t).replaceAll("@PH_K", k);
    } else {
      with_template_starrcoks = with_template_starrcoks.replaceAll("@PH_ITE", ite).replaceAll("@PH_Y", this.y);
      with_template_starrcoks = with_template_starrcoks.replaceAll("@PH_T", this.t).replaceAll("@PH_K", k);

      func_template_strarocks_true = func_template_strarocks_true.replaceAll("@PH_ITE", ite).replaceAll("@PH_Y", this.y);
      func_template_strarocks_true = func_template_strarocks_true.replaceAll("@PH_T", this.t).replaceAll("@PH_K", k);

      func_template_starrocks_false = func_template_starrocks_false.replaceAll("@PH_ITE", ite).replaceAll("@PH_Y", this.y);
      func_template_starrocks_false = func_template_starrocks_false.replaceAll("@PH_T", this.t).replaceAll("@PH_K", k);
    }
    this.causal_function_name = "lift";
  }

  @Override public SqlOperator getOperator() {
    return null;
  }

  @Override
  public void unparseClickHouse(SqlWriter writer, int leftPrec, int rightPrec) {
    withs.add(with_template);
    if (Boolean.parseBoolean(flag))
      replace_sql = func_template_true;
    else
      replace_sql = func_template_false;
  }

  @Override
  public void unparseStarRocks(SqlWriter writer, int leftPrec, int rightPrec) {
    withs.add(with_template_starrcoks);
    if (Boolean.parseBoolean(flag))
      replace_sql = func_template_strarocks_true;
    else
      replace_sql = func_template_starrocks_false;
  }

  @Override public List<SqlNode> getOperandList() {
    return null;
  }
}
