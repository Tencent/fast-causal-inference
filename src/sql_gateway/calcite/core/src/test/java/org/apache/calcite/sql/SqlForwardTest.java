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

import org.apache.calcite.sql.olap.SqlForward;
import org.apache.calcite.sql.parser.SqlParseException;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class SqlForwardTest {
  String sqlForward(String sql) throws SqlParseException {
    SqlForward sql_forward = new SqlForward(sql);
    return sql_forward.getForwardSql();
  }

  @Test void testDeltamethod() throws SqlParseException {
    String sql = "sElect deltamethod(avg(if(a,b,c))/avg(8)+avg(e/f)) from tBl";
    assertEquals(sqlForward(sql),
        "SELECT Deltamethod('x1/x2+x3')(if(a, b, c),8,e / f)\n"
  +
            "FROM tBl");
    sql = "SELECT\n" +
        "    groupname,\n" +
        "    count(*) as cnt, -- 样本量\n" +
        "    avg(numerator)/avg(denominator) as mean, -- 指标均值\n" +
        "    deltamethod(avg(numerator)/avg(denominator)) as std, -- 指标均值的标准差\n" +
        "    deltamethod(avg(numerator)/avg(denominator)) * SQRT(sum(denominator)) AS " +
        "sample_std\n" +
        "FROM\n" +
        "    tbl\n" +
        "where\n" +
        "        metric_id = 8371\n" +
        "group by\n" +
        "    groupname\n";
    assertEquals(sqlForward(sql),
        "SELECT groupname, COUNT(*) AS cnt, AVG(numerator) / AVG(denominator) AS mean, " +
            "Deltamethod('x1/x2')(numerator,denominator) AS std, Deltamethod('x1/x2')(numerator," +
            "denominator) * SQRT(SUM(denominator)) AS sample_std\n" +
            "FROM tbl\n" +
            "WHERE metric_id = 8371\n" +
            "GROUP BY groupname");
  }

  @Test void testOls() throws SqlParseException {
    String sql = "select ols(y ~ x1 + x2 + X3) from tbl";
    assertEquals(sqlForward(sql),
        "SELECT Ols(true)(y,x1,x2,X3)\n"
  +
            "FROM tbl");
    sql = "select ols(y ~ x1 + x2 + X3, false) from tbl";
    assertEquals(sqlForward(sql),
        "SELECT Ols(FALSE )(y,x1,x2,X3)\n"
  +
            "FROM tbl");
    sql = "SELECT ols(y~ x1 +x2+ c*X3) \n" +
        "FROM tbl";
    assertEquals(sqlForward(sql),
        "SELECT Ols(true)(y,x1,x2,c*X3)\n" +
            "FROM tbl");
    System.out.println(sqlForward(sql));
  }
  @Test void testWith() throws SqlParseException {
    String sql = "with t1 as (\n" +
        "SELECT \n" +
        "    uin,treatment_dummy,week_1_vv,\n" +
        "    new_sex\n" +
        "FROM creator_production_data_final_small_creator\n" +
        "where new_sex!=0),\n" +
        "t2 as (\n" +
        "select *,\n" +
        " if(new_sex=1,1,0) AS male\n" +
        " from t1\n" +
        ")\n" +
        "\n" +
        "SELECT ols(week_1_vv~ treatment_dummy + male + treatment_dummy*male) AS res\n" +
        "FROM t2";
    assertEquals(sqlForward(sql),
        "with t1 as (SELECT uin, treatment_dummy, week_1_vv, new_sex\n" +
            "FROM creator_production_data_final_small_creator\n" +
            "WHERE new_sex <> 0 ) ,\n" +
            "t2 as (SELECT *, if(new_sex = 1, 1, 0) AS male\n" +
            "FROM t1 ) SELECT Ols(true)(week_1_vv,treatment_dummy,male,treatment_dummy*male) AS " +
            "res\n" +
            "FROM t2");
    //System.out.println(sqlForward(sql));
  }

  @Test void testBasicOp() throws SqlParseException {
    String sql = "select *,caliperMatching(if(search_num_tc=1,1,-1),score,0.1) AS matchingIndex " +
        "\n" +
        "from dapan_user_fea_all_v1_1695820401_matched where matchingIndex!=0";
    System.out.println(sqlForward(sql));
  }
  @Test void testPredict() throws SqlParseException {
    String sql = "SELECT\n" +
        "                                    predict(olsState(y+x1+x2),x1,x2) AS res \n" +
        "FROM test_data_small LIMIT 10";
    assertEquals(sqlForward(sql),
        "with (select OlsState(y , x1 , x2) from test_data_small) as model \n" +
            "SELECT evalMLMethod(model,x1,x2) AS res\n" +
            "FROM test_data_small\n" +
            "LIMIT 10");
    sql = "SELECT\n" +
        "    predict(ols(numerator~numerator_pre+psex), numerator_pre, psex) from " +
        "tbl\n" +
        "where\n" +
        "    metric_id = 8371";
    assertEquals(sqlForward(sql),
        "with (select OlsState(numerator , numerator_pre , psex) from " +
            "tbl) as model \n" +
            "SELECT evalMLMethod(model,numerator_pre,psex)\n" +
            "FROM tbl\n" +
            "WHERE metric_id = 8371");
    sql = "SELECT\n" +
        "    predict(ols(numerator~numerator_pre+psex),'confidence',0.95,numerator_pre,psex) from" +
        " tbl\n" +
        "where\n" +
        "    metric_id = 8371";
    assertEquals(sqlForward(sql), "" +
        "with (select OlsIntervalState(numerator , numerator_pre , psex) from " +
        "tbl) as model \n" +
        "SELECT evalMLMethod(model,'confidence',0.95,numerator_pre,psex)\n" +
        "FROM tbl\n" +
        "WHERE metric_id = 8371");
  }

  @Test void tmpTest() throws SqlParseException {
     String sql = "select corr(a, b) from tbl";
     //String sql = "select caliperMatching(if(groupname = 'A1', -1, 1), propensity_score, 0.01) as matchingIndex FROM test_tmp_shichao_1102_1026_ios_observational_data where matchingIndex!=0";

     System.out.println(sqlForward(sql));
  }

  @Test void testLongTerm() throws SqlParseException {
    String sql = "select recursiveForcasting([(pull_qv1, click_times_clickidqv1, click_jump_times1, change_query_qv1), (pull_qv2, click_times_clickidqv2, click_jump_times2, change_query_qv2), (pull_qv3, click_times_clickidqv3, click_jump_times3, change_query_qv3), (pull_qv4, click_times_clickidqv4, click_jump_times4, change_query_qv4), (pull_qv5, click_times_clickidqv5, click_jump_times5, change_query_qv5), (pull_qv6, click_times_clickidqv6, click_jump_times6, change_query_qv6)],3 ~ 2,4-5, S1, groupid, 'Ols(True)','{PH}',500,1) from tbl";
    System.out.println(sqlForward(sql));
    /*
    assertEquals(sqlForward(sql),
        "with ( \n" +
            "  SELECT tuple(BootStrapOlsState('Ols(True)', 500,1,'{PH}')(pull_qv3,pull_qv2," +
            "click_times_clickidqv2,click_jump_times2,change_query_qv2),BootStrapOlsState('Ols" +
            "(True)', 500,1,'{PH}')(click_times_clickidqv3,pull_qv2,click_times_clickidqv2," +
            "click_jump_times2,change_query_qv2),BootStrapOlsState('Ols(True)', 500,1,'{PH}')" +
            "(click_jump_times3,pull_qv2,click_times_clickidqv2,click_jump_times2," +
            "change_query_qv2),BootStrapOlsState('Ols(True)', 500,1,'{PH}')(change_query_qv3," +
            "pull_qv2,click_times_clickidqv2,click_jump_times2,change_query_qv2)) FROM tbl where " +
            "groupid = 0 \n" +
            "  ) as model0, \n" +
            "  ( \n" +
            "  SELECT tuple(BootStrapOlsState('Ols(True)', 500,1,'{PH}')(pull_qv3,pull_qv2," +
            "click_times_clickidqv2,click_jump_times2,change_query_qv2),BootStrapOlsState('Ols" +
            "(True)', 500,1,'{PH}')(click_times_clickidqv3,pull_qv2,click_times_clickidqv2," +
            "click_jump_times2,change_query_qv2),BootStrapOlsState('Ols(True)', 500,1,'{PH}')" +
            "(click_jump_times3,pull_qv2,click_times_clickidqv2,click_jump_times2," +
            "change_query_qv2),BootStrapOlsState('Ols(True)', 500,1,'{PH}')(change_query_qv3," +
            "pull_qv2,click_times_clickidqv2,click_jump_times2,change_query_qv2)) FROM tbl where " +
            "groupid = 1 \n" +
            "  ) as model1  \n" +
            " select  BootStrapMerge('Ttest_2samp(\"x1\", \"two-sided\")',500, 1,  '{PH}')(s4)," +
            "BootStrapMerge('Ttest_2samp(\"x1\", \"two-sided\")',500, 1,  '{PH}')(s5)  FROM " +
            "(SELECT \n" +
            "BootStrapState('Ttest_2samp(\"x1\", \"two-sided\")',500, 1,  '{PH}')(x41, group) as " +
            "s4,BootStrapState('Ttest_2samp(\"x1\", \"two-sided\")',500, 1,  '{PH}')(x51, group) " +
            "as s5 FROM (SELECT *, groupid as group, evalMLMethod(model0.1,pull_qv3," +
            "click_times_clickidqv3,click_jump_times3,change_query_qv3) as x41,\n" +
            "evalMLMethod(model0.2,pull_qv3,click_times_clickidqv3,click_jump_times3," +
            "change_query_qv3) as x42,\n" +
            "evalMLMethod(model0.3,pull_qv3,click_times_clickidqv3,click_jump_times3," +
            "change_query_qv3) as x43,\n" +
            "evalMLMethod(model0.4,pull_qv3,click_times_clickidqv3,click_jump_times3," +
            "change_query_qv3) as x44,\n" +
            "evalMLMethod(model0.1,pull_qv4,click_times_clickidqv4,click_jump_times4," +
            "change_query_qv4) as x51,\n" +
            "evalMLMethod(model0.2,pull_qv4,click_times_clickidqv4,click_jump_times4," +
            "change_query_qv4) as x52,\n" +
            "evalMLMethod(model0.3,pull_qv4,click_times_clickidqv4,click_jump_times4," +
            "change_query_qv4) as x53,\n" +
            "evalMLMethod(model0.4,pull_qv4,click_times_clickidqv4,click_jump_times4," +
            "change_query_qv4) as x54 FROM tbl where group = 0) union all SELECT \n" +
            "BootStrapState('Ttest_2samp(\"x1\", \"two-sided\")',500, 1,  '{PH}')(x41, group) as " +
            "s4,BootStrapState('Ttest_2samp(\"x1\", \"two-sided\")',500, 1,  '{PH}')(x51, group) " +
            "as s5 FROM (SELECT *, groupid as group, evalMLMethod(model1.1,pull_qv3," +
            "click_times_clickidqv3,click_jump_times3,change_query_qv3) as x41,\n" +
            "evalMLMethod(model1.2,pull_qv3,click_times_clickidqv3,click_jump_times3," +
            "change_query_qv3) as x42,\n" +
            "evalMLMethod(model1.3,pull_qv3,click_times_clickidqv3,click_jump_times3," +
            "change_query_qv3) as x43,\n" +
            "evalMLMethod(model1.4,pull_qv3,click_times_clickidqv3,click_jump_times3," +
            "change_query_qv3) as x44,\n" +
            "evalMLMethod(model1.1,pull_qv4,click_times_clickidqv4,click_jump_times4," +
            "change_query_qv4) as x51,\n" +
            "evalMLMethod(model1.2,pull_qv4,click_times_clickidqv4,click_jump_times4," +
            "change_query_qv4) as x52,\n" +
            "evalMLMethod(model1.3,pull_qv4,click_times_clickidqv4,click_jump_times4," +
            "change_query_qv4) as x53,\n" +
            "evalMLMethod(model1.4,pull_qv4,click_times_clickidqv4,click_jump_times4," +
            "change_query_qv4) as x54 FROM tbl where group = 1))");

     */
  }

  @Test void testWls() throws SqlParseException {
    String sql = "select wls(y ~ x1 + x2 + X3, weight) from tbl";
    assertEquals(sqlForward(sql),
        "SELECT Wls(y, x1, x2, X3, weight)\n"
  +
            "FROM tbl");
  }

  @Test void testTtest_1samp() throws SqlParseException {
    String sql = "select ttest_1samp(avg(num/deno), 'less', 0, avg(num) +  avg(denom)) from tbl";
    assertEquals(sqlForward(sql), "" +
        "SELECT Ttest_1samp('x1','less',0,'X=x2+x3')(num/deno,num,denom)\n"
  +
        "FROM tbl");
    sql = "select ttest_1samp(avg(num/deno), 'greater') from tbl";
    assertEquals(sqlForward(sql),
        "SELECT Ttest_1samp('x1','greater',0)(num/deno)\n"
  +
            "FROM tbl");
    sql = "select ttest_1samp(avg(num)) from tbl";
    assertEquals(sqlForward(sql),
        "SELECT Ttest_1samp('x1','two-sided',0)(num)\n"
  +
            "FROM tbl");
  }

  @Test void testTtest_2samp() throws SqlParseException {
    String sql = "select ttest_2samp(avg(num)+avg(deno), index, 'two-sided',avg(a)/avg(b)) from tbl";
    assertEquals(sqlForward(sql),
        "SELECT Ttest_2samp('x1+x2','two-sided','X=x3/x4')(num,deno,a,b,index)\n"
  +
            "FROM tbl");
    sql = "select ttest_2samp(avg(rand(1))+avg(rand(2)), mod(rand(3),2), 'two-sided', pse = mod(rand(6), 2) + 123) from tbl";
    assertEquals(sqlForward(sql),
        "SELECT Ttest_2samp('x1+x2','two-sided',2)(RAND(1),RAND(2),MOD(RAND(6), 2),123,MOD(RAND" +
            "(3), 2))\n" +
            "FROM tbl");
    sql = "select ttest_2samp(avg(rand(1))+avg(rand(2)), mod(rand(3),2), 'two-sided', avg(rand()), pse = mod(rand(6), 2) + 123) from tbl";
    assertEquals(sqlForward(sql),
        "SELECT Ttest_2samp('x1+x2','two-sided',2,'X=x3')(RAND(1),RAND(2),RAND(),MOD(RAND(6), 2)," +
            "123,MOD(RAND(3), 2))\n" +
            "FROM tbl");
    sql = "select\n" +
        "    ttest_2samp(avg(numerator)/avg(denominator),T ,'two-sided',\n" +
        "                      avg(numerator_pre)/avg(denominator)+avg(page)/avg(denominator)) as" +
        " res\n" +
        "    from (\n" +
        "        select bucketsrc_hit,\n" +
        "               if(groupname = 'B19', 1, 0) as T,\n" +
        "               sum(numerator) as numerator,\n" +
        "               sum(denominator) as denominator,\n" +
        "               sum(page) as page,\n" +
        "               sum(numerator_pre) as numerator_pre,\n" +
        "               sum(denominator_pre) as denominator_pre\n" +
        "        from expt_detail_57360319_adamdeng_1697704507283\n" +
        "        where metric_id = 10223 and groupname in ('A1','A2','B19')\n" +
        "        group by bucketsrc_hit, if(groupname = 'B19', 1, 0)\n" +
        "    )";
    assertEquals(sqlForward(sql),
        "SELECT Ttest_2samp('x1/x2','two-sided','X=x3/x4+x5/x6')(numerator,denominator," +
            "numerator_pre,denominator,page,denominator,T) AS res\n" +
            "FROM (SELECT bucketsrc_hit, if(groupname = 'B19', 1, 0) AS T, SUM(numerator) AS " +
            "numerator, SUM(denominator) AS denominator, SUM(page) AS page, SUM(numerator_pre) AS" +
            " numerator_pre, SUM(denominator_pre) AS denominator_pre\n" +
            "FROM expt_detail_57360319_adamdeng_1697704507283\n" +
            "WHERE metric_id = 10223 AND groupname IN ('A1', 'A2', 'B19')\n" +
            "GROUP BY bucketsrc_hit, if(groupname = 'B19', 1, 0))");
    sql = "select\n" +
        "    ttest_2samp(avg(numerator)/avg(denominator),if(groupname='B2',1,0),'two-sided') as " +
        "ttest_result\n" +
        "FROM\n" +
        "    tbl\n" +
        "where\n" +
        "        metric_id = 8371\n" +
        "  and groupname in ('A1','A2','B2')";
    assertEquals(sqlForward(sql),
        "SELECT Ttest_2samp('x1/x2','two-sided')(numerator,denominator,if(groupname = 'B2', 1, 0)" +
            ") AS ttest_result\n" +
            "FROM tbl\n" +
            "WHERE metric_id = 8371 AND groupname IN ('A1', 'A2', 'B2')");
    sql = "select\n" +
        "    ttest_2samp(avg(numerator)/avg(denominator),if(groupname='B2',1,0),'two-sided',avg" +
        "(numerator_pre)/avg(denominator_pre)+avg(psex)) AS ttest_result\n" +
        "FROM\n" +
        "    tbl\n" +
        "where\n" +
        "        metric_id = 8371\n" +
        "  and groupname in ('A1','A2','B2')";
    assertEquals(sqlForward(sql),
        "SELECT Ttest_2samp('x1/x2','two-sided','X=x3/x4+x5')(numerator,denominator," +
            "numerator_pre,denominator_pre,psex,if(groupname = 'B2', 1, 0)) AS ttest_result\n" +
            "FROM tbl\n" +
            "WHERE metric_id = 8371 AND groupname IN ('A1', 'A2', 'B2')");
    sql = "select\n" +
        "    psex,\n" +
        "    ttest_2samp(avg(numerator)/avg(denominator),if(groupname='B2',1,0),'two-sided') as " +
        "ttest_result\n" +
        "FROM\n" +
        "    tbl\n" +
        "where\n" +
        "    metric_id = 8371\n" +
        "    and groupname in ('A1','A2','B2')\n" +
        "group by\n" +
        "    psex";
    assertEquals(sqlForward(sql),
        "SELECT psex, Ttest_2samp('x1/x2','two-sided')(numerator,denominator,if(groupname = 'B2'," +
            " 1, 0)) AS ttest_result\n" +
            "FROM tbl\n" +
            "WHERE metric_id = 8371 AND groupname IN ('A1', 'A2', 'B2')\n" +
            "GROUP BY psex");
  }

  @Test void testDid() throws SqlParseException {
    String sql = "select diD(numerator, exptid, uin, gid) from tbl";
    assertEquals(sqlForward(sql),
        "SELECT Ols(numerator,exptid,uin,exptid*uin,gid)\n"
  +
            "FROM tbl");
  }


  @Test void testXexpt_ttest_2samp() throws SqlParseException {
    String sql = "select xexpt_ttest_2samp(num, deno, if(A,B,C), uin, 0.1, 0.2, avg(n/d) + avg(xx), 0.3) from tbl where aa = bb";
    assertEquals(sqlForward(sql),
        "SELECT Xexpt_Ttest_2samp(0.1,0.2,0.3,'X=x3+x4')(num,deno,n/d,xx,uin,if(A, B, C))\n"
  +
            "FROM tbl\n"
  +
            "WHERE aa = bb");
    sql = "select xexpt_ttest_2samp(num, deno, if(A,B,C), uin, 0.1, 0.2) from tbl where aa = bb";
    assertEquals(sqlForward(sql),
        "SELECT Xexpt_Ttest_2samp(0.1,0.2,0.08)(num,deno,uin,if(A, B, C))\n"
  +
            "FROM tbl\n"
  +
            "WHERE aa = bb");
    sql = "SELECT \n" +
        "    xexpt_ttest_2samp(numerator, denominator, if(groupname = 'B2','B','A'), uin, 0.05, 0" +
        ".005, 0.8) -- 0.05代表显著性水平， 0.005代表MDE，0.8代表power，三个参数可省略\n" +
        "FROM\n" +
        "    tbl\n" +
        "where\n" +
        "    metric_id = 8371\n" +
        "    and groupname in ('A1','A2','B2')";
    assertEquals(sqlForward(sql),
        "SELECT Xexpt_Ttest_2samp(0.05,0.005,0.8)(numerator,denominator,uin,if(groupname = 'B2', " +
            "'B', 'A'))\n" +
            "FROM tbl\n" +
            "WHERE metric_id = 8371 AND groupname IN ('A1', 'A2', 'B2')");
    sql = "SELECT \n" +
        "    xexpt_ttest_2samp(numerator, denominator, if(groupname = 'B2', 'B', 'A'), uin, avg" +
        "(numerator_pre)/avg(denominator_pre), 0.05, 0.005, 0.8)\n" +
        "FROM\n" +
        "    tbl\n" +
        "where\n" +
        "    metric_id = 8371\n" +
        "    and groupname in ('A1','A2','B2')";
    assertEquals(sqlForward(sql),
        "SELECT Xexpt_Ttest_2samp(0.05,0.005,0.8,'X=x3/x4')(numerator,denominator,numerator_pre," +
            "denominator_pre,uin,if(groupname = 'B2', 'B', 'A'))\n" +
            "FROM tbl\n" +
            "WHERE metric_id = 8371 AND groupname IN ('A1', 'A2', 'B2')");
  }

  @Test void testNestedQuery() throws SqlParseException {
    String sql = "select page,\n" +
        "xexpt_ttest_2samp(numerator, denominator, if(groupname = 'B1', 'B', 'A'), bucketsrc_hit," +
        " avg(numerator_pre)/avg(denominator_pre), 0.05, 0.005, 0.8)\n" +
        "from (\n" +
        "    select bucketsrc_hit,\n" +
        "           groupname,\n" +
        "           page,\n" +
        "           sum(numerator) as numerator,\n" +
        "           sum(denominator) as denominator,\n" +
        "           sum(numerator_pre) as numerator_pre,\n" +
        "           sum(denominator_pre) as denominator_pre\n" +
        "    from expt_detail_16338862_adamdeng_1695710466997\n" +
        "    where metric_id = 8377 and groupname in ('A1','A2','B1')\n" +
        "    group by bucketsrc_hit, groupname,page\n" +
        ")group by\n" +
        "    page";
    System.out.println(sqlForward(sql));
    assertEquals(sqlForward(sql),
        "SELECT page, Xexpt_Ttest_2samp(0.05,0.005,0.8,'X=x3/x4')(numerator,denominator," +
            "numerator_pre,denominator_pre,bucketsrc_hit,if(groupname = 'B1', 'B', 'A'))\n" +
            "FROM (SELECT bucketsrc_hit, groupname, page, SUM(numerator) AS numerator, SUM" +
            "(denominator) AS denominator, SUM(numerator_pre) AS numerator_pre, SUM" +
            "(denominator_pre) AS denominator_pre\n" +
            "FROM expt_detail_16338862_adamdeng_1695710466997\n" +
            "WHERE metric_id = 8377 AND groupname IN ('A1', 'A2', 'B1')\n" +
            "GROUP BY bucketsrc_hit, groupname, page)\n" +
            "GROUP BY page");
  }

  @Test void testLift() throws SqlParseException {
    String sql = "select lift(1,rand()%2,rand()%2,100,false) from test_data_small_local";
    System.out.println(sqlForward("select * from test_data_small limit 5"));
  }

  @Test void testLinearDML() throws SqlParseException {
    String sql = "select linearDML(y,treatment,x1+x2,x7_needcut,model_y='Ols',model_t=stochasticLogisticRegression(1.0, 1.0, 10, 'SGD'),cv=2 ) from test_data_small";
    assertEquals(sqlForward(sql),
        "with   (\n" +
            "    SELECT\n" +
            "    (\n" +
            "     tuple(\n" +
            "        OlsStateIf(toFloat64(y) ,x1,x2 ,x7_needcut, rowNumberInAllBlocks()%2 != 0)," +
            "\n" +
            "OlsStateIf(toFloat64(y) ,x1,x2 ,x7_needcut, rowNumberInAllBlocks()%2 != 1) \n" +
            "\n" +
            "),\n" +
            "     tuple(\n" +
            "        stochasticLogisticRegressionStateIf(1.0, 1.0, 10, 'SGD')(toFloat64" +
            "(treatment) ,x1,x2 ,x7_needcut, rowNumberInAllBlocks()%2 != 0),\n" +
            "stochasticLogisticRegressionStateIf(1.0, 1.0, 10, 'SGD')(toFloat64(treatment) ,x1,x2" +
            " ,x7_needcut, rowNumberInAllBlocks()%2 != 1) \n" +
            "\n" +
            "     ) \n" +
            "    )\n" +
            "    FROM test_data_small\n" +
            "  ) as mm_models\n" +
            ",\n" +
            " (\n" +
            "   SELECT Ols(false)(mm_y ,x1*mm_t,x2*mm_t, mm_t) FROM \n" +
            "   ( \n" +
            "          SELECT *, treatment - evalMLMethod(mm_models.2.1 ,x1,x2 ,x7_needcut) as " +
            "mm_t, y - evalMLMethod(mm_models.1.1 ,x1,x2 ,x7_needcut) as mm_y from " +
            "test_data_small where rowNumberInAllBlocks()%2 = 0\n" +
            " union all \n" +
            " SELECT *, treatment - evalMLMethod(mm_models.2.2 ,x1,x2 ,x7_needcut) as mm_t, y - " +
            "evalMLMethod(mm_models.1.2 ,x1,x2 ,x7_needcut) as mm_y from test_data_small where " +
            "rowNumberInAllBlocks()%2 = 1\n" +
            "\n" +
            "    ) \n" +
            "  ) as final_model \n" +
            "select final_model");
  }

  @Test void testNonParamDML() throws SqlParseException {
    String sql = "select nonParamDML(numerator,treatment,flast_used_device+modelclass+psex+page+fcity_level+fgrade+fans_level_int,model_y='Ols',model_t='Ols',cv=2) from test_data_small_load3";
    System.out.println(sqlForward(sql));
    assertEquals(sqlForward(sql),
        "with   (\n" +
            "    SELECT\n" +
            "    (\n" +
            "     tuple(\n" +
            "        OlsStateIf(toFloat64(numerator) ,flast_used_device,modelclass,psex,page," +
            "fcity_level,fgrade,fans_level_int , rowNumberInAllBlocks()%2 != 0),\n" +
            "OlsStateIf(toFloat64(numerator) ,flast_used_device,modelclass,psex,page,fcity_level," +
            "fgrade,fans_level_int , rowNumberInAllBlocks()%2 != 1) \n" +
            "\n" +
            "),\n" +
            "     tuple(\n" +
            "        OlsStateIf(toFloat64(treatment) ,flast_used_device,modelclass,psex,page," +
            "fcity_level,fgrade,fans_level_int , rowNumberInAllBlocks()%2 != 0),\n" +
            "OlsStateIf(toFloat64(treatment) ,flast_used_device,modelclass,psex,page,fcity_level," +
            "fgrade,fans_level_int , rowNumberInAllBlocks()%2 != 1) \n" +
            "\n" +
            "), \n" +
            "     count()\n" +
            "    )\n" +
            "    FROM test_data_small_load3\n" +
            "  ) as mm_models\n" +
            ",\n" +
            "  (\n" +
            "   SELECT sum(mm_t * mm_t) from\n" +
            "   (\n" +
            "          SELECT *, treatment - evalMLMethod(mm_models.2.1 ,flast_used_device," +
            "modelclass,psex,page,fcity_level,fgrade,fans_level_int ) as mm_t, numerator - " +
            "evalMLMethod(mm_models.1.1 ,flast_used_device,modelclass,psex,page,fcity_level," +
            "fgrade,fans_level_int ) as mm_y from test_data_small_load3 where " +
            "rowNumberInAllBlocks()%2 = 0\n" +
            " union all \n" +
            " SELECT *, treatment - evalMLMethod(mm_models.2.2 ,flast_used_device,modelclass," +
            "psex,page,fcity_level,fgrade,fans_level_int ) as mm_t, numerator - evalMLMethod" +
            "(mm_models.1.2 ,flast_used_device,modelclass,psex,page,fcity_level,fgrade," +
            "fans_level_int ) as mm_y from test_data_small_load3 where rowNumberInAllBlocks()%2 =" +
            " 1\n" +
            "\n" +
            "    )\n" +
            "  ) as weight_sum_nullable,\n" +
            " (\n" +
            "   SELECT Ols(false)(mm_y / mm_t * sqrt_weight ,flast_used_device*sqrt_weight," +
            "modelclass*sqrt_weight,psex*sqrt_weight,page*sqrt_weight,fcity_level*sqrt_weight," +
            "fgrade*sqrt_weight,fans_level_int*sqrt_weight, sqrt_weight) FROM \n" +
            "   ( \n" +
            "          SELECT Cast(weight_sum_nullable as Float64) as weight_sum, *, treatment - " +
            "evalMLMethod(mm_models.2.1 ,flast_used_device,modelclass,psex,page,fcity_level," +
            "fgrade,fans_level_int ) as mm_t, numerator - evalMLMethod(mm_models.1.1 ," +
            "flast_used_device,modelclass,psex,page,fcity_level,fgrade,fans_level_int ) as mm_y, " +
            "sqrt(mm_t * mm_t * mm_models.3 / weight_sum) as sqrt_weight from " +
            "test_data_small_load3 where rowNumberInAllBlocks()%2 = 0\n" +
            " union all \n" +
            " SELECT Cast(weight_sum_nullable as Float64) as weight_sum, *, treatment - " +
            "evalMLMethod(mm_models.2.2 ,flast_used_device,modelclass,psex,page,fcity_level," +
            "fgrade,fans_level_int ) as mm_t, numerator - evalMLMethod(mm_models.1.2 ," +
            "flast_used_device,modelclass,psex,page,fcity_level,fgrade,fans_level_int ) as mm_y, " +
            "sqrt(mm_t * mm_t * mm_models.3 / weight_sum) as sqrt_weight from " +
            "test_data_small_load3 where rowNumberInAllBlocks()%2 = 1\n" +
            "\n" +
            "    ) \n" +
            "  ) as final_model \n" +
            "select final_model");
  }

  @Test void testIvRegression() throws SqlParseException {
    String sql = "SELECT\n" +
        "                              ivregression(y~(x3~treatment)+x1+x2)\n" +
        "                        FROM\n" +
        "                              test_data_small";
    assertEquals(sqlForward(sql),
        "with \n" +
            "    ( \n" +
            "        SELECT OlsState(true)(toFloat64(x3), toFloat64(treatment),x1,x2) \n" +
            "        FROM test_data_small \n" +
            "    ) AS model1, \n" +
            "    ( \n" +
            "        SELECT OlsState(true)(toFloat64(y), evalMLMethod(model1, toFloat64" +
            "(treatment),x1,x2),x1,x2) \n" +
            "        FROM test_data_small \n" +
            "    ) AS model_final, \n" +
            "    ( \n" +
            "      select  \n" +
            "      MatrixMultiplication(true, false)(1, evalMLMethod(model1, toFloat64(treatment)" +
            ",x1,x2),x1,x2) from test_data_small \n" +
            "    ) as xx_inverse, \n" +
            "    ( \n" +
            "      select  \n" +
            "      MatrixMultiplication(false, true)(1, evalMLMethod(model1, toFloat64(treatment)" +
            ",x1,x2),x1,x2, ABS(toFloat64(y) - evalMLMethod(model_final, toFloat64(x3),x1,x2))) " +
            "from test_data_small \n" +
            "    )  as xx_weighted  \n" +
            "SELECT Ols(true, false, toString(xx_inverse), toString(xx_weighted))(toFloat64(y)," +
            "evalMLMethod(model1, toFloat64(treatment),x1,x2),x1,x2)\n" +
            "FROM test_data_small");
  }

  @Test void testCutBins() throws SqlParseException {
    String sql = "select cutbins(uin, [0, 1000, 2000], true) from test_demo";
    assertEquals(sqlForward(sql),
        "SELECT multiIf(uin<0, '0',uin>=0 and uin<1000, '[0,1000)',uin>=1000 and uin<2000, " +
            "'[1000,2000)', '>=2000')\n"
  +
            "FROM test_demo");
    sql = "select cutbins(uin, [0, 1000, 2000], false) from test_demo";
    assertEquals(sqlForward(sql),
        "SELECT multiIf(uin<0, 1,uin>=0 and uin<1000, 2,uin>=1000 and uin<2000, 3, 4)\n"
  +
            "FROM test_demo");
    sql = "select cutbins(uin, [0, 1000, 2000]) from test_demo";
    assertEquals(sqlForward(sql),
        "SELECT multiIf(uin<0, '0',uin>=0 and uin<1000, '[0,1000)',uin>=1000 and uin<2000, " +
            "'[1000,2000)', '>=2000')\n"
  +
            "FROM test_demo");
  }

  @Test void testCaliperMatching() throws SqlParseException  {
    String sql = "SELECT uin, if(groupname='B10', 1, -1), caliperMatching(if(groupname = 'A1', -1, 1), numerator, 0.2) as index, rand() as rand FROM tbl";
    assertEquals(sqlForward(sql),
        "with    (\n"
  +
            "        SELECT 0.2\n"
  +
            "    ) AS VAR_WEIGHT,\n"
  +
            "     table_1 as (\n"
  +
            "        select *, \n"
  +
            "        if(groupname = 'A1', -1, 1) as m_caliper_t,\n"
  +
            "        numerator as m_caliper_v\n"
  +
            "        from tbl\n"
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
            "            sum(multiIf(m_caliper_cnt < 2, 0, m_caliper_index_min > " +
            "m_caliper_index_max, m_caliper_index_max, m_caliper_index_min)) OVER (ROWS BETWEEN " +
            "UNBOUNDED PRECEDING AND 1 PRECEDING) AS m_caliper_matching_index,\n"
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
            "    ) \n"
  +
            "SELECT uin, if(groupname = 'B10', 1, -1), m_caliper_index AS index, RAND() AS rand\n"
  +
            "FROM final_table");
    //sql = "SELECT uin, if(groupname='B10', 1, -1), caliperMatching(if(groupname = 'A1', -1, 1), numerator, 0.2) as index, rand() as rand FROM tbl";
    sql = "SELECT\n" +
        "                                 treatment,weight,caliperMatching(if(treatment=1,-1,1),weight,0.2) AS matchingIndex\n" +
        "                           FROM\n" +
        "                                 test_data_small\n where matchingIndex != 0 order by abs(matchingIndex) limit 50";
    System.out.println(sqlForward(sql));
  }

  @Test void testExactMatching() throws SqlParseException  {
    String sql = "SELECT\n" +
        "                                 treatment,x2,exactMatching(if(treatment=1,-1,1),x2) as " +
        "matchingIndex\n" +
        "                           FROM\n" +
        "                                 test_data_small where matchingIndex != 0 order by abs(matchingIndex)\n" +
        "                           limit 20";
    assertEquals(sqlForward(sql),
        "with table_1 AS\n" +
            "    (\n" +
            "        SELECT\n" +
            "            *,\n" +
            "            if(treatment = 1, -1, 1) AS mm_t,\n" +
            "            x2 as l1 \n" +
            "        FROM test_data_small\n" +
            "),\n" +
            "    table_2 AS\n" +
            "    (\n" +
            "        SELECT\n" +
            "            count() as mm_cnt, \n" +
            "            min(mm_num) AS mm_index_min,\n" +
            "            max(mm_num) AS mm_index_max,\n" +
            "            sum(multiIf(mm_cnt < 2, 0, mm_index_min > mm_index_max, mm_index_max, " +
            "mm_index_min)) OVER (ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS " +
            "mm_matching_index,\n" +
            "            l1  \n" +
            "        FROM\n" +
            "        (\n" +
            "            SELECT\n" +
            "                count(1) AS mm_num,\n" +
            "                mm_t,\n" +
            "                l1  \n" +
            "            FROM table_1\n" +
            "            GROUP BY\n" +
            "                mm_t,\n" +
            "                l1  \n" +
            "        )\n" +
            "        GROUP BY l1  \n" +
            "),\n" +
            "    final_table AS\n" +
            "    (\n" +
            "        SELECT\n" +
            "            *,\n" +
            "            row_number() OVER (PARTITION BY mm_t, l1  ) AS mm_rn,\n" +
            "            if(mm_rn <= if(mm_cnt < 2, 0, mm_index_min), (mm_rn + mm_matching_index)" +
            " * mm_t, 0) as mm_index \n" +
            "        FROM table_1 AS m_a\n" +
            "        LEFT JOIN\n" +
            "        (\n" +
            "            SELECT\n" +
            "                mm_cnt, \n" +
            "                mm_index_min,\n" +
            "                mm_matching_index,\n" +
            "                l1  \n" +
            "            FROM table_2\n" +
            "        ) AS m_b ON m_a.l1 = m_b.l1\n" +
            "    ) \n" +
            "SELECT treatment, x2, mm_index AS matchingIndex\n" +
            "FROM final_table\n" +
            "WHERE matchingIndex <> 0\n" +
            "ORDER BY ABS(matchingIndex)\n" +
            "LIMIT 20");
    System.out.println(sqlForward(sql));
  }



}
