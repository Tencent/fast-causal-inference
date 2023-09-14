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
  }

  @Test void testOls() throws SqlParseException {
    String sql = "select ols(y ~ x1 + x2 + X3) from tbl";
    assertEquals(sqlForward(sql),
        "SELECT Ols(true)(y, x1, x2, X3)\n"
  +
            "FROM tbl");
    sql = "select ols(y ~ x1 + x2 + X3, false) from tbl";
    assertEquals(sqlForward(sql),
        "SELECT Ols(FALSE )(y, x1, x2, X3)\n"
  +
            "FROM tbl");
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

    System.out.println(sqlForward(sql));
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
  }

  @Test void testLift() throws SqlParseException {
    String sql = "select lift(1,rand()%2,rand()%2,100,false) from test_data_small_local";
    System.out.println(sqlForward("select * from test_data_small limit 5"));
  }

  @Test void testLinearDML() throws SqlParseException {
    String sql = "select linearDML(y,treatment,x1+x2,x7_needcut,model_y='Ols',model_t=stochasticLogisticRegression(1.0, 1.0, 10, 'SGD'),cv=2 ) from test_data_small";
    /*
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
            "select final_model\n");

     */
    System.out.println(sqlForward(sql));
  }

  @Test void testNonParamDML() throws SqlParseException {
    String sql = "select nonParamDML(y,treatment,x1+x2,x7_needcut,model_y='Ols',model_t='Ols',cv=2) from test_data_small";
    System.out.println(sqlForward(sql));
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
