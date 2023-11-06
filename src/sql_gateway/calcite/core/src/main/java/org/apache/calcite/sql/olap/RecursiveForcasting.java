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
import java.util.Arrays;
import java.util.List;

public class RecursiveForcasting extends SqlCallCausal {

  String template_models = "( \n" +
      "  SELECT tuple(@PH_MODEL) FROM @TBL where @PH_GROUP = 0 \n" +
      "  ) as model0, \n" +
      "  ( \n" +
      "  SELECT tuple(@PH_MODEL) FROM @TBL where @PH_GROUP = 1 \n" +
      "  ) as model1 ";


  String func_template = " @PH_TTESTS FROM (@PH_TBLS)";

  String template_tbl = "SELECT *, @PH_GROUP as group, @PH_PREDICT FROM @TBL where group = ";

  String template_ttest = " @PH_INDEX, Ttest_2samp('x1', 'two-sided')(@PH_X, group), '\\n'";

  String template_no_ttest = " @PH_X as @PH_X_pred ";

  ArrayList<ArrayList<String>> surrogates = new ArrayList<ArrayList<String>>();
  ArrayList<String> formula;
  String predict;
  String y;
  String group;
  String model;
  String bs_param;
  String sample_num;
  String bs_num;


  public RecursiveForcasting(SqlParserPos pos) {
    super(pos);
  }

  public RecursiveForcasting(SqlParserPos pos, ArrayList<ArrayList<String>> surrogates, String formula, String predict, String y, String group,
      String model, String bs_param, String sample_num, String bs_num) {
    super(pos);
    this.causal_function_name = "recursiveForcasting";
    this.surrogates = surrogates;
    this.formula = new ArrayList<String> (Arrays.asList(formula.split("\\+")));
    this.predict = predict;
    this.y = y.replaceAll("S", "").trim();
    this.group = group;
    this.model = model.substring(1, model.length() - 1);
    this.bs_param = bs_param;
    this.sample_num = sample_num;
    this.bs_num = bs_num;

  }

  @Override public SqlOperator getOperator() {
    return null;
  }

  @Override public void unparse(SqlWriter writer, int leftPrec, int rightPrec) {
    int predict_start = 0;
    int predict_end = 0;
    if (predict.contains("-")) {
      String[] split_tmp = predict.split("-");
      predict_start = Integer.parseInt(split_tmp[0].trim());
      predict_end = Integer.parseInt(split_tmp[1].trim());
    }

    boolean useTtest = true;

    boolean use_bs = true;

    if (use_bs) {
      this.model = "BootStrapOlsState('" + model + "', " + sample_num + "," + "1," + bs_param + ")";
    }

    String PH_MODEL;
    int model_num = this.surrogates.get(0).size();
    int data_num = surrogates.size();
    int x_num = formula.size() - 1;
    String XS = "";
    {
      String models = "";
      for (int i = 0; i < model_num; i++) {
        int y_i = Integer.valueOf(formula.get(0).trim()) - 1;
        String tmp_model = model + "(";
        tmp_model += surrogates.get(y_i).get(i) + ",";

        for (int j = 1; j <= x_num; j++) {
          int x_i = Integer.valueOf(formula.get(j).trim()) - 1;
          for (String v : surrogates.get(x_i))
            tmp_model += v + ",";
        }
        tmp_model = tmp_model.substring(0, tmp_model.length() - 1);
        tmp_model += XS + ")";
        models += tmp_model + ",";
      }
      models = models.substring(0, models.length() - 1);
      PH_MODEL = models;
    }
    String with = template_models;

    with = with.replaceAll("@PH_GROUP", group);
    with = with.replaceAll("@PH_MODEL", PH_MODEL);

    String PH_PREDICT0 = "";
    String PH_PREDICT1 = "";
    int l = 1, r = surrogates.size();
      for (int i = predict_start; i <= predict_end; i++) {
        for (int j = 1; j <= model_num; j++) {
          String tmp0 = "evalMLMethod(";
          String tmp1 = "evalMLMethod(";
          tmp0 +=  "model0." + String.valueOf(j);
          tmp1 +=  "model1." + String.valueOf(j);
          String tmp = "";
          for (int k = x_num; k >= 1; k--) {
            for (int z = 0; z < model_num; z++) {
              if (i - k >= l && i - k <= r)
                tmp += "," + surrogates.get(i-k-1).get(z);
              else tmp += ", x" + String.valueOf(i - k) + String.valueOf(z  + 1);
            }
          }
          if (use_bs)
            tmp += XS + ") as x" + String.valueOf(i) + String.valueOf(j) + ",\n";
        else
          tmp += XS + ") as x" + String.valueOf(i) + String.valueOf(j) + ",\n";
          PH_PREDICT0 += tmp0 + tmp;
          PH_PREDICT1 += tmp1 + tmp;
        }
      }

      PH_PREDICT0 = PH_PREDICT0.substring(0, PH_PREDICT0.length() - 2);
      PH_PREDICT1 = PH_PREDICT1.substring(0, PH_PREDICT1.length() - 2);

    String tmp_tbl0 = template_tbl;
    tmp_tbl0 = tmp_tbl0.replaceAll("@PH_PREDICT", PH_PREDICT0);
    String tmp_tbl1 = template_tbl;
    tmp_tbl1 = tmp_tbl1.replaceAll("@PH_PREDICT", PH_PREDICT1);

    String PH_TBLS = tmp_tbl0 + " 0 " + " union all " + tmp_tbl1 + " 1 ";
    if (use_bs) {
      tmp_tbl0 += "0";
      tmp_tbl1 += "1";
      String prefix = "SELECT \n";
      for (int i = predict_start; i <= predict_end; i++) {
        String tmp = "BootStrapState(\'Ttest_2samp(\"x1\", \"two-sided\")\'," + sample_num +  ", " + bs_num +  ",  " + bs_param +  ")(@PH_X, group) as s" + String.valueOf(i) + ",";
        tmp = tmp.replaceAll("@PH_X", "x" + String.valueOf(i) + String.valueOf(this.y));
        prefix += tmp;
      }
      prefix = prefix.substring(0, prefix.length() - 1) + ' ';
      prefix += "FROM (";
      PH_TBLS = prefix + tmp_tbl0 + ") union all " + prefix + tmp_tbl1 + ")";
    }

    String PH_TTESTS = "";

    for (int i = predict_start; i <= predict_end; i++) {
      String tmp = useTtest ? template_ttest : template_no_ttest;
      if (use_bs) {
        tmp = "BootStrapMerge(\'Ttest_2samp(\"x1\", \"two-sided\")\'," + sample_num +  ", " + bs_num +  ",  " + bs_param +  ")(s" + String.valueOf(i) + ")";
      }
      tmp = tmp.replaceAll("@PH_INDEX", String.valueOf(i));
      tmp = tmp.replaceAll("@PH_X", "x" + String.valueOf(i) + String.valueOf(y));
      PH_TTESTS += tmp + ",";
    }
    PH_TTESTS = PH_TTESTS.substring(0, PH_TTESTS.length() - 1) + ' ';

    this.withs.add(with);
    func_template = func_template.replaceAll("@PH_TBLS", PH_TBLS);
    func_template = func_template.replaceAll("@PH_TTESTS", PH_TTESTS);
    func_template = func_template.replaceAll("@PH_GROUP", group);
    this.replace_sql = " select " + func_template;
  }

  @Override public List<SqlNode> getOperandList() {
    return null;
  }
}
