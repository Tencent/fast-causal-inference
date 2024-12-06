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

public class NonParamDMLParser extends SqlCallCausal {
  private String Y, T, model_y, model_t, cv;
  private ArrayList<String> xs, ws;

  static String with_template_models = "  (\n"
  +
      "    SELECT\n"
  +
      "    (\n"
  +
      "     tuple(\n"
  +
      "        @PH_MODELYS\n"
  +
      "),\n"
  +
      "     tuple(\n"
  +
      "        @PH_MODELTS\n"
  +
      "), \n"
  +
      "     count()\n"
  +
      "    )\n"
  +
      "    FROM @TBL\n"
  +
      "  ) as mm_models\n";

  static String with_template_single_model = "@PH_MODEL(toFloat64(@PH_YT) @PH_X @PH_W, rowNumberInAllBlocks()%@PH_CV != @PH_INDEX)";

  static String with_template_weight_sum = "  (\n"
  +
      "   SELECT sum(mm_t * mm_t) from\n"
  +
      "   (\n"
  +
      "          @PH_EACH_UNION\n"
  +
      "    )\n"
  +
      "  ) as weight_sum_nullable";

  static String with_template_final_model = " (\n"
  +
      "   SELECT Ols(false)(mm_y / mm_t * sqrt_weight @PH_X2, sqrt_weight) FROM \n"
  +
      "   ( \n"
  +
      "          @PH_EACH_UNION\n"
  +
      "    ) \n"
  +
      "  ) as final_model";

  static String with_each_union_with_weight = "SELECT Cast(weight_sum_nullable as Float64) as weight_sum, *, @PH_T - evalMLMethod(mm_models.2.@PH_INDEX_1 @PH_X @PH_W) as mm_t, @PH_Y - evalMLMethod(mm_models.1.@PH_INDEX_1 @PH_X @PH_W) as mm_y, sqrt(mm_t * mm_t * mm_models.3 / weight_sum) as sqrt_weight from @TBL where rowNumberInAllBlocks()%@PH_CV = @PH_INDEX\n";

  static String with_each_union = "SELECT *, @PH_T - evalMLMethod(mm_models.2.@PH_INDEX_1 @PH_X @PH_W) as mm_t, @PH_Y - evalMLMethod(mm_models.1.@PH_INDEX_1 @PH_X @PH_W) as mm_y from @TBL where rowNumberInAllBlocks()%@PH_CV = @PH_INDEX\n";

  static String func_template = "select final_model";


  public NonParamDMLParser(SqlParserPos pos, String Y, String T, String model_y, String model_t, String cv, ArrayList<String> xs, ArrayList<String> ws, EngineType engineType) {
    super(pos, engineType);
    this.Y = SqlForwardUtil.exchangIdentity(Y);
    this.T = SqlForwardUtil.exchangIdentity(T);
    this.model_y = SqlForwardUtil.exchangIdentity(model_y);
    this.model_t = SqlForwardUtil.exchangIdentity(model_t);
    this.cv = SqlForwardUtil.exchangIdentity(cv);
    this.xs = xs;
    this.ws = ws;
    if (this.model_t.toUpperCase().equals("ols"))
      this.model_t = 'O' + this.model_t.substring(1);
    if (this.model_y.toUpperCase().equals("ols"))
      this.model_y = 'O' + this.model_y.substring(1);
    this.causal_function_name = "nonParamDML";
  }

  @Override public SqlOperator getOperator() {
    return null;
  }

  @Override public void unparse(SqlWriter writer, int leftPrec, int rightPrec) {
    if (model_y.indexOf("(") != -1) {
      model_y = model_y.substring(0, model_y.indexOf("(")) + "StateIf" + model_y.substring(model_y.indexOf("("));
    } else {
      model_y += "StateIf";
    }
    if (model_t.indexOf("(") != -1) {
      model_t = model_t.substring(0, model_t.indexOf("(")) + "StateIf" + model_t.substring(model_t.indexOf("("));
    } else {
      model_t += "StateIf";
    }

    StringBuffer ph_modelys = new StringBuffer();
    StringBuffer ph_modelts = new StringBuffer();
    {
      for (int i = 0; i < Integer.valueOf(cv); i++) {
        String tmp = with_template_single_model;
        tmp = tmp.replaceAll("@PH_YT", Y);
        tmp = tmp.replaceAll( "@PH_MODEL", model_y);
        tmp = tmp.replaceAll( "@PH_INDEX", String.valueOf(i));
        ph_modelys.append(tmp).append(",\n");
      }
      for (int i = 0; i < Integer.valueOf(cv); i++) {
        String tmp = with_template_single_model;
        tmp = tmp.replaceAll("@PH_YT", T);
        tmp = tmp.replaceAll( "@PH_MODEL", model_t);
        tmp = tmp.replaceAll( "@PH_INDEX", String.valueOf(i));
        ph_modelts.append(tmp).append(",\n");
      }
      ph_modelys.setCharAt(ph_modelys.length() - 2, ' ');
      ph_modelts.setCharAt(ph_modelts.length() - 2, ' ');
    }

    String with_models = with_template_models;
    with_models = with_models.replaceAll("@PH_MODELYS", ph_modelys.toString());
    with_models = with_models.replaceAll("@PH_MODELTS", ph_modelts.toString());

    String ph_each_union = "";
    String ph_each_union2 = "";
    for (int i = 0; i < Integer.valueOf(cv); i++) {
      String tmp = with_each_union;
      tmp = tmp.replaceAll("@PH_INDEX_1", String.valueOf(i+1));
      tmp = tmp.replaceAll("@PH_INDEX", String.valueOf(i));
      if (i != 0) ph_each_union += " union all \n ";
      ph_each_union += tmp;

      String tmp2 = with_each_union_with_weight;
      tmp2 = tmp2.replaceAll("@PH_INDEX_1", String.valueOf(i+1));
      tmp2 = tmp2.replaceAll("@PH_INDEX", String.valueOf(i));
      if (i != 0) ph_each_union2 += " union all \n ";
      ph_each_union2 += tmp2;
    }

    String with_final_model = with_template_final_model;
    String with_weight_sum = with_template_weight_sum;
    with_final_model = with_final_model.replaceAll("@PH_EACH_UNION", ph_each_union2);
    with_weight_sum = with_weight_sum.replaceAll("@PH_EACH_UNION", ph_each_union);

    String ph_x2 = "";
    String ph_x = "";
    for (String x : xs) {
      ph_x2 = ph_x2 + "," + x + "*sqrt_weight";
      ph_x += "," + x;
    }
    String ph_w = "";
    for (String w : ws) {
      ph_w += "," + w;
    }
    with_final_model = with_final_model.replaceAll("@PH_X2", ph_x2);
    with_weight_sum = with_weight_sum.replaceAll("@PH_X2", ph_x2);

    String with = with_models + ",\n"
  + with_weight_sum + ",\n"
  + with_final_model;
    with = with.replaceAll("@PH_MODELYS", ph_modelys.toString());
    with = with.replaceAll("@PH_MODELTS", ph_modelts.toString());
    with = with.replaceAll("@PH_X", ph_x);
    with = with.replaceAll("@PH_W", ph_w);
    with = with.replaceAll("@PH_CV", cv);
    with = with.replaceAll("@PH_T", T);
    with = with.replaceAll("@PH_Y", Y);
    withs.add(with);
    replace_sql = func_template;
  }

  @Override public List<SqlNode> getOperandList() {
    return null;
  }

}
