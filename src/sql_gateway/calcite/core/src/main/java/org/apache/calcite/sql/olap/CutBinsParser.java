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

public class CutBinsParser extends SqlCallCausal {

  private String target;
  private ArrayList<String> buckets;
  private boolean use_string;

  public CutBinsParser(SqlParserPos pos) {
    super(pos);
  }

  public CutBinsParser(SqlParserPos pos, String target, ArrayList<String> buckets, String use_string) {
    super(pos);
    causal_function_name = "cutBins";
    this.target = SqlForwardUtil.exchangIdentity(target);
    this.buckets = buckets;
    this.use_string = SqlForwardUtil.exchangIdentity(use_string).toLowerCase().trim().equals("true");
  }

  @Override public SqlOperator getOperator() {
    return null;
  }

  @Override public void unparse(SqlWriter writer, int leftPrec, int rightPrec) {
    writer.print("multiIf(");
    writer.print(target + "<" + buckets.get(0));
    if (use_string)
      writer.print(", '" + buckets.get(0) + "'");
    else
      writer.print(", 1");

    for (int i = 0; i < buckets.size() - 1; i++) {
      writer.print("," + target + ">=" + buckets.get(i) + " and " + target + "<" + buckets.get(i+1));
      if (use_string)
        writer.print(", '[" + buckets.get(i) + "," + buckets.get(i+1) + ")'");
      else
        writer.print(", " + String.valueOf(i + 2));
    }
    if (use_string)
      writer.print(", '>=" + buckets.get(buckets.size() - 1) + "'");
    else
      writer.print(", " + String.valueOf(buckets.size() + 1));
    writer.print(")");
  }

  @Override public List<SqlNode> getOperandList() {
    return null;
  }
}
