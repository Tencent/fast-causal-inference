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
package org.apache.calcite.sql.fun;

import org.apache.calcite.sql.SqlCall;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlOperatorBinding;
import org.apache.calcite.sql.SqlSpecialOperator;
import org.apache.calcite.sql.SqlWriter;
import org.apache.calcite.sql.type.InferTypes;
import org.apache.calcite.sql.type.IntervalSqlType;
import org.apache.calcite.sql.type.OperandTypes;
import org.apache.calcite.sql.type.SqlReturnTypeInference;
import org.apache.calcite.sql.type.SqlTypeTransforms;
import org.apache.calcite.sql.validate.SqlMonotonicity;

/**
 * Operator that adds an INTERVAL to a DATETIME.
 */
public class SqlDatetimePlusOperator extends SqlSpecialOperator {
  private static final SqlReturnTypeInference RETURN_TYPE_INFERENCE =
      opBinding ->
          SqlTimestampAddFunction.deduceType(opBinding.getTypeFactory(),
              ((IntervalSqlType) opBinding.getOperandType(1))
                  .getIntervalQualifier().getStartUnit(),
              opBinding.getOperandType(0));

  //~ Constructors -----------------------------------------------------------

  SqlDatetimePlusOperator() {
    super("+", SqlKind.PLUS, 40, true,
        RETURN_TYPE_INFERENCE.andThen(SqlTypeTransforms.TO_NULLABLE),
        InferTypes.FIRST_KNOWN, OperandTypes.MINUS_DATE_OPERATOR);
  }

  //~ Methods ----------------------------------------------------------------

  @Override public void unparse(
      SqlWriter writer,
      SqlCall call,
      int leftPrec,
      int rightPrec) {
    writer.getDialect().unparseSqlDatetimeArithmetic(
        writer, call, SqlKind.PLUS, leftPrec, rightPrec);
  }

  @Override public SqlMonotonicity getMonotonicity(SqlOperatorBinding call) {
    return SqlStdOperatorTable.PLUS.getMonotonicity(call);
  }
}
