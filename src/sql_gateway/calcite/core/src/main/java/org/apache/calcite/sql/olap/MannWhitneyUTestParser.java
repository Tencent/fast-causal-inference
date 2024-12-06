//
// Source code recreated from a .class file by IntelliJ IDEA
// (powered by FernFlower decompiler)
//

package org.apache.calcite.sql.olap;

import java.util.List;

import org.apache.calcite.sql.SqlCallCausal;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlWriter;
import org.apache.calcite.sql.parser.SqlParserPos;

public class MannWhitneyUTestParser extends SqlCallCausal {
  String Y;
  String T;
  String alternative;
  String continuity_correction;

  public MannWhitneyUTestParser(SqlParserPos pos, String Y, String T, String alternative,
      String continuity_correction, EngineType engineType) {
    super(pos, engineType);
    this.Y = Y;
    this.T = T;
    this.alternative = alternative;
    this.continuity_correction = continuity_correction;
    this.causal_function_name = "mann_whitney_u_test";
  }

  public SqlOperator getOperator() {
    return null;
  }

  public void unparseClickHouse(SqlWriter writer, int leftPrec, int rightPrec) {
    writer.print(String.format(
        " countIf(%s = 0) * countIf(%s = 1) - 2 * mannWhitneyUTest(%s, %s)(%s, %s).1 as " +
            "statistic, " + "mannWhitneyUTest(%s, %s)(%s, %s).2 as p_value", T, T, alternative,
        continuity_correction, Y, T, alternative, continuity_correction, Y, T));
  }

  public void unparseStarRocks(SqlWriter writer, int leftPrec, int rightPrec) {
    writer.print(String.format(
        " count(if(%s = 0, 1, NULL)) * count(if(%s = 1, 1, NULL)) - 2 * get_json_double" +
            "(mann_whitney_u_test(%s, %s, %s, %s), \"$[0]\") as statistic, " +
            " get_json_double(mann_whitney_u_test(%s, %s, %s, %s), \"$[1]\") as p_value", T, T, Y,
        T, alternative, continuity_correction, Y, T, alternative, continuity_correction));
  }

  public List<SqlNode> getOperandList() {
    return null;
  }
}
