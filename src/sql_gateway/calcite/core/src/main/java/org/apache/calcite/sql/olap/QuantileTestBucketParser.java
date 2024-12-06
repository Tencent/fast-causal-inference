//
// Source code recreated from a .class file by IntelliJ IDEA
// (powered by FernFlower decompiler)
//

package org.apache.calcite.sql.olap;

import java.util.ArrayList;
import java.util.List;

import org.apache.calcite.sql.SqlCallCausal;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlWriter;
import org.apache.calcite.sql.parser.SqlParserPos;

import org.apache.commons.lang.NotImplementedException;

public class QuantileTestBucketParser extends SqlCallCausal {
  String Y;
  String T;
  ArrayList<String> percentiles;
  String uin;
  String bootstrapNum;
  String alpha;
  String power;
  String mde;

  public QuantileTestBucketParser(SqlParserPos pos, String Y, String T,
      ArrayList<String> percentiles, String uin, String bootstrapNum, String alpha, String power,
      String mde, EngineType engineType) {
    super(pos, engineType);
    this.Y = Y;
    this.T = T;
    this.percentiles = percentiles;
    this.uin = uin;
    this.bootstrapNum = bootstrapNum;
    this.alpha = alpha;
    this.power = power;
    this.mde = mde;
    this.causal_function_name = "quantileTestBucket";
  }

  public SqlOperator getOperator() {
    return null;
  }

  public String cdf(String x) {
    return String.format("((1 + erf((%s) / sqrt(2))) / 2)", x);
  }

  public void unparseClickHouse(SqlWriter writer, int leftPrec, int rightPrec) {
    ArrayList<String> allWiths = new ArrayList<>();
    String t1 = String.format(
        " __t1__ as (select groupnew, quantile_percentile_zip.2 as percentile, " +
            "quantile_percentile_zip.1 as quantile, count from  ( select groupnew, count, " +
            "arrayJoin(quantile_percentile_zips) as quantile_percentile_zip from ( select %s as " +
            "groupnew, arrayZip(quantilesExactExclusive(%s)(%s), %s) as quantile_percentile_zips," +
            " count(*) as count from @TBL group by %s ) ) )", this.T,
        String.join(",", this.percentiles), this.Y, this.percentiles, this.T);
    allWiths.add(t1);
    String t2 = String.format(
        " __t2__ as (select toUInt8(murmurHash3_32(toInt32(%s),0) / toInt32(pow(2, 25))) AS " +
            "bucket_id, %s as groupnew, arrayZip(quantilesExactExclusive(%s)(%s), %s) as " +
            "quantile_percentile_zips from @TBL group by bucket_id, %s ) ", this.uin, this.T,
        String.join(",", this.percentiles), this.Y, this.percentiles, this.T);
    allWiths.add(t2);
    String quantilesSql =
        " __quantiles__ as (select groupnew, tupleElement(quantile_percentile_zip, 1) as " +
            "quantile, tupleElement(quantile_percentile_zip, 2) as percentile from (select " +
            "groupnew, arrayJoin(quantile_percentile_zips) as quantile_percentile_zip from " +
            "__t2__) ) ";
    allWiths.add(quantilesSql);

    String dfSql;
    for (int treatment = 0; treatment <= 1; ++treatment) {
      for (int percentile_idx = 0; percentile_idx < this.percentiles.size(); ++percentile_idx) {
        dfSql = String.format(
            " __sub_quantiles_%d_%d__ as (select quantile from __quantiles__ where percentile = " +
                "%s and groupnew = %s) ", treatment, percentile_idx,
            this.percentiles.get(percentile_idx), treatment);
        allWiths.add(dfSql);
      }
    }

    String bsParam = "concat('{\"', hostName(), '\":128,\"random_seed\":0}')";
    ArrayList<String> quantsStdSqls = new ArrayList<>();

    for (int treatment = 0; treatment <= 1; ++treatment) {
      for (int percentile_idx = 0; percentile_idx < this.percentiles.size(); ++percentile_idx) {
        String subBootStrapSql = String.format(
            " ( select arrayReduce('stddevPop', BootStrap('AVG', 128, %s, %s)(quantile)) as " +
                "quants_std from __sub_quantiles_%d_%d__  ) ", this.bootstrapNum, bsParam,
            treatment, percentile_idx);
        quantsStdSqls.add(
            String.format("(%d, %s, %s)", treatment, this.percentiles.get(percentile_idx),
                subBootStrapSql));
      }
    }

    allWiths.add(String.format(
        " __quants_std__ as ( select t.1 as groupnew, t.2 as percentile, t.3 as quants_std from " +
            "(select arrayJoin(%s) as t) ) ", quantsStdSqls));
    dfSql =
        " __pre_df__ as ( select l.groupnew, l.percentile, l.count, l.quantile, r.quants_std from" +
            " __t1__ as l inner join __quants_std__ as r on l.groupnew = r.groupnew and l" +
            ".percentile = r.percentile ) ";
    allWiths.add(dfSql);
    String joinedDfSql =
        " __df__ as ( select l.percentile, l.groupnew as groupnew_x, l.quantile as quantiles_x, l" +
            ".count as count_x, l.quants_std as quants_std_x, r.groupnew as groupnew_y, r" +
            ".quantile as quantiles_y, r.count as count_y, r.quants_std as quants_std_y from " +
            "(select * from __pre_df__ where groupnew = 0) as l inner join __pre_df__ as r on l" +
            ".percentile = r.percentile )";
    allWiths.add(joinedDfSql);
    String std_samp = "quants_std_y * sqrt(count_y)";
    String abs_diff = "(quantiles_y - quantiles_x)";
    String rela_diff =
        "if(quantiles_x != 0, quantiles_y / quantiles_x - 1, if(quantiles_y=0,0,inf))";
    String rela_diff_se =
        String.format("(sqrt(quants_std_x*quants_std_x+quants_std_y*quants_std_y)*(%s + 1))",
            rela_diff);
    String pvalue =
        String.format("if((%s) = 0, 1, if(%s=0,0,(1 - (0.5 * (1 + erf(abs(%s / " +
            "%s) / sqrt(2))))) * 2))", rela_diff, rela_diff_se, rela_diff, rela_diff_se);
    String abs_diff_confidence_interval_width =
        String.format("PercentPointFunction(1 - %s / 2) * %s * quantiles_x", this.alpha,
            rela_diff_se);
    String abs_diff_confidence_interval =
        String.format("(%s - (%s), %s + %s)", abs_diff, abs_diff_confidence_interval_width,
            abs_diff, abs_diff_confidence_interval_width);
    String rela_diff_confidence_interval_width =
        String.format("PercentPointFunction(1 - %s / 2) * %s", this.alpha, rela_diff_se);
    String rela_diff_confidence_interval = String.format("(((%s) - (%s)), ((%s) + (%s)))", rela_diff,
        rela_diff_confidence_interval_width, rela_diff, rela_diff_confidence_interval_width);
    String test_power_temp = String.format("%s / %s", mde, rela_diff_se);
    String test_power = String.format("if(%s = 0, 1, 1 - %s + %s)", rela_diff_se,
        this.cdf(String.format("PercentPointFunction(1 - %s / 2) - %s", alpha, test_power_temp)),
        this.cdf(String.format("PercentPointFunction(%s / 2) - %s", alpha, test_power_temp)));
    String point =
        String.format("PercentPointFunction(1 - %s / 2) - PercentPointFunction(1 - %s)", alpha,
            power);
    String recom_sample_size = String.format(
        "if(isInfinite(%s), NULL, toInt64(ceil(%s * %s * count_y * (%s) * (%s) / (%s * %s))))",
        rela_diff, rela_diff_se, rela_diff_se, point, point, mde, mde);
    this.replace_sql = String.format("select \n" +
            " percentile,\n" +
            " groupnew_y as treatment,\n" +
            " quantiles_y as quantile_qtb,\n" +
            " %s as std_samp_qtb,\n" +
            " if(treatment=0,'',cast(%s as String)) as p_value,\n" +
            " if(treatment=0,'',cast(%s as String)) as abs_diff,\n" +
            " if(treatment=0,'',cast(%s as String)) as abs_diff_confidence_interval_qtb,\n" +
            " if(treatment=0,'',cast(%s as String)) as rela_diff_qtb,\n" +
            " if(treatment=0,'',cast(%s as String)) as rela_diff_confidence_interval_qtb,\n" +
            " if(treatment=0,'',cast(%s as String)) as test_power,\n" +
            " if(treatment=0,'',cast(%s as String)) as recom_sample_size\n" +
            " from __df__ order by percentile, treatment ",
        std_samp, pvalue, abs_diff, abs_diff_confidence_interval, rela_diff,
        rela_diff_confidence_interval, test_power, recom_sample_size);
    this.withs.add(String.join(",\n", allWiths));
  }

  public void unparseStarRocks(SqlWriter writer, int leftPrec, int rightPrec) {
    throw new NotImplementedException();
  }

  public List<SqlNode> getOperandList() {
    return null;
  }

  @Override 
  public boolean disableGroupBy() {
    return true;
  }
}
