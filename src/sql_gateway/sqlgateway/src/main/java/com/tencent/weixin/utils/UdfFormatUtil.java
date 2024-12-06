package com.tencent.weixin.utils;

import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.Map;
import com.github.vertical_blank.sqlformatter.SqlFormatter;
import java.math.BigDecimal;
import java.math.RoundingMode;

public class UdfFormatUtil {
    public static JSON formatUdfResult(JSON input) {
        ArrayList<String> column_name = new ArrayList<>();
        ArrayList<ArrayList<String>> data = new ArrayList<>();

        boolean isXexptData = input.toString().toLowerCase().contains("recommend_samples")
                && input.toString().toLowerCase().contains("mde");
        JSONArray inputArray = (JSONArray) input;

        for (int i = 0; i < inputArray.size(); i++) {
            ArrayList<ArrayList<String>> transform_row_datas = new ArrayList<>();
            transform_row_datas.add(new ArrayList<String>());
            if (isXexptData) {
                transform_row_datas.add(new ArrayList<String>());
            }

            JSONObject row = (JSONObject) inputArray.get(i);
            for (String key : row.keySet()) {
                String value = row.getString(key);
                boolean isXexptColumn = value.toLowerCase().contains("recommend_samples")
                        && value.toLowerCase().contains("mde");
                if (i == 0) {
                    if (isXexptColumn) {
                        String[] xexpt_value = value.split("\n");
                        // second line and five line is column name
                        String line = xexpt_value[1];
                        String[] column = line.split("\\s+");
                        for (String col : column) {
                            column_name.add(col);
                        }
                        line = xexpt_value[5];
                        column = line.split("\\s+");
                        for (String col : column) {
                            column_name.add(col);
                        }
                    } else if (value.contains("\n")) {
                        String[] column = value.split("\n");
                        Integer k = 0;
                        while (k < column.length) {
                            if (column[k].isEmpty() || column[k].contains("warning"))
                                k++;
                            else {
                                String[] column_split = column[k].split("\\s+");
                                for (String col : column_split)
                                    column_name.add(col);
                                break;
                            }
                        }
                    } else {
                        column_name.add(key);
                    }
                }

                if (isXexptColumn) {
                    String[] xexpt_value = value.split("\n");
                    String lineA = xexpt_value[2];
                    String lineB = xexpt_value[3];
                    String[] columnA = lineA.split("\\s+");
                    String[] columnB = lineB.split("\\s+");
                    for (String col : columnA) {
                        transform_row_datas.get(0).add(col);
                    }
                    for (String col : columnB) {
                        transform_row_datas.get(1).add(col);
                    }
                    String columnTotal = xexpt_value[6];
                    String[] column = columnTotal.split("\\s+");
                    for (String col : column) {
                        transform_row_datas.get(1).add(col);
                    }
                    for (String col : column) {
                        transform_row_datas.get(0).add("");
                    }
                } else if (value.contains("\n")) {
                    // split with '\n' and get last line
                    String[] split_value = value.split("\n");
                    String last_line = split_value[split_value.length - 1];
                    String[] column = last_line.split("\\s+");
                    for (ArrayList<String> transform_row_data : transform_row_datas) {
                        for (String col : column) {
                            transform_row_data.add(col);
                        }
                    }
                } else {
                    for (ArrayList<String> transform_row_data : transform_row_datas) {
                        transform_row_data.add(value);
                    }
                }
            }
            data.addAll(transform_row_datas);
        }
        // get jsonArray result with the same order
        JSONArray result = new JSONArray();
        for (ArrayList<String> row : data) {
            Map<String, Object> rowObj = new LinkedHashMap<>();
            for (int i = 0; i < column_name.size(); i++) {
                String columnName = column_name.get(i);
                String value = "";
                if (i < row.size()) {
                    value = row.get(i);
                }
                value = tryFormatFloat(value);
                value = tryFormatQuantileTestBucketResult(value, columnName);
                columnName = tryReplaceColumnName(columnName);
                rowObj.put(columnName, value);
            }
            result.add(new JSONObject(rowObj));
        }
        return result;
    }

    public static String trySqlFormat(String sql) {
        try {
            return SqlFormatter.format(sql);
        } catch (Exception e) {
            return sql;
        }
    }

    private static String tryReplaceColumnName(String columnName) {
        if (columnName.equals("p_value")) {
            return "pvalue";
        }
        if (columnName.equals("quantile_qtb")) {
            return "quantile";
        }
        if (columnName.equals("std_samp_qtb")) {
            return "std_samp";
        }
        if (columnName.equals("rela_diff_qtb")) {
            return "rela_diff";
        }
        if (columnName.equals("rela_diff_confidence_interval_qtb")) {
            return "rela_diff_confidence_interval";
        }
        if (columnName.equals("abs_diff_qtb")) {
            return "abs_diff";
        }
        if (columnName.equals("abs_diff_confidence_interval_qtb")) {
            return "abs_diff_confidence_interval";
        }
        return columnName;
    }

    private static String tryFormatFloat(String value, int precision) {
        // if it's a Long value, then return directly
        if (value.matches("^-?\\d+$")) {
            return value;
        }
        try {
            BigDecimal bd = new BigDecimal(value);
            bd = bd.setScale(precision, RoundingMode.HALF_UP); // 设置小数点后6位，并四舍五入
            bd = bd.stripTrailingZeros(); // 去掉后缀0
            String result = bd.toPlainString(); // 返回不带指数形式的字符串

            // 如果结果是整数，则去掉小数点
            if (result.contains(".") && result.endsWith("0")) {
                result = result.substring(0, result.length() - 1);
            }
            return result;
        } catch (Exception e) {
            return value;
        }
    }

    private static String tryFormatFloat(String value) {
        return tryFormatFloat(value, 6);
    }

    private static String tryFormatQuantileTestBucketResult(String value, String columnName) {
        if (columnName.equals("std_samp_qtb")) {
            value = tryFormatFloat(value, 2);
        }
        if (columnName.equals("quantile_qtb")) {
            value = tryFormatFloat(value, 5);
        }
        if (columnName.equals("rela_diff_qtb")) {
            value = tryFormatAsPercentile(value);
        }
        if (columnName.equals("rela_diff_confidence_interval_qtb")
                || columnName.equals("abs_diff_confidence_interval_qtb")) {
            value = tryReplaceParentheses(value);
        }
        if (columnName.equals("rela_diff_confidence_interval_qtb")) {
            value = tryFormatPercentileInterval(value);
        }
        if (columnName.equals("abs_diff_confidence_interval_qtb")) {
            value = tryFormatInterval(value);
        }
        return value;
    }

    private static String tryReplaceParentheses(String value) {
        if (value.contains("(") && value.contains(")")) {
            return value.replace("(", "[").replace(")", "]");
        }
        return value;
    }

    private static String tryFormatAsPercentile(String value, int precision) {
        if (value.contains("%")) {
            return value;
        }
        try {
            BigDecimal bd = new BigDecimal(Double.parseDouble(value) * 100);
            bd = bd.setScale(precision, RoundingMode.HALF_UP); // 设置小数点后6位，并四舍五入
            bd = bd.stripTrailingZeros(); // 去掉后缀0
            String result = bd.toPlainString(); // 返回不带指数形式的字符串

            // 如果结果是整数，则去掉小数点
            if (result.contains(".") && result.endsWith("0")) {
                result = result.substring(0, result.length() - 1);
            }
            return result + "%";
        } catch (Exception e) {
            return value;
        }
    }

    private static String tryFormatAsPercentile(String value) {
        return tryFormatAsPercentile(value, 3);
    }

    private static String tryFormatPercentileInterval(String value) {
        try {
            String[] split = value.replace("[", "").replace("]", "").split(",");
            return "[" + tryFormatAsPercentile(tryFormatFloat(split[0])) + ","
                    + tryFormatAsPercentile(tryFormatFloat(split[1])) + "]";
        } catch (Exception e) {
            return value;
        }
    }

    private static String tryFormatInterval(String value) {
        try {
            String[] split = value.replace("[", "").replace("]", "").split(",");
            return "[" + tryFormatFloat(split[0]) + "," + tryFormatFloat(split[1]) + "]";
        } catch (Exception e) {
            return value;
        }
    }
}
