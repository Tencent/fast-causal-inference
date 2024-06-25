package com.tencent.weixin.utils;

import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.Map;

public class UdfFormatUtil {
    public static JSON formatUdfResult(JSON input) {
        ArrayList<String> column_name = new ArrayList<>();
        ArrayList<ArrayList<String>> data = new ArrayList<>();

        boolean isXexptData = input.toString().contains("recommend_samples") && input.toString().contains("mde");
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
                boolean isXexptColumn = value.contains("recommend_samples") && value.contains("mde");
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
                    } else {
                        column_name.add(key);
                    }
                }

                if (!isXexptColumn) {
                    for (ArrayList<String> transform_row_data : transform_row_datas) {
                        transform_row_data.add(value);
                    }
                } else {
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
                        transform_row_datas.get(0).add(col);
                    }
                    for (String col : column) {
                        transform_row_datas.get(1).add("");
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
                String value = row.get(i);
                rowObj.put(columnName, value);
            }
            result.add(new JSONObject(rowObj));
        }
        return result;
    }
}
