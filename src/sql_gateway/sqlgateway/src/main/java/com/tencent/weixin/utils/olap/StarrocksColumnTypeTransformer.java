package com.tencent.weixin.utils.olap;

import java.util.HashMap;

public class StarrocksColumnTypeTransformer extends OlapColumnTypeTransformer {
    private static final HashMap<String, String> dict;

    static {
        dict = new HashMap<>();
        // 初始化dict
        dict.put("bigint", "Int");
        dict.put("int", "Int");
        dict.put("double", "Float");
        dict.put("boolean", "Bool");
        dict.put("date", "Date");
        dict.put("datetime", "DateTime");
        dict.put("time", "Time");
    }

    @Override
    public String toDataFrameColumnType(String columnType) {
        columnType = columnType.toLowerCase();
        if (columnType.contains("varchar(")) {
            return "String";
        }
        if (columnType.contains("int(")) {
            return "Int";
        }
        if (columnType.contains("tinyint(")) {
            return "Int";
        }
        if (columnType.contains("array")) {
            return "Array";
        }
        String dfColumnType = dict.get(columnType);
        if (dfColumnType == null) {
            return columnType;
        }
        return dfColumnType;
    }
}
