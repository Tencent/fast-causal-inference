package com.tencent.weixin.utils;

import java.util.Map;

public class Utils {

    public static Map<String, Object> getStringObjectMap(Object object, Map<String, Object> resultMap, int retCode) {
        if (retCode == 0) {
            resultMap.put("code", -1);
            resultMap.put("message", "操作失败");
        } else {
            resultMap.put("code", 0);
            resultMap.put("message", "操作成功");
        }
        resultMap.put("results", object.toString());
        return resultMap;
    }

    public static Boolean isNotEmpty(String ref) {
        return ref != null && !ref.isEmpty();
    }

    public static Boolean isEmpty(String ref) {
        return !isNotEmpty(ref);
    }
}