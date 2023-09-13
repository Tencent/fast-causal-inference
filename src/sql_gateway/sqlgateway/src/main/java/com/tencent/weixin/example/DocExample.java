package com.tencent.weixin.example;

public class DocExample {
    public static final String CONTROLLER_EXAMPLE_GET_200 = "" +
            "{\n" +
            "   \"status\": 200,\n" +
            "   \"message\": \"ok\",\n" +
            "   \"data\": [\n" +
            "    {\n" +
            "        \"id\": 19,\n" +
            "            \"rawSql\": \"select * from a\",\n" +
            "            \"executeSql\": \"select * from a\",\n" +
            "            \"calciteSqlCostTime\": null,\n" +
            "            \"calciteSqlCostTimeReadable\": null,\n" +
            "            \"executeSqlCostTime\": null,\n" +
            "            \"executeSqlCostTimeReadable\": null,\n" +
            "            \"totalTime\": null,\n" +
            "            \"totalTimeReadable\": null,\n" +
            "            \"creator\": \"bearlyhuang\",\n" +
            "            \"retcode\": -1,\n" +
            "            \"createTime\": \"2023-08-21 14:18:56\",\n" +
            "            \"updateTime\": \"2023-08-21 14:18:56\"\n" +
            "    },\n" +
            "    {\n" +
            "        \"id\": 20,\n" +
            "            \"rawSql\": \"select * from a\",\n" +
            "            \"executeSql\": \"select * from a\",\n" +
            "            \"calciteSqlCostTime\": null,\n" +
            "            \"calciteSqlCostTimeReadable\": null,\n" +
            "            \"executeSqlCostTime\": null,\n" +
            "            \"executeSqlCostTimeReadable\": null,\n" +
            "            \"totalTime\": null,\n" +
            "            \"totalTimeReadable\": null,\n" +
            "            \"creator\": \"bearlyhuang\",\n" +
            "            \"retcode\": -1,\n" +
            "            \"createTime\": \"2023-08-21 15:09:35\",\n" +
            "            \"updateTime\": \"2023-08-21 15:09:35\"\n" +
            "    }\n" +
            "  ],\n" +
            "  \"timestamp\": \"2023-08-21 15:09:38\"\n" +
            "}";
    public static final String CONTROLLER_EXAMPLE_GET_400 = "{\n" +
            "  \"status\": 400,\n" +
            "  \"message\": \"获取失败, 没有对应的主键id可获取\",\n" +
            "  \"data\": null,\n" +
            "  \"timestamp\": \"2022-12-12 14:53:01\"\n" +
            "}";
    public static final String CONTROLLER_EXAMPLE_GET_500 = "{\n" +
            "  \"status\": 500,\n" +
            "  \"message\": \"服务器系统异常\",\n" +
            "  \"data\": null,\n" +
            "  \"timestamp\": \"2022-12-12 14:53:01\"\n" +
            "}";

    public static final String CONTROLLER_POST_EXAMPLE_200 = "" +
            "{\n" +
            "  \"status\": 204,\n" +
            "  \"message\": \"no content 用户删除数据成功\",\n" +
            "  \"data\": null,\n" +
            "  \"timestamp\": \"2023-08-21 15:42:08\"\n" +
            "}\n";

    public static void main(String[] args) {
        System.out.println(CONTROLLER_EXAMPLE_GET_200);
    }
}
