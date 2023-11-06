package com.tencent.weixin.utils;


import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;
import lombok.Getter;
import lombok.Setter;
import lombok.ToString;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;
import org.yaml.snakeyaml.Yaml;

import java.io.InputStream;
import java.sql.*;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

@Component
@ConfigurationProperties(prefix = "clickhouse")
@Getter
@Setter
@ToString
public class ClickhouseUtil {
    private Logger logger = LoggerFactory.getLogger(this.getClass());
    private String driver;
    private ArrayList<Map<String, Object>> devices;

    public Connection getClickHouseConnection(String database, Integer deviceId, String launcherIp) {
        if (deviceId == null) {
            throw new RuntimeException("deviceId is arg error");
        }
        Connection conn = null;
        try {
            Class.forName(driver);
            for (Map<String, Object> deviceInfo : devices) {
                if ((int) deviceInfo.get("id") == deviceId.intValue()) {
                    String clickHouseUrl = (String) deviceInfo.get("url");
                    String urlDatabase = clickHouseUrl.substring(clickHouseUrl.lastIndexOf("/", clickHouseUrl.indexOf("?")) + "/".length(), clickHouseUrl.indexOf("?"));
                    String urlIp = clickHouseUrl.substring(clickHouseUrl.indexOf("//") + "//".length(), clickHouseUrl.indexOf(":", clickHouseUrl.indexOf("//")));
                    clickHouseUrl = clickHouseUrl.replace(urlDatabase, database);
                    if (launcherIp != null) {
                        clickHouseUrl = clickHouseUrl.replace(urlIp, launcherIp);
                    }
                    String clickHouseUser = (String) deviceInfo.get("user");
                    String clickHousePassword = (String) deviceInfo.get("password");
                    logger.info("clickHouseUrl=" + clickHouseUrl);
                    conn = DriverManager.getConnection(clickHouseUrl, clickHouseUser, clickHousePassword);
                    return conn;
                }
            }
            throw new RuntimeException("deviceId is arg error");
        } catch (SQLException | ClassNotFoundException e) {
            e.printStackTrace();
        }
        return null;
    }

    public Connection getClickHouseConnection(String database, Integer deviceId) {
        return getClickHouseConnection(database, deviceId, null);
    }

//    private static String clickHouseUrl;
//
//    private static String clickHouseUser;
//
//    private static String clickHousePassword;
//
//    private static Map<String, Object> clickhouseConf;

//    static {
//        try {
//            Yaml yaml = new Yaml();
//            InputStream inputStream = ClickhouseUtil.class.getClassLoader().getResourceAsStream("application.yml");
//            Map<String, Object> ymlObject = yaml.load(inputStream);
//            clickhouseConf = (Map<String, Object>) ymlObject.get("clickhouse");
//            String driver = (String) clickhouseConf.get("driver");
//            Class.forName(driver);
//        } catch (Exception e) {
//            e.printStackTrace();
//            throw new RuntimeException(e);
//        }
//    }
//
//    public static Connection getOnlineClickHouseConnection(String database) {
//        Connection conn = null;
//        try {
//            Map<String, String> clickhouseEnv = (Map<String, String>) clickhouseConf.get("online");
//            clickHouseUrl = clickhouseEnv.get("url");
//            clickHouseUrl = clickHouseUrl.replace("all_in_sql", database);
//            clickHouseUser = clickhouseEnv.get("user");
//            clickHousePassword = clickhouseEnv.get("password");
//            conn = DriverManager.getConnection(clickHouseUrl, clickHouseUser, clickHousePassword);
//        } catch (SQLException e) {
//            e.printStackTrace();
//        }
//        return conn;
//    }


    public void closeAll(AutoCloseable... closeAbles) {
        for (AutoCloseable autoCloseable : closeAbles) {
            try {
                if (autoCloseable != null) {
                    autoCloseable.close();
                }
            } catch (Exception e) {
                e.printStackTrace();

            }
        }
    }

    public JSONObject resultSetToJson(ResultSet resultSet) throws SQLException {
        ResultSetMetaData metaData = resultSet.getMetaData();
        int columnCount = metaData.getColumnCount();
        JSONObject jsonObject = new JSONObject();
        JSONArray schemaArray = new JSONArray();
        for (int i = 1; i <= columnCount; i++) {
            JSONObject columnInfo = new JSONObject();
            columnInfo.put("name", metaData.getColumnName(i));
            columnInfo.put("type", metaData.getColumnTypeName(i));
            schemaArray.add(columnInfo);
        }
        jsonObject.put("schema", schemaArray);

        JSONArray rowsArray = new JSONArray();
        while (resultSet.next()) {
            JSONArray rowArray = new JSONArray();
            for (int i = 1; i <= columnCount; i++) {
                rowArray.add(resultSet.getObject(i));
            }
            rowsArray.add(rowArray);
        }
        jsonObject.put("rows", rowsArray);

        return jsonObject;
    }

    public JSON resultSetToJsonDataFrame(ResultSet resultSet) {
        JSONArray jsonArray = new JSONArray();
        try {
            ResultSetMetaData rsmd = resultSet.getMetaData();
            while (resultSet.next()) {
                LinkedHashMap rowObj = new LinkedHashMap<>();
                int columnCount = rsmd.getColumnCount();
                for (int i = 1; i <= columnCount; i++) {
                    String columnName = rsmd.getColumnName(i);
                    String value = resultSet.getString(columnName);
                    rowObj.put(columnName, value);
                }
                jsonArray.add(new JSONObject(rowObj));
            }
        } catch (SQLException e) {
            e.printStackTrace();
        }
        return jsonArray;
    }

}
