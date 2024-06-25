package com.tencent.weixin.utils.olap;

import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;
import com.tencent.weixin.proto.AisDataframe;
import lombok.Getter;
import lombok.Setter;
import lombok.ToString;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.sql.*;
import java.text.DecimalFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@Component
@Getter
@Setter
@ToString
public class OlapUtil {
    private Logger logger = LoggerFactory.getLogger(this.getClass());

    @Autowired
    private AllOlapProperties allOlapProperties;

    public Connection getConnection(String database, Integer deviceId, String launcherIp, EngineType engineType)
            throws ClassNotFoundException, SQLException {
        if (deviceId == null) {
            throw new RuntimeException("Unknown deviceId.");
        }
        OlapProperties olapProperties = allOlapProperties.getOlapProperties(engineType);
        Class.forName(olapProperties.getDriver());
        for (Device deviceInfo : olapProperties.getDevices()) {
            if (deviceInfo.getId() == deviceId) {
                String olapUrl = getOlapUrl(database, launcherIp, deviceInfo);
                String olapUser = deviceInfo.getUser();
                String olapPassword = deviceInfo.getPassword();
                return DriverManager.getConnection(olapUrl, olapUser, olapPassword);
            }
        }
        throw new RuntimeException("Unknown deviceId.");
    }

    private static String getOlapUrl(String database, String launcherIp, Device deviceInfo) {
        String olapUrl = deviceInfo.getUrl();
        String urlDatabase =
                olapUrl.substring(olapUrl.lastIndexOf("/", olapUrl.indexOf("?")) + "/".length(), olapUrl.indexOf("?"));
        String urlIp =
                olapUrl.substring(olapUrl.indexOf("//") + "//".length(), olapUrl.indexOf(":", olapUrl.indexOf("//")));
        olapUrl = olapUrl.replace(urlDatabase, database);
        if (launcherIp == null) {
            List<String> ips = deviceInfo.getIp();
            // get a random ip
            launcherIp = ips.get((int) (Math.random() * ips.size()));
        }
        if (launcherIp != null) {
            olapUrl = olapUrl.replace(urlIp, launcherIp);
        }
        return olapUrl;
    }

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
        logger.info("start resultSetToJson");
        ResultSetMetaData metaData = resultSet.getMetaData();
        int columnCount = metaData.getColumnCount();
        JSONObject jsonObject = new JSONObject();
        JSONArray schemaArray = new JSONArray();
        for (int i = 1; i <= columnCount; i++) {
            if (metaData.getColumnName(i).equals("day_"))
                continue;
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
                if (metaData.getColumnName(i).equals("day_"))
                    continue;
                rowArray.add(resultSet.getObject(i));
            }
            rowsArray.add(rowArray);
        }
        jsonObject.put("rows", rowsArray);

        return jsonObject;
    }

    public JSON resultSetToJsonDataFrame(ResultSet resultSet) {
        logger.info("resultSetToJsonDataFrame");
        JSONArray jsonArray = new JSONArray();
        try {
            ResultSetMetaData rsmd = resultSet.getMetaData();
            DecimalFormat df = new DecimalFormat("#");
            df.setMaximumFractionDigits(9); // 设置小数点后最大位数

            while (resultSet.next()) {
                Map<String, Object> rowObj = new LinkedHashMap<>();
                int columnCount = rsmd.getColumnCount();
                for (int i = 1; i <= columnCount; i++) {
                    String columnName = rsmd.getColumnName(i);

                    int columnType = rsmd.getColumnType(i);
                    if (columnType == Types.INTEGER) {
                        int value = resultSet.getInt(columnName);
                        rowObj.put(columnName, value);
                    } else if (columnType == Types.DOUBLE || columnType == Types.FLOAT) {
                        double value = resultSet.getDouble(columnName);
                        String formattedValue = df.format(value);
                        rowObj.put(columnName, formattedValue);
                    } else {
                        String value = resultSet.getString(columnName);
                        rowObj.put(columnName, value);
                    }
                }
                jsonArray.add(new JSONObject(rowObj));
            }
        } catch (SQLException e) {
            e.printStackTrace();
        }
        return jsonArray;
    }

    public String getDefaultStatementSettings(EngineType engineType) {
        if (engineType == EngineType.Clickhouse) {
            return "";
        }
        if (engineType == EngineType.Starrocks) {
            return "";
        }
        throw new IllegalArgumentException(String.format("%s is not a valid engine type.", engineType));
    }

    private static OlapColumnTypeTransformer getColumnTypeTransformer(EngineType engineType) {
        if (engineType == EngineType.Clickhouse) {
            return new ClickhouseColumnTypeTransformer();
        }
        if (engineType == EngineType.Starrocks) {
            return new StarrocksColumnTypeTransformer();
        }
        throw new IllegalArgumentException(String.format("invalid engine type %s", engineType));
    }

    private static String getFieldIdentifier(EngineType engineType) {
        if (engineType == EngineType.Clickhouse) {
            return "name";
        }
        if (engineType == EngineType.Starrocks) {
            return "Field";
        }
        throw new IllegalArgumentException(String.format("invalid engine type %s", engineType));
    }

    private static String getTypeIdentifier(EngineType engineType) {
        if (engineType == EngineType.Clickhouse) {
            return "type";
        }
        if (engineType == EngineType.Starrocks) {
            return "Type";
        }
        throw new IllegalArgumentException(String.format("invalid engine type %s", engineType));
    }

    public static AisDataframe.Column toDataFrameColumnType(EngineType engineType, JSONObject jsonObject) {
        OlapColumnTypeTransformer typeTransformer = OlapUtil.getColumnTypeTransformer(engineType);
        return AisDataframe.Column.newBuilder().setName(jsonObject.getString(getFieldIdentifier(engineType)))
                .setType(typeTransformer.toDataFrameColumnType(jsonObject.getString(getTypeIdentifier(engineType))))
                .build();
    }
}
