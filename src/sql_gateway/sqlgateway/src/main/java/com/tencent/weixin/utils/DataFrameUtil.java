package com.tencent.weixin.utils;

import com.tencent.weixin.proto.AisDataframe;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

public class DataFrameUtil {
    static private Logger logger = LoggerFactory.getLogger(DataFrameUtil.class);

    static private Integer MAX_LIMIT = 1000000;

    public static String transformDataFrameToSql(AisDataframe.DataFrame dataframe) throws Exception {
        logger.info("transformDataFrameToSql: {}", dataframe);

        String sql = "";
        // fill cte
        String cte = dataframe.getCte();
        if (cte != null && !cte.isEmpty()) {
            sql += cte + "\n";
        }

        sql += "SELECT\n";

        // fill selectitems
        List<AisDataframe.Column> columns = dataframe.getColumnsList();
        if (columns.size() == 0) {
            throw new Exception("transformDataFrameToSql, columns is empty");
        }
        for (AisDataframe.Column column : columns) {
            sql += column.getName();
            if (column.getAlias() != null && !column.getAlias().isEmpty()) {
                sql += " AS " + column.getAlias();
            }
            sql += ", ";
        }
        sql = sql.substring(0, sql.length() - 2);

        // fill from
        AisDataframe.Source source = dataframe.getSource();
        if (source.getType() == AisDataframe.SourceType.ClickHouse) {
            AisDataframe.ClickHouseSource clickHouseSource = source.getClickhouse();
            if (clickHouseSource.getTableName() == null || clickHouseSource.getTableName().isEmpty()) {
                throw new Exception("transformDataFrameToSql, clickHouseSource.tableName is empty");
            }
            sql += "\nFROM ";
            String dataBase = "";
            if (clickHouseSource.getDatabase() != null && !clickHouseSource.getDatabase().isEmpty()) {
                dataBase = clickHouseSource.getDatabase();
            }
            String tableName = clickHouseSource.getTableName();
            if (dataBase.equals("Nested")) {
                sql += " ( " + tableName + " ) ";
            } else {
                if (dataBase.isEmpty()) {
                    sql += tableName;
                } else {
                    sql += dataBase + "." + tableName;
                }
            }
        } else if (source.getType() == AisDataframe.SourceType.StarRocks) {
            AisDataframe.StarRocksSource starRocksSource = source.getStarrocks();
            if (starRocksSource.getTableName().isEmpty()) {
                throw new Exception("transformDataFrameToSql, starRocksSource.tableName is empty");
            }
            sql += "\nFROM ";
            String dataBase = starRocksSource.getDatabase();
            String tableName = starRocksSource.getTableName();
            if (dataBase.equals("Nested")) {
                sql += " ( " + tableName + " ) as df_tmp_tbl_" + tableName.length();
            } else {
                if (dataBase.isEmpty()) {
                    sql += tableName;
                } else {
                    sql += dataBase + "." + tableName;
                }
            }
        } else {
            throw new Exception("transformDataFrameToSql, source.type is unknown");
        }

        // fill where
        List<String> filters = dataframe.getFiltersList();
        if (filters.size() > 0) {
            sql += "\nWHERE ";
            for (String filter : filters) {
                sql += '(' + filter + ") AND ";
            }
            sql = sql.substring(0, sql.length() - 5);
        }

        // fill group by
        List<AisDataframe.Column> groupBy = dataframe.getGroupByList();
        if (groupBy.size() > 0) {
            sql += "\nGROUP BY ";
            for (AisDataframe.Column column : groupBy) {
                sql += column.getName() + ", ";
            }
            sql = sql.substring(0, sql.length() - 2);
        }

        // fill order by
        List<AisDataframe.Order> orderBy = dataframe.getOrderByList();
        if (orderBy.size() > 0) {
            sql += "\nORDER BY ";
            for (AisDataframe.Order order : orderBy) {
                sql += order.getColumn().getName();
                if (order.getDesc()) {
                    sql += " DESC, ";
                } else {
                    sql += " ASC, ";
                }
            }
            sql = sql.substring(0, sql.length() - 2);
        }

        // fill limit
        AisDataframe.Limit limit = dataframe.getLimit();

        // if limit is 0, then limit 200
        if (limit.getLimit() <= 0) {
            limit = AisDataframe.Limit.newBuilder().setLimit(200).build();
        }

        if (limit.getLimit() > 0) {
            sql += "\nLIMIT " + (limit.getLimit() > MAX_LIMIT ? MAX_LIMIT : limit.getLimit());
            if (limit.getOffset() > 0) {
                sql += " OFFSET " + limit.getOffset();
            }
        }

        logger.info("transformDataFrameToSql: {}", sql);
        return sql;
    }
}
