package com.tencent.weixin.utils.olap;

public enum EngineType {
    Unknown,
    Clickhouse,
    Starrocks;

    public org.apache.calcite.sql.olap.EngineType toCalciteEngineType() {
        if (this.equals(Clickhouse)) {
            return org.apache.calcite.sql.olap.EngineType.ClickHouse;
        }
        if (this.equals(Starrocks)) {
            return org.apache.calcite.sql.olap.EngineType.StarRocks;
        }
        throw new IllegalArgumentException(String.format("%s is an invalid engine type.", this));
    }
}
