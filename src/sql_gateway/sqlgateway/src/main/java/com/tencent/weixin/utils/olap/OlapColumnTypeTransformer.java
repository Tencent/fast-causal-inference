package com.tencent.weixin.utils.olap;

public abstract class OlapColumnTypeTransformer {
    public abstract String toDataFrameColumnType(String columnType);
}
