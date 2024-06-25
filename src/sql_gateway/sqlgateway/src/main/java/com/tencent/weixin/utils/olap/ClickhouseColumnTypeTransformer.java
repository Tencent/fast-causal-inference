package com.tencent.weixin.utils.olap;

public class ClickhouseColumnTypeTransformer extends OlapColumnTypeTransformer {
    @Override
    public String toDataFrameColumnType(String columnType) {
        return columnType;
    }
}
