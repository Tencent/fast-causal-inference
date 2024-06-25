package com.tencent.weixin.utils.olap;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

@Component
public class AllOlapProperties {
    @Autowired
    private ClickhouseProperties clickhouse;
    @Autowired
    private StarrocksProperties starrocks;

    public OlapProperties getOlapProperties(EngineType engineType) {
        if (engineType == EngineType.Clickhouse) {
            return clickhouse;
        }
        if (engineType == EngineType.Starrocks) {
            return starrocks;
        }
        throw new IllegalArgumentException(String.format("%s is not a valid engine type.", engineType));
    }
}
