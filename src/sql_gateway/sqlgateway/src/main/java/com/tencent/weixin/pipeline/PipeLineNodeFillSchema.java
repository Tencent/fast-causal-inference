package com.tencent.weixin.pipeline;

import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;
import com.tencent.weixin.proto.AisDataframe;
import com.tencent.weixin.service.OlapExecuteService;
import com.tencent.weixin.utils.olap.EngineType;
import com.tencent.weixin.utils.olap.OlapUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class PipeLineNodeFillSchema extends PipeLineNode<AisDataframe.DataFrame> {

    private final OlapExecuteService olapExecuteService;

    private final Logger logger = LoggerFactory.getLogger(this.getClass());

    public PipeLineNodeFillSchema(String dataBase, int deviceId, OlapExecuteService olapExecuteService,
                                  EngineType engineType, String rtx, String user, String password) {
        this.dataBase = dataBase;
        this.deviceId = deviceId;
        this.olapExecuteService = olapExecuteService;
        this.engineType = engineType;
        this.rtx = rtx;
        this.user = user;
        this.password = password;
    }

    public PipeLineNodeFillSchema(String dataBase, int deviceId, OlapExecuteService olapExecuteService,
                                  EngineType engineType, String rtx) {
        this(dataBase, deviceId, olapExecuteService, engineType, rtx, null, null);
    }

    @Override
    public String getName() {
        return "FillSchema";
    }

    @Override
    public void workImpl() throws Exception {
        logger.info("FillSchema workImpl, dataBase: " + dataBase + ", deviceId: " + deviceId + " data: " + data);
        try {
            if (!(engineType == EngineType.Clickhouse || engineType == EngineType.Starrocks)) {
                throw new Exception("FillSchema workImpl, data source type is not supported.");
            }
            String table_name;

            if (engineType == EngineType.Clickhouse) {
                table_name = data.getSource().getClickhouse().getTableName();
            } else {
                table_name = data.getSource().getStarrocks().getTableName();
            }
            if (table_name.isEmpty()) {
                throw new Exception("FillSchema workImpl, data source clickhouse table name is empty");
            }

            logger.info("FillSchema workImpl, data source clickhouse table name: " + table_name);
            JSON result = olapExecuteService.execute(deviceId, dataBase, "desc " + table_name, null, true, engineType, this.rtx, this.user, this.password, true, 600);
            logger.info("Query ClickHouse Finish, Result: " + result);
            AisDataframe.DataFrame.Builder dataBuilder = data.toBuilder();

            JSONArray jsonArray = JSON.parseArray(result.toJSONString());
            for (int i = 0; i < jsonArray.size(); i++) {
                JSONObject jsonObject = jsonArray.getJSONObject(i);
                AisDataframe.Column column = OlapUtil.toDataFrameColumnType(engineType, jsonObject);
                dataBuilder.addColumns(column);
            }
            data = dataBuilder.build();
        } catch (Exception e) {
            logger.error("FillSchema workImpl, error: " + e.getMessage());
            throw e;
        }
    }

    private final String dataBase;
    private final int deviceId;
    private final EngineType engineType;
    private final String rtx;
    private final String user;
    private final String password;
}
