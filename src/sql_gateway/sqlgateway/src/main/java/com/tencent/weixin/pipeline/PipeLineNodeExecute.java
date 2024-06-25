package com.tencent.weixin.pipeline;

import com.alibaba.fastjson.JSON;
import com.tencent.weixin.model.SqlUdfModel;
import com.tencent.weixin.proto.AisDataframe;
import com.tencent.weixin.service.OlapExecuteService;
import com.tencent.weixin.service.SqlUdfService;
import com.tencent.weixin.utils.DataFrameUtil;
import com.tencent.weixin.utils.olap.EngineType;
import org.apache.calcite.sql.olap.SqlForward;
import org.apache.calcite.sql.parser.SqlParseException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class PipeLineNodeExecute extends PipeLineNode<AisDataframe.DataFrame> {
    private final OlapExecuteService olapExecuteService;

    private final SqlUdfService sqlUdfService;

    private final Logger logger = LoggerFactory.getLogger(this.getClass());

    static private Integer MAX_LIMIT = 1000000;

    public PipeLineNodeExecute(String dataBase, int deviceId, OlapExecuteService olapExecuteService,
                               SqlUdfService sqlUdfService, EngineType engineType) {
        this.dataBase = dataBase;
        this.deviceId = deviceId;
        this.olapExecuteService = olapExecuteService;
        this.sqlUdfService = sqlUdfService;
        this.engineType = engineType;
    }

    @Override
    public String getName() {
        return "Select";
    }

    @Override
    public void workImpl() throws Exception {
        logger.info("Select workImpl, dataBase: " + dataBase + ", deviceId: " + deviceId + " data: " + data);
        try {
            String executeSql = DataFrameUtil.transformDataFrameToSql(data);
            logger.info("Select workImpl, sql: " + executeSql);
            long calciteSqlCostTime = 0;

            for (SqlUdfModel udfName : sqlUdfService.select()) {
                if (executeSql.toUpperCase().contains(udfName.getUdf().toUpperCase() + "(")) {
                    if (udfName.getIsDisable()) {
                        throw new Exception("udf " + udfName.getUdf() + " is disable");
                    }
                    logger.info("hit all in sql udf :" + udfName.getUdf());
                    long startTime = System.currentTimeMillis();
                    try {
                        logger.info("sql parse rawSql :" + executeSql);
                        SqlForward sqlForward = new SqlForward(executeSql, engineType.toCalciteEngineType());
                        executeSql = sqlForward.getForwardSql();
                        calciteSqlCostTime = System.currentTimeMillis() - startTime;
                        logger.info("sql parse executeSql :" + executeSql + ", calcite sql cost time :" +
                                calciteSqlCostTime + " ms");
                    } catch (SqlParseException e) {
                        logger.info("change status fail");
                        e.printStackTrace();
                        throw new Exception("sql解析异常, 请检查sql:\n" + executeSql + "\nerrmsg:\n" + e.getMessage());
                    }
                    break;
                }
            }
            logger.info("Select workImpl, forward sql: " + executeSql);

            String retExecuteSql = executeSql;
            if (engineType.equals(EngineType.Clickhouse)) {
                executeSql += " settings max_result_rows = " + MAX_LIMIT +
                        ", result_overflow_mode = 'break', max_block_size=2000, distributed_ddl_task_timeout = 1800, max_execution_time = 1800, max_parser_depth = 3000";
            }

            JSON result = olapExecuteService.execute(deviceId, dataBase, executeSql, null, true, engineType);
            logger.info("Query Olap Finish, Result: " + result);
            AisDataframe.DataFrame.Builder dataBuilder = data.toBuilder();
            // fill data's result
            dataBuilder.setResult(result.toJSONString());
            dataBuilder.setExecuteSql(retExecuteSql);
            data = dataBuilder.build();
        } catch (Exception e) {
            logger.error("Select workImpl error: " + e);
            throw e;
        }
    }

    private final String dataBase;
    private final int deviceId;
    private final EngineType engineType;
}
