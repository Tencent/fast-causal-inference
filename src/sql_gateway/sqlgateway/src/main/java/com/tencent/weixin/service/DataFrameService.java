package com.tencent.weixin.service;

import com.google.protobuf.util.JsonFormat;
import com.tencent.weixin.dao.mysql.SqlDetailMapper;
import com.tencent.weixin.model.SqlDetailModel;
import com.tencent.weixin.pipeline.*;
import com.tencent.weixin.proto.AisDataframe;
import com.tencent.weixin.utils.olap.EngineType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class DataFrameService {
    private final Logger logger = LoggerFactory.getLogger(this.getClass());

    @Autowired
    private OlapExecuteService olapExecuteService;

    @Autowired
    private SqlUdfService sqlUdfService;

    @Autowired
    private SqlDetailMapper sqlDetailMapper;

    public AisDataframe.DataFrameRequest parseDataFrameRequestFromBase64String(String base64String) {
        try {
            byte[] binaryData = java.util.Base64.getDecoder().decode(base64String);
            AisDataframe.DataFrameRequest request = AisDataframe.DataFrameRequest.parseFrom(binaryData);
            logger.info("parseDataFrameRequest: {}", request);
            return request;
        } catch (Exception e) {
            logger.error("parseDataFrameRequest error: {}", e);
            return null;
        }
    }

    public AisDataframe.DataFrameResponse dataFrameExecuteBase64(String base64String) {
        AisDataframe.DataFrameRequest request = parseDataFrameRequestFromBase64String(base64String);
        if (request == null) {
            return null;
        }
        return dataFrameExecuteImpl(request);
    }

    public AisDataframe.DataFrameResponse dataFrameExecuteJson(String json) {
        System.out.println("dataFrameExecuteJson: " + json);
        try {
            AisDataframe.DataFrameRequest.Builder builder = AisDataframe.DataFrameRequest.newBuilder();
            JsonFormat.Parser parser = JsonFormat.parser();
            parser.merge(json, builder);
            AisDataframe.DataFrameRequest request = builder.build();
            if (request == null) {
                return retError("dataFrameExecuteJson null: {}");
            }
            return dataFrameExecuteImpl(request);
        } catch (Exception e) {
            logger.error("dataFrameExecuteJson error: {}", e);
            return retError("dataFrameExecuteJson error, errmsg: " + e);
        }
    }

    public AisDataframe.DataFrameResponse retError(String msg, AisDataframe.DataFrameResponse.Builder df_resp_builder) {
        df_resp_builder.setStatus(AisDataframe.RetStatus.FAIL);
        df_resp_builder.setMsg(msg);
        return df_resp_builder.build();
    }

    public AisDataframe.DataFrameResponse retError(String msg) {
        AisDataframe.DataFrameResponse.Builder df_resp_builder = AisDataframe.DataFrameResponse.newBuilder();
        return retError(msg, df_resp_builder);
    }

    public AisDataframe.DataFrameResponse dataFrameExecuteImpl(AisDataframe.DataFrameRequest df_req) {
        logger.info("dataFrameExecuteImpl: {}", df_req);
        SqlDetailModel sqlDetailModel = new SqlDetailModel();
        sqlDetailModel.setDeviceId(df_req.getDeviceId());
        sqlDetailModel.setDatabase(df_req.getDatabase());
        sqlDetailModel.setCreator(df_req.getRtx());
        long startTime = System.currentTimeMillis();

        System.out.println(df_req);
        AisDataframe.DataFrameResponse.Builder df_resp_builder = AisDataframe.DataFrameResponse.newBuilder();
        df_resp_builder.setStatus(AisDataframe.RetStatus.SUCC);

        AisDataframe.SourceType sourceType = df_req.getDf().getSource().getType();
        EngineType engineType = EngineType.Unknown;
        if (sourceType == AisDataframe.SourceType.ClickHouse) {
            engineType = EngineType.Clickhouse;
        } else if (sourceType == AisDataframe.SourceType.StarRocks) {
            engineType = EngineType.Starrocks;
        }

        try {
            PipeLineNode<AisDataframe.DataFrame> pipeLineNodeSource = new PipeLineNodeSource();
            pipeLineNodeSource.setData(df_req.getDf());
            PipeLineNode<AisDataframe.DataFrame> pipeLineNodeSink = new PipeLineNodeSink();

            if (df_req.getTaskType() == AisDataframe.TaskType.FILL_SCHEMA) {
                // if not database and deviceId, error
                if (df_req.getDatabase() == null || df_req.getDatabase().isEmpty() || df_req.getDeviceId() == 0) {
                    return retError("database or deviceId is empty", df_resp_builder);
                }
                PipeLineNode<AisDataframe.DataFrame> pipeLineNodeFillSchema =
                        new PipeLineNodeFillSchema(df_req.getDatabase(), df_req.getDeviceId(),
                                olapExecuteService, engineType);
                pipeLineNodeSource.addOutput(pipeLineNodeFillSchema);
                pipeLineNodeFillSchema.addOutput(pipeLineNodeSink);
            } else if (df_req.getTaskType() == AisDataframe.TaskType.EXECUTE) {
                PipeLineNode<AisDataframe.DataFrame> pipeLineNodeExecute =
                        new PipeLineNodeExecute(df_req.getDatabase(), df_req.getDeviceId(), olapExecuteService,
                                sqlUdfService, engineType);
                pipeLineNodeSource.addOutput(pipeLineNodeExecute);
                pipeLineNodeExecute.addOutput(pipeLineNodeSink);
            }
            logger.info("create pipeline finish, pipeLineNodeSource: {}", pipeLineNodeSource);
            logger.info("pipeline executor start work");

            PipeLineExecutor<AisDataframe.DataFrame> pipeLineExecutor = new PipeLineExecutor<>(pipeLineNodeSource);
            pipeLineExecutor.work();
            AisDataframe.DataFrame df = pipeLineExecutor.getData();
            System.out.println("result dataframe: " + df);
            if (df == null) {
                return retError("result dataframe is empty", df_resp_builder);
            }
            df_resp_builder.setDf(df);

            long totalTime = System.currentTimeMillis() - startTime;
            sqlDetailModel.setTotalTime((int) totalTime);
            sqlDetailModel.setTotalTimeReadable(totalTime/1000 + "s");
            sqlDetailModel.setExecuteSql(df.getExecuteSql());
            sqlDetailModel.setRetcode(5);
            sqlDetailModel.setRawSql(df.getResult().substring(0, Math.min(df.getResult().length(), 256)));
            return df_resp_builder.build();
        } catch (Exception e) {
            logger.error("dataFrameExecuteImpl error: {}", e);
            return retError(e.getMessage(), df_resp_builder);
        }
    }

}
