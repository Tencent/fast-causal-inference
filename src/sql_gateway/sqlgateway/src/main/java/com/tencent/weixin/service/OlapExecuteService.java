package com.tencent.weixin.service;

import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONObject;
import com.tencent.weixin.model.SqlUdfModel;
import com.tencent.weixin.utils.UdfFormatUtil;
import com.tencent.weixin.utils.olap.AllOlapProperties;
import com.tencent.weixin.utils.olap.Device;
import com.tencent.weixin.utils.olap.EngineType;
import com.tencent.weixin.utils.olap.OlapProperties;
import com.tencent.weixin.utils.olap.OlapUtil;
import org.apache.calcite.sql.olap.SqlForward;
import org.apache.calcite.sql.parser.SqlParseException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

@Service
public class OlapExecuteService {
    private Logger logger = LoggerFactory.getLogger(this.getClass());

    static private Integer MAX_LIMIT = 1000000;

    @Autowired
    private OlapUtil olapUtil;

    @Autowired
    private AllOlapProperties allOlapProperties;

    @Autowired
    private SqlUdfService sqlUdfService;

    public String sqlPretreatment(String executeSql) {
        logger.info("sql pretreatment raw executeSql:" + executeSql);
        if (executeSql.toUpperCase().contains("DROP") && !executeSql.toUpperCase().contains("DELETE")) {
            throw new RuntimeException("sql not can exists drop/delete");
        }
        logger.info("sql pretreatment executeSql:" + executeSql);
        return executeSql;
    }

    public JSON execute(Integer deviceId, String database, String executeSql, String launcherIp,
                        Boolean isDataFrameOutput, EngineType engineType, String rtx, boolean isQuery, int maxQueryTime) throws Exception {
        return execute(deviceId, database, executeSql, launcherIp, isDataFrameOutput, engineType, rtx, null, null, isQuery, maxQueryTime);
    }

    public JSON execute(Integer deviceId, String database, String executeSql, String launcherIp,
                        Boolean isDataFrameOutput, EngineType engineType, String rtx, String user, String password, boolean isQuery, int maxQueryTime) throws Exception {
        logger.info(
                "deviceId: " + deviceId + ", database: " + database + ", executeSql: " + executeSql + ", launcherIp: " +
                        launcherIp + ", isDataFrameOutput: " + isDataFrameOutput);
        executeSql = executeSql.trim();
        if (executeSql.toUpperCase().startsWith("SELECT") && engineType.equals(EngineType.Clickhouse)) {
            executeSql += " settings max_result_rows = " + MAX_LIMIT +
                    ", result_overflow_mode = 'break', max_block_size=2000, distributed_ddl_task_timeout = 1800, max_execution_time = 1800, max_parser_depth = 6000, max_expanded_ast_elements = 15000000, max_ast_elements=15000000";
        }
        if (engineType.equals(EngineType.Starrocks)) {
            executeSql = "set query_timeout = " + maxQueryTime + ";" + executeSql;
        }
        
        Connection connection = null;
        ResultSet rst = null;
        PreparedStatement pst = null;
        long startTime = System.currentTimeMillis();
        int retry = 3;
        while (retry-- > 0) {
            try {
                connection = olapUtil.getConnection(database, deviceId, launcherIp, engineType, user, password);
                connection.setAutoCommit(false);

                if (isQuery) {
                    Statement statement = connection.createStatement();
                    String[] sqls = executeSql.split(";");
                    int end = sqls.length - 1;
                    while (end >= 0 && sqls[end].trim().isEmpty()) {
                        end--;
                    }
                    for (int i = 0; i < end; i++) {
                        String sql = sqls[i];
                        if (sql.trim().isEmpty()) {
                            continue;
                        }
                        statement.execute(sql);
                    }
                    rst = statement.executeQuery(sqls[end]);
                } else {
                    Statement statement = connection.createStatement();
                    String[] sqlStatements = executeSql.split(";");
                    // 遍历SQL语句列表并添加到批处理中
                    for (String sql : sqlStatements) {
                        if (sql.trim().isEmpty()) {
                            continue;
                        }
                        logger.info("execute sql: " + sql);
                        statement.execute(sql);
                    }
                    connection.commit();
                    statement.close();
                    connection.close();
                    return new JSONObject();
                }
                //logger.info("execute sql cost: " + (System.currentTimeMillis() - startTime) + " ms");
                if (isDataFrameOutput != null && isDataFrameOutput) {
                    return olapUtil.resultSetToJsonDataFrame(rst);
                } else {
                    return olapUtil.resultSetToJson(rst);
                }
            } catch (Exception e) {
                if (e.getMessage().contains("Access denied for user")) {
                    throw new RuntimeException("用户" + user +"没有" + database +"数据库的权限, 请联系cooperxiong开通.");
                }
                StringWriter sw = new StringWriter();
                PrintWriter pw = new PrintWriter(sw);
                e.printStackTrace(pw);
                logger.info(sw.toString());
                if (retry == 0 || !sw.toString().contains("NetException")) {
                    throw new RuntimeException("Olap执行sql失败，sql:\n" + executeSql + "\nerrmsg: \n" + sw);
                }
            } finally {
                logger.info("sql: " + executeSql + ", execute cost: " + (System.currentTimeMillis() - startTime) + " ms");
                olapUtil.closeAll(rst, pst, connection);
            }

            try {
                Thread.sleep(2000);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }

        }
        return null;
    }

    public JSON executeWithSqlForward(Integer deviceId, String database, String executeSql, String launcherIp,
                                      Boolean isDataFrameOutput, EngineType engineType, String rtx, boolean isQuery) throws Exception {

        long calciteSqlCostTime = 0;

        for (SqlUdfModel udfName : sqlUdfService.select()) {
            if (executeSql.toUpperCase().contains(udfName.getUdf().toUpperCase())) {
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
                    logger.info(
                            "sql parse executeSql :" + executeSql + ", calcite sql cost time :" + calciteSqlCostTime +
                                    " ms");
                } catch (SqlParseException e) {
                    e.printStackTrace();
                    throw new Exception("sql解析异常, 请检查sql:\n" + executeSql + "\nerrmsg:\n" + e.getMessage());
                }
                break;
            }
        }
        logger.info("Select workImpl, forward sql: " + executeSql);
        String retExecuteSql = executeSql;

        JSON result = execute(deviceId, database, executeSql, null, true, engineType, rtx, isQuery, 300);
        result = UdfFormatUtil.formatUdfResult(result);
        logger.info("Query Olap Finish, Result: " + result);
        return result;
    }

    public void executeParallel(Integer deviceId, String database, String executeSql, Boolean isDataFrameOutput,
                                EngineType engineType, String rtx) {
        //logger.info("executeParallel deviceId: " + deviceId + ", database: " + database + ", executeSql: " + executeSql + ", isDataFrameOutput: " + isDataFrameOutput);
        OlapProperties olapProperties = allOlapProperties.getOlapProperties(engineType);
        List<String> ipLists = null;
        for (Device deviceInfo : olapProperties.getDevices()) {
            if (deviceInfo.getId() == deviceId) {
                ipLists = deviceInfo.getIp();
                break;
            }
        }
        int ipSize = ipLists.size();
        ExecutorService executorService = Executors.newFixedThreadPool(ipSize);
        ArrayList<Future<?>> futures = new ArrayList<>();
        for (String ip : ipLists) {
            futures.add(executorService.submit(() -> {
                try {
                    int retry = 3;
                    while (retry-- > 0) {
                        try {
                            int start = (int) System.currentTimeMillis();
                            JSON res = execute(deviceId, database, executeSql, ip, isDataFrameOutput, engineType, rtx, true, 300);
                            int end = (int) System.currentTimeMillis();
                            //logger.info("executeParallel res: " + res + ", ip: " + ip + ", cost: " + (end - start) + " ms");
                            break;
                        } catch (Exception e) {
                            logger.info(e.toString());
                            if (retry == 0) {
                                throw new RuntimeException("Olap执行sql失败，sql:\n" + executeSql + "\nerrmsg: \n" + e);
                            }
                        }
                    }
                    int end = (int) System.currentTimeMillis();
                } catch (Exception e) {
                    logger.info(e.toString());
                    throw new RuntimeException("Olap执行sql失败，sql:\n" + executeSql + "\nerrmsg: \n" + e);
                }
            }));
        }
        for (Future<?> future : futures) {
            try {
                future.get();
            } catch (Exception e) {
                logger.info(e.toString());
                throw new RuntimeException("Olap执行sql失败，sql:\n" + executeSql + "\nerrmsg: \n" + e);
            }
        }
        executorService.shutdown();
    }
}