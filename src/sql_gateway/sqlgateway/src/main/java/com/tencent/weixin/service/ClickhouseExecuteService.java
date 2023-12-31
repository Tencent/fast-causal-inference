package com.tencent.weixin.service;

import com.alibaba.fastjson.JSON;
import com.github.pagehelper.PageHelper;
import com.github.pagehelper.PageInfo;
import com.tencent.weixin.dao.clickhouse.ClickhouseExecuteMapper;
import com.tencent.weixin.model.ClickhouseExecuteModel;
import com.tencent.weixin.utils.ClickhouseUtil;
import com.tencent.weixin.utils.PageResult;
import com.tencent.weixin.utils.PageUtils;
import org.apache.commons.lang.StringEscapeUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.Statement;
import java.util.HashSet;
import java.util.List;

@Service
public class ClickhouseExecuteService {
    private static final Integer MAX_RET_RECORD = 1000;
    private Logger logger = LoggerFactory.getLogger(this.getClass());

    @Autowired
    ClickhouseExecuteMapper clickhouseExecuteMapper;

    @Autowired
    private ClickhouseUtil clickhouseUtil;

    public PageResult select(Integer pageNum, Integer pageSize) {
        PageHelper.startPage(pageNum, pageSize);
        List<ClickhouseExecuteModel> clickhouseExecuteModels = clickhouseExecuteMapper.select();
        return PageUtils.getPageResult(new PageInfo<>(clickhouseExecuteModels));
    }
    
    public String sqlPretreatment(String executeSql) {
        logger.info("sql pretreatment raw executeSql:" + executeSql);
        if (executeSql.toUpperCase().contains("DROP") && !executeSql.toUpperCase().contains("DELETE")) {
            throw new RuntimeException("sql not can exists drop/delete");
        }
        logger.info("sql pretreatment executeSql:" + executeSql);
        return executeSql;
    }
    
    public JSON execute(Integer deviceId, String database, String executeSql, String launcherIp, Boolean isDataFrameOutput) {
        Connection connection = null;
        Statement st = null;
        ResultSet rst = null;
        long startTime = System.currentTimeMillis();
        try {
            connection = clickhouseUtil.getClickHouseConnection(database, deviceId, launcherIp);
            st = connection.createStatement();
            rst = st.executeQuery("SET max_result_rows = 1000, result_overflow_mode = 'break', distributed_ddl_task_timeout = 1800, max_execution_time = 1800; " + executeSql);
            if (isDataFrameOutput != null && isDataFrameOutput) {
                return clickhouseUtil.resultSetToJsonDataFrame(rst);
            } else {
                return clickhouseUtil.resultSetToJson(rst);
            }
        } catch (Exception e) {
            logger.info(e.getMessage());
            throw new RuntimeException(e);
        } finally {
            logger.info("sql: " + executeSql + ", execute cost: " + (System.currentTimeMillis() - startTime) + " ms");
            clickhouseUtil.closeAll(rst, st, connection);
        }
    }
}