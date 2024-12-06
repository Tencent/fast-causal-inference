package com.tencent.weixin.utils;

import java.net.InetAddress;
import java.net.UnknownHostException;
import java.sql.Timestamp;
import java.time.LocalDateTime;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import com.tencent.weixin.dao.mysql.SqlAuditMapper;
import com.tencent.weixin.model.SqlAuditModel;

@Component
public class AuditUtil {
    @Autowired
    private SqlAuditMapper sqlAuditMapper;

    Logger logger = LoggerFactory.getLogger(AuditUtil.class);

    public void audit(String rtx, String taskType, String engine, String database, String sql, String parsedSql,
            String sqlBrief, Integer durationMs, String status, String result, String extraTaskInfo) {
        String ipAddress = "";
        try {
            InetAddress localhost = InetAddress.getLocalHost();
            ipAddress = localhost.getHostAddress();
        } catch (UnknownHostException e) {
            e.printStackTrace();
        }
        try {
            sql = sql.length() > 1024 ? sql.substring(0, 1024) : sql;
            parsedSql = parsedSql.length() > 1024 ? parsedSql.substring(0, 1024) : parsedSql;
            sqlBrief = sqlBrief.length() > 1024 ? sqlBrief.substring(0, 1024) : sqlBrief;
            result = result.length() > 1024 ? result.substring(0, 1024) : result;
            extraTaskInfo = extraTaskInfo.length() > 1024 ? extraTaskInfo.substring(0, 1024) : extraTaskInfo;
            engine = engine == null ? "clickhouse" : engine;
            SqlAuditModel sqlAuditModel = new SqlAuditModel(rtx, taskType, engine, database, sql, parsedSql, sqlBrief,
                    Timestamp.valueOf(LocalDateTime.now()), ipAddress, durationMs, status, result, extraTaskInfo);
            sqlAuditMapper.insert(sqlAuditModel);
            logger.info("audit succ.");
        } catch (Exception e) {
            logger.error("audit error: {}", e);
        }
    }
}
