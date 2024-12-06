package com.tencent.weixin.model;

import lombok.*;

import java.sql.Timestamp;

@Getter
@Setter
@RequiredArgsConstructor
@NoArgsConstructor
@ToString
public class SqlAuditModel {
    @NonNull
    private String rtx; // 用户名
    @NonNull
    private String taskType; // sql 来源 (sql-run/expt-detail/dataframe)
    @NonNull
    private String engine; // olap引擎类型 (clickhouse/starrocks)
    @NonNull
    private String database; // allinsql 数据库名
    @NonNull
    private String sql; // sql
    @NonNull
    private String parsedSql; // sql类型 (select/insert/update/delete)
    @NonNull
    private String sqlBrief; // sql语句中的关键词，udf函数名等，加速搜索
    @NonNull
    private Timestamp submitTime; // 时间戳
    @NonNull
    private String serverIp; // 服务器ip
    @NonNull
    private Integer durationMs; // 耗时
    @NonNull
    private String status; // 任务状态：成功 or 失败
    @NonNull
    private String result; // 结果/报错
    @NonNull
    private String extraTaskInfo; // 额外任务信息
}
