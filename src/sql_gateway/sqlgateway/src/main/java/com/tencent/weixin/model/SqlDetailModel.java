package com.tencent.weixin.model;

import lombok.*;

import javax.validation.constraints.NotBlank;
import java.sql.Timestamp;

@Getter
@Setter
@AllArgsConstructor
@RequiredArgsConstructor
@NoArgsConstructor
@ToString
public class SqlDetailModel {
    private Integer id;
    @NonNull
    private Integer deviceId;
    @NonNull
    private String database;
    @NonNull
    @NotBlank(message = "rawSql string can not be null or empty")  // 作用于字符串类型
    private String rawSql;
    @NonNull
    private String executeSql;
    private Integer calciteSqlCostTime;
    private String calciteSqlCostTimeReadable;
    private Integer executeSqlCostTime;
    private String executeSqlCostTimeReadable;
    private Integer totalTime;
    private String totalTimeReadable;
    @NonNull
    private String creator;
    private Integer retcode;
    private Timestamp createTime;
    private Timestamp updateTime;
}