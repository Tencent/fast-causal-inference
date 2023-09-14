CREATE TABLE if not exists mmexptdataplatform.`sql_detail`
(
    `id`                             int(11) NOT NULL AUTO_INCREMENT COMMENT 'sql主键id',
    `device_id`                      int(11) NOT NULL COMMENT '数据源id',
    `database`                       varchar(64) NOT NULL COMMENT '数据库名称',
    `raw_sql`                        varchar(512) NOT NULL COMMENT '原始sql',
    `execute_sql`                    varchar(512) NOT NULL COMMENT '可执行sql',
    `calcite_sql_cost_time`          int(11) DEFAULT NULL COMMENT 'calcite sql转换花费时间,单位秒',
    `calcite_sql_cost_time_readable` varchar(56)           DEFAULT NULL COMMENT '可读calcite_sql_cost_time',
    `execute_sql_cost_time`          int(11) DEFAULT NULL COMMENT '执行花费时间,单位秒',
    `execute_sql_cost_time_readable` varchar(56)           DEFAULT NULL COMMENT '可读execute_sql_cost_time',
    `total_time`                     int(11) DEFAULT NULL COMMENT '总时间,单位秒',
    `total_time_readable`            varchar(56)           DEFAULT NULL COMMENT '可读total_time',
    `creator`                        varchar(128) NOT NULL COMMENT '用户rtx',
    `retcode`                        int(8) NOT NULL default -1 comment "状态码 0运行中 1成功 2失败",
    `retmsg`                         varchar(512)          DEFAULT NULL comment "返回信息摘要",
    `create_time`                    timestamp    NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '记录创建时间',
    `update_time`                    timestamp    NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '记录最后更新时间',
    PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=19 DEFAULT CHARSET=utf8 COMMENT='sql detail表';

CREATE TABLE if not exists mmexptdataplatform.`sql_udf_dict`
(
    `id`         int(11) NOT NULL AUTO_INCREMENT COMMENT '主键id',
    `udf`        varchar(512) NOT NULL COMMENT 'udf标识',
    `udf_desc`   varchar(512) NOT NULL COMMENT 'udf描述',
    `is_disable` tinyint(1) default 0 comment "是否禁用 0不禁用 1禁用",
    PRIMARY KEY (`id`),
    UNIQUE KEY `unique_key` (`udf`)
) ENGINE=InnoDB AUTO_INCREMENT=19 DEFAULT CHARSET=utf8 COMMENT='sql udf字典表';

CREATE TABLE if not exists mmexptdataplatform.`sql_detail_udf_map`
(
    `id`     int(11) NOT NULL AUTO_INCREMENT COMMENT '主键id',
    `udf_id` int(11) NOT NULL COMMENT 'udf id',
    `sql_id` int(11) NOT NULL COMMENT 'sql id',
    PRIMARY KEY (`id`),
    UNIQUE KEY `unique_key` (`udf_id`,`sql_id`)
) ENGINE=InnoDB AUTO_INCREMENT=19 DEFAULT CHARSET=utf8 COMMENT='sql命中udf关系表';


insert into mmexptdataplatform.sql_udf_dict(udf,udf_desc) values('deltamethod','deltamethod');
insert into mmexptdataplatform.sql_udf_dict(udf,udf_desc) values('predict','predict');
insert into mmexptdataplatform.sql_udf_dict(udf,udf_desc) values('olsState','olsState');
insert into mmexptdataplatform.sql_udf_dict(udf,udf_desc) values('ols','ols');
insert into mmexptdataplatform.sql_udf_dict(udf,udf_desc) values('ivregression','ivregression');
insert into mmexptdataplatform.sql_udf_dict(udf,udf_desc) values('wls','wls');
insert into mmexptdataplatform.sql_udf_dict(udf,udf_desc) values('xexpt_ttest_2samp','xexpt_ttest_2samp');
insert into mmexptdataplatform.sql_udf_dict(udf,udf_desc) values('ttest_1samp','ttest_1samp');
insert into mmexptdataplatform.sql_udf_dict(udf,udf_desc) values('ttest_2samp','ttest_2samp');
insert into mmexptdataplatform.sql_udf_dict(udf,udf_desc) values('did','did');
insert into mmexptdataplatform.sql_udf_dict(udf,udf_desc) values('lift','lift');
insert into mmexptdataplatform.sql_udf_dict(udf,udf_desc) values('linearDML','linearDML');
insert into mmexptdataplatform.sql_udf_dict(udf,udf_desc) values('nonParamDML','nonParamDML');
insert into mmexptdataplatform.sql_udf_dict(udf,udf_desc) values('cutbins','cutbins');
insert into mmexptdataplatform.sql_udf_dict(udf,udf_desc) values('caliperMatching','caliperMatching');
insert into mmexptdataplatform.sql_udf_dict(udf,udf_desc) values('exactMatching','exactMatching');
