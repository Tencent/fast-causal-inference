create database if not exists all_in_sql;
drop table if exists all_in_sql.test_data_small;
CREATE TABLE if not exists all_in_sql.test_data_small
(
    `id` varchar(36),
    `x1` double,
    `x2` double,
    `x3` double,
    `x4` double,
    `x5` double,
    `x_long_tail1` double,
    `x_long_tail2` double,
    `x_cat1` varchar(4),
    `treatment` bigint,
    `t_ob` bigint,
    `y` double,
    `y_ob` double,
    `numerator_pre` double,
    `numerator` double,
    `denominator_pre` bigint,
    `denominator` bigint,
    `weight` double,
    `day_` Date
)
ENGINE=olap
PROPERTIES
(
    "replication_num"="1"
);
