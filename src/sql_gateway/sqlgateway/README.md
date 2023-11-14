SQLGateway server
1. raw sql -> calcite sql -> 下沉到引擎层clickhouse执行 (返回结果1000行限制)
2. 审计
3. 系统保护、高并发：熔断(过载保护，防雪崩 熔断器-hystrix)、限流(同时处在运行中的sql不超过20个,超过的sql在等待队列中排队执行)、udf拦截
4. 智能调度: 多stage拆分，及stage调度到具体引擎执行能力
5. 服务注册服务发现，负载均衡


mysql -h${host} -u${user} -p${password} ${database_name} < resources/sql/schema.sql

curl -H "Accept: application/json" -H "Content-Type: application/json" -X GET 'http://127.0.0.1:9099/api/v1/sqlgateway/sql-run'|jq
curl -H "Accept: application/json" -H "Content-Type: application/json" -X POST -d"{\"rawSql\": \"select * from test_data_small limit 5\",\"creator\": \"bearlyhuang\",\"database\": \"all_in_sql_expt\",\"deviceId\": 61}" 'http://127.0.0.1:9099/api/v1/sqlgateway/sql-run'|jq
指定运行的ip:
curl -H "Accept: application/json" -H "Content-Type: application/json" -X POST -d"{\"rawSql\": \"select * from test_data_small limit 5\",\"creator\": \"bearlyhuang\",\"database\": \"all_in_sql_expt\",\"deviceId\": 61,\"launcherIp\":\"9.146.224.77\"}" 'http://127.0.0.1:9099/api/v1/sqlgateway/sql-run'|jq
指定dataframe输出格式:
curl -H "Accept: application/json" -H "Content-Type: application/json" -X POST -d"{\"rawSql\": \"select * from test_data_small limit 5\",\"creator\": \"bearlyhuang\",\"database\": \"all_in_sql_expt\",\"deviceId\": 61,\"isDataframeOutput\":true}" 'http://127.0.0.1:9099/api/v1/sqlgateway/sql-run'|jq
curl -H "Accept: application/json" -H "Content-Type: application/json" -X POST -d"{\"rawSql\": \"SELECT  xexpt_ttest_2samp(numerator, denominator, if(groupname = 'B1','B','A'), uin, 0.05, 0.005, 0.8) FROM expt_detail_8740718_shichaohan_1693799227533 where groupname in ('A1','A2','B1');\",\"creator\": \"bearlyhuang\",\"database\": \"all_in_sql\",\"deviceId\": 556}" 'http://127.0.0.1:9099/api/v1/sqlgateway/sql-run'|jq
仅做calcite sqlparse:
curl -H "Accept: application/json" -H "Content-Type: application/json" -X POST -d"{\"rawSql\": \"SELECT  xexpt_ttest_2samp(numerator, denominator, if(groupname = 'B1','B','A'), uin, 0.05, 0.005, 0.8) FROM expt_detail_8740718_shichaohan_1693799227533 where groupname in ('A1','A2','B1');\",\"creator\": \"bearlyhuang\",\"database\": \"all_in_sql\",\"deviceId\": 556,\"isCalciteParse\":true}" 'http://127.0.0.1:9099/api/v1/sqlgateway/sql-run'|jq
nohup java -Xms10g -Xmx25g -jar sqlgateway-0.0.1-SNAPSHOT.jar --spring.config.location=./sqlgateway_application.yml &> sqlgateway.log &

