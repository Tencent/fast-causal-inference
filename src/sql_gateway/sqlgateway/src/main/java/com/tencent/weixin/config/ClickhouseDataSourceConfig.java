package com.tencent.weixin.config;

import org.apache.ibatis.session.SqlSessionFactory;
import org.mybatis.spring.SqlSessionFactoryBean;
import org.mybatis.spring.SqlSessionTemplate;
import org.mybatis.spring.annotation.MapperScan;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.jdbc.DataSourceBuilder;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.io.support.PathMatchingResourcePatternResolver;
import org.springframework.jdbc.datasource.DataSourceTransactionManager;

import javax.sql.DataSource;

@Configuration
@MapperScan(value = "com.tencent.weixin.dao.clickhouse",sqlSessionFactoryRef = "ClickhouseSqlSessionFactory")
public class ClickhouseDataSourceConfig {
    
    @Bean(name="ClickhouseDataSource")
    @ConfigurationProperties(prefix = "spring.datasource.clickhouse")
    public DataSource getDateSource() {
        return DataSourceBuilder.create().build();
    }

    @Bean(name = "ClickhouseSqlSessionFactory")
    public SqlSessionFactory sqlSessionFactory(@Qualifier("ClickhouseDataSource") DataSource datasource) throws Exception {
        SqlSessionFactoryBean bean = new SqlSessionFactoryBean();
        bean.setDataSource(datasource);
        bean.setMapperLocations(new PathMatchingResourcePatternResolver().getResources("classpath:dao/clickhouse/*.xml"));
        bean.getObject().getConfiguration().setMapUnderscoreToCamelCase(true);
        return bean.getObject();
    }

    @Bean(name = "ClickhouseTransactionManager")
    public DataSourceTransactionManager transactionManager(@Qualifier("ClickhouseDataSource") DataSource dataSource) {
        return new DataSourceTransactionManager(dataSource);
    }

    @Bean("ClickhouseSqlSessionTemplate")
    public SqlSessionTemplate sqlSessionTemplate(@Qualifier("ClickhouseSqlSessionFactory") SqlSessionFactory sessionfactory) {
        return new SqlSessionTemplate(sessionfactory);
    }
}