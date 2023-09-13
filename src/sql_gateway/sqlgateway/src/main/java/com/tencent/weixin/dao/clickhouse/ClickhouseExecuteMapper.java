package com.tencent.weixin.dao.clickhouse;

import com.tencent.weixin.model.ClickhouseExecuteModel;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface ClickhouseExecuteMapper {
    List<ClickhouseExecuteModel> select();
}