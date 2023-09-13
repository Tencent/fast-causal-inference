package com.tencent.weixin.dao.mysql;

import com.tencent.weixin.model.SqlDetailModel;
import com.tencent.weixin.model.SqlUdfModel;
import org.apache.ibatis.annotations.Param;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface SqlUdfMapper {

    List<SqlUdfModel> select();

    int insert(@Param("udfId") Integer udfId, @Param("sqlId") Integer sqlId);
}
