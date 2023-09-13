package com.tencent.weixin.dao.mysql;

import com.tencent.weixin.model.SqlDetailModel;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface SqlDetailMapper {

    List<SqlDetailModel> select();

    SqlDetailModel selectById(int id);

    SqlDetailModel selectByCreator(String creator);

    SqlDetailModel selectByCode(int retcode);

    int insert(SqlDetailModel exptDetailTask);

    int delete(int id);

    int update(SqlDetailModel exptDetailTask);
}
