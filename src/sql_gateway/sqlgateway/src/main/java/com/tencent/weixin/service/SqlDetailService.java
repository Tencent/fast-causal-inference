package com.tencent.weixin.service;

import com.github.pagehelper.PageHelper;
import com.github.pagehelper.PageInfo;
import com.tencent.weixin.dao.mysql.SqlDetailMapper;
import com.tencent.weixin.model.SqlDetailModel;
import com.tencent.weixin.utils.PageResult;
import com.tencent.weixin.utils.PageUtils;
import com.tencent.weixin.utils.SqlRetCode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.sql.Timestamp;
import java.util.List;

@Service
public class SqlDetailService {

    private Logger logger = LoggerFactory.getLogger(this.getClass());

    @Autowired
    SqlDetailMapper sqlDetailMapper;

    @Value("${spring.datasource.is-open}")
    private Boolean isDatasourceOpen;

    public PageResult select(Integer pageNum, Integer pageSize) {
        PageHelper.startPage(pageNum, pageSize);
        List<SqlDetailModel> exptDetailTasks = sqlDetailMapper.select();
        return PageUtils.getPageResult(new PageInfo<>(exptDetailTasks));
    }
    
    public SqlDetailModel selectById(int id) {
        return sqlDetailMapper.selectById(id);
    }

    public SqlDetailModel selectByCreator(String creator) {
        return sqlDetailMapper.selectByCreator(creator);
    }

    public SqlDetailModel selectByCode(int retcode) {
        return sqlDetailMapper.selectByCode(retcode);
    }

    public int insert(SqlDetailModel sqlDetailModel) {
        if (!isDatasourceOpen) {
            return 0;
        } else {
            sqlDetailModel.setRetcode(SqlRetCode.INIT.getCode());
            sqlDetailModel.setCreateTime(new Timestamp(System.currentTimeMillis()));
            sqlDetailModel.setUpdateTime(new Timestamp(System.currentTimeMillis()));
            int retCode = sqlDetailMapper.insert(sqlDetailModel);
            if (retCode > 0) {
                return retCode;
            } else {
                logger.error(sqlDetailModel.toString());
                throw new RuntimeException("insert error");
            } 
        }
    }

    public int delete(int id) {
        int retCode = sqlDetailMapper.delete(id);
        if (retCode > 0) {
            return retCode;
        } else {
            logger.error("error id = "+ id);
            throw new RuntimeException("delete error");
        }
    }

    public SqlDetailModel update(SqlDetailModel sqlDetailModel) {
        if (!isDatasourceOpen) {
            return sqlDetailModel;
        } else {
            sqlDetailModel.setUpdateTime(new Timestamp(System.currentTimeMillis()));
            int retCode = sqlDetailMapper.update(sqlDetailModel);
            if (retCode > 0) {
                return sqlDetailModel;
            } else {
                throw new RuntimeException("update error");
            }
        }
    }
}