package com.tencent.weixin.service;

import com.tencent.weixin.dao.mysql.SqlUdfMapper;
import com.tencent.weixin.model.SqlDetailModel;
import com.tencent.weixin.model.SqlUdfModel;
import com.tencent.weixin.utils.SqlRetCode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.sql.Timestamp;
import java.util.Arrays;
import java.util.List;

@Service
public class SqlUdfService {

    private Logger logger = LoggerFactory.getLogger(this.getClass());

    @Autowired
    SqlUdfMapper sqlUdfMapper;

    @Value("${spring.datasource.is-open}")
    private Boolean isDatasourceOpen;

    private static final List<SqlUdfModel> sqlUdfDict = Arrays.asList(
            new SqlUdfModel(1, "deltamethod", "deltamethod", false),
            new SqlUdfModel(16, "predict", "predict", false),
            new SqlUdfModel(2, "olsState", "olsState", false),
            new SqlUdfModel(3, "ols", "ols", false),
            new SqlUdfModel(4, "ivregression", "ivregression", false),
            new SqlUdfModel(5, "wls", "wls", false),
            new SqlUdfModel(6, "xexpt_ttest_2samp", "xexpt_ttest_2samp", false),
            new SqlUdfModel(7, "ttest_1samp", "ttest_1samp", false),
            new SqlUdfModel(8, "ttest_2samp", "ttest_2samp", false),
            new SqlUdfModel(9, "did", "did", false),
            new SqlUdfModel(10, "lift", "lift", false),
            new SqlUdfModel(11, "linearDML", "linearDML", false),
            new SqlUdfModel(12, "nonParamDML", "nonParamDML", false),
            new SqlUdfModel(13, "cutbins", "cutbins", false),
            new SqlUdfModel(14, "caliperMatching", "caliperMatching", false),
            new SqlUdfModel(15, "exactMatching", "exactMatching", false));

    public List<SqlUdfModel> select() {
        if (!isDatasourceOpen) {
            return sqlUdfDict;
        } else {
            List<SqlUdfModel> exptDetailTasks = sqlUdfMapper.select();
            return exptDetailTasks;
        }
    }

    public int insert(Integer udfId, Integer sqlId) {
        if (!isDatasourceOpen) {
            return 0;
        } else {
            int retCode = sqlUdfMapper.insert(udfId, sqlId);
            if (retCode > 0) {
                return retCode;
            } else {
                throw new RuntimeException("insert error");
            }
        }
    }
}