package com.tencent.weixin.controller;

import com.alibaba.fastjson.JSON;
import com.google.common.base.Strings;
import com.tencent.weixin.example.DocExample;
import com.tencent.weixin.model.SqlDetailModel;
import com.tencent.weixin.model.SqlUdfModel;
import com.tencent.weixin.service.ClickhouseExecuteService;
import com.tencent.weixin.service.SqlDetailService;
import com.tencent.weixin.service.SqlUdfService;
import com.tencent.weixin.utils.*;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.responses.ApiResponses;
import io.swagger.v3.oas.annotations.tags.Tag;
import org.apache.calcite.sql.olap.SqlForward;
import org.apache.calcite.sql.parser.SqlParseException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;

import javax.validation.Valid;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.io.Writer;
import java.sql.Timestamp;
import java.util.*;


@Tag(name = "sql_run", description = "sql记录管理")  //springdoc api主题
@RestController
// consumes: 消费消息，和Content-Type对应， 指定处理请求时的提交内容类型
// produces: 生产消息，和Accept对应， 指定返回的内容类型，仅当request header中Accept类型包含该指定类型时才返回
@RequestMapping(path = "/api/v1/sqlgateway/sql-run", produces = MediaType.APPLICATION_JSON_VALUE)
public class SqlRunController {

    private Logger logger = LoggerFactory.getLogger(this.getClass());

    @Autowired
    private SqlDetailService sqlDetailService;

    @Autowired
    private SqlUdfService sqlUdfService;

    @Autowired
    private ClickhouseExecuteService clickhouseExecuteService;

    private static final String MODEL_DESC = "sql-run desc";

    @Operation(summary = "查询sql明细列表", description = "获取相应条件的sql明细列表")  //springdoc api方法主题
    @ApiResponses(value = {
            @ApiResponse(responseCode = "200", description = "成功获取sql明细对象List",
                    content = @Content(schema = @Schema(type = "string", example = DocExample.CONTROLLER_EXAMPLE_GET_200))),
            @ApiResponse(responseCode = "400", description = "参数解释异常",
                    content = @Content(schema = @Schema(type = "string", example = DocExample.CONTROLLER_EXAMPLE_GET_400))),
            @ApiResponse(responseCode = "500", description = "服务器系统异常",
                    content = @Content(schema = @Schema(example = DocExample.CONTROLLER_EXAMPLE_GET_500)))
    })     //springdoc api方法返回
    @GetMapping
    public ResponseData get(@RequestParam(name = "page", required = false, defaultValue = "0") @Parameter(description = "页数") Integer page,     //Parameter springdoc api方法参数
                            @RequestParam(name = "page_size", required = false, defaultValue = "100") @Parameter(description = "每页大小") Integer pageSize,
                            @RequestParam(name = "id", required = false) @Parameter(description = "主键id", example = "19") Integer id,
                            @RequestParam(name = "retcode", required = false) @Parameter(description = "状态码 0运行中 1成功 2失败", example = "1") Integer retcode,
                            @RequestParam(name = "creator", required = false) @Parameter(description = "用户rtx", example = "bearlyhuang") String creator) {
        if (id == null && retcode == null && creator == null) {
            PageResult exptDetailTasks = sqlDetailService.select(page, pageSize);
            return ResponseData.success(exptDetailTasks.getContent());
        } else if (id != null) {
            SqlDetailModel sqlDetailModel = sqlDetailService.selectById(id);
            return ResponseData.success(sqlDetailModel);
        } else if (retcode != null) {
            SqlDetailModel sqlDetailModel = sqlDetailService.selectByCode(retcode);
            return ResponseData.success(sqlDetailModel);
        } else if (creator != null) {
            SqlDetailModel sqlDetailModel = sqlDetailService.selectByCreator(creator);
            return ResponseData.success(sqlDetailModel);
        } else {
            return ResponseData.error(RetCode.PARAMETER_ERROR);
        }
    }

    @Operation(summary = "查询单个sql明细记录", description = "根据传入主键id获取相应sql明细记录")
    @ApiResponses(value = {@ApiResponse(responseCode = "200", description = "sql明细对象")})
    @GetMapping(path = "/{resource-id}")
    public ResponseData get(@PathVariable("resource-id") @Parameter(description = "id主键") int id) {
        SqlDetailModel sqlDetailModel = sqlDetailService.selectById(id);
        if (sqlDetailModel == null) {
            return ResponseData.error(400, "获取失败, 没有对应的主键id可获取");
        } else {
            return ResponseData.success(sqlDetailModel);
        }
    }

    @Operation(summary = "sql转发任务", description = "用于sql转发, " + MODEL_DESC)
    @ApiResponses(value = {
            @ApiResponse(responseCode = "201", description = "转发后的sql内容",
                    content = @Content(schema = @Schema(type = "string", example = DocExample.CONTROLLER_POST_EXAMPLE_200)))
    })
    @PostMapping(consumes = MediaType.APPLICATION_JSON_VALUE)
    public ResponseData post(@RequestBody @Valid SqlDetailModel sqlDetailModel) {
        try {
            String rtx = sqlDetailModel.getCreator();
            String rawSql = sqlDetailModel.getRawSql();
            if (Strings.isNullOrEmpty(rtx)) {
                return ResponseData.error(400, "操作失败, 用户传参rtx为空");
            } else if (Strings.isNullOrEmpty(rawSql)) {
                return ResponseData.error(400, "操作失败, 用户传参sql为空");
            } else {
                // 初始化
                long totalStartTime = System.currentTimeMillis();
                String executeSql = rawSql.replaceAll(";", "");
                sqlDetailModel.setExecuteSql(executeSql);
                sqlDetailModel.setRetcode(SqlRetCode.INIT.getCode());
                sqlDetailService.insert(sqlDetailModel);

                if (sqlDetailModel.getIsCalciteParse() == null || ! sqlDetailModel.getIsCalciteParse()) {
                    executeSql = clickhouseExecuteService.sqlPretreatment(executeSql);
                }
                //sql rebuild后, 改为执行状态
                long calciteSqlCostTime = 0;
                for (SqlUdfModel udfName : sqlUdfService.select()) {
                    if (executeSql.toUpperCase().contains(udfName.getUdf().toUpperCase() + "(")) {
                        if (udfName.getIsDisable()) {
                            sqlDetailModel.setRetcode(SqlRetCode.FAIL.getCode());
                            sqlDetailService.update(sqlDetailModel);
                            return ResponseData.error(500, udfName.getUdf() + " 已经被管理员禁用");
                        }
                        logger.info("hit all in sql udf :" + udfName.getUdf());
                        long startTime = System.currentTimeMillis();
                        sqlUdfService.insert(udfName.getId(), sqlDetailModel.getId());
                        try {
                            logger.info("sql parse rawSql :" + executeSql);
                            SqlForward sqlForward = new SqlForward(executeSql);
                            executeSql = sqlForward.getForwardSql();
                            calciteSqlCostTime = System.currentTimeMillis() - startTime;
                            logger.info("sql parse executeSql :" + executeSql + ", calcite sql cost time :" + calciteSqlCostTime + " ms");
                        } catch (SqlParseException e) {
                            logger.info("change status fail");
                            e.printStackTrace();
                            sqlDetailModel.setRetcode(SqlRetCode.FAIL.getCode());
                            sqlDetailService.update(sqlDetailModel);
                            return ResponseData.error(500, "sql解析异常, 请检查sql, " + e.getMessage());
                        }
//                        if (!executeSql.toUpperCase().contains("WITH ")) {
//                            
//                        } else {
//                            logger.info("break sql parse");
//                        }
                        break;
                    }
                }
                sqlDetailModel.setCalciteSqlCostTime((int) calciteSqlCostTime);
                sqlDetailModel.setCalciteSqlCostTimeReadable((calciteSqlCostTime / 1000 % 60) + " sec");
                if (sqlDetailModel.getIsCalciteParse() != null && sqlDetailModel.getIsCalciteParse()) {
                    sqlDetailModel.setRetcode(SqlRetCode.SUCCESS.getCode());
                    long totalCostTime = System.currentTimeMillis() - totalStartTime;
                    sqlDetailModel.setTotalTime((int) totalCostTime);
                    sqlDetailModel.setTotalTimeReadable((totalCostTime / 1000 / 60) + " min," + (totalCostTime / 1000 % 60) + " sec");
                    sqlDetailService.update(sqlDetailModel);
                    String finalRawSql = rawSql;
                    String finalExecuteSql = executeSql;
                    return ResponseData.success(RetCode.API_YES, new HashMap<String, Object>() {{
                        put("rawSql", finalRawSql);
                        put("executeSql", finalExecuteSql);
                        put("result", null);
                    }});
                } else {
                    logger.info("change status running");
                    sqlDetailModel.setExecuteSql(executeSql);
                    sqlDetailModel.setRetcode(SqlRetCode.RUNNING.getCode());
                    sqlDetailService.update(sqlDetailModel);
                    //execute后, 改为最终状态
                    try {
                        long startTime = System.currentTimeMillis();
                        JSON retJson = clickhouseExecuteService.execute(sqlDetailModel.getDeviceId(), sqlDetailModel.getDatabase(), executeSql, sqlDetailModel.getLauncherIp(), sqlDetailModel.getIsDataframeOutput());
                        long executeSqlCostTime = System.currentTimeMillis() - startTime;
                        logger.info("rawSql :" + rawSql + ", executeSql :" + executeSql);
                        JSON finalRetJson = retJson;
                        sqlDetailModel.setRetcode(SqlRetCode.SUCCESS.getCode());
                        sqlDetailModel.setExecuteSqlCostTime((int) executeSqlCostTime);
                        sqlDetailModel.setExecuteSqlCostTimeReadable((executeSqlCostTime / 1000 / 60) + " min," + (executeSqlCostTime / 1000 % 60) + " sec");
                        long totalCostTime = System.currentTimeMillis() - totalStartTime;
                        sqlDetailModel.setTotalTime((int) totalCostTime);
                        sqlDetailModel.setTotalTimeReadable((totalCostTime / 1000 / 60) + " min," + (totalCostTime / 1000 % 60) + " sec");
                        sqlDetailService.update(sqlDetailModel);
                        logger.info("change status success");
                        String finalRawSql = rawSql;
                        String finalExecuteSql = executeSql;
                        return ResponseData.success(RetCode.API_YES, new HashMap<String, Object>() {{
                            put("rawSql", finalRawSql);
                            put("executeSql", finalExecuteSql);
                            put("result", finalRetJson);
                        }});
                    } catch (Exception e) {
                        logger.info("change status fail");
                        e.printStackTrace();
                        sqlDetailModel.setRetcode(SqlRetCode.FAIL.getCode());
                        sqlDetailService.update(sqlDetailModel);
                        return ResponseData.error(500, "sql执行异常, 请检查sql, " + e.getMessage());
                    }
                }
            }
        } catch (Exception e) {
            ResponseData responseData = null;
            try {
                if (sqlDetailModel.getId() != null && sqlDetailModel.getId() != 0) {
                    sqlDetailModel.setRetcode(SqlRetCode.FAIL.getCode());
                    sqlDetailService.update(sqlDetailModel);
                }
                final Writer result = new StringWriter();
                final PrintWriter printWriter = new PrintWriter(result);
                e.printStackTrace(printWriter);
                logger.info(result.toString());
                responseData = ResponseData.error(500, result.toString());
                result.close();
                printWriter.close();
            } catch (IOException e1) {
            }
            return responseData;
        }
    }

    @Operation(summary = "更新sql记录", description = "用于更新sql记录,Request body中所有参数都为必须参数, 局部更新建议使用patch方法;" + MODEL_DESC)
    @ApiResponses(value = {
            @ApiResponse(responseCode = "201", description = "已更新成功的sql对象内容",
                    content = @Content(schema = @Schema(type = "string", example = DocExample.CONTROLLER_POST_EXAMPLE_200)))
    })
    @PutMapping(path = "/{resource-id}", consumes = MediaType.APPLICATION_JSON_VALUE)
    public ResponseData put(@RequestBody @Valid SqlDetailModel sqlDetailModel, @PathVariable(name = "resource-id") @Parameter(description = "id主键") Integer id) {
        SqlDetailModel exitSqlDetailModel = sqlDetailService.selectById(id);
        if (exitSqlDetailModel == null) {
            return ResponseData.error(400, "操作失败, 该任务主键id不存在, 请检查待更新任务");
        }
        try {
            sqlDetailModel.setId(id);
            sqlDetailService.update(sqlDetailModel);
            return ResponseData.success(RetCode.CREATED_SUCCESS, sqlDetailModel);
        } catch (Exception e) {
            ResponseData responseData = null;
            try {
                final Writer result = new StringWriter();
                final PrintWriter printWriter = new PrintWriter(result);
                e.printStackTrace(printWriter);
                logger.info(result.toString());
                responseData = ResponseData.error(500, result.toString());
                result.close();
                printWriter.close();
            } catch (IOException e1) {
            }
            return responseData;
        }
    }

    @Operation(summary = "局部更新sql记录", description = "用于局部更新sql记录,Request body中id必须参数, 其余都为可选参数")
    @ApiResponses(value = {
            @ApiResponse(responseCode = "201", description = "已局部更新成功的sql记录",
                    content = @Content(schema = @Schema(type = "string", example = DocExample.CONTROLLER_POST_EXAMPLE_200)))
    })
    @PatchMapping(path = "/{resource-id}", consumes = MediaType.APPLICATION_JSON_VALUE)
    public ResponseData patch(@RequestBody @Valid SqlDetailModel sqlDetailModel, @PathVariable(name = "resource-id") @Parameter(description = "id主键") Integer id) {
        SqlDetailModel exitSqlDetailModel = sqlDetailService.selectById(id);
        if (exitSqlDetailModel == null) {
            return ResponseData.error(400, "操作失败, 该任务主键id不存在, 请检查待更新任务");
        }
        logger.info(exitSqlDetailModel.toString());
        try {
            sqlDetailModel.setId(id);
            if (sqlDetailModel.getCreator() == null) {
                sqlDetailModel.setCreator(exitSqlDetailModel.getCreator());
            }
            if (sqlDetailModel.getCreateTime() == null) {
                sqlDetailModel.setCreateTime(new Timestamp(System.currentTimeMillis()));
            }
            sqlDetailService.update(sqlDetailModel);
            return ResponseData.success(RetCode.CREATED_SUCCESS, sqlDetailModel);
        } catch (Exception e) {
            ResponseData responseData = null;
            try {
                final Writer result = new StringWriter();
                final PrintWriter printWriter = new PrintWriter(result);
                e.printStackTrace(printWriter);
                logger.info(result.toString());
                responseData = ResponseData.error(500, result.toString());
                result.close();
                printWriter.close();
            } catch (IOException e1) {
            }
            return responseData;
        }
    }

    @Operation(summary = "删除sql记录", description = "用于删除sql记录")
    @ApiResponses(value = {
            @ApiResponse(responseCode = "204", description = "已删除成功的空标识",
                    content = @Content(schema = @Schema(type = "string", example = DocExample.CONTROLLER_POST_EXAMPLE_200)))
    })
    @DeleteMapping(path = "/{resource-id}", consumes = MediaType.APPLICATION_JSON_VALUE)
    public ResponseData delete(@PathVariable(name = "resource-id") @Parameter(description = "id主键") Integer id) {
        SqlDetailModel exitSqlDetailModel = sqlDetailService.selectById(id);
        if (exitSqlDetailModel == null) {
            return ResponseData.error(RetCode.NOT_FOUND_ERROR);
        }
        sqlDetailService.delete(id);
        return ResponseData.success(RetCode.NO_CONTENT_SUCCESS, null);
    }

}
