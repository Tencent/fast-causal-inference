package com.tencent.weixin.controller;

import com.alibaba.fastjson.JSON;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.google.protobuf.InvalidProtocolBufferException;
import com.google.protobuf.util.JsonFormat;
import com.tencent.weixin.example.DocExample;
import com.tencent.weixin.proto.AisDataframe;
import com.tencent.weixin.service.*;
import com.tencent.weixin.utils.ResponseData;
import com.tencent.weixin.utils.RetCode;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.responses.ApiResponses;
import io.swagger.v3.oas.annotations.tags.Tag;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;

import javax.validation.Valid;

@Tag(name = "dataframe", description = "")
@RestController
// consumes: 消费消息，和Content-Type对应， 指定处理请求时的提交内容类型
// produces: 生产消息，和Accept对应， 指定返回的内容类型，仅当request header中Accept类型包含该指定类型时才返回
@RequestMapping(path = "/api/v1/sqlgateway/dataframe-run", produces = MediaType.APPLICATION_JSON_VALUE)

public class DataFrameController {
    private Logger logger = LoggerFactory.getLogger(this.getClass());

    @Autowired
    private DataFrameService dataFrameService;

    private static final String MODEL_DESC = "dataframe-run desc";

    @Operation(summary = "dataframe", description = "用于sql转发, " + MODEL_DESC)
    @ApiResponses(value = {
            @ApiResponse(responseCode = "201", description = "转发后的sql内容",
                    content = @Content(schema = @Schema(type = "string", example = DocExample.CONTROLLER_POST_EXAMPLE_200)))
    })

    @PostMapping(value = "/protobuf/json", consumes = MediaType.APPLICATION_JSON_VALUE)
    public String post(@RequestBody @Valid String protobufJson) {
        logger.info("protobufJson: " + protobufJson);
        AisDataframe.DataFrameResponse resp = dataFrameService.dataFrameExecuteJson(protobufJson);
        JsonFormat.Printer printer = JsonFormat.printer();
        try {
            String resp_str = printer.print(resp);
            return resp_str;
        } catch (InvalidProtocolBufferException e) {
            e.printStackTrace();
            return "{" +
                    "\"status\": \"FAIL\"," +
                    "\"msg\": \"protobuf parse error\"" +
                    "}";
        }
    }
}
