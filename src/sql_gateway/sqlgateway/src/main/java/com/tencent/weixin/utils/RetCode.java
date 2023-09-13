package com.tencent.weixin.utils;

// 状态码
public enum RetCode {
    // 操作成功
    API_YES(0, "success"),
    OK(200, "ok"),
    CREATED_SUCCESS(201, "created 用户新建或修改数据成功"),
    NO_CONTENT_SUCCESS(204, "no content 用户删除数据成功"),
    // 客户端错误
    PARAMETER_ERROR(400, "bad request 参数解释异常"),
    AUTHORIZED_ERROR(401, "unauthorized 未认证错误"),
    FORBIDDEN_ERROR(403, "forbidden 访问是被禁止"),
    NOT_FOUND_ERROR(404, "not found 资源不存在"),
    // 服务器错误
    SYSTEM_ERROR(500, "internal server error 服务器系统异常"),
    SERVICE_ERROR(503, "service unavailable 服务端当前无法处理请求");


    private int status;
    private String message;

    RetCode(int status, String message) {
        this.status = status;
        this.message = message;
    }

    public int getStatus() {
        return status;
    }

    public String getMessage() {
        return message;
    }
}
