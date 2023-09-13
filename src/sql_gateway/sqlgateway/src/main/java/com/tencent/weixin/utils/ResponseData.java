package com.tencent.weixin.utils;


import lombok.*;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
public class ResponseData<T> {

    private int status;
    private String message;
    private T data;
    private String timestamp;

    private void parserRetCode(RetCode retCode) {
        this.status = retCode.getStatus();
        this.message = retCode.getMessage();
    }

    public static <T> ResponseData<T> success(T data) {
        ResponseData<T> responseData = new ResponseData<T>();
        responseData.parserRetCode(RetCode.OK);
        responseData.setData(data);
        responseData.setTimestamp(DateUtil.getNow());
        return responseData;
    }

    public static <T> ResponseData<T> success(RetCode retCode, T data) {
        ResponseData<T> responseData = new ResponseData<T>();
        responseData.parserRetCode(retCode);
        responseData.setData(data);
        responseData.setTimestamp(DateUtil.getNow());
        return responseData;
    }

    public static <T> ResponseData<T> error(RetCode retCode) {
        ResponseData<T> responseData = new ResponseData<T>();
        responseData.parserRetCode(retCode);
        responseData.setTimestamp(DateUtil.getNow());
        return responseData;
    }

    public static <T> ResponseData<T> error(int status, String message) {
        ResponseData<T> responseData = new ResponseData<T>();
        responseData.setStatus(status);
        responseData.setMessage(message);
        responseData.setTimestamp(DateUtil.getNow());
        return responseData;
    }
}