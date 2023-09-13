package com.tencent.weixin.utils;

public enum SqlRetCode {
    RUNNING(0),
    SUCCESS(1),
    FAIL(2),
    INIT(-1);
    private int code;
    
    SqlRetCode(int code) {
        this.code = code;
    }

    public int getCode() {
        return code;
    }
}