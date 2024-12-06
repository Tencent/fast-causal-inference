package com.tencent.weixin.exception;


import com.tencent.weixin.utils.ResponseData;
import com.tencent.weixin.utils.RetCode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.validation.BindingResult;
import org.springframework.validation.FieldError;
import org.springframework.validation.ObjectError;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.MissingServletRequestParameterException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.ResponseStatus;
import org.springframework.web.bind.annotation.RestControllerAdvice;

import java.util.List;

@RestControllerAdvice("com.tencent.weixin.controller")
public class GlobalExceptionHandler {

    private Logger logger = LoggerFactory.getLogger(this.getClass());

    /**
     * 忽略参数异常处理器
     *
     * @param e 忽略参数异常
     * @return ResponseResult
     */
    @ResponseStatus(HttpStatus.BAD_REQUEST)
    @ExceptionHandler(MissingServletRequestParameterException.class)
    public ResponseData parameterMissingExceptionHandler(MissingServletRequestParameterException e) {
        logger.error("MissingServletRequestParameterException：", e);
        return ResponseData.error(RetCode.PARAMETER_ERROR.getStatus(), "请求参数 " + e.getParameterName() + " 不能为空");
    }

    /**
     * 参数效验异常处理器
     *
     * @param e 参数验证异常
     * @return ResponseInfo
     */
    @ResponseStatus(HttpStatus.BAD_REQUEST)
    @ExceptionHandler(MethodArgumentNotValidException.class)
    public ResponseData parameterExceptionHandler(MethodArgumentNotValidException e) {
        logger.error("parameterExceptionHandler：", e);
        // 获取异常信息
        BindingResult exceptions = e.getBindingResult();
        // 判断异常中是否有错误信息，如果存在就使用异常中的消息，否则使用默认消息
        if (exceptions.hasErrors()) {
            List<ObjectError> errors = exceptions.getAllErrors();
            if (!errors.isEmpty()) {
                // 这里列出了全部错误参数，按正常逻辑，只需要第一条错误即可
                FieldError fieldError = (FieldError) errors.get(0);
                return ResponseData.error(RetCode.PARAMETER_ERROR.getStatus(), fieldError.getDefaultMessage());
            }
        }
        return ResponseData.error(RetCode.PARAMETER_ERROR.getStatus(), "请求参数效验异常");
    }

}
