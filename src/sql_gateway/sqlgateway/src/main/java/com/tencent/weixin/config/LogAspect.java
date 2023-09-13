package com.tencent.weixin.config;

import com.google.gson.Gson;
import org.aspectj.lang.JoinPoint;
import org.aspectj.lang.ProceedingJoinPoint;
import org.aspectj.lang.annotation.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;
import org.springframework.web.context.request.RequestContextHolder;
import org.springframework.web.context.request.ServletRequestAttributes;

import javax.servlet.http.HttpServletRequest;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.List;

@Aspect
@Component
public class LogAspect {

    private final static Logger logger = LoggerFactory.getLogger(LogAspect.class);

    @Pointcut("execution(public * com.tencent.weixin.controller..*.*(..))")
    public void webLog() {
    }

    @Before("webLog()")
    public void doBefore(JoinPoint joinPoint) {
        ServletRequestAttributes attributes = (ServletRequestAttributes) RequestContextHolder.getRequestAttributes();
        HttpServletRequest request = attributes.getRequest();
        logger.info("URL            : {}", request.getRequestURL().toString());
        logger.info("HTTP Method    : {}", request.getMethod());
        logger.info("Class Method   : {}.{}", joinPoint.getSignature().getDeclaringTypeName(), joinPoint.getSignature().getName());
        logger.info("IP             : {}", request.getRemoteAddr());
        Enumeration<String> headerNames = request.getHeaderNames();
        logger.info("Headers        :");
        List<String> validHeader = Arrays.asList("host", "user-agent", "referer", "staffid", "staffname", "x-client-ip-port", "x-real-ip");
        while (headerNames.hasMoreElements()) {
            String name = headerNames.nextElement();
            if (validHeader.contains(name)) {
                logger.info("                 {}", name + ":" + request.getHeader(name) + ";");
            }
        }
//        Cookie[] cookies = request.getCookies();
//        List<String> validCookie= Arrays.asList("t_uid");
//        logger.info("Cookies        : ");
//        if(cookies != null) {
//            for (Cookie cookie : cookies) {
//                if (validCookie.contains(cookie.getName())) {
//                    logger.info("                 {}", cookie.getName() +  ":"  + cookie.getValue() +  ";");
//                }
//            }
//        }
        logger.info("Request Args   : {}", new Gson().toJson(joinPoint.getArgs()));
    }

    @After("webLog()")
    public void doAfter() {
    }

    @Around("webLog()")
    public Object doAround(ProceedingJoinPoint proceedingJoinPoint) throws Throwable {
        long startTime = System.currentTimeMillis();
        logger.info("========================================== Start ==========================================");
        Object result = proceedingJoinPoint.proceed();
        logger.info("Response Args  : {}", new Gson().toJson(result));
        logger.info("Time-Consuming : {} ms", System.currentTimeMillis() - startTime);
        logger.info("=========================================== End ===========================================");
        return result;
    }

}

