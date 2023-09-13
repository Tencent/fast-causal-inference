package com.tencent.weixin.model;

import lombok.*;

import javax.validation.constraints.NotBlank;
import java.sql.Timestamp;

@Getter
@Setter
@AllArgsConstructor
@RequiredArgsConstructor
@NoArgsConstructor
@ToString
public class SqlUdfModel {
    private Integer id;
    @NonNull
    @NotBlank(message = "udf string can not be null or empty")  // 作用于字符串类型
    private String udf;
    @NonNull
    @NotBlank(message = "udf_desc string can not be null or empty")  // 作用于字符串类型
    private String udfDesc;
    
    private Boolean isDisable;
}

