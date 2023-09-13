package com.tencent.weixin.config;

import io.swagger.v3.oas.models.ExternalDocumentation;
import io.swagger.v3.oas.models.OpenAPI;
import io.swagger.v3.oas.models.info.Info;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class SpringDocConfig {
    @Bean
    public OpenAPI exptOpenAPI(){
        return new OpenAPI()
                .info(new Info().title("X实验平台 RESTful API文档")
                        .version("v1.0.0"))
                .externalDocs(new ExternalDocumentation()
                        .description("X实验平台主页")
                        .url(""));
    }
}
