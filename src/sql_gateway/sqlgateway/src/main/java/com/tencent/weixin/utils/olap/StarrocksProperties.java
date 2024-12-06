package com.tencent.weixin.utils.olap;

import lombok.Getter;
import lombok.Setter;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

import java.util.List;

@Component
@Getter
@Setter
@ConfigurationProperties(prefix = "olap.starrocks")
public class StarrocksProperties implements OlapProperties {
    private String driver;
    private List<Device> devices;
}
