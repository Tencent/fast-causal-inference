package com.tencent.weixin.utils.olap;

import lombok.Getter;

import java.util.List;

public interface OlapProperties {
    String getDriver();
    List<Device> getDevices();
}
