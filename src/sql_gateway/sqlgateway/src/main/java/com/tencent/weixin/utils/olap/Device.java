package com.tencent.weixin.utils.olap;

import lombok.Getter;
import lombok.Setter;

import java.util.List;

@Getter
@Setter
public class Device {
    private int id;
    private String url;
    private String user;
    private String password;
    private List<String> ip;
}
