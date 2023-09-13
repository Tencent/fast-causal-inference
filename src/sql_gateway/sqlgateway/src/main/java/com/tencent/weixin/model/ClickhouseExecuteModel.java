package com.tencent.weixin.model;

import lombok.*;

@Getter
@Setter
@AllArgsConstructor
@RequiredArgsConstructor
@NoArgsConstructor
@ToString
public class ClickhouseExecuteModel {
    @NonNull
    private Integer uin;
    private Integer ds;
}

