package com.tencent.weixin.pipeline;

import com.tencent.weixin.proto.AisDataframe;

public class PipeLineNodeSink extends PipeLineNode<AisDataframe.DataFrame> {
    @Override
    public String getName() {
        return "Sink";
    }

    @Override
    public void workImpl() throws Exception {
        // do nothing
    }
}
