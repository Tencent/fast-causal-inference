package com.tencent.weixin.pipeline;

import com.tencent.weixin.proto.AisDataframe;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class PipeLineNodeSource extends PipeLineNode<AisDataframe.DataFrame> {

    private Logger logger = LoggerFactory.getLogger(this.getClass());

    public PipeLineNodeSource() {
        super();
        status = PipeLineNodeStatus.Ready;
    }

    @Override
    public String getName() {
        return "Source";
    }

    @Override
    public void workImpl() throws Exception {
        logger.info("Source workImpl");
    }
}
