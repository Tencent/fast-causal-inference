package com.tencent.weixin.pipeline;

import com.google.common.collect.ForwardingQueue;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.LinkedList;
import java.util.Queue;

// TODO 多线程改造
public class PipeLineExecutor<T> {

    public PipeLineExecutor(PipeLineNode<T> source) {
        readyQueue.add(source);
    }

    private Logger logger = LoggerFactory.getLogger(this.getClass());

    public void addNode(PipeLineNode<T> node) {
        readyQueue.add(node);
    }

    public void work() throws Exception {
        try {
            while (!readyQueue.isEmpty()) {
                PipeLineNode<T> node = readyQueue.poll();

                if (node.getName() == "Sink") {
                    logger.info("work finish: " + node.getName());
                    data = node.getData();
                    return;
                }

                node.work();
                node.transfromData();

                for (PipeLineNode<T> outputNode : node.output) {
                    if (outputNode.status == PipeLineNodeStatus.Ready) {
                        readyQueue.add(outputNode);
                    }
                }
            }
        }
        catch (Exception e) {
            logger.error("PipeLineExecutor work error: " + e);
            throw e;
        }
    }

    public T getData() {
        return data;
    }

    private LinkedList<PipeLineNode<T>> readyQueue = new LinkedList<>();
    T data;
}
