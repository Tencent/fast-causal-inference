package com.tencent.weixin.pipeline;

import org.springframework.stereotype.Component;

import java.util.ArrayList;

enum PipeLineNodeStatus {
    Wait,
    Ready,
    Working,
    WorkingFinish,
    TransformFinish,
    Async,
    Error,
}

// pipeline 节点的抽象
@Component
public abstract class PipeLineNode<T> {
    public abstract String getName();

    public void work() throws Exception {
        status = PipeLineNodeStatus.Working;
        if (data == null) {
            status = PipeLineNodeStatus.Error;
            throw new Exception("data is null");
        }
        try {
            workImpl();
        } catch (Exception e) {
            status = PipeLineNodeStatus.Error;
            throw e;
        }
        status = PipeLineNodeStatus.WorkingFinish;
    }
    public abstract void workImpl() throws Exception;

    public void addOutput(PipeLineNode<T> node) {
        output.add(node);
        node.preNodeCount++;
    }

    // 默认方式强行覆盖 data， 适用于单输入单输出的情况
    // 如果是并行的 PipeLineNode，需要重写这个方法， 对 output 中的 data 进行数据归并
    public void updateData(T data) {
        this.data = data;
    }

    public void updateDataToOutput(T data) {
        for (PipeLineNode<T> node : output) {
            node.updateData(data);
            node.preNodeCount--;
            if (node.preNodeCount == 0) {
                node.status = PipeLineNodeStatus.Ready;
            }
        }
    }

    public void transfromData() {
        updateDataToOutput(data);
        status = PipeLineNodeStatus.TransformFinish;
    }

    public T getData() {
        return data;
    }

    public void setData(T data) {
        this.data = data;
    }

    public String toString() {
        String str = "PipeLineNode: " + getName() + " status: " + status + "\n";
        str += "preNodeCount: " + preNodeCount + "\n";
        str += "output: " + output + "\n";
        str += "data: " + data + "\n";
        return str;
    }

    public T data;
    public int preNodeCount = 0;
    public ArrayList<PipeLineNode<T>> output = new ArrayList<>();
    public PipeLineNodeStatus status = PipeLineNodeStatus.Wait;
}
