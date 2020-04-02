/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * License); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*
 * Copyright (c) 2017, Open AI Lab
 * Author: haitao@openailab.com
 */
#ifndef __GRAPH_EXECUTOR_HPP__
#define __GRAPH_EXECUTOR_HPP__

#include "static_graph.hpp"
#include "graph.hpp"
#include "exec_engine.hpp"
#include "exec_attr.hpp"
#include "attr_io.hpp"

namespace TEngine {

class GraphExecutor;

using get_graph_attr_func_t = std::function<bool(GraphExecutor*, const char* name, void* val, int size)>;
using set_graph_attr_func_t = std::function<bool(GraphExecutor*, const char* name, const void* val, int size)>;

class GraphExecutor
{
public:
    GraphExecutor()
    {
        graph_ = nullptr;
        graph_attached_ = false;
        exec_handle_ = nullptr;
        prerun_done_ = false;
        optimize_only = 0;

        InitAttrIO();
    }

    ~GraphExecutor()
    {
        if(graph_ && !graph_attached_)
            ReleaseGraph();
        if(exec_handle_)
            ReleaseExecHandle();
    }

    bool CreateGraph(void* context, const char* graph_name, const char* model_name);

    bool AttachGraph(void* context, Graph* graph_);

    Graph* GetGraph(void)
    {
        return graph_;
    }
    Graph* GetOptimizedGraph(void);

    const std::string& GetGraphName(void)
    {
        return graph_->GetName();
    }
    const std::string& GetModelName(void)
    {
        return model_name_;
    }

    bool SetGraphInputNode(const std::vector<std::string>& node_name);
    bool SetGraphOutputNode(const std::vector<std::string>& node_name);

    int GetGraphInputNodeNum(void);
    const std::string& GetGraphInputNodeName(int idx);
    int GetNodeInputNum(const std::string& node_name);
    const std::string& GetNodeInputTensor(const std::string& node_name, int idx);

    int GetGraphOutputNodeNum(void);
    const std::string& GetGraphOutputNodeName(int idx);
    int GetNodeOutputNum(const std::string& node_name);
    const std::string& GetNodeOutputTensor(const std::string& node_name, int idx);

    Tensor* FindTensor(const std::string& name);
    Node* FindNode(const std::string& name);

    bool SetTensorBuffer(Tensor* tensor, void* buffer, int buffer_size);
    void* GetTensorBuffer(Tensor* tensor);

    bool SetTensorData(Tensor* tensor, const void* input_data, int data_size);
    bool GetTensorData(Tensor* tensor, void* output_data, int data_size);

    Tensor* GetInputNodeTensor(unsigned int node_idx, unsigned int tensor_idx);
    Tensor* GetOutputNodeTensor(unsigned int node_idx, unsigned int tensor_idx);

    int GetExecStatus(void);
    bool SetEventHook(int event, event_handler_t cb_func, void* cb_arg);

    bool InferShape(void);

    bool Prerun(void);

    bool Run(int block);
    bool SyncRun(void);

    int WaitGraph(int try_wait);

    bool Postrun(void);

    ExecAttr* GetExecAttr(void)
    {
        return &exec_attr_;
    }

    int SetGraphAttr(const char* name, const void* val, int size)
    {
        if(attr_io_.SetAttr(name, val, size))
            return 0;
        else
            return -1;
    }

    int GetGraphAttr(const char* name, void* val, int size)
    {
        if(attr_io_.GetAttr(name, val, size))
            return 0;
        else
            return -1;
    }

    bool GetOptimizeOnly(const char* name, void* val, int size);
    bool SetOptimizeOnly(const char* name, const void* val, int size);

    bool GetExecAttrEntry(const char* name, void* val, int size);
    bool SetExecAttrEntry(const char* name, const void* val, int size);

    bool BailoutSetAttr(const char* name, const void* val, int size);
    bool BailoutGetAttr(const char* name, void* val, int size);

    void InitAttrIO(void);

    bool PrerunDone(void)
    {
        return prerun_done_;
    }

protected:
    void ReleaseGraph(void);
    void ReleaseExecHandle(void);
    bool PrepareExec(void* context, Graph* graph, StaticGraph* static_graph);
    bool SetExecParam(Graph* graph);

private:
    std::string model_name_;

    Graph* graph_;
    bool graph_attached_;

    ExecAttr exec_attr_;

    ExecEnginePtr exec_engine_;
    exec_handle_t exec_handle_;
    exec_event_t exec_event_;

    AttrIO attr_io_;
    bool prerun_done_;
    int optimize_only;
};

}    // namespace TEngine

#endif
