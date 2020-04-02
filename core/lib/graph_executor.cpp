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
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <functional>

#include "tengine_c_api.h"
#include "exec_context.hpp"
#include "graph_executor.hpp"
#include "tengine_config.hpp"
#include "tengine_errno.hpp"

namespace TEngine {

bool GraphExecutor::CreateGraph(void* exec_context, const char* graph_name, const char* model_name)
{
    StaticGraphPtr static_graph;
    Graph* graph = nullptr;

    /* create empty graph */
    if(model_name == nullptr)
    {
        graph = new Graph(graph_name);
        graph->SetModelFormat(MODEL_FORMAT_TENGINE);
        graph->SetLayout(TENGINE_LAYOUT_NCHW);
        graph->SetModelLayout(TENGINE_LAYOUT_NCHW);
    }
    else
    {
        if(!StaticGraphManager::SafeGet(model_name, static_graph))
        {
            set_tengine_errno(ENOENT);
            return false;
        }

        graph = Graph::CreateFromStatic(graph_name, static_graph);

        if(graph == nullptr)
            return false;

        model_name_ = model_name;
    }

    graph_ = graph;

    return PrepareExec(exec_context, graph_, static_graph.get());
}

bool GraphExecutor::PrepareExec(void* exec_context, Graph* graph, StaticGraph* static_graph)
{
    std::string exec_engine_name;

    TEngineConfig::Get("exec.engine", exec_engine_name);

    if(!ExecEngineManager::SafeGet(exec_engine_name, exec_engine_))
    {
        LOG_ERROR() << "No executor engine registered with name: " << exec_engine_name << "\n";
        set_tengine_errno(ENODEV);
        return false;
    }

    exec_handle_ = exec_engine_->AddGraphExecutor(this);

    if(exec_handle_ == nullptr)
    {
        set_tengine_errno(EFAULT);
        return false;
    }

    if(static_graph != nullptr)
    {
        /* set dev handle */
        exec_attr_.dev_handle = static_graph->dev_handle;

        if(exec_context != static_graph->exec_context)
        {
            XLOG_INFO() << "create runtime graph from different context model\n";
        }
    }

    exec_attr_.exec_context = exec_context;

    return true;
}

bool GraphExecutor::SetExecParam(Graph* graph)
{
#if 0
    int model_format = graph->GetModelFormat();

    /* set proper layout */
    if(model_format == MODEL_FORMAT_CAFFE || model_format == MODEL_FORMAT_ONNX ||
       model_format == MODEL_FORMAT_TENSORFLOW || model_format == MODEL_FORMAT_MXNET ||
       model_format == MODEL_FORMAT_TENGINE)
    {
        exec_attr_.graph_layout = TENGINE_LAYOUT_NCHW;

        if(model_format == MODEL_FORMAT_TENSORFLOW)
            exec_attr_.model_layout = TENGINE_LAYOUT_NHWC;
        else
            exec_attr_.model_layout = TENGINE_LAYOUT_NCHW;
    }
    else if(model_format == MODEL_FORMAT_TFLITE)
    {
        exec_attr_.graph_layout = TENGINE_LAYOUT_NHWC;
        exec_attr_.model_layout = TENGINE_LAYOUT_NHWC;
    }
    else
    {
        XLOG_INFO() << "unkown model format: " << model_format << " to set layout \n";
        return false;
    }

    exec_attr_.model_format = model_format;
#else

#endif

    exec_attr_.graph_layout = graph->GetLayout();
    exec_attr_.model_layout = graph->GetModelLayout();
    exec_attr_.model_format = graph->GetModelFormat();

    if(exec_attr_.graph_layout < 0)
    {
        LOG_ERROR() << "why graph layout is: " << exec_attr_.graph_layout << "\n";
    }

    if(exec_attr_.model_layout < 0)
    {
        LOG_ERROR() << "why model layout is: " << exec_attr_.model_layout << "\n";
    }

    if(exec_attr_.model_format < 0)
    {
        LOG_ERROR() << "why model format is: " << exec_attr_.model_format << "\n";
    }

    // check graph layout variable
    const char* layout_str = std::getenv("GRAPH_LAYOUT");
    if(layout_str)
    {
        int layout = strtoul(layout_str, NULL, 10);

        if(layout == TENGINE_LAYOUT_NCHW || layout == TENGINE_LAYOUT_NCHW)
        {
            exec_attr_.graph_layout = layout;
            LOG_INFO() << "ENV set graph layout: [" << layout << "]\n";
        }
    }

    // check kernel mode variable
    const char* mode = std::getenv("KERNEL_MODE");

    if(mode)
    {
        int kernel_mode = strtoul(mode, NULL, 10);

        LOG_INFO() << "ENV set kernel mode: [" << kernel_mode << "]\n";

        exec_attr_.kernel_mode = kernel_mode;
    }

    // check low_mem_mode env var

    const char* mem = std::getenv("LOW_MEM_MODE");

    if(mem)
    {
        if(mem[0] == '0')
            exec_attr_.low_mem_mode = false;
        else
            exec_attr_.low_mem_mode = true;
    }

    return true;
}

bool GraphExecutor::AttachGraph(void* context, Graph* graph)
{
    model_name_ = "none model";
    graph_attached_ = true;
    graph_ = graph;

    return PrepareExec(context, graph, nullptr);
}

int GraphExecutor::GetGraphInputNodeNum(void)
{
    return graph_->input_nodes.size();
}

const std::string& GraphExecutor::GetGraphInputNodeName(int idx)
{
    std::vector<Node*>& inputs = graph_->input_nodes;
    Node* node = inputs[idx];

    return node->GetName();
}

int GraphExecutor::GetNodeInputNum(const std::string& node_name)
{
    Node* node = FindNode(node_name);

    if(node == nullptr)
        return -1;

    return node->GetInputNum();
}

const std::string& GraphExecutor::GetNodeInputTensor(const std::string& node_name, int idx)
{
    Node* node = FindNode(node_name);

    const Tensor* tensor = node->GetInputTensor(idx);

    return tensor->GetName();
}

int GraphExecutor::GetGraphOutputNodeNum(void)
{
    Graph* cur_graph = GetOptimizedGraph();

    return cur_graph->output_nodes.size();
}

const std::string& GraphExecutor::GetGraphOutputNodeName(int idx)
{
    Graph* cur_graph = GetOptimizedGraph();

    std::vector<Node*>& outputs = cur_graph->output_nodes;
    Node* node = outputs[idx];

    return node->GetName();
}

int GraphExecutor::GetNodeOutputNum(const std::string& node_name)
{
    Node* node = FindNode(node_name);

    if(node == nullptr)
        return -1;

    return node->GetOutputNum();
}

const std::string& GraphExecutor::GetNodeOutputTensor(const std::string& node_name, int idx)
{
    Node* node = FindNode(node_name);

    const Tensor* tensor = node->GetOutputTensor(idx);

    return tensor->GetName();
}

bool GraphExecutor::SetGraphInputNode(const std::vector<std::string>& node_name_list)
{
    graph_->ResetInputNode();

    for(unsigned int i = 0; i < node_name_list.size(); i++)
    {
        if(!graph_->AddInputNode(node_name_list[i]))
            return false;
    }

    return true;
}

bool GraphExecutor::SetGraphOutputNode(const std::vector<std::string>& node_name_list)
{
    graph_->ResetOutputNode();

    for(unsigned int i = 0; i < node_name_list.size(); i++)
    {
        if(!graph_->AddOutputNode(node_name_list[i]))
            return false;
    }

    graph_->StripGraph();

    return true;
}

Node* GraphExecutor::FindNode(const std::string& name)
{
    Graph* cur_graph = GetOptimizedGraph();

    Node* node = cur_graph->FindNode(name);
    if(node)
        return node;
    else
        return graph_->FindNode(name);
}

Tensor* GraphExecutor::FindTensor(const std::string& name)
{
    // try to search in optmized graph first

    Graph* cur_graph = GetOptimizedGraph();

    Tensor* tensor = cur_graph->FindTensor(name);
    if(tensor)
        return tensor;
    else
        return graph_->FindTensor(name);
}

bool GraphExecutor::InferShape(void)
{
    int node_number = graph_->seq_nodes.size();
    Node* node;

    for(int i = 0; i < node_number; i++)
    {
        node = graph_->seq_nodes[i];

        Operator* op = node->GetOp();

        // std::cout<<"Process Node: "<<node->GetName()<<" Op: "<<op->GetName()<<std::endl;

        if(op->GetName() == "Const" || op->GetName() == "Input")
            continue;

        if(node->IsDynamicShape())
            continue;

        bool skip = false;
        unsigned int j;

        for(j = 0; j < node->GetInputNum(); j++)
        {
            Tensor* input = node->GetInputTensor(j);
            TShape& shape = input->GetShape();
            if(shape.GetSize() == 0)
            {
                XLOG_ERROR() << "infer shape failed on node: " << node->GetName()
                             << " due to input: " << input->GetName() << " size is zero\n";
                return false;
            }
            if(shape.GetSize() < 0)
            {
                skip = true;
                break;
            }

            if(input->Reshaped())
                input->UpdateReshapeCount();
        }

        if(skip == true)
        {
            XLOG_ERROR() << "infer shape failed on node: " << node->GetName()
                         << " due to input: " << node->GetInputTensor(j)->GetName() << " not ready\n";
            return false;
        }

        std::vector<TShape> inputs;

        for(unsigned int i = 0; i < node->GetInputNum(); i++)
        {
            Tensor* tensor = node->GetInputTensor(i);

            inputs.push_back(tensor->GetShape());
        }

        std::vector<TShape> outputs;

        outputs.resize(node->GetOutputNum());

        if(!op->InferShape(inputs, outputs, exec_attr_.graph_layout))
        {
            std::cout << "infer shaped for node: " << node->GetName() << " op: " << op->GetName() << " failed\n";
            return false;
        }

        for(unsigned int i = 0; i < node->GetOutputNum(); i++)
        {
            Tensor* tensor = node->GetOutputTensor(i);
            TShape& shape = tensor->GetShape();
            TShape& new_shape = outputs[i];

            if(new_shape.GetSize())
                shape = new_shape;
        }
    }

    return true;
}

Tensor* GraphExecutor::GetInputNodeTensor(unsigned int node_idx, unsigned int tensor_idx)
{
    if(node_idx >= graph_->input_nodes.size())
        return nullptr;

    Node* node = graph_->input_nodes[node_idx];

    if(tensor_idx >= node->GetOutputNum())
        return nullptr;

    return node->GetOutputTensor(tensor_idx);
}

Tensor* GraphExecutor::GetOutputNodeTensor(unsigned int node_idx, unsigned int tensor_idx)
{
    if(node_idx >= graph_->output_nodes.size())
        return nullptr;

    Node* node = graph_->output_nodes[node_idx];

    if(tensor_idx >= node->GetOutputNum())
        return nullptr;

    return node->GetOutputTensor(tensor_idx);
}

bool GraphExecutor::SetTensorBuffer(Tensor* tensor, void* input_data, int data_size)
{
    return exec_engine_->SetTensorBuffer(tensor, input_data, data_size, exec_handle_);
}

void* GraphExecutor::GetTensorBuffer(Tensor* tensor)
{
    return exec_engine_->GetTensorBuffer(tensor, exec_handle_);
}

bool GraphExecutor::SetTensorData(Tensor* tensor, const void* input_data, int data_size)
{
    int tensor_size = tensor->GetTotalSize();

    if(tensor_size != data_size)
        return false;

    void* tensor_addr = GetTensorBuffer(tensor);

    if(tensor_addr == nullptr)
        return false;

    std::memcpy(tensor_addr, input_data, data_size);

    return true;
}

bool GraphExecutor::GetTensorData(Tensor* tensor, void* output_data, int data_size)
{
    int tensor_size = tensor->GetTotalSize();

    if(tensor_size != data_size)
        return false;

    void* tensor_addr = GetTensorBuffer(tensor);

    if(tensor_addr == nullptr)
        return false;

    std::memcpy(output_data, tensor_addr, data_size);

    return true;
}

static int MapExecStatus(int internal)
{
    switch(internal)
    {
        case EXEC_STATUS_CREATED:
        case EXEC_STATUS_INITED:
            return GRAPH_STAT_CREATED;
        case EXEC_STATUS_READY:
            return GRAPH_STAT_READY;
        case EXEC_STATUS_DONE:
            return GRAPH_STAT_DONE;
        case EXEC_STATUS_WAIT:
        case EXEC_STATUS_RUN:
            return GRAPH_STAT_RUNNING;
        case EXEC_STATUS_BAD:
        case EXEC_STATUS_INVALID:
            return GRAPH_STAT_ERROR;

        default:
            set_tengine_errno(EINVAL);
            return -1;
    }
}

int GraphExecutor::GetExecStatus(void)
{
    const exec_status_t& ret = exec_engine_->GetStatus(exec_handle_);

    int code = exec_engine_->GetStatusCode(ret);

    if(code < 0)
    {
        set_tengine_errno(EFAULT);
        return -1;
    }

    return MapExecStatus(code);
}

bool GraphExecutor::SetEventHook(int event, event_handler_t cb_func, void* cb_arg)
{
    return exec_engine_->SetEventHook(exec_handle_, event, cb_func, cb_arg);
}

bool GraphExecutor::Prerun(void)
{
    graph_->SanitizeGraph();

    SetExecParam(graph_);

    int optimize_only = 0;

    GetGraphAttr("optimize_only", &optimize_only, sizeof(int));

    if(optimize_only)
    {
        if(exec_engine_->Prerun(exec_handle_))
            return true;
        else
            return false;
    }

    if(InferShape() && exec_engine_->Prerun(exec_handle_))
    {
        prerun_done_ = true;
        return true;
    }

    return false;
}

bool GraphExecutor::Postrun(void)
{
    return exec_engine_->Postrun(exec_handle_);
}

int GraphExecutor::WaitGraph(int try_wait)
{
    return exec_engine_->Wait(exec_handle_, exec_event_, try_wait);
}

bool GraphExecutor::SyncRun(void)
{
    return exec_engine_->SyncRun(exec_handle_);
}

bool GraphExecutor::Run(int block)
{
    if(!exec_engine_->Run(exec_handle_, exec_event_))
        return false;

    if(block)
        exec_engine_->Wait(exec_handle_, exec_event_, 0);

    return true;
}

void GraphExecutor::ReleaseGraph(void)
{
    delete graph_;
}

void GraphExecutor::ReleaseExecHandle(void)
{
    exec_engine_->RemoveGraphExecutor(exec_handle_);
}

Graph* GraphExecutor::GetOptimizedGraph(void)
{
    if(exec_handle_ == nullptr || exec_engine_ == nullptr)
        return nullptr;

    Graph* graph = exec_engine_->GetOptimizedGraph(exec_handle_);
    if(graph == nullptr)
        graph = graph_;

    return graph;
}

bool GraphExecutor::GetOptimizeOnly(const char* name, void* val, int size)
{
    if(size != sizeof(int))
        return false;

    *( int* )val = optimize_only;

    return 0;
}

bool GraphExecutor::SetOptimizeOnly(const char* name, const void* val, int size)
{
    const int* int_ptr = ( const int* )val;

    optimize_only = int_ptr[0];

    return true;
}

bool GraphExecutor::SetExecAttrEntry(const char* name, const void* val, int size)
{
    if(!strcmp("exec_policy", name))
    {
        exec_attr_.policy = (exec_policy_t)(*( int* )val);
    }
    else if(!strcmp("exec_priority", name))
    {
        exec_attr_.priority = *( int* )val;
    }
    else if(!strcmp("kernel_mode", name))
    {
        exec_attr_.kernel_mode = *( int* )val;
    }
    else if(!strcmp("low_mem_mode", name))
    {
        int n = *( int* )val;

        if(n)
            exec_attr_.low_mem_mode = true;
        else
            exec_attr_.low_mem_mode = false;
    }
    else if(!strcmp("fc_mt", name))
    {
        int n = *( int* )val;
        if(n)
            exec_attr_.fc_mt = true;
        else
            exec_attr_.fc_mt = false;
    }
    else if(!strcmp("pooling_mt", name))
    {
        int n = *( int* )val;
        if(n)
            exec_attr_.pooling_mt = true;
        else
            exec_attr_.pooling_mt = false;
    }
    else
    {
        return false;
    }

    return true;
}

bool GraphExecutor::GetExecAttrEntry(const char* name, void* val, int size)
{
    if(!strcmp("exec_policy", name))
    {
        *( int* )val = exec_attr_.policy;
    }
    else if(!strcmp("exec_priority", name))
    {
        *( int* )val = exec_attr_.priority;
    }
    else if(!strcmp("kernel_mode", name))
    {
        *( int* )val = exec_attr_.kernel_mode;
    }
    else if(!strcmp("low_mem_mode", name))
    {
        if(exec_attr_.low_mem_mode)
            *( int* )val = 1;
        else
            *( int* )val = 0;
    }
    else if(!strcmp("fc_mt", name))
    {
        if(exec_attr_.fc_mt)
            *( int* )val = 1;
        else
            *( int* )val = 0;
    }
    else
    {
        return false;
    }

    return true;
}

bool GraphExecutor::BailoutSetAttr(const char* name, const void* val, int size)
{
    return exec_engine_->SetGraphAttr(exec_handle_, name, val, size);
}

bool GraphExecutor::BailoutGetAttr(const char* name, void* val, int size)
{
    return exec_engine_->GetGraphAttr(exec_handle_, name, val, size);
}

void GraphExecutor::InitAttrIO(void)
{
    auto get_func = std::bind(&GraphExecutor::GetExecAttrEntry, this, std::placeholders::_1, std::placeholders::_2,
                              std::placeholders::_3);

    attr_io_.RegGetFunc("exec_policy", get_func);
    attr_io_.RegGetFunc("exec_priority", get_func);
    attr_io_.RegGetFunc("kernel_mode", get_func);
    attr_io_.RegGetFunc("low_mem_mode", get_func);
    attr_io_.RegGetFunc("fc_mt", get_func);
    attr_io_.RegGetFunc("pooling_mt", get_func);

    auto set_func = std::bind(&GraphExecutor::SetExecAttrEntry, this, std::placeholders::_1, std::placeholders::_2,
                              std::placeholders::_3);

    attr_io_.RegSetFunc("exec_policy", set_func);
    attr_io_.RegSetFunc("exec_priority", set_func);
    attr_io_.RegSetFunc("kernel_mode", set_func);
    attr_io_.RegSetFunc("low_mem_mode", set_func);
    attr_io_.RegSetFunc("fc_mt", set_func);
    attr_io_.RegSetFunc("pooling_mt", set_func);

    auto set_opt_only_func = std::bind(&GraphExecutor::SetOptimizeOnly, this, std::placeholders::_1,
                                       std::placeholders::_2, std::placeholders::_3);

    auto get_opt_only_func = std::bind(&GraphExecutor::GetOptimizeOnly, this, std::placeholders::_1,
                                       std::placeholders::_2, std::placeholders::_3);

    attr_io_.RegSetFunc("optimize_only", set_opt_only_func);
    attr_io_.RegGetFunc("optimize_only", get_opt_only_func);

    // bailout
    auto set_func2 = std::bind(&GraphExecutor::BailoutSetAttr, this, std::placeholders::_1, std::placeholders::_2,
                               std::placeholders::_3);

    auto get_func2 = std::bind(&GraphExecutor::BailoutGetAttr, this, std::placeholders::_1, std::placeholders::_2,
                               std::placeholders::_3);

    attr_io_.RegSetFunc(nullptr, set_func2);

    attr_io_.RegGetFunc(nullptr, get_func2);
}

}    // namespace TEngine
