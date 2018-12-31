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
 * Copyright (c) 2018, Open AI Lab
 * Author: haitao@openailab.com
 */
#include "data_type.hpp"
#include "exec_context.hpp"
#include "graph.hpp"
#include "tensor_mem.hpp"
#include "operator/convolution.hpp"

#include "tengine_test_api.h"
#include "node_ops.hpp"
#include "cpu_driver.hpp"
#include "graph_executor.hpp"

using namespace TEngine;

test_node_t create_convolution_test_node(int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h0, int pad_h1,
                                         int pad_w0, int pad_w1, int dilation_h, int dilation_w, int input_channel,
                                         int output_channel, int group)
{
    /* create op */

    Operator* op = OpManager::CreateOp("Convolution");
    Convolution* conv_op = dynamic_cast<Convolution*>(op);

    ConvParam* param = conv_op->GetParam();

    param->kernel_h = kernel_h;
    param->kernel_w = kernel_w;
    param->stride_h = stride_h;
    param->stride_w = stride_w;
    param->output_channel = output_channel;
    param->group = group;
    param->dilation_h = dilation_h;
    param->dilation_w = dilation_w;

    param->pad_h = -1;
    param->pad_w = -1;

    param->pads.resize(4);
    param->pads[0] = pad_h0;
    param->pads[1] = pad_w0;
    param->pads[2] = pad_h1;
    param->pads[3] = pad_w1;

    /* create node */

    Node* node = new Node("test_convolution");

    node->SetOp(conv_op);

    return node;
}

static int test_conv_node_set_input(Node* node, float* input_data[], int* input_shape[], int input_number)
{
    // input

    Tensor* tensor = new Tensor("input");

    tensor->SetDataType(DataType::GetTypeID("float32"));
    tensor->SetType(kConstTensor);
    tensor->SetMemAddr(input_data[0]);

    int* input_dim = input_shape[0];

    std::vector<int> input_dims = {input_dim[0], input_dim[1], input_dim[2], input_dim[3]};

    TShape& intput_shape = tensor->GetShape();

    intput_shape.SetDataLayout("NCHW");
    intput_shape.SetDim(input_dims);

    node->AddInputTensor(tensor);

    // weight

    tensor = new Tensor("weight");

    tensor->SetDataType(DataType::GetTypeID("float32"));
    tensor->SetType(kConstTensor);
    tensor->SetMemAddr(input_data[1]);

    input_dim = input_shape[1];

    std::vector<int> weight_dims = {input_dim[0], input_dim[1], input_dim[2], input_dim[3]};

    TShape& weight_shape = tensor->GetShape();

    weight_shape.SetDataLayout("NCHW");
    weight_shape.SetDim(weight_dims);

    node->AddInputTensor(tensor);

    if(input_number == 2)
        return 0;

    // bias

    tensor = new Tensor("bias");

    tensor->SetDataType(DataType::GetTypeID("float32"));
    tensor->SetType(kConstTensor);
    tensor->SetMemAddr(input_data[2]);

    input_dim = input_shape[2];

    std::vector<int> bias_dims = {input_dim[0]};

    TShape& bias_shape = tensor->GetShape();

    bias_shape.SetDataLayout("W");
    bias_shape.SetDim(bias_dims);

    node->AddInputTensor(tensor);

    return 0;
}

int test_node_set_input(test_node_t node, float* input_data[], int* input_shape[], int input_number)
{
    Node* test_node = ( Node* )node;

    Operator* op = test_node->GetOp();

    if(op->GetName() == "Convolution")
        return test_conv_node_set_input(test_node, input_data, input_shape, input_number);

    return -1;
}

static int test_conv_node_set_output(Node* node, float* output_data, int* output_shape)
{
    Tensor* tensor = new Tensor("output");

    tensor->SetDataType(DataType::GetTypeID("float32"));
    tensor->SetType(kConstTensor);
    tensor->SetMemAddr(output_data);

    int* output_dim = output_shape;

    std::vector<int> output_dims = {output_dim[0], output_dim[1], output_dim[2], output_dim[3]};

    TShape& shape = tensor->GetShape();

    shape.SetDataLayout("NCHW");
    shape.SetDim(output_dims);

    node->AddOutputTensor(tensor);

    return 0;
}

int test_node_set_output(test_node_t node, float* output_data[], int* output_shape[], int output_number)
{
    Node* test_node = ( Node* )node;

    Operator* op = test_node->GetOp();

    if(op->GetName() == "Convolution")
        return test_conv_node_set_output(test_node, output_data[0], output_shape[0]);

    return -1;
}

static Graph* create_test_graph(Node* node)
{
    Graph* graph = new Graph(node->GetName());

    node->SetNodeIndex(0);
    graph->seq_nodes.push_back(node);

    graph->AddInputNode(node);
    graph->AddOutputNode(node);

    /* for all tensors */

    for(unsigned int i = 0; i < node->GetInputNum(); i++)
    {
        Tensor* tensor = node->GetInputTensor(i);
        graph->AddTensorMap(tensor->GetName(), tensor);
    }

    for(unsigned int i = 0; i < node->GetOutputNum(); i++)
    {
        Tensor* tensor = node->GetOutputTensor(i);
        graph->AddTensorMap(tensor->GetName(), tensor);
    }

    return graph;
}

int test_node_prerun(test_node_t node)
{
    Node* test_node = ( Node* )node;

    // create graph for this node

    Graph* graph = create_test_graph(test_node);

    GraphExecutor* executor = new GraphExecutor();
    ExecContext* exec_context = ExecContext::GetDefaultContext();

    if(!executor->AttachGraph(exec_context, graph) || !executor->Prerun())
    {
        std::cout << "Prerun failed\n";
        return -1;
    }

    test_node->SetAttr("TEST_EXECUTOR", executor);

    return 0;

    /*
        NodeOps * node_ops=NodeOpsRegistryManager::FindNodeOps(cpu_dev->GetCPUInfo(),test_node);

        if(node_ops==nullptr)
              return -1;

        auto dispatch=std::bind(&CPUDevice::PushAiderTask,cpu_dev,std::placeholders::_1,
                                            std::placeholders::_2);

        auto wait=std::bind(&CPUDevice::WaitDone,cpu_dev);

        node_ops->SetHelper(std::malloc,std::free,dispatch,wait);


        if(!node_ops->Prerun(test_node))
        {
            std::cout<<"Prerun failed\n";
            return -1;
        }

        test_node->SetAttr(ATTR_NODE_OPS,node_ops);
    */

    return 0;
}

int test_node_run(test_node_t node)
{
    Node* test_node = ( Node* )node;

    GraphExecutor* executor = any_cast<GraphExecutor*>(test_node->GetAttr("TEST_EXECUTOR"));

    if(!executor->SyncRun())
    {
        std::cout << "Run failed\n";
        return -1;
    }

    return 0;

    /*
        NodeOps * node_ops=any_cast<NodeOps *>(test_node->GetAttr(ATTR_NODE_OPS));

        if(!node_ops->Run(test_node))
        {
            std::cout<<"Run failed\n";
            return -1;
        }
    */

    return 0;
}

int test_node_postrun(test_node_t node)
{
    Node* test_node = ( Node* )node;

    GraphExecutor* executor = any_cast<GraphExecutor*>(test_node->GetAttr("TEST_EXECUTOR"));

    if(!executor->Postrun())
    {
        std::cout << "Postrun failed\n";
        return -1;
    }

    return 0;

    /*
        NodeOps * node_ops=any_cast<NodeOps *>(test_node->GetAttr(ATTR_NODE_OPS));

        if(!node_ops->Postrun(test_node))
        {
            std::cout<<"Postrun failed\n";
            return -1;
        }
    */

    return 0;
}

void destroy_test_node(test_node_t node)
{
    Node* test_node = ( Node* )node;

    /* releaset graph executor & graph */

    GraphExecutor* executor = any_cast<GraphExecutor*>(test_node->GetAttr("TEST_EXECUTOR"));

    Graph* graph = executor->GetGraph();

    delete executor;
    delete graph;

    /* free tensor */

    for(unsigned int i = 0; i < test_node->GetInputNum(); i++)
    {
        Tensor* tensor = test_node->GetInputTensor(i);

        delete tensor;
    }

    for(unsigned int i = 0; i < test_node->GetOutputNum(); i++)
    {
        Tensor* tensor = test_node->GetOutputTensor(i);

        delete tensor;
    }

    /* free node */

    delete test_node;
}
