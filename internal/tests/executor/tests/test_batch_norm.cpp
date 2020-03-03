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
#include <iostream>
#include <functional>
#include <unistd.h>
#include <sys/time.h>

#include "share_lib_parser.hpp"
#include "tensor_mem.hpp"
#include "graph_executor.hpp"
#include "graph.hpp"
#include "operator/batch_norm.hpp"
#include "prof_utils.hpp"
#include "debug_utils.hpp"
#include "node_ops.hpp"
#include "test_soc_info.hpp"

using namespace TEngine;
void init_tensor_data(float* addr, int number, int fixed_val)
{
    for(int i = 0; i < number; i++)
    {
        if(fixed_val >= 0)
            addr[i] = fixed_val;
        else
            addr[i] = (i % 64) / 100.f;
    }
}
float op_fops;

Node* create_batch_norm_node(int n, int c, int h, int w)
{
    Operator* op = OpManager::CreateOp("BatchNormalization");
    BatchNorm* bn_op = dynamic_cast<BatchNorm*>(op);

    BatchNormParam* param = bn_op->GetParam();
    param->caffe_flavor = 1;
    param->rescale_factor = 1.2;
    param->eps = 1e-5;

    /* input shape */
    int input_h = h;
    int input_w = w;
    int input_c = c;
    int input_n = n;

    std::vector<int> input_dims = {input_n, input_c, input_h, input_w};
    std::vector<int> channel_dims = {input_c};

    op_fops = 1.0 * input_n * input_c * input_h * input_w * 2;

    Node* node = new Node("test_convolution");

    node->SetOp(bn_op);

    // prepare tensor: input/gmma/beta/mean/vars
    Tensor* tensor;
    int mem_size;
    void* addr;

    tensor = new Tensor("input");

    tensor->SetDataType("float32");
    tensor->SetType(kVarTensor);

    TShape* shape = &tensor->GetShape();

    shape->SetDataLayout("NCHW");
    shape->SetDim(input_dims);

    node->SetInputPort(0, tensor);

    mem_size = tensor->GetTotalSize();
    addr = std::malloc(mem_size);
    init_tensor_data(( float* )addr, mem_size / sizeof(float), -1);

    set_tensor_mem(tensor, addr, mem_size, std::free);

    tensor = new Tensor("gamma");

    tensor->SetDataType("float32");
    tensor->SetType(kVarTensor);

    shape = &tensor->GetShape();

    shape->SetDataLayout("W");
    shape->SetDim(channel_dims);

    node->SetInputPort(1, tensor);

    mem_size = tensor->GetTotalSize();
    addr = std::malloc(mem_size);
    init_tensor_data(( float* )addr, mem_size / sizeof(float), -1);
    set_tensor_mem(tensor, addr, mem_size, std::free);

    /*** beta ***/

    tensor = new Tensor("beta");

    tensor->SetDataType("float32");
    tensor->SetType(kVarTensor);

    shape = &tensor->GetShape();

    shape->SetDataLayout("W");
    shape->SetDim(channel_dims);

    node->SetInputPort(2, tensor);

    mem_size = tensor->GetTotalSize();
    addr = std::malloc(mem_size);
    init_tensor_data(( float* )addr, mem_size / sizeof(float), -1);
    set_tensor_mem(tensor, addr, mem_size, std::free);

    /*** means ***/

    tensor = new Tensor("means");

    tensor->SetDataType("float32");
    tensor->SetType(kVarTensor);

    shape = &tensor->GetShape();

    shape->SetDataLayout("W");
    shape->SetDim(channel_dims);

    node->SetInputPort(3, tensor);

    mem_size = tensor->GetTotalSize();
    addr = std::malloc(mem_size);
    init_tensor_data(( float* )addr, mem_size / sizeof(float), -1);
    set_tensor_mem(tensor, addr, mem_size, std::free);

    /*** vars */
    tensor = new Tensor("vars");

    tensor->SetDataType("float32");
    tensor->SetType(kVarTensor);

    shape = &tensor->GetShape();

    shape->SetDataLayout("W");
    shape->SetDim(channel_dims);

    node->SetInputPort(4, tensor);

    mem_size = tensor->GetTotalSize();
    addr = std::malloc(mem_size);
    init_tensor_data(( float* )addr, mem_size / sizeof(float), -1);
    set_tensor_mem(tensor, addr, mem_size, std::free);

    /*** output ***/

    tensor = new Tensor("output");

    tensor->SetDataType("float32");
    tensor->SetType(kVarTensor);

    shape = &tensor->GetShape();

    shape->SetDataLayout("NCHW");
    shape->SetDim(input_dims);

    node->SetOutputPort(0, tensor);

    mem_size = tensor->GetTotalSize();
    addr = std::malloc(mem_size);
    init_tensor_data(( float* )addr, mem_size / sizeof(float), 0);
    set_tensor_mem(tensor, addr, mem_size, std::free);

    return node;
}

namespace TEngine {

extern bool caffe_run_batch_norm(Node* node);
}

void test_batch_norm(int rep)
{
    Node* node = create_batch_norm_node(1, 64, 112, 112);

    unsigned long start = get_cur_time();

    for(int i = 0; i < rep; i++)
        caffe_run_batch_norm(node);

    unsigned long end = get_cur_time();

    std::cout << "rep=" << rep << " used time: " << end - start << "us \n";
    std::printf("total fops: %.2f M, per op fops: %.2f K\n", op_fops * rep / 1000000, op_fops / 1000);
    std::printf("performance rate: %.2f Mops\n", op_fops * rep / (end - start));
}

void test_new_operator(int input_n, int input_c, int input_h, int input_w)
{
    // caffe
    Node* node0 = create_batch_norm_node(input_n, input_c, input_h, input_w);
    caffe_run_batch_norm(node0);

    Tensor* tensor = node0->GetOutputTensor(0);
    TShape* shape = &tensor->GetShape();

    float* data0 = ( float* )get_tensor_mem(tensor);
    std::cout << "caffe output [";
    for(int i = 0; i < 9; i++)
        std::cout << data0[i] << ",";
    std::cout << "]\n";

    // node1
    Node* node1 = create_batch_norm_node(input_n, input_c, input_h, input_w);

    SocInfo* soc_info = TestGetSocInfo();
    NodeOps* bn_ops = NodeOpsRegistryManager::FindNodeOps(soc_info, node1);

    bn_ops->SetHelper(std::malloc, std::free, nullptr);

    if(!bn_ops->Prerun(node1))
    {
        std::cout << "Prerun failed\n";
    }
    if(!bn_ops->Run(node1))
    {
        std::cout << "Run failed\n";
    }
    Tensor* otensor = node1->GetOutputTensor(0);
    float* data_out = ( float* )get_tensor_mem(otensor);
    std::cout << "\nmy output [";
    for(int i = 0; i < 9; i++)
        std::cout << data_out[i] << ",";
    std::cout << "]\n";

    /* compare date */
    CalcMaxError(data_out, data0, shape->GetSize());

    /* performance benchmark ... */
    struct timeval t0, t1;
    float sum = 0.f;
    for(int i = 0; i < 60; i++)
    {
        gettimeofday(&t0, NULL);
        bn_ops->Run(node1);
        gettimeofday(&t1, NULL);
        float mytime = ( float )((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;
        // std::cout<<"time is "<<mytime<<"\n";
        if(i >= 10)
            sum += mytime;
    }
    std::cout << "avg time is " << sum / 50. << std::endl;

    if(!bn_ops->Postrun(node1))
    {
        std::cout << "Postrun failed\n";
    }

    bn_ops->Release();
}

void sys_init(void)
{
    ShareLibParser p0("./build/operator/liboperator.so");
    p0.ExcecuteFunc<int()>("tengine_plugin_init");

    ShareLibParser p2("./build/executor/libexecutor.so");
    p2.ExcecuteFunc<int()>("tengine_plugin_init");
}

int main(int argc, char* argv[])
{
    int res;
    int rep = 1;

    while((res = getopt(argc, argv, "r:")) != -1)
    {
        switch(res)
        {
            case 'r':
                rep = strtoul(optarg, NULL, 10);
                break;
            default:
                break;
        }
    }

    sys_init();

    test_batch_norm(rep);

    test_new_operator(1, 64, 112, 112);

    test_new_operator(1, 64, 112, 112);

    return 0;
}
