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
#include <cstring>

#include "share_lib_parser.hpp"
#include "tensor_mem.hpp"
#include "graph_executor.hpp"
#include "graph.hpp"
#include "operator/fully_connected.hpp"
#include "prof_utils.hpp"
#include "debug_utils.hpp"
#include "node_ops.hpp"

using namespace TEngine;

float op_fops;

void init_tensor_data(float* addr, int number, int fixed_val)
{
    for(int i = 0; i < number; i++)
    {
        if(fixed_val >= 0)
            addr[i] = fixed_val;
        else
            addr[i] = i % 64;
    }
}

Node* create_fc_node(void)
{
    Operator* op = OpManager::CreateOp("FullyConnected");
    FullyConnected* fc_op = dynamic_cast<FullyConnected*>(op);

    FCParam* param = fc_op->GetParam();

    /* calculate shapes */

#if 1
    param->num_output = 2;
    int input_n = 1;
    int input_c = 128;
#else
    param->num_output = 12544;
    int input_n = 32;
    int input_c = 288;
#endif

    int input_h = 1;
    int input_w = 1;

    int N = input_n;
    int K = input_h * input_w * input_c;
    int M = param->num_output;

    std::vector<int> input_dims = {input_n, input_c, input_h, input_w};
    std::vector<int> weight_dims = {M, K};
    std::vector<int> bias_dims = {M};
    std::vector<int> output_dims = {N, M};

    op_fops = 1.0 * N * M * (2 * K);

    std::cout << "Input n: " << input_n << " c: " << input_c << " h: " << input_h << " w: " << input_w << "\n";
    std::cout << "weight M: " << M << " K: " << K << "\n";
    std::cout << "N: " << N << " M: " << M << " K: " << K << "\n";

    Node* node = new Node("test_fc");

    node->SetOp(fc_op);

    // prepare tensor: input/weight/bias/output
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
    set_tensor_mem(tensor, addr, mem_size, std::free);

    init_tensor_data(( float* )addr, mem_size / sizeof(float), -1);

    tensor = new Tensor("weight");

    tensor->SetDataType("float32");
    tensor->SetType(kVarTensor);

    shape = &tensor->GetShape();

    shape->SetDataLayout("HW");
    shape->SetDim(weight_dims);

    node->SetInputPort(1, tensor);

    mem_size = tensor->GetTotalSize();
    addr = std::malloc(mem_size);
    set_tensor_mem(tensor, addr, mem_size, std::free);

    init_tensor_data(( float* )addr, mem_size / sizeof(float), -1);

#if 0
    tensor=new Tensor("bias");
    
    tensor->SetDataType("float32");
    tensor->SetType(kVarTensor);
         
    shape=&tensor->GetShape();

    shape->SetDataLayout("W");
    shape->SetDim(bias_dims);

    node->SetInputPort(2,tensor);

    mem_size=tensor->GetTotalSize();
    addr=std::malloc(mem_size);
    set_tensor_mem(tensor,addr,mem_size,std::free);

    init_tensor_data((float *)addr,mem_size/sizeof(float),-1);
#endif

    tensor = new Tensor("output");

    tensor->SetDataType("float32");
    tensor->SetType(kVarTensor);

    shape = &tensor->GetShape();

    shape->SetDataLayout("NCHW");
    shape->SetDim(output_dims);

    node->SetOutputPort(0, tensor);

    mem_size = tensor->GetTotalSize();
    addr = std::malloc(mem_size);
    set_tensor_mem(tensor, addr, mem_size, std::free);

    init_tensor_data(( float* )addr, mem_size / sizeof(float), 0);

    return node;
}

namespace TEngine {

extern bool caffe_run_fully_connected(Node* node, int rep, unsigned long* time);
}

void test_fully_connected(int rep)
{
    Node* node = create_fc_node();

    unsigned long time;

    caffe_run_fully_connected(node, rep, &time);

    std::cout << "rep=" << rep << " used time: " << time << "us \n";
    std::printf("total fops: %.2f M, per op fops: %.2f K\n", op_fops * rep / 1000000, op_fops / 1000);
    std::printf("performance rate: %.2f Mops\n", op_fops * rep / (time));
}

void test_new_operator(int rep)
{
    Node* node0 = create_fc_node();
    Node* node1 = create_fc_node();

    caffe_run_fully_connected(node0, 0, nullptr);

    SocInfo* soc_info = TestGetSocInfo();
    NodeOps* fc_ops = NodeOpsRegistryManager::FindNodeOps(soc_info, node1);

    fc_ops->SetHelper(std::malloc, std::free, nullptr);

    if(!fc_ops->Prerun(node1))
    {
        std::cout << "Prerun failed\n";
    }

    if(!fc_ops->Run(node1))
    {
        std::cout << "Run failed\n";
    }

    /* compare date */

    Tensor* tensor = node0->GetOutputTensor(0);
    float* data0 = ( float* )get_tensor_mem(tensor);

    tensor = node1->GetOutputTensor(0);

    float* data1 = ( float* )get_tensor_mem(tensor);

    std::vector<int> output_dim = tensor->GetShape().GetDim();

    std::vector<int> mismatch;

    if(!CompareFloatTensor(data0, data1, output_dim, mismatch))
    {
        std::cout << "MISMATCH: ";
        for(unsigned int i = 0; i < mismatch.size(); i++)
            std::cout << " " << mismatch[i];
        std::cout << "\n";

        DumpFloat("/tmp/data0", data0, tensor->GetShape().GetSize());
        DumpFloat("/tmp/data1", data1, tensor->GetShape().GetSize());

        return;
    }

    /* performance benchmark ... */

    unsigned long start = get_cur_time();

    for(int i = 0; i < rep; i++)
        fc_ops->Run(node1);

    unsigned long end = get_cur_time();

    std::printf("OUR IMPLEMENTATION ...\n");
    std::cout << "rep=" << rep << " used time: " << end - start << "us \n";
    std::printf("total fops: %.2f M, per op fops: %.2f K\n", op_fops * rep / 1000000, op_fops / 1000);
    std::printf("performance rate: %.2f Mops\n", op_fops * rep / (end - start));

    if(!fc_ops->Postrun(node1))
    {
        std::cout << "Postrun failed\n";
    }

    fc_ops->Release();
}

void sys_init(void)
{
    ShareLibParser p0("./build/operator/liboperator.so");
    p0.ExcecuteFunc<int()>("tengine_plugin_init");

    ShareLibParser p1("./build/serializer/libserializer.so");
    p1.ExcecuteFunc<int()>("tengine_plugin_init");

    ShareLibParser p2("./build/executor/libexecutor.so");
    p2.ExcecuteFunc<int()>("tengine_plugin_init");
}

int main(int argc, char* argv[])
{
    int rep = 1;
    int res;

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

    test_fully_connected(rep);

    test_new_operator(rep);
    return 0;
}
