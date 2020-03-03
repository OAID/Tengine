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
#include <cmath>
#include <unistd.h>
#include <omp.h>
#include "share_lib_parser.hpp"
#include "tensor_mem.hpp"
#include "graph_executor.hpp"
#include "graph.hpp"
#include "operator/convolution.hpp"
#include "operator/pooling.hpp"
#include "prof_utils.hpp"
#include "debug_utils.hpp"
#include "node_ops.hpp"
#include "test_soc_info.hpp"

#include <sys/time.h>
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

Node* create_pooling_node(int input_n, int input_c, int input_h, int input_w, int ksize, int stride,
                          PoolArg type = kPoolMax, int pad = 0, int global = 0)
{
    Operator* op = OpManager::CreateOp("Pooling");
    Pooling* pool_op = dynamic_cast<Pooling*>(op);
    // param, padding use default value 0
    PoolParam* param = pool_op->GetParam();

    param->pads.resize(4);
    param->pads[0] = pad;
    param->pads[1] = pad;
    param->pads[2] = pad;
    param->pads[3] = pad;

    param->kernel_shape.resize(2);
    param->kernel_shape[0] = ksize;
    param->kernel_shape[1] = ksize;
    // param->kernel_shape[1]=3;
    param->strides.resize(2);
    param->strides[0] = stride;
    param->strides[1] = stride;
    param->caffe_flavor = 1;
    param->global = global;
    param->alg = type;
    // param->alg=kPoolAvg;

    /* calculate shapes */
    std::vector<int> input_dims = {input_n, input_c, input_h, input_w};

    Node* node = new Node("test_pool");
    node->SetOp(pool_op);
    // prepare tensor: input/output
    Tensor* tensor;
    int mem_size;
    void* addr;

    tensor = new Tensor("input");
    tensor->SetDataType("float32");
    tensor->SetType(kVarTensor);
    TShape& shape = tensor->GetShape();
    shape.SetDataLayout("NCHW");
    shape.SetDim(input_dims);
    node->SetInputPort(0, tensor);
    mem_size = tensor->GetTotalSize();
    addr = std::malloc(mem_size);
    init_tensor_data(( float* )addr, mem_size / sizeof(float), -1);
    set_tensor_mem(tensor, addr, mem_size, std::free);

    tensor = new Tensor("output");
    tensor->SetDataType("float32");
    tensor->SetType(kVarTensor);
    TShape& oshape = tensor->GetShape();
    oshape.SetDataLayout("NCHW");

    std::vector<TShape> input_shape;
    input_shape.push_back(shape);

    std::vector<TShape> output_shape;
    output_shape.resize(1);

    pool_op->InferShape(input_shape, output_shape);

    oshape = output_shape[0];

    node->SetOutputPort(0, tensor);
    mem_size = tensor->GetTotalSize();
    addr = std::malloc(mem_size);
    init_tensor_data(( float* )addr, mem_size / sizeof(float), 0);
    set_tensor_mem(tensor, addr, mem_size, std::free);

    std::cout << "\n----------------------------------------------\n";
    std::cout << "Pooling Settings:\n";
    std::cout << "kernel_h : " << param->kernel_shape[0] << " kernel_w: " << param->kernel_shape[1] << "\n";
    std::cout << " stride: " << param->strides[0] << " pad: " << param->pads[0];
    std::cout << " global: " << param->global << " alg: " << param->alg << " caffe_style: " << param->caffe_flavor
              << "\n";
    std::cout << "input  n: " << input_n << " c: " << input_c << " h: " << input_h << " w: " << input_w << "\n";
    std::cout << "output n: " << oshape.GetN() << " c: " << oshape.GetC() << " h: " << oshape.GetH()
              << " w: " << oshape.GetW() << "\n";

    op_fops =
        1.0 * input_n * oshape.GetH() * oshape.GetW() * input_c * (param->kernel_shape[0] * param->kernel_shape[1]);

    return node;
}

namespace TEngine {

extern bool caffe_run_pooling(Node* node);
}

void test_new_operator(int rep, int input_n, int input_c, int input_h, int input_w, int ksize, int stride,
                       PoolArg type = kPoolMax, int pad = 0, int global = 0)
{
    // caffe
    Node* node0 = create_pooling_node(input_n, input_c, input_h, input_w, ksize, stride, type, pad, global);
    caffe_run_pooling(node0);

    Tensor* tensor = node0->GetOutputTensor(0);
    TShape* shape = &tensor->GetShape();

    float* data0 = ( float* )get_tensor_mem(tensor);
    // std::cout<<"caffe output [";
    //  for(int i=0;i<25;i++)std::cout<< data0[i]<<",";
    // std::cout<<"]\n";

    // pooling_run
    Node* node1 = create_pooling_node(input_n, input_c, input_h, input_w, ksize, stride, type, pad, global);

    SocInfo* soc_info = TestGetSocInfo();
    NodeOps* pooling_ops = NodeOpsRegistryManager::FindNodeOps(soc_info, node1);

    pooling_ops->SetHelper(std::malloc, std::free, nullptr);

    if(!pooling_ops->Run(node1))
    {
        std::cout << "Run failed\n";
    }
    Tensor* otensor = node1->GetOutputTensor(0);
    float* data_out = ( float* )get_tensor_mem(otensor);
    // std::cout<<"\nmy output [";
    // for(int i=0;i<25;i++)std::cout<< data_out[i]<<",";
    // std::cout<<"]\n";

    /* compare date */
    CalcMaxError(data_out, data0, shape->GetSize());

    /* performance benchmark ... */
    struct timeval t0, t1;
    float sum = 0.f;
    for(int i = 0; i < 30; i++)
    {
        gettimeofday(&t0, NULL);
        pooling_ops->Run(node1);
        gettimeofday(&t1, NULL);
        float mytime = ( float )((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;
        // std::cout<<"time is "<<mytime<<"\n";
        if(i >= 10)
            sum += mytime;
    }
    std::cout << "avg time is " << sum / 20. << std::endl;

    pooling_ops->Release();
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
    sys_init();

    // "divide exactly" means (input_h - ksize_h )= n * stride_h
    // ------------------------------------------------------------
    //                    n,c ,h, w, ksize,stride,Mehtod,pad,global
    test_new_operator(rep, 1, 64, 35, 35, 3, 1, kPoolMax, 2);

    // global
    test_new_operator(rep, 1, 1000, 16, 16, 3, 2, kPoolMax, 0, 1);
    test_new_operator(rep, 2, 2048, 7, 7, 3, 2, kPoolMax, 0, 1);
    test_new_operator(rep, 1, 1000, 16, 16, 3, 2, kPoolAvg, 0, 1);
    test_new_operator(rep, 2, 2048, 7, 7, 3, 2, kPoolAvg, 0, 1);
    //  max
    test_new_operator(rep, 1, 25, 16, 16, 3, 1, kPoolMax, 0);
    test_new_operator(rep, 2, 15, 45, 45, 3, 1, kPoolMax, 1);
    test_new_operator(rep, 1, 34, 16, 16, 3, 2, kPoolMax, 0);
    test_new_operator(rep, 2, 300, 45, 45, 3, 2, kPoolMax, 1);
    //  avg
    test_new_operator(rep, 1, 25, 16, 16, 3, 1, kPoolAvg, 0);
    test_new_operator(rep, 2, 15, 45, 45, 3, 1, kPoolAvg, 1);
    test_new_operator(rep, 1, 34, 16, 16, 3, 2, kPoolAvg, 0);
    test_new_operator(rep, 2, 300, 45, 45, 3, 2, kPoolAvg, 1);

    return 0;
}
