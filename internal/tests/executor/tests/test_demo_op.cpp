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
#include <iostream>
#include <functional>
#include <unistd.h>
#include <cstring>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>

#include "share_lib_parser.hpp"
#include "operator/demo_op.hpp"
#include "node_ops.hpp"
#include "graph.hpp"
#include "tengine_c_api.h"
#include "cpu_driver.hpp"

using namespace TEngine;

Node* create_demo_op_node(bool is_float)
{
    Operator* op = OpManager::CreateOp("DemoOp");
    DemoOp* demo_op = dynamic_cast<DemoOp*>(op);

    Node* node = new Node("test_demo_op");

    node->SetOp(demo_op);

    Tensor* tensor = new Tensor("input");

    if(is_float)
        tensor->SetDataType("float32");
    else
        tensor->SetDataType("int8");

    tensor->SetType(kVarTensor);

    node->SetInputPort(0, tensor);

    tensor = new Tensor("output");

    if(is_float)
        tensor->SetDataType("float32");
    else
        tensor->SetDataType("int8");

    tensor->SetType(kVarTensor);

    node->SetOutputPort(0, tensor);

    return node;
}

void test_mt_mode(void)
{
    Node* float_node = create_demo_op_node(true);
    Node* int_node = create_demo_op_node(false);

    CPUDevice* cpu_dev = dynamic_cast<CPUDevice*>(DriverManager::GetDefaultDevice());

    NodeOps* float_ops = NodeOpsRegistryManager::FindNodeOps(cpu_dev->GetCPUInfo(), float_node);

    // setup helper
    auto dispatch = std::bind(&CPUDevice::PushAiderTask, cpu_dev, std::placeholders::_1, std::placeholders::_2);

    auto wait = std::bind(&CPUDevice::WaitDone, cpu_dev);

    float_ops->SetHelper(std::malloc, std::free, dispatch, wait);

    if(!float_ops->Prerun(float_node))
    {
        std::cout << "Prerun failed\n";
    }

    if(!float_ops->Run(float_node))
    {
        std::cout << "Run failed\n";
    }

    if(!float_ops->Postrun(float_node))
    {
        std::cout << "Postrun failed\n";
    }

    std::cout << "FLOAT TEST DONE\n";

    NodeOps* int_ops = NodeOpsRegistryManager::FindNodeOps(cpu_dev->GetCPUInfo(), int_node);

    int_ops->SetHelper(std::malloc, std::free, dispatch, wait);

    if(!int_ops->Prerun(int_node))
    {
        std::cout << "Prerun failed\n";
    }

    if(!int_ops->Run(int_node))
    {
        std::cout << "Run failed\n";
    }

    if(!int_ops->Postrun(int_node))
    {
        std::cout << "Postrun failed\n";
    }

    std::cout << "INT TEST DONE\n";

    delete float_node;
    delete int_node;
}

void test_st_mode(void)
{
    Node* float_node = create_demo_op_node(true);

    CPUDevice* cpu_dev = dynamic_cast<CPUDevice*>(DriverManager::GetDefaultDevice());

    NodeOps* float_ops = NodeOpsRegistryManager::FindNodeOps(cpu_dev->GetCPUInfo(), float_node);

    // setup helper
    auto dispatch = std::bind(&CPUDevice::PushAiderTask, cpu_dev, std::placeholders::_1, std::placeholders::_2);

    auto wait = std::bind(&CPUDevice::WaitDone, cpu_dev);

    float_ops->SetHelper(std::malloc, std::free, dispatch, wait);

    if(!float_ops->Prerun(float_node))
    {
        std::cout << "Prerun failed\n";
    }

    if(!float_ops->Run(float_node))
    {
        std::cout << "Run failed\n";
    }

    if(!float_ops->Postrun(float_node))
    {
        std::cout << "Postrun failed\n";
    }

    std::cout << "FLOAT TEST DONE\n";

    delete float_node;
}

void sys_init(void)
{
    init_tengine_library();
}

int main(int argc, char* argv[])
{
    sys_init();

    test_mt_mode();
    test_st_mode();

    return 0;
}
