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
 * Copyright (c) 2021, OPEN AI LAB
 * Author: hbshi@openailab.com
 */
#pragma once
#ifndef TENGINE_LITE_SOURCE_DEVICE_OPENCL_OCL_NODE_HPP_
#define TENGINE_LITE_SOURCE_DEVICE_OPENCL_OCL_NODE_HPP_
#include <vector>
#include "ocl_cpp_helper.hpp"
class OCLEngine;
class ocl_node
{
public:
    ocl_node() = delete;
    explicit ocl_node(OCLEngine* engine, struct node* ir_node)
    {
        this->ir_node = ir_node;
        this->engine = engine;
    };
    virtual void pre_run() = 0;
    virtual void run(struct subgraph* subgraph) = 0;
    virtual ~ocl_node() = default;

    void debug_data();
    // care do not copy
private:
protected:
    OCLEngine* engine;
    struct node* ir_node;
    void run_node_2d(std::vector<uint32_t> global_work_size, std::vector<uint32_t> local_work_size, cl::Kernel& kernel, cl::Event* event = nullptr);
    void run_node_3d(std::vector<uint32_t> global_work_size, std::vector<uint32_t> local_work_size, cl::Kernel& kernel, cl::Event* event = nullptr);
};

const std::vector<uint32_t> find_local_group_2d(std::vector<uint32_t> global_work_size, uint32_t max_group_work_size, OCLEngine* engine, cl::Kernel& kernel, const std::string& kernel_name);
const std::vector<uint32_t> find_local_group_3d(std::vector<uint32_t> global_work_size, uint32_t max_group_work_size, OCLEngine* engine, cl::Kernel& kernel, const std::string& kernel_name);
void print_data_file(struct tensor* tensor, std::string name, float* tensor_data);
#endif //TENGINE_LITE_SOURCE_DEVICE_OPENCL_OCL_NODE_HPP_
