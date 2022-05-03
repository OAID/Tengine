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
 * Copyright (c) 2021, Open AI Lab
 * Author: hbshi@openailab.com
 */
#pragma once

#include "ocl_cpp_helper.hpp"

typedef uint64_t gpu_mem_handle;
class ocl_node_creator;
class ocl_convertor;
class ocl_node;

class OCLEngine
{
public:
    OCLEngine();
    ~OCLEngine();

    int OCLEnginePreRun(struct subgraph* subgraph);
    int OCLEngineRun(struct subgraph* subgraph);
    void OCLEnginePostRun();

public:
    uint64_t get_max_work_group_size(const cl::Kernel& kernel);
    std::vector<uint32_t> get_max_work_item_sizes();
    std::vector<uint32_t> get_max_image_size();
    cl::Kernel build_kernel(const std::string& program_name, const std::string& kernel_name, const std::set<std::string>& options);

private:
    bool init();
    void build_tensor_gpu_map(struct subgraph* subgraph);
    void allocate_gpu_mem(struct tensor* ir_tensor, int tensor_index, cl_mem_flags);

private:
    std::shared_ptr<cl::Context> engine_context;
    std::shared_ptr<cl::Device> engine_device;
    std::shared_ptr<cl::CommandQueue> engine_command_queue;
    std::shared_ptr<ocl_convertor> engine_convertor;

public:
    const cl::CommandQueue& get_command_queue() const;
    const cl::Context& get_context() const;
    ocl_convertor& get_converter();
    uint64_t gpu_global_memory_cache_size;
    uint32_t gpu_compute_unit;
    uint32_t gpu_max_frequent;

private:
    std::map<int, uint64_t> gpu_mem_map;
    std::pair<int, std::shared_ptr<cl::Buffer> > temp_buffer_up_down;

    std::shared_ptr<cl_cache> gpu_cache;

public:
    std::vector<std::shared_ptr<ocl_node> > exe_ocl_node_list;

public:
    static std::map<int, ocl_node_creator*>* s_ocl_node_creator_map;
    static void add_ocl_node_creator(int op_type, ocl_node_creator* creator);
    void upload_input_nc4hw(tensor* ir_tensor, int ir_tensor_idx);
    uint64_t get_gpu_mem_by_idx(int idx);
    void set_gpu_mem_by_idx(int idx, uint64_t mem_handle);
    double get_cost_time(cl::Event* event);
    void download_output(tensor* tensor, int ir_tensor_idx);
    void open_command_queue_profile();
    void close_command_queue_profile();
    void alloc_temp_buffer(int len);

    int add_cache_auto_tune(const auto_tune& tune);
    int get_cache_auto_tune(auto_tune* tune);
    int load_cache(const std::string& path);
    int store_cache(const std::string& path);
};

class ocl_node_creator
{
public:
    ocl_node_creator() = default;
    virtual ~ocl_node_creator() = default;
    virtual ocl_node* creator(OCLEngine* engine, struct node* ir_node) = 0;
};

#define REGISTER_OCL_OP(type, T)                  \
    void ocl_##type##_creator()                   \
    {                                             \
        T* t = new T;                             \
        OCLEngine::add_ocl_node_creator(type, t); \
    }
