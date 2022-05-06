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

#include "ocl_executor.hpp"
#include "ocl_node.hpp"
#include "ocl_convertor.hpp"
#include "ocl_helper.hpp"
#include "cache/cache.hpp"
#include <../examples/common/common.h>
#include <string>

extern "C" {
#include "operator/op.h"
#include "convolution_param.h"
#include "ocl_define.h"
}

void register_all_ocl_creator();

std::map<int, ocl_node_creator*>* OCLEngine::s_ocl_node_creator_map = new std::map<int, ocl_node_creator*>();

extern const std::map<std::string, std::vector<unsigned char> > opencl_program_map;

bool OCLEngine::init()
{
    std::vector<cl::Platform> platforms;
    cl_int res = cl::Platform::get(&platforms);
    if (platforms.empty() || res != CL_SUCCESS)
    {
        return false;
    }
    cl::Platform::setDefault(platforms[0]);
    std::vector<cl::Device> all_devices;
    res = platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &all_devices);
    if (all_devices.empty() || res != CL_SUCCESS)
    {
        return false;
    }
    engine_device = std::make_shared<cl::Device>(all_devices[0]);
    engine_context = std::make_shared<cl::Context>(*engine_device, nullptr, nullptr, nullptr, &res);
    engine_command_queue = std::make_shared<cl::CommandQueue>(*engine_context, *engine_device, 0, &res);
    engine_convertor = std::make_shared<ocl_convertor>(this);
    gpu_cache = std::make_shared<cl_cache>();

    const std::string device_name = engine_device->getInfo<CL_DEVICE_NAME>();
    const std::string vendor_name = engine_device->getInfo<CL_DEVICE_VENDOR>();
    engine_device->getInfo(CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, &gpu_global_memory_cache_size);
    engine_device->getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &gpu_compute_unit);
    engine_device->getInfo(CL_DEVICE_MAX_CLOCK_FREQUENCY, &gpu_max_frequent);

    TLOG_ERR("device name:%s device_vendor:%s \n", device_name.c_str(), vendor_name.c_str());
    return true;
}

int OCLEngine::OCLEnginePreRun(struct subgraph* subgraph)
{
    //set_log_level(LOG_INFO);
    //    dump_sub_graph(subgraph);
    struct graph* ir_graph = subgraph->graph;
    // allocate var type tensor into map
    build_tensor_gpu_map(subgraph);
    open_command_queue_profile();
    // new node
    for (int i = 0; i < subgraph->node_num; ++i)
    {
        uint16_t node_id = subgraph->node_list[i];
        struct node* ir_node = get_ir_graph_node(ir_graph, node_id);
        auto op_type = ir_node->op.type;
        if (op_type == OP_CONST || op_type == OP_INPUT)
        {
            continue;
        }
        auto ocl_creator = s_ocl_node_creator_map->find(op_type);
        if (ocl_creator == s_ocl_node_creator_map->end())
        {
            TLOG_ERR("ocl do not support type:%d current now \n", op_type);
            continue;
        }
        exe_ocl_node_list.push_back(std::shared_ptr<ocl_node>(ocl_creator->second->creator(this, ir_node)));
    }

    // node prerun
    for (auto& _ocl_node : exe_ocl_node_list)
    {
        _ocl_node->pre_run();
    }
#ifndef OPENCL_PROFILE_TIME
    close_command_queue_profile();
#endif
    return 0;
}

int OCLEngine::OCLEngineRun(struct subgraph* subgraph)
{
    struct graph* ir_graph = subgraph->graph;
    for (int i = 0; i < subgraph->input_num; ++i)
    {
        int ir_tensor_idx = subgraph->input_tensor_list[i];
        struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_tensor_idx);
        upload_input_nc4hw(input_tensor, ir_tensor_idx);
    }

    for (auto& _ocl_node : exe_ocl_node_list)
    {
        _ocl_node->run(subgraph);
    }

    for (int i = 0; i < subgraph->output_num; ++i)
    {
        int ir_tensor_idx = subgraph->output_tensor_list[i];
        struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_tensor_idx);
        if (output_tensor->data == nullptr)
        {
            //TLOG_INFO("Log:download data malloc \n");
            auto* fp32_data = (float*)malloc(output_tensor->elem_size * output_tensor->elem_num);
            output_tensor->data = fp32_data;
        }
        download_output(output_tensor, ir_tensor_idx);
    }
    engine_command_queue->finish();
    engine_command_queue->flush();
    return 0;
}

void OCLEngine::OCLEnginePostRun()
{
}

OCLEngine::OCLEngine()
{
    init();
    register_all_ocl_creator();
}

void OCLEngine::build_tensor_gpu_map(struct subgraph* subgraph)
{
    struct graph* ir_graph = subgraph->graph;

    for (uint8_t i = 0; i < subgraph->input_num; i++)
    {
        int ir_tensor_idx = subgraph->input_tensor_list[i];
        struct tensor* ir_tensor = get_ir_graph_tensor(ir_graph, ir_tensor_idx);
        this->allocate_gpu_mem(ir_tensor, ir_tensor_idx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR);
    }

    for (int i = 0; i < subgraph->node_num; i++)
    {
        uint16_t node_id = subgraph->node_list[i];
        struct node* ir_node = get_ir_graph_node(ir_graph, node_id);
        for (int j = 0; j < ir_node->input_num; j++)
        {
            int ir_tensor_idx = ir_node->input_tensors[j];
            struct tensor* ir_tensor = get_ir_graph_tensor(ir_graph, ir_tensor_idx);
            this->allocate_gpu_mem(ir_tensor, ir_tensor_idx, CL_MEM_READ_WRITE);
        }
        for (int j = 0; j < ir_node->output_num; j++)
        {
            int ir_tensor_idx = ir_node->output_tensors[j];
            struct tensor* ir_tensor = get_ir_graph_tensor(ir_graph, ir_tensor_idx);
            this->allocate_gpu_mem(ir_tensor, ir_tensor_idx, CL_MEM_READ_WRITE);
        }
    }
}

void OCLEngine::add_ocl_node_creator(int op_type, ocl_node_creator* creator)
{
    if (s_ocl_node_creator_map->find(op_type) != s_ocl_node_creator_map->end())
    {
        TLOG_INFO("add_ocl_node_creator: %d befor", op_type);
        return;
    }
    TLOG_INFO("add_ocl_node_creator: %d \n", op_type);
    s_ocl_node_creator_map->insert(std::make_pair(op_type, creator));
}

void OCLEngine::allocate_gpu_mem(struct tensor* ir_tensor, int tensor_index, cl_mem_flags flags)
{
    auto iter = gpu_mem_map.find(tensor_index);
    if (iter != gpu_mem_map.end())
    {
        TLOG_INFO("has allocate gpu_mem: %d before \n", tensor_index);
        return;
    }
    auto type = ir_tensor->tensor_type;
    if (TENSOR_TYPE_CONST == type || TENSOR_TYPE_DEP == type || TENSOR_TYPE_UNKNOWN)
    {
        return;
    }
    int N, C, H, W;
    N = ir_tensor->dims[0];
    C = ir_tensor->dims[1];
    H = ir_tensor->dims[2];
    W = ir_tensor->dims[3];
    size_t image_width = UP_DIV(C, 4) * W;
    size_t image_height = N * H;
    cl_channel_type data_type = CL_FLOAT;
    auto image = new cl::Image2D(*this->engine_context,
                                 flags,
                                 cl::ImageFormat(CL_RGBA, data_type),
                                 image_width,
                                 image_height,
                                 0,
                                 nullptr,
                                 nullptr);
    gpu_mem_map.insert(std::make_pair(tensor_index, (gpu_mem_handle)image));
}

cl::Kernel OCLEngine::build_kernel(const std::string& program_name,
                                   const std::string& kernel_name,
                                   const std::set<std::string>& options)
{
    std::string build_option_str;
    build_option_str = "-DFLOAT=float -DFLOAT4=float4 -DFLOAT8=float8 -DRI_F=read_imagef -DFLOAT16=float16 -DWI_F=write_imagef -DCONVERT_FLOAT4=convert_float4";
    for (auto& option : options)
    {
        build_option_str += " " + option;
    }
    build_option_str += " -cl-mad-enable";
    auto it_source = opencl_program_map.find(program_name);
    if (it_source == opencl_program_map.end())
    {
        TLOG_ERR("build %s fail cannot find source \n", kernel_name.c_str());
    }
    cl::Program::Sources sources;
    std::string source_binary(it_source->second.begin(), it_source->second.end());
    sources.push_back(source_binary);
    cl::Program program = cl::Program(*engine_context, sources);
    cl_int res;
    res = program.build({*engine_device}, build_option_str.c_str());
    if (res != CL_SUCCESS)
    {
        if (program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(*engine_device) == CL_BUILD_ERROR)
        {
            std::string build_log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(*engine_device);
            TLOG_ERR("program build log: %s \n", build_log.c_str());
        }
    }
    cl::Kernel kernel = cl::Kernel(program, kernel_name.c_str(), &res);
    if (res == CL_SUCCESS)
    {
        TLOG_INFO("build %s success \n", kernel_name.c_str());
    }
    else
    {
        TLOG_ERR("build %s fail res:%d \n", kernel_name.c_str(), res);
    }
    return kernel;
}

uint64_t OCLEngine::get_max_work_group_size(const cl::Kernel& kernel)
{
    uint64_t max_work_group_size = 0;
    kernel.getWorkGroupInfo(*engine_device, CL_KERNEL_WORK_GROUP_SIZE, &max_work_group_size);
    return max_work_group_size;
}

const cl::CommandQueue& OCLEngine::get_command_queue() const
{
    return *engine_command_queue;
}

void OCLEngine::upload_input_nc4hw(tensor* ir_tensor, int ir_tensor_idx)
{
    auto need_size = (int)ir_tensor->elem_num * ir_tensor->elem_size;
    alloc_temp_buffer(need_size);
    engine_command_queue->enqueueWriteBuffer(*temp_buffer_up_down.second, CL_TRUE, 0, need_size, ir_tensor->data);

    if (gpu_mem_map.find(ir_tensor_idx) == gpu_mem_map.end())
    {
        TLOG_ERR("error in find input tensor gpu mem \n");
        return;
    }
    auto input_gpu_mem = gpu_mem_map.find(ir_tensor_idx)->second;
    get_converter().nchw_buffer_to_image(ir_tensor, temp_buffer_up_down.second.get(), (cl::Image*)input_gpu_mem, false);
}

void OCLEngine::download_output(struct tensor* ir_tensor, int ir_tensor_idx)
{
    auto need_size = (int)ir_tensor->elem_num * ir_tensor->elem_size;
    alloc_temp_buffer(need_size);
    if (gpu_mem_map.find(ir_tensor_idx) == gpu_mem_map.end())
    {
        TLOG_ERR("error in find output tensor gpu mem why???\n");
        return;
    }
    auto input_gpu_mem = gpu_mem_map.find(ir_tensor_idx)->second;
    int N, C, H, W;
    N = ir_tensor->dims[0];
    C = ir_tensor->dims[1];
    H = ir_tensor->dims[2];
    W = ir_tensor->dims[3];
    int image_width = UP_DIV(C, 4) * W;
    int image_height = N * H;
    get_converter().image_to_buffer(ir_tensor,
                                    (cl::Image*)input_gpu_mem,
                                    temp_buffer_up_down.second.get(),
                                    image_width,
                                    image_height);
    engine_command_queue->enqueueReadBuffer(*temp_buffer_up_down.second,
                                            CL_TRUE,
                                            0,
                                            need_size,
                                            ir_tensor->data,
                                            nullptr,
                                            nullptr);
}

const cl::Context& OCLEngine::get_context() const
{
    return *engine_context;
}

ocl_convertor& OCLEngine::get_converter()
{
    return *engine_convertor;
}

uint64_t OCLEngine::get_gpu_mem_by_idx(int idx)
{
    if (gpu_mem_map.find(idx) == gpu_mem_map.end())
    {
        TLOG_ERR("eror find gpu mem do not init before, why?");
        return 0;
    }
    return gpu_mem_map.find(idx)->second;
}

std::vector<uint32_t> OCLEngine::get_max_image_size()
{
    size_t height, width;
    cl_int res = engine_device->getInfo(CL_DEVICE_IMAGE2D_MAX_HEIGHT, &height);
    if (res != CL_SUCCESS)
    {
        TLOG_ERR("getInfo(CL_DEVICE_IMAGE2D_MAX_HEIGHT error height %d\n", res);
    }
    res = engine_device->getInfo(CL_DEVICE_IMAGE2D_MAX_WIDTH, &width);
    if (res != CL_SUCCESS)
    {
        TLOG_ERR("getInfo(CL_DEVICE_IMAGE2D_MAX_HEIGHT error width %d\n", res);
    }
    return {(uint32_t)width, (uint32_t)height};
}

std::vector<uint32_t> OCLEngine::get_max_work_item_sizes()
{
    int dims = 3;
    cl_int res = engine_device->getInfo(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, &dims);
    if (res != CL_SUCCESS)
    {
        TLOG_ERR("getInfo(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS error \n");
    }
    if (dims > 3)
    {
        std::vector<uint32_t> work_item(3, 8);
        return work_item;
    }
    cl::vector<cl::size_type> _work_item(dims, 1);
    res = engine_device->getInfo(CL_DEVICE_MAX_WORK_ITEM_SIZES, &_work_item);
    if (res != CL_SUCCESS)
    {
        TLOG_ERR("getInfo(CL_DEVICE_MAX_WORK_ITEM_SIZES error \n");
    }

    std::vector<uint32_t> work_item(dims, 1);
    for (int i = 0; i < dims; ++i)
    {
        work_item[i] = _work_item[i];
    }
    return work_item;
}

double OCLEngine::get_cost_time(cl::Event* event)
{
    if (nullptr == event)
    {
        TLOG_ERR("why event nullptr? \n");
        return 0;
    }
    cl_int res = event->wait();
    if (res != CL_SUCCESS)
    {
        TLOG_ERR("error in event->wait get_cost_time \n");
    }
    double start_nano = event->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    double end_nano = event->getProfilingInfo<CL_PROFILING_COMMAND_END>();
    return (end_nano - start_nano) / 1000.0;
}
void OCLEngine::open_command_queue_profile()
{
    engine_command_queue->finish();
    engine_command_queue.reset();
    cl_command_queue_properties properties = CL_QUEUE_PROFILING_ENABLE;
    cl_int res;
    engine_command_queue = std::make_shared<cl::CommandQueue>(*engine_context, *engine_device, properties, &res);
    if (res != CL_SUCCESS)
    {
        TLOG_ERR("OCLEngine::open_command_queue_profile error \n");
    }
}
void OCLEngine::close_command_queue_profile()
{
    engine_command_queue->finish();
    engine_command_queue.reset();
    cl_command_queue_properties properties = 0;
    cl_int res;
    engine_command_queue = std::make_shared<cl::CommandQueue>(*engine_context, *engine_device, properties, &res);
    if (res != CL_SUCCESS)
    {
        TLOG_ERR("OCLEngine::open_command_queue_profile error \n");
    }
}
void OCLEngine::set_gpu_mem_by_idx(int idx, uint64_t handle)
{
    if (gpu_mem_map.find(idx) == gpu_mem_map.end())
    {
        TLOG_ERR("cannot find gpu mem set mem no mean \n");
    }
    auto iter = gpu_mem_map.find(idx);
    if (handle != iter->second)
    {
        auto ptr = (cl::Image2D*)iter->second;
        auto mem = (cl_mem)ptr->get();
        clReleaseMemObject(mem);
    }
    iter->second = handle;
}

void OCLEngine::alloc_temp_buffer(int len)
{
    if (temp_buffer_up_down.second != nullptr && temp_buffer_up_down.first >= len)
    {
        return;
    }
    temp_buffer_up_down.first = len;
    temp_buffer_up_down.second = std::make_shared<cl::Buffer>(get_context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, len);
}

OCLEngine::~OCLEngine()
{
    temp_buffer_up_down.second.reset();
    std::set<uint64_t> mem_release_set;
    for (auto& iter : gpu_mem_map)
    {
        mem_release_set.insert(iter.second);
    }
    for (auto handle : mem_release_set)
    {
        auto ptr = (cl::Memory*)handle;
        auto mem = (cl_mem)ptr->get();
        clReleaseMemObject(mem);
    }
    gpu_mem_map.clear();
    exe_ocl_node_list.clear();
}

int OCLEngine::get_cache_auto_tune(auto_tune* tune)
{
    if (gpu_cache->get_cache_tune(tune->key, tune) == 0)
    {
        return 0;
    }
    else
    {
        return -1;
    }
}

int OCLEngine::add_cache_auto_tune(const auto_tune& tune)
{
    gpu_cache->set_auto_tune(tune);
    return 0;
}

int OCLEngine::load_cache(const std::string& path)
{
    TLOG_ERR("load cache from path: %s \n", path.c_str());
    gpu_cache->de_serializer(path);

    return 0;
}

int OCLEngine::store_cache(const std::string& path)
{
    TLOG_ERR("store cache to path: %s \n", path.c_str());
    gpu_cache->serializer(path);
    return 0;
}