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

#include "ocl_node.hpp"
#include <fcntl.h>
#include <sys/stat.h>
#include "ocl_cpp_helper.hpp"
#include "ocl_executor.hpp"

void ocl_node::run_node_2d(std::vector<uint32_t> global_work_size, std::vector<uint32_t> local_work_size, cl::Kernel& kernel, cl::Event* event)
{
    std::vector<uint32_t> final_global_size = global_work_size;
    for (int i = 0; i < 2; ++i)
    {
        final_global_size[i] = ROUND_UP(global_work_size[i], std::max((uint32_t)1, local_work_size[i]));
    }

    cl_int res = CL_SUCCESS;
    if (local_work_size[0] == 0 && local_work_size[1] == 0)
    {
        res = engine->get_command_queue().enqueueNDRangeKernel(
            kernel, cl::NullRange, cl::NDRange(final_global_size[0], final_global_size[1]),
            cl::NullRange, nullptr, event);
    }
    else
    {
        res = engine->get_command_queue().enqueueNDRangeKernel(
            kernel, cl::NullRange, cl::NDRange(final_global_size[0], final_global_size[1]),
            cl::NDRange(local_work_size[0], local_work_size[1]), nullptr, event);
    }
    if (res != CL_SUCCESS)
    {
        TLOG_ERR("error run in %s \n", ir_node->name);
    }
}

void ocl_node::run_node_3d(std::vector<uint32_t> global_work_size, std::vector<uint32_t> local_work_size, cl::Kernel& kernel, cl::Event* event)
{
    std::vector<uint32_t> final_global_size = global_work_size;
    for (int i = 0; i < 3; ++i)
    {
        final_global_size[i] = ROUND_UP(global_work_size[i], std::max((uint32_t)1, local_work_size[i]));
    }
    cl_int res = CL_SUCCESS;
    if (local_work_size[0] == 0 || local_work_size[1] == 0 || local_work_size[2] == 0)
    {
        res = engine->get_command_queue().enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(final_global_size[0], final_global_size[1], final_global_size[2]),
                                                               cl::NullRange, nullptr, event);
    }
    else
    {
        res = engine->get_command_queue().enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(final_global_size[0], final_global_size[1], final_global_size[2]),
                                                               cl::NDRange(local_work_size[0], local_work_size[1], local_work_size[2]), nullptr, event);
    }
    if (res != CL_SUCCESS)
    {
        TLOG_ERR("error run in %s \n", ir_node->name);
    }
}

const std::vector<uint32_t> find_local_group_2d(std::vector<uint32_t> global_work_size, uint32_t max_group_work_size, OCLEngine* engine, cl::Kernel& kernel, const std::string& kernel_name)
{
    std::vector<uint32_t> lws(3, 1);
    std::vector<uint32_t> lws_prefer(2, 1);
    auto max_work_item_size = engine->get_max_work_item_sizes();
    uint32_t min_cost = UINT32_MAX;

    auto_tune tune;
    tune.key = kernel_name;
    if (engine->get_cache_auto_tune(&tune) == 0)
    {
        lws_prefer[0] = tune.local_size[0];
        lws_prefer[1] = tune.local_size[1];
        return lws_prefer;
    }

    if (false)
    {
        while (lws[1] <= global_work_size[1] || lws[1] <= 6)
        {
            lws[0] = 1;
            while (lws[0] <= global_work_size[0] || lws[0] <= 6)
            {
                if (lws[0] <= max_work_item_size[0] && lws[1] <= max_work_item_size[1] && lws[0] * lws[1] <= max_group_work_size)
                {
                    cl::Event event;
                    std::vector<uint32_t> internalGlobalWS(2, 1);
                    for (size_t i = 0; i < global_work_size.size(); ++i)
                    {
                        internalGlobalWS[i] = ROUND_UP(global_work_size[i], std::max((uint32_t)1, lws[i]));
                    }
                    cl_int res = engine->get_command_queue().enqueueNDRangeKernel(
                        kernel, cl::NullRange,
                        cl::NDRange(internalGlobalWS[0], internalGlobalWS[1]),
                        cl::NDRange(lws[0], lws[1]),
                        nullptr, &event);

                    if (res != CL_SUCCESS)
                    {
                        TLOG_ERR("lws tune res %s\n", kernel_name.c_str());
                    }

                    int cost_time = (int)engine->get_cost_time(&event);
                    if (cost_time < min_cost)
                    {
                        min_cost = cost_time;
                        lws_prefer[0] = lws[0];
                        lws_prefer[1] = lws[1];
                    }
                }
                lws[0]++;
            }
        }
    }
    else
    {
        while (lws[1] <= global_work_size[1] || lws[1] <= 6)
        {
            lws[0] = 1;
            while (lws[0] <= global_work_size[0] || lws[0] <= 6)
            {
                if (lws[0] <= max_work_item_size[0] && lws[1] <= max_work_item_size[1] && lws[0] * lws[1] <= max_group_work_size)
                {
                    cl::Event event;
                    std::vector<uint32_t> internalGlobalWS(2, 1);
                    for (size_t i = 0; i < global_work_size.size(); ++i)
                    {
                        internalGlobalWS[i] = ROUND_UP(global_work_size[i], std::max((uint32_t)1, lws[i]));
                    }
                    cl_int res = engine->get_command_queue().enqueueNDRangeKernel(
                        kernel, cl::NullRange,
                        cl::NDRange(internalGlobalWS[0], internalGlobalWS[1]),
                        cl::NDRange(lws[0], lws[1]),
                        nullptr, &event);

                    int cost_time = (int)engine->get_cost_time(&event);

                    if (res != CL_SUCCESS)
                    {
                        TLOG_ERR("lws tune res %s\n", kernel_name.c_str());
                    }

                    if (cost_time < min_cost)
                    {
                        min_cost = cost_time;
                        lws_prefer[0] = lws[0];
                        lws_prefer[1] = lws[1];
                    }
                }
                do {
                    lws[0]++;
                } while (((2 * global_work_size[0]) % lws[0] > 1) && (lws[0] & (lws[0] - 1)) != 0 && (lws[0] <= global_work_size[0]) && (lws[0] > 6)); //divisible powOfTwo lessThanSix
            }
            do {
                lws[1]++;
            } while (((2 * global_work_size[1]) % lws[1] > 1) && (lws[1] & (lws[1] - 1)) != 0 && (lws[1] <= global_work_size[1]) && (lws[1] > 6)); //divisible powOfTwo lessThanSix
        }
    }

    cl::Event event;
    cl_int res = engine->get_command_queue().enqueueNDRangeKernel(
        kernel, cl::NullRange,
        cl::NDRange(global_work_size[0], global_work_size[1]),
        cl::NullRange,
        nullptr, &event);
    if (res != CL_SUCCESS)
    {
        TLOG_ERR("3D lws null res %s\n", kernel_name.c_str());
    }

    int cost_time = (int)engine->get_cost_time(&event);
    if (cost_time < min_cost)
    {
        lws_prefer[0] = 0;
        lws_prefer[1] = 0;
        lws_prefer[2] = 0;
        min_cost = cost_time;
    }

    auto_tune tune_save;
    tune_save.key = kernel_name;
    tune_save.global_size[0] = global_work_size[0];
    tune_save.global_size[1] = global_work_size[1];
    tune_save.global_size[2] = 1;
    tune_save.local_size[0] = lws_prefer[0];
    tune_save.local_size[1] = lws_prefer[1];
    tune_save.local_size[2] = 1;
    engine->add_cache_auto_tune(tune_save);

    return lws_prefer;
}

const std::vector<uint32_t> find_local_group_3d(std::vector<uint32_t> global_work_size, uint32_t max_group_work_size, OCLEngine* engine, cl::Kernel& kernel, const std::string& kernel_name)
{
    std::vector<uint32_t> lws(3, 1);
    std::vector<uint32_t> lws_prefer(4, 1);
    uint32_t min_cost = UINT32_MAX;
    auto max_work_item_size = engine->get_max_work_item_sizes();

    auto_tune tune;
    tune.key = kernel_name;
    if (engine->get_cache_auto_tune(&tune) == 0)
    {
        lws_prefer[0] = tune.local_size[0];
        lws_prefer[1] = tune.local_size[1];
        lws_prefer[2] = tune.local_size[2];
        return lws_prefer;
    }

    while (lws[2] <= global_work_size[2] || lws[2] <= 6)
    {
        lws[1] = 1;
        while (lws[1] <= global_work_size[1] || lws[1] <= 6)
        {
            lws[0] = 1;
            while (lws[0] <= global_work_size[0] || lws[0] <= 6)
            {
                if (lws[0] <= max_work_item_size[0] && lws[1] <= max_work_item_size[1] && lws[2] <= max_work_item_size[2] && lws[0] * lws[1] * lws[2] <= max_group_work_size)
                {
                    cl::Event event;
                    std::vector<uint32_t> internalGlobalWS(3, 1);
                    for (size_t i = 0; i < global_work_size.size(); ++i)
                    {
                        internalGlobalWS[i] = ROUND_UP(global_work_size[i], std::max((uint32_t)1, lws[i]));
                    }
                    cl_int res = engine->get_command_queue().enqueueNDRangeKernel(
                        kernel, cl::NullRange,
                        cl::NDRange(internalGlobalWS[0], internalGlobalWS[1], internalGlobalWS[2]),
                        cl::NDRange(lws[0], lws[1], lws[2]),
                        nullptr, &event);
                    int cost_time = (int)engine->get_cost_time(&event);
                    if (res != CL_SUCCESS)
                    {
                        TLOG_ERR("lws tune res %s\n", kernel_name.c_str());
                    }
                    else
                    {
                        //TLOG_ERR("%s lws tune res:cost:%d  %d,%d,%d\n", kernel_name.c_str(), cost_time, lws[0], lws[1], lws[2]);
                    }
                    if (cost_time < min_cost)
                    {
                        min_cost = cost_time;
                        lws_prefer[0] = lws[0];
                        lws_prefer[1] = lws[1];
                        lws_prefer[2] = lws[2];
                    }
                }
                do {
                    lws[0]++;
                } while (((2 * global_work_size[0]) % lws[0] > 1) && (lws[0] & (lws[0] - 1)) != 0 && (lws[0] <= global_work_size[0]) && (lws[0] > 6)); //divisible powOfTwo lessThanSix
            }
            do {
                lws[1]++;
            } while (((2 * global_work_size[1]) % lws[1] > 1) && (lws[1] & (lws[1] - 1)) != 0 && (lws[1] <= global_work_size[1]) && (lws[1] > 6)); //divisible powOfTwo lessThanSix
        }
        do {
            lws[2]++;
        } while (((2 * global_work_size[2]) % lws[2] > 1) && (lws[2] & (lws[2] - 1)) != 0 && (lws[2] <= global_work_size[2]) && (lws[2] > 6)); //divisible powOfTwo lessThanSix
    }

    cl::Event event;
    cl_int res = engine->get_command_queue().enqueueNDRangeKernel(
        kernel, cl::NullRange,
        cl::NDRange(global_work_size[0], global_work_size[1], global_work_size[2]),
        cl::NullRange,
        nullptr, &event);
    if (res != CL_SUCCESS)
    {
        TLOG_ERR("3D lws null res %s\n", kernel_name.c_str());
    }

    int cost_time = (int)engine->get_cost_time(&event);
    if (cost_time < min_cost)
    {
        lws_prefer[0] = 0;
        lws_prefer[1] = 0;
        lws_prefer[2] = 0;
        min_cost = cost_time;
    }

    auto_tune tune_save;
    tune_save.key = kernel_name;
    tune_save.global_size[0] = global_work_size[0];
    tune_save.global_size[1] = global_work_size[1];
    tune_save.global_size[2] = global_work_size[2];
    tune_save.local_size[0] = lws_prefer[0];
    tune_save.local_size[1] = lws_prefer[1];
    tune_save.local_size[2] = lws_prefer[2];
    engine->add_cache_auto_tune(tune_save);

    return lws_prefer;
}

void print_data_file(struct tensor* tensor, std::string name, float* tensor_data)
{
    mkdir("/Users/hebingshi/stnn/tenginetest/Tengine/cmake-build-debuggcc/examples/cl_output", S_IRWXU | S_IRGRP | S_IWGRP | S_IROTH);
    std::string filename = std::string("/Users/hebingshi/stnn/tenginetest/Tengine/cmake-build-debuggcc/examples/cl_output") + "/" + name + ".txt";
    FILE* file = fopen(filename.c_str(), "w");
    if (NULL == file)
    {
        fprintf(stderr, "Tengine: Open file %s failed, skip dump\n", filename.c_str());
        return;
    }

    int batch = tensor->dims[0], channel = 0, height = 0, width = 0;

    if (TENGINE_LAYOUT_NCHW == tensor->layout)
    {
        channel = tensor->dims[1];
        height = tensor->dims[2];
        width = tensor->dims[3];
    }
    if (TENGINE_LAYOUT_NHWC == tensor->layout)
    {
        height = tensor->dims[1];
        width = tensor->dims[2];
        channel = tensor->dims[3];
    }

    fprintf(file, "Shape is {%d %d %d %d}, data type is fp32\n", batch, channel, height, width);

    for (int n = 0; n < batch; n++)
    {
        fprintf(file, "Batch %d:\n", n);

        for (int ch = 0; ch < channel; ch++)
        {
            fprintf(file, "\tChannel %d:\n", ch);

            for (int h = 0; h < height; h++)
            {
                fprintf(file, "\t\t");

                for (int w = 0; w < width; w++)
                {
                    int offset = 0;
                    offset += n * channel * height * width;
                    offset += ch * height * width;
                    offset += h * width;
                    offset += w;

                    float* base_ptr = tensor_data;
                    float val = base_ptr[offset];
                    if (val < 0)
                        fprintf(file, "%.4f ", val);
                    else
                        fprintf(file, " %.4f ", val);
                }
                fprintf(file, "\n");
            }
            fprintf(file, "\n");
        }
        fprintf(file, "\n");
    }
}

void ocl_node::debug_data()
{
    //return;
    struct graph* ir_graph = ir_node->graph;

    int ir_tensor_idx_input = ir_node->input_tensors[0];
    int ir_tensor_idx_output = ir_node->output_tensors[0];

    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_tensor_idx_input);
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_tensor_idx_output);

    uint64_t handle_input = engine->get_gpu_mem_by_idx(ir_tensor_idx_input);
    uint64_t handle_output = engine->get_gpu_mem_by_idx(ir_tensor_idx_output);

    TLOG_ERR("handle_input: %lld  handle_output: %lld \n", handle_input, handle_output);

    int height = output_tensor->dims[2];
    int width = output_tensor->dims[3];
    int input_height = input_tensor->dims[2];
    int input_width = input_tensor->dims[3];
    int input_channel = input_tensor->dims[1];
    int input_channel_block = UP_DIV(input_channel, 4);
    int output_channel = output_tensor->dims[1];
    int output_height = output_tensor->dims[2];

    uint32_t input_w = input_width * input_channel_block;
    uint32_t input_h = input_height;
    std::vector<float> input_debug(input_w * input_h * 4);
    engine->get_command_queue().enqueueReadImage(*(cl::Image*)handle_input, CL_TRUE, {0, 0, 0}, {input_w, input_h, 1}, input_w * sizeof(float) * 4, 0, input_debug.data());
    int idx_debug_input = 0;
    std::vector<float> input_debug_nchw(input_tensor->elem_num);
    for (int i = 0; i < input_tensor->dims[1]; ++i)
    {
        for (int j = 0; j < input_tensor->dims[2]; ++j)
        {
            for (int k = 0; k < input_tensor->dims[3]; ++k)
            {
                int index_nchw = i * input_width * input_height + j * input_width + k;
                int from_index = j * input_w * 4 + (i / 4) * (input_width * 4) + k * 4 + i % 4;
                input_debug_nchw[index_nchw] = input_debug[from_index];
            }
        }
    }
    std::string input_name = std::string(ir_node->name) + "input";
    print_data_file(input_tensor, input_name, input_debug_nchw.data());

    uint32_t output_w = width * UP_DIV(output_channel, 4);
    uint32_t output_h = height;
    std::vector<float> output_debug(output_w * output_h * 4);
    engine->get_command_queue().enqueueReadImage(*(cl::Image*)handle_output, CL_TRUE, {0, 0, 0}, {output_w, output_h, 1}, output_w * sizeof(float) * 4, 0, output_debug.data());
    std::vector<float> output_debug_nchw(output_tensor->elem_num);
    for (int i = 0; i < output_tensor->dims[1]; ++i)
    {
        for (int j = 0; j < output_tensor->dims[2]; ++j)
        {
            for (int k = 0; k < output_tensor->dims[3]; ++k)
            {
                int index_nchw = i * width * height + j * width + k;
                int from_index = j * output_w * 4 + (i / 4) * (width * 4) + k * 4 + i % 4;
                output_debug_nchw[index_nchw] = output_debug[from_index];
            }
        }
    }

    std::string output_name = std::string(ir_node->name) + "output";
    print_data_file(output_tensor, output_name, output_debug_nchw.data());
}
