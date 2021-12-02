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

//#include <libc.h>
#include <stdlib.h>

#include <memory>
#include "ocl_convertor.hpp"
#include "ocl_conv2d.hpp"
#include "ocl_winograd.hpp"
#include "ocl_dwconv.hpp"

void ocl_conv2d::pre_run()
{
    struct graph* ir_graph = ir_node->graph;

    int ir_tensor_idx_input = ir_node->input_tensors[0];
    int ir_tensor_idx_output = ir_node->output_tensors[0];

    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_tensor_idx_input);
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_tensor_idx_output);

    uint64_t handle_input = engine->get_gpu_mem_by_idx(ir_tensor_idx_input);
    uint64_t handle_output = engine->get_gpu_mem_by_idx(ir_tensor_idx_output);

    // TLOG_ERR("handle_input: %lld  handle_output: %lld \n", handle_input, handle_output);

    int height = output_tensor->dims[2];
    int width = output_tensor->dims[3];
    int input_height = input_tensor->dims[2];
    int input_width = input_tensor->dims[3];
    int input_channel = input_tensor->dims[1];
    int input_channel_block = UP_DIV(input_channel, 4);
    int output_channel = output_tensor->dims[1];
    int output_height = output_tensor->dims[2];
    int kernel_width = this->conv2d_param->kernel_w;
    int kernel_height = this->conv2d_param->kernel_h;
    int global_0 = UP_DIV(width, 4) * UP_DIV(output_channel, 4);
    int global_1 = output_height;

    int input_image_shape[2] = {input_height, input_width};
    int output_image_shape[2] = {height, width};
    int kernel_shape[2] = {kernel_height, kernel_width};
    int stride_shape[2] = {strides[0], strides[1]};
    int padding_shape[2] = {paddings[0], paddings[1]};
    int dilation_shape[2] = {dilations[0], dilations[1]};

    uint32_t idx = 0;
    auto kernel = &conv2d_kernel;
    kernel->setArg(idx++, global_0);
    kernel->setArg(idx++, global_1);

    kernel->setArg(idx++, *(cl::Image*)handle_input);
    kernel->setArg(idx++, *gpu_weight);
    if (2 < ir_node->input_num)
    {
        kernel->setArg(idx++, *gpu_bias);
    }
    kernel->setArg(idx++, *(cl::Image*)handle_output);
    kernel->setArg(idx++, sizeof(input_image_shape), input_image_shape);
    kernel->setArg(idx++, input_channel_block);
    kernel->setArg(idx++, sizeof(output_image_shape), output_image_shape);
    kernel->setArg(idx++, sizeof(kernel_shape), kernel_shape);
    kernel->setArg(idx++, sizeof(stride_shape), stride_shape);
    kernel->setArg(idx++, sizeof(padding_shape), padding_shape);
    kernel->setArg(idx++, sizeof(dilation_shape), dilation_shape);
    kernel->setArg(idx, UP_DIV(width, 4));

    global_work_size = {(uint32_t)global_0, (uint32_t)global_1};
    local_work_size = find_local_group_2d(global_work_size, max_work_group_size, engine, conv2d_kernel, ir_node->name);
}

void ocl_conv2d::run(struct subgraph* subgraph)
{
#ifdef OPENCL_PROFILE_TIME
    cl::Event event;
    run_node_2d(global_work_size, local_work_size, conv2d_kernel, &event);
    int cost = (int)engine->get_cost_time(&event);
    TLOG_ERR("cost: %d conv2d:%s \n", cost, ir_node->name);
#else
    run_node_2d(global_work_size, local_work_size, conv2d_kernel);
#endif
#if 0
    // print input
    printf("ocl_conv2d::run :input ------------------ \n");
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

    printf("#### %s ocl_conv2d::run :output ------------------ \n", ir_node->name);
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

#endif
}

ocl_conv2d::ocl_conv2d(OCLEngine* engine, struct node* ir_node)
    : ocl_node(engine, ir_node)
{
    struct graph* ir_graph = ir_node->graph;
    struct tensor* weight_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    struct tensor* bias_tensor;
    if (2 < ir_node->input_num)
    {
        bias_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[2]);
        upload_bias_gpu(bias_tensor);
    }

    auto* conv_2d_param = (struct conv_param*)ir_node->op.param_mem;
    this->conv2d_param = conv_2d_param;
    strides = {conv_2d_param->stride_h, conv_2d_param->stride_w};
    dilations = {conv_2d_param->dilation_h, conv_2d_param->dilation_w};
    paddings = {conv_2d_param->pad_h0, conv_2d_param->pad_w0};
    int kernel_width = conv_2d_param->kernel_w;
    int kernel_height = conv_2d_param->kernel_h;
    int out_channel = conv_2d_param->output_channel;
    int input_channel = conv_2d_param->input_channel;

    int filter_size = kernel_height * kernel_width * out_channel * input_channel;
    int filter_buffer_size = filter_size * sizeof(float);
    cl::Buffer filter_buffer(engine->get_context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, filter_buffer_size);
    cl_int error;
    auto filter_ptr_gpu = engine->get_command_queue().enqueueMapBuffer(filter_buffer, true, CL_MAP_WRITE, 0, filter_buffer_size, nullptr, nullptr, &error);
    if (filter_ptr_gpu != nullptr && error == CL_SUCCESS)
    {
        ::memset(filter_ptr_gpu, 0, filter_buffer_size);
        ::memcpy(filter_ptr_gpu, weight_tensor->data, filter_buffer_size);
    }
    else
    {
        TLOG_ERR("error in filter_ptr_gpu");
    }
    engine->get_command_queue().enqueueUnmapMemObject(filter_buffer, filter_ptr_gpu);
    gpu_weight = std::make_shared<cl::Image2D>(engine->get_context(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, CL_FLOAT), input_channel, UP_DIV(out_channel, 4) * kernel_width * kernel_height);
    engine->get_converter().conv2d_buffer_to_image(conv_2d_param, &filter_buffer, gpu_weight.get());

    std::set<std::string> buildOption;
    if (2 < ir_node->input_num)
    {
        buildOption.emplace("-DBIAS");
    }
    if (conv2d_param->activation == 0)
    {
        buildOption.emplace("-DRELU");
    }
    else if (conv_2d_param->activation == 6)
    {
        buildOption.emplace("-DRELU6");
    }
    conv2d_kernel = engine->build_kernel("conv_2d_2d", "conv_2d", buildOption);
    max_work_group_size = engine->get_max_work_group_size(conv2d_kernel);

#if 0
    std::vector<float> debugData;
    debugData.resize(3 * 9 * 4);
    engine->get_command_queue().enqueueReadImage(*gpu_weight,
                                                 CL_TRUE, {0, 0, 0}, {3, 9, 1}, 3 * sizeof(float) * 4,
                                                 0, debugData.data());
    int debugIndex = 0;
    printf("\n");
    for (int i = 0; i < 9; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            for (int k = 0; k < 4; ++k)
            {
                printf("%.4f,", debugData[debugIndex]);
                debugIndex++;
            }
            printf("  ");
        }
        printf("\n");
    }
#endif
}

void ocl_conv2d::upload_bias_gpu(struct tensor* ir_tensor)
{
    int bias_size = ir_tensor->elem_num;
    int buffer_size = ROUND_UP(bias_size, 4);
    cl::Buffer bias_buffer(engine->get_context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, bias_size * sizeof(float));
    cl_int error;
    auto* bias_ptr_gpu = (float*)engine->get_command_queue().enqueueMapBuffer(bias_buffer, true, CL_MAP_WRITE, 0, bias_size * sizeof(float), nullptr, nullptr, &error);
    if (bias_ptr_gpu != nullptr && error == CL_SUCCESS)
    {
        ::memset(bias_ptr_gpu, 0, bias_size * sizeof(float));
        ::memcpy(bias_ptr_gpu, ir_tensor->data, bias_size * sizeof(float));
    }
    engine->get_command_queue().enqueueUnmapMemObject(bias_buffer, bias_ptr_gpu);
    gpu_bias = std::make_shared<cl::Image2D>(engine->get_context(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, CL_FLOAT), UP_DIV(bias_size, 4), 1);
    engine->get_converter().buffer_to_image(&bias_buffer, gpu_bias.get(), UP_DIV(bias_size, 4), 1);
}

class ocl_conv2d_creator : public ocl_node_creator
{
public:
    ocl_node* creator(OCLEngine* engine, struct node* ir_node) override
    {
        auto* conv_2d_param = (struct conv_param*)ir_node->op.param_mem;
        struct graph* ir_graph = ir_node->graph;
        struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);

        bool use_wino = false;
        bool is_dwconv = false;

        if (conv_2d_param->group == input_tensor->dims[1] && conv_2d_param->group != 1)
        {
            is_dwconv = true;
        }
        else if (conv_2d_param->kernel_w == 3 && conv_2d_param->kernel_h == 3 && conv_2d_param->dilation_w == 1 && conv_2d_param->dilation_h == 1
                 && conv_2d_param->stride_w == 1 && conv_2d_param->stride_h == 1
                 && conv_2d_param->input_channel >= 32
                 && conv_2d_param->input_channel >= 32)
        {
            use_wino = true;
        }

        if (is_dwconv)
        {
            return new ocl_dwconv(engine, ir_node);
        }
        else if (use_wino)
        {
            return new ocl_winograd(engine, ir_node);
        }
        else
        {
            return new ocl_conv2d(engine, ir_node);
        }
    }
};

REGISTER_OCL_OP(OP_CONV, ocl_conv2d_creator);