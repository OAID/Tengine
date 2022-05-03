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

#ifdef OPENCL_DEBUG_DATA
    debug_data();
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

static bool use_image_winograd(struct node* ir_node, OCLEngine* engine)
{
    std::vector<uint32_t> w_h_limit = engine->get_max_image_size();
    struct graph* ir_graph = ir_node->graph;

    int ir_tensor_idx_input = ir_node->input_tensors[0];
    int ir_tensor_idx_output = ir_node->output_tensors[0];

    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_tensor_idx_input);
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_tensor_idx_output);

    int height = output_tensor->dims[2];
    int width = output_tensor->dims[3];
    int input_channel = input_tensor->dims[1];
    int output_channel = output_tensor->dims[1];

    bool res = true;
    auto w_unit = UP_DIV(width, 2);
    auto h_unit = UP_DIV(height, 2);
    if (w_h_limit[0] < 16 * UP_DIV(w_unit * h_unit, 4) || w_h_limit[0] < UP_DIV(input_channel, 4) * 4)
    {
        return false;
    }

    if (w_h_limit[1] < 16 * UP_DIV(w_unit * h_unit, 4) || w_h_limit[1] < 4 * UP_DIV(output_channel, 4))
    {
        return false;
    }

    return res;
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
                 && conv_2d_param->input_channel >= 32 && use_image_winograd(ir_node, engine))
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