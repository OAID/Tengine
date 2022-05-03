//
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

#include "ocl_winograd.hpp"

#include "ocl_executor.hpp"
#include "ocl_convertor.hpp"
#include "ocl_node.hpp"
ocl_winograd::ocl_winograd(OCLEngine* engine, struct node* ir_node)
    : ocl_node(engine, ir_node)
{
    struct graph* ir_graph = ir_node->graph;
    int ir_tensor_idx_output = ir_node->output_tensors[0];
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_tensor_idx_output);
    float* bias_data = nullptr;
    int bias_elem = output_tensor->dims[1];
    if (2 < ir_node->input_num)
    {
        struct tensor* bias_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[2]);
        bias_data = (float*)bias_tensor->data;
    }
    upload_bias_gpu(bias_data, bias_elem);

    // weight process
    struct tensor* weight_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    int weight_wino_size = ALIGN_UP4(weight_tensor->dims[0]) * ALIGN_UP4(weight_tensor->dims[1]) * 16;
    auto weight_wino = new float[weight_wino_size];
    weight_transform(weight_tensor, weight_wino);

    cl::Buffer weight_buffer(engine->get_context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, weight_wino_size * sizeof(float));
    cl_int error;
    auto weight_ptr_cl = engine->get_command_queue().enqueueMapBuffer(weight_buffer, CL_TRUE, CL_MAP_WRITE, 0, weight_wino_size * sizeof(float), nullptr, nullptr, &error);
    if (weight_ptr_cl != nullptr && error == CL_SUCCESS)
    {
        memcpy(weight_ptr_cl, weight_wino, weight_wino_size * sizeof(float));
    }
    else
    {
        TLOG_ERR("error map wino weight gpu \n");
    }
    engine->get_command_queue().enqueueUnmapMemObject(weight_buffer, weight_ptr_cl);
    int ci_block = UP_DIV(weight_tensor->dims[1], 4);
    int co_block = UP_DIV(weight_tensor->dims[0], 4);
    gpu_weight = std::make_shared<cl::Image2D>(engine->get_context(), CL_MEM_READ_WRITE,
                                               cl::ImageFormat(CL_RGBA, CL_FLOAT), ci_block * 4, co_block * 16, 0, nullptr, nullptr);
    engine->get_converter().buffer_to_image(&weight_buffer, gpu_weight.get(), ci_block * 4, co_block * 16);
    delete[] weight_wino;
}

void ocl_winograd::upload_bias_gpu(const float* bias_data, int bias_size)
{
    cl::Buffer bias_buffer(engine->get_context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, bias_size * sizeof(float));
    cl_int error;
    auto* bias_ptr_gpu = (float*)engine->get_command_queue().enqueueMapBuffer(bias_buffer, true, CL_MAP_WRITE, 0, bias_size * sizeof(float), nullptr, nullptr, &error);
    if (bias_ptr_gpu != nullptr && error == CL_SUCCESS)
    {
        ::memset(bias_ptr_gpu, 0, bias_size * sizeof(float));
        if (bias_data != nullptr)
        {
            ::memcpy(bias_ptr_gpu, bias_data, bias_size * sizeof(float));
        }
    }
    engine->get_command_queue().enqueueUnmapMemObject(bias_buffer, bias_ptr_gpu);
    gpu_bias = std::make_shared<cl::Image2D>(engine->get_context(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, CL_FLOAT), UP_DIV(bias_size, 4), 1);
    engine->get_converter().buffer_to_image(&bias_buffer, gpu_bias.get(), UP_DIV(bias_size, 4), 1);
}

void ocl_winograd::pre_run()
{
    auto* conv_2d_param = (struct conv_param*)ir_node->op.param_mem;
    this->conv2d_param = conv_2d_param;
    strides = {conv_2d_param->stride_h, conv_2d_param->stride_w};
    dilations = {conv_2d_param->dilation_h, conv_2d_param->dilation_w};
    paddings = {conv_2d_param->pad_h0, conv_2d_param->pad_w0};

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
    int output_channel = output_tensor->dims[1];

    auto w_unit = UP_DIV(width, 2);
    auto h_unit = UP_DIV(height, 2);

    gpu_source = std::make_shared<cl::Image2D>(engine->get_context(), CL_MEM_READ_WRITE,
                                               cl::ImageFormat(CL_RGBA, CL_FLOAT), UP_DIV(input_channel, 4) * 4, 16 * UP_DIV(w_unit * h_unit, 4));
    gpu_dest = std::make_shared<cl::Image2D>(engine->get_context(), CL_MEM_READ_WRITE,
                                             cl::ImageFormat(CL_RGBA, CL_FLOAT), 16 * UP_DIV(w_unit * h_unit, 4), 4 * UP_DIV(output_channel, 4));

    int ic_block = UP_DIV(input_channel, 4);
    int oc_block = UP_DIV(output_channel, 4);

    std::set<std::string> basic;
    source_transform = engine->build_kernel("winogradTransformSource2_3_1", "winogradTransformSource", basic);
    max_work_group_size_source = engine->get_max_work_group_size(source_transform);
    dot_mul = engine->build_kernel("gemm", "gemm", basic);
    max_work_group_size_dot = engine->get_max_work_group_size(dot_mul);
    if (conv2d_param->activation == 0)
    {
        basic.emplace("-DRELU");
    }
    else if (conv_2d_param->activation == 6)
    {
        basic.emplace("-DRELU6");
    }
    dest_transform = engine->build_kernel("winogradTransformDest2_3_1", "winogradTransformDest", basic);
    max_work_group_size_dest = engine->get_max_work_group_size(dest_transform);

    //source transform
    source_transform.setArg(0, *(cl::Image2D*)handle_input);
    source_transform.setArg(1, *gpu_source);
    source_transform.setArg(2, w_unit);
    source_transform.setArg(3, h_unit);
    source_transform.setArg(4, paddings[1]);
    source_transform.setArg(5, paddings[0]);
    source_transform.setArg(6, input_width);
    source_transform.setArg(7, input_height);
    source_transform.setArg(8, ic_block);
    source_transform.setArg(9, 0);
    source_transform.setArg(10, 0);
    source_transform.setArg(11, 0);

    global_work_size_source = {(uint32_t)w_unit * h_unit, (uint32_t)ic_block};
    local_work_size_source = find_local_group_2d(global_work_size_source, max_work_group_size_source, engine, source_transform, std::string(ir_node->name) + "wino_source_transform");

    // dot mul
    dot_mul.setArg(0, *gpu_source);
    dot_mul.setArg(1, *gpu_weight);
    dot_mul.setArg(2, *gpu_dest);
    int gemm_width = UP_DIV(w_unit * h_unit, 4);
    dot_mul.setArg(3, gemm_width);
    dot_mul.setArg(4, oc_block);
    dot_mul.setArg(5, ic_block);
    dot_mul.setArg(6, 16);

    global_work_size_dot = {(uint32_t)gemm_width * oc_block, 16};
    local_work_size_dot = find_local_group_2d(global_work_size_dot, max_work_group_size_dot, engine, dot_mul, std::string(ir_node->name) + "wino_dot_mul");

    // dest
    dest_transform.setArg(0, *gpu_dest);
    dest_transform.setArg(1, *gpu_bias);
    dest_transform.setArg(2, *(cl::Image2D*)handle_output);
    dest_transform.setArg(3, w_unit);
    dest_transform.setArg(4, h_unit);
    dest_transform.setArg(5, width);
    dest_transform.setArg(6, height);
    dest_transform.setArg(7, oc_block);
    dest_transform.setArg(8, 0);
    dest_transform.setArg(9, 0);
    dest_transform.setArg(10, 0);

    global_work_size_dest = {(uint32_t)w_unit * h_unit, (uint32_t)oc_block};
    local_work_size_dest = find_local_group_2d(global_work_size_dest, max_work_group_size_dest, engine, dest_transform, std::string(ir_node->name) + "wino_dest_transform");
}
void ocl_winograd::run(struct subgraph* subgraph)
{
    {
#ifdef OPENCL_PROFILE_TIME
        cl::Event event;
        run_node_2d(global_work_size_source, local_work_size_source, source_transform, &event);
        int cost = (int)engine->get_cost_time(&event);
        TLOG_ERR("cost: %d conv2d wino_source:%s \n", cost, ir_node->name);
#else
        run_node_2d(global_work_size_source, local_work_size_source, source_transform);
#endif
    }
    {
#ifdef OPENCL_PROFILE_TIME
        cl::Event event;
        run_node_2d(global_work_size_dot, local_work_size_dot, dot_mul, &event);
        int cost = (int)engine->get_cost_time(&event);
        TLOG_ERR("cost: %d conv2d wino_dot:%s \n", cost, ir_node->name);
#else
        run_node_2d(global_work_size_dot, local_work_size_dot, dot_mul);
#endif
    }

    {
#ifdef OPENCL_PROFILE_TIME
        cl::Event event;
        run_node_2d(global_work_size_dest, local_work_size_dest, dest_transform, &event);
        int cost = (int)engine->get_cost_time(&event);
        TLOG_ERR("cost: %d conv2d wino_dest:%s \n", cost, ir_node->name);
#else
        run_node_2d(global_work_size_dest, local_work_size_dest, dest_transform);
#endif
    }
}

void ocl_winograd::weight_transform(struct tensor* weight_tensor, float* weight_dst)
{
    int channel_out = weight_tensor->dims[0];
    int channel_in = weight_tensor->dims[1];
    int channel_in_align = ALIGN_UP4(channel_in);
    int channel_out_align = ALIGN_UP4(channel_out);

    int stride_0 = channel_in_align * channel_out_align;
    int stride_1 = channel_in_align * 4;

    if (channel_in % 4 != 0 || channel_out % 4 != 0)
    {
        memset(weight_dst, 0, ALIGN_UP4(weight_tensor->dims[0]) * ALIGN_UP4(weight_tensor->dims[1]) * 16 * sizeof(float));
    }

    auto kernel_trans = new float[16];
    for (int i = 0; i < channel_out; ++i)
    {
        int channel_out_c4 = i / 4;
        int channel_out_r4 = i % 4;
        for (int j = 0; j < channel_in; ++j)
        {
            auto kernel_src = (float*)weight_tensor->data + i * channel_in * 9 + j * 9;
            trans_kernel(kernel_src, kernel_trans);

            int start_pos = stride_1 * channel_out_c4 + j * 4 + channel_out_r4;
            for (int k = 0; k < 16; ++k)
            {
                weight_dst[start_pos + k * stride_0] = kernel_trans[k];
            }
        }
    }

    delete[] kernel_trans;
}

void ocl_winograd::trans_kernel(const float* src, float* dest)
{
    float m00 = src[0], m01 = src[1], m02 = src[2];

    float m10 = src[0] + src[3] + src[6];
    float m11 = src[1] + src[4] + src[7];
    float m12 = src[2] + src[5] + src[8];

    float m20 = src[0] - src[3] + src[6];
    float m21 = src[1] - src[4] + src[7];
    float m22 = src[2] - src[5] + src[8];

    float m30 = src[6], m31 = src[7], m32 = src[8];

    dest[0] = m00, dest[4] = m10, dest[8] = m20, dest[12] = m30;

    dest[1] = m00 + m01 + m02;
    dest[5] = m10 + m11 + m12;
    dest[9] = m20 + m21 + m22;
    dest[13] = m30 + m31 + m32;

    dest[2] = m00 - m01 + m02;
    dest[6] = m10 - m11 + m12;
    dest[10] = m20 - m21 + m22;
    dest[14] = m30 - m31 + m32;

    dest[3] = m02, dest[7] = m12, dest[11] = m22, dest[15] = m32;
}
