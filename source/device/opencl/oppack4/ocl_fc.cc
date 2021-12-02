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

#include "ocl_fc.hpp"
#include "ocl_convertor.hpp"

ocl_fc::ocl_fc(OCLEngine* engine, struct node* ir_node)
    : ocl_node(engine, ir_node)
{
    std::set<std::string> build_options;
    ocl_fc_kernel = engine->build_kernel("fc", "fc", build_options);
    max_work_group_size = engine->get_max_work_group_size(ocl_fc_kernel);
}

void ocl_fc::upload_bias_gpu(const float* bias_data, int bias_size)
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

void ocl_fc::upload_weight_gpu(struct tensor* tensor)
{
    int elem_size = tensor->elem_num;
    cl::Buffer weight_buffer(engine->get_context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, elem_size * sizeof(float));
    cl_int error;
    auto* weight_ptr_gpu = (float*)engine->get_command_queue().enqueueMapBuffer(weight_buffer, true, CL_MAP_WRITE, 0, elem_size * sizeof(float), nullptr, nullptr, &error);
    if (weight_ptr_gpu != nullptr && error == CL_SUCCESS)
    {
        ::memset(weight_ptr_gpu, 0, elem_size * sizeof(float));
        ::memcpy(weight_ptr_gpu, tensor->data, elem_size * sizeof(float));
    }
    engine->get_command_queue().enqueueUnmapMemObject(weight_buffer, weight_ptr_gpu);
    gpu_weight = std::make_shared<cl::Image2D>(engine->get_context(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, CL_FLOAT), UP_DIV(tensor->dims[1], 4), tensor->dims[0]);
    engine->get_converter().buffer_to_image(&weight_buffer, gpu_weight.get(), UP_DIV(tensor->dims[1], 4), tensor->dims[0]);

#if 1
//    printf("fc weight \n");
//    uint32_t input_w = UP_DIV(tensor->dims[1], 4);
//    uint32_t input_h = tensor->dims[0];
//    std::vector<float> input_debug(input_w * input_h * 4);
//    engine->get_command_queue().enqueueReadImage(*gpu_weight, CL_TRUE, {0, 0, 0}, {input_w, input_h, 1}, input_w * sizeof(float) * 4, 0, input_debug.data());
//    int idx_debug_input = 0;
//    auto ptr = (float *)tensor->data;
//    for (int j = 0; j < 4; ++j)
//    {
//        for (int k = 0; k < input_w; ++k)
//        {
//            for (int i = 0; i < 4; ++i)
//            {
//                printf("%.4f,", input_debug[idx_debug_input++]);
//                //printf("%.4f,", ptr[idx_debug_input++]);
//            }
//            printf(" ");
//        }
//        printf("\n");
//    }
#endif
}

void ocl_fc::pre_run()
{
    struct graph* ir_graph = ir_node->graph;

    int ir_tensor_idx_input = ir_node->input_tensors[0];
    int ir_tensor_idx_weight = ir_node->input_tensors[1];
    int ir_tensor_idx_bias = ir_node->input_tensors[2];
    int ir_tensor_idx_output = ir_node->output_tensors[0];

    struct tensor* weight_tensor = get_ir_graph_tensor(ir_graph, ir_tensor_idx_weight);
    struct tensor* bias_tensor = get_ir_graph_tensor(ir_graph, ir_tensor_idx_bias);

    uint64_t handle_input = engine->get_gpu_mem_by_idx(ir_tensor_idx_input);
    uint64_t handle_output = engine->get_gpu_mem_by_idx(ir_tensor_idx_output);

    ocl_fc_param = (fc_param*)ir_node->op.param_mem;

    float* bias_data = nullptr;
    int bias_elem = (int)bias_tensor->elem_num;
    if (2 < ir_node->input_num)
    {
        bias_data = (float*)bias_tensor->data;
    }
    upload_bias_gpu(bias_data, bias_elem);
    upload_weight_gpu(weight_tensor);

    uint32_t idx = 0;
    uint32_t global_0 = UP_DIV(ocl_fc_param->num_output, 4);
    ocl_fc_kernel.setArg(idx++, global_0);
    TLOG_ERR("%lld \n", handle_input);
    ocl_fc_kernel.setArg(idx++, *(cl::Image2D*)handle_input);
    ocl_fc_kernel.setArg(idx++, *gpu_weight);
    ocl_fc_kernel.setArg(idx++, *gpu_bias);
    ocl_fc_kernel.setArg(idx++, *(cl::Image2D*)handle_output);
    ocl_fc_kernel.setArg(idx, UP_DIV(weight_tensor->dims[1], 4));

    global_work_size = {global_0};
    local_work_size = {max_work_group_size};
}
void ocl_fc::run(struct subgraph* subgraph)
{
#ifdef OPENCL_PROFILE_TIME
    cl::Event event;
    uint32_t final_global_size = ROUND_UP(global_work_size[0], local_work_size[0]);
    engine->get_command_queue().enqueueNDRangeKernel(ocl_fc_kernel, cl::NullRange, {final_global_size},
                                                     {local_work_size[0]}, nullptr, &event);
    TLOG_ERR("cost: %d %s \n", (int)engine->get_cost_time(&event), ir_node->name);
//    debug_data();
#else
    engine->get_command_queue().enqueueNDRangeKernel(ocl_fc_kernel, cl::NullRange, {ROUND_UP(global_work_size[0], local_work_size[0])},
                                                     {local_work_size[0]}, nullptr, nullptr);
#endif
}

class ocl_fc_creator : public ocl_node_creator
{
public:
    ocl_node* creator(OCLEngine* engine, struct node* ir_node) override
    {
        return new ocl_fc(engine, ir_node);
    }
};

REGISTER_OCL_OP(OP_FC, ocl_fc_creator);
