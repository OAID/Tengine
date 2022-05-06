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

#include "ocl_concat.hpp"
#include "ocl_cpp_helper.hpp"
#include "ocl_executor.hpp"

extern "C" {
#include "concat_param.h"
}

ocl_concat::ocl_concat(OCLEngine* engine, node* ir_node)
    : ocl_node(engine, ir_node)
{
    for (int i = 0; i < ir_node->input_num - 1; ++i)
    {
        auto temp_tensor = get_ir_graph_tensor(ir_node->graph, ir_node->input_tensors[i]);
        if (temp_tensor->dims[1] % 4 != 0)
        {
            use_image_anyway = false;
        }
    }

    if (ir_node->input_num == 2)
    {
        if (use_image_anyway)
        {
            type_concat = 0;
        }
        else
        {
            type_concat = 1;
        }
    }
    else
    {
        if (use_image_anyway)
        {
            type_concat = 2;
        }
        else
        {
            type_concat = 3;
        }
    }
    if (ir_node->input_num > 1)
    {
        build_kernel();
    }
}

void ocl_concat::build_kernel()
{
    if (type_concat == 0)
    {
        std::set<std::string> build_options;
        concat_kernel = engine->build_kernel("concat", "ConcatChannel4X", build_options);
        max_group_work_item = engine->get_max_work_group_size(concat_kernel);
    }
    else if (type_concat == 1)
    {
        std::set<std::string> build_options;
        concat_kernel = engine->build_kernel("concat", "ConcatChannel", build_options);
        max_group_work_item = engine->get_max_work_group_size(concat_kernel);
    }
    else if (type_concat == 2)
    {
        concat_multi_kernels.resize(ir_node->input_num);
        concat_multi_max_group_size.resize(ir_node->input_num);
        std::set<std::string> build_options;
        for (int i = 0; i < ir_node->input_num; ++i)
        {
            concat_multi_kernels[i] = engine->build_kernel("copy", "CopyImage", build_options);
            concat_multi_max_group_size[i] = engine->get_max_work_group_size(concat_multi_kernels[i]);
        }
    }
}

void ocl_concat::pre_run()
{
    if (ir_node->input_num == 1)
    {
        int ir_tensor_idx_output = ir_node->output_tensors[0];
        int ir_tensor_idx_input = ir_node->input_tensors[0];
        uint64_t handle_input = engine->get_gpu_mem_by_idx(ir_tensor_idx_input);
        engine->set_gpu_mem_by_idx(ir_tensor_idx_output, handle_input);
        return;
    }
    if (type_concat == 0)
    {
        pre_run_type_concat_0();
    }
    else if (type_concat == 1)
    {
        //todo
    }
    else if (type_concat == 2)
    {
        pre_run_type_concat_2();
    }
    else if (type_concat == 3)
    {
        // todo
    }
}

void ocl_concat::run(struct subgraph* subgraph)
{
    if (ir_node->input_num == 1)
    {
        return;
    }
    if (type_concat == 0)
    {
        run_type_concat_0();
    }
    else if (type_concat == 2)
    {
        run_type_concat_2();
    }

#ifdef OPENCL_DEBUG_DATA
    debug_data();
#endif
}

void ocl_concat::pre_run_type_concat_0()
{
    struct graph* ir_graph = ir_node->graph;

    int ir_tensor_idx_input = ir_node->input_tensors[0];
    int ir_tensor_idx_input_1 = ir_node->input_tensors[1];
    int ir_tensor_idx_output = ir_node->output_tensors[0];

    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_tensor_idx_input);
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_tensor_idx_output);

    uint64_t handle_input = engine->get_gpu_mem_by_idx(ir_tensor_idx_input);
    uint64_t handle_input1 = engine->get_gpu_mem_by_idx(ir_tensor_idx_input_1);
    uint64_t handle_output = engine->get_gpu_mem_by_idx(ir_tensor_idx_output);

    int out_width = output_tensor->dims[3];
    int out_height = output_tensor->dims[2];
    int out_channel = output_tensor->dims[1];

    global_work_size = {UP_DIV((uint32_t)out_channel, 4), (uint32_t)out_width, (uint32_t)out_height};

    uint32_t idx = 0;
    concat_kernel.setArg(idx++, global_work_size[0]);
    concat_kernel.setArg(idx++, global_work_size[1]);
    concat_kernel.setArg(idx++, global_work_size[2]);
    concat_kernel.setArg(idx++, *(cl::Image*)handle_input);
    concat_kernel.setArg(idx++, *(cl::Image*)handle_input1);
    concat_kernel.setArg(idx++, input_tensor->dims[1]);
    concat_kernel.setArg(idx++, output_tensor->dims[1]);
    concat_kernel.setArg(idx, *(cl::Image*)handle_output);

    local_work_size = find_local_group_3d(global_work_size, max_group_work_item, engine, concat_kernel, ir_node->name);
}

void ocl_concat::pre_run_type_concat_2()
{
    auto output_tensor = get_ir_graph_tensor(ir_node->graph, ir_node->output_tensors[0]);
    uint64_t handle_output = engine->get_gpu_mem_by_idx(ir_node->output_tensors[0]);
    int input_offset[] = {0, 0, 0, 0};
    int output_offset[] = {0, 0, 0, 0};
    int output_wh[] = {output_tensor->dims[3], output_tensor->dims[2]};
    concat_multi_local_size.resize(concat_multi_kernels.size());
    concat_multi_global_size.resize(concat_multi_kernels.size());
    for (int i = 0; i < concat_multi_kernels.size(); ++i)
    {
        auto input_tensor = get_ir_graph_tensor(ir_node->graph, ir_node->input_tensors[i]);
        uint64_t handle_input = engine->get_gpu_mem_by_idx(ir_node->input_tensors[i]);
        int input_wh[] = {input_tensor->dims[3], input_tensor->dims[2]};
        int input_region[] = {input_tensor->dims[0], UP_DIV(input_tensor->dims[1], 4), input_tensor->dims[2], input_tensor->dims[3]};

        concat_multi_local_size[i] = {std::max((uint32_t)1, concat_multi_max_group_size[i] / 16), 16};
        concat_multi_global_size[i] = {(uint32_t)UP_DIV(input_tensor->dims[1], 4) * input_tensor->dims[3], (uint32_t)input_tensor->dims[2]};

        auto& kernel = concat_multi_kernels[i];
        uint32_t idx = 0;
        kernel.setArg(idx++, concat_multi_global_size[i][0]);
        kernel.setArg(idx++, concat_multi_global_size[i][1]);
        kernel.setArg(idx++, *(cl::Image*)handle_input);
        kernel.setArg(idx++, *(cl::Image*)handle_output);
        kernel.setArg(idx++, input_offset);
        kernel.setArg(idx++, output_offset);
        kernel.setArg(idx++, input_wh);
        kernel.setArg(idx++, output_wh);
        kernel.setArg(idx, input_wh);

        output_offset[1] += input_region[1];
    }
}
void ocl_concat::run_type_concat_0()
{
#ifdef OPENCL_PROFILE_TIME
    cl::Event event;
    run_node_3d(global_work_size, local_work_size, concat_kernel, &event);
    TLOG_ERR("cost: %d %s \n", (int)engine->get_cost_time(&event), ir_node->name);
#else
    run_node_3d(global_work_size, local_work_size, concat_kernel);
#endif
}

void ocl_concat::run_type_concat_2()
{
#ifdef OPENCL_PROFILE_TIME
    cl::Event event;
#endif
    for (int i = 0; i < concat_multi_kernels.size(); ++i)
    {
#ifdef OPENCL_PROFILE_TIME
        run_node_2d(concat_multi_global_size[i], concat_multi_local_size[i], concat_multi_kernels[i], &event);
        TLOG_ERR("cost: %d %s %d\n", (int)engine->get_cost_time(&event), ir_node->name, i);
#else
        run_node_2d(concat_multi_global_size[i], concat_multi_local_size[i], concat_multi_kernels[i], nullptr);
#endif
    }
}

class ocl_concat_creator : public ocl_node_creator
{
public:
    ocl_node* creator(OCLEngine* engine, struct node* ir_node) override
    {
        //pre decide which conv way
        return new ocl_concat(engine, ir_node);
    }
};

REGISTER_OCL_OP(OP_CONCAT, ocl_concat_creator);
