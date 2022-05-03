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

#include "ocl_upsample.hpp"
#include "ocl_executor.hpp"

extern "C" {
#include "upsample_param.h"
#include "operator/op.h"
}

ocl_upsample::ocl_upsample(OCLEngine* engine, struct node* ir_node)
    : ocl_node(engine, ir_node)
{
    std::set<std::string> build_options;
    ocl_upsample_kernel = engine->build_kernel("upsample", "Nearest", build_options);
    max_work_group_size = engine->get_max_work_group_size(ocl_upsample_kernel);
}
void ocl_upsample::pre_run()
{
    struct graph* ir_graph = ir_node->graph;

    int ir_tensor_idx_input = ir_node->input_tensors[0];
    int ir_tensor_idx_output = ir_node->output_tensors[0];

    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_tensor_idx_input);
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_tensor_idx_input);

    uint64_t handle_input = engine->get_gpu_mem_by_idx(ir_tensor_idx_input);
    uint64_t handle_output = engine->get_gpu_mem_by_idx(ir_tensor_idx_output);

    int out_width = output_tensor->dims[3];
    int out_height = output_tensor->dims[2];
    int out_channel = output_tensor->dims[1];

    int input_width = input_tensor->dims[3];
    int input_height = input_tensor->dims[2];

    global_work_size = {UP_DIV((uint32_t)out_channel, 4) * out_width, (uint32_t)out_height};

    float height_scale = (float)input_height / (float)out_height;
    float width_scale = (float)input_width / (float)out_width;

    uint32_t idx = 0;
    ocl_upsample_kernel.setArg(idx++, global_work_size[0]);
    ocl_upsample_kernel.setArg(idx++, global_work_size[1]);
    ocl_upsample_kernel.setArg(idx++, *(cl::Image*)handle_input);
    ocl_upsample_kernel.setArg(idx++, *(cl::Image*)handle_output);
    ocl_upsample_kernel.setArg(idx++, height_scale);
    ocl_upsample_kernel.setArg(idx++, width_scale);
    ocl_upsample_kernel.setArg(idx++, input_height);
    ocl_upsample_kernel.setArg(idx++, input_width);
    ocl_upsample_kernel.setArg(idx++, out_height);
    ocl_upsample_kernel.setArg(idx++, out_height);

    local_work_size = find_local_group_2d(global_work_size, max_work_group_size, engine, ocl_upsample_kernel, ir_node->name);
}

void ocl_upsample::run(struct subgraph* subgraph)
{
#ifdef OPENCL_PROFILE_TIME
    cl::Event event;
    run_node_2d(global_work_size, local_work_size, ocl_upsample_kernel, &event);
    int cost_time = (int)engine->get_cost_time(&event);
    TLOG_ERR("cost:%d %s \n", cost_time, ir_node->name);
#else
    run_node_2d(global_work_size, local_work_size, ocl_upsample_kernel);
#endif
}

class ocl_upsample_creator : public ocl_node_creator
{
public:
    ocl_node* creator(OCLEngine* engine, struct node* ir_node) override
    {
        //pre decide which conv way
        return new ocl_upsample(engine, ir_node);
    }
};

REGISTER_OCL_OP(OP_UPSAMPLE, ocl_upsample_creator);
REGISTER_OCL_OP(OP_INTERP, ocl_upsample_creator)
