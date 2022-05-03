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

#include "ocl_relu.hpp"
#include "ocl_executor.hpp"

extern "C" {
#include "relu_param.h"
#include "operator/op.h"
}

void ocl_relu::run(struct subgraph* subgraph)
{
#ifdef OPENCL_PROFILE_TIME
    cl::Event event;
    run_node_3d(global_work_size, local_work_size, leaky_relu_kernel, &event);
    TLOG_ERR("cost: %d %s \n", (int)engine->get_cost_time(&event), ir_node->name);
#else
    run_node_3d(global_work_size, local_work_size, leaky_relu_kernel);
#endif
#ifdef OPENCL_DEBUG_DATA
    debug_data();
#endif
}
ocl_relu::ocl_relu(OCLEngine* engine, struct node* ir_node)
    : ocl_node(engine, ir_node)
{
    auto type = ir_node->op.type;

    auto* relu_param = (struct relu_param*)ir_node->op.param_mem;
    std::set<std::string> build_options;
    std::string compute;
    char tmp_str[30] = {};
    if (type == OP_RELU1)
    {
        float min = 0.0f;
        float max = 1.0f;
        std::string format = "clamp(in,(FLOAT4)((FLOAT)%f),(FLOAT4)((FLOAT)%f))";
        sprintf(tmp_str, format.c_str(), min, max);
        compute = tmp_str;
    }
    else if (type == OP_RELU6)
    {
        float min = 0.0f;
        float max = 6.0f;
        std::string format = "clamp(in,(FLOAT4)((FLOAT)%f),(FLOAT4)((FLOAT)%f))";
        sprintf(tmp_str, format.c_str(), min, max);
        compute = tmp_str;
    }
    else if (type == OP_RELU && relu_param->negative_slope == 0.0f)
    {
        compute = "fmax(in,(FLOAT4)((FLOAT)0))";
    }
    else if (type == OP_RELU && relu_param->negative_slope > 0.0f)
    {
        sprintf(tmp_str, "%.8f", relu_param->negative_slope);
        std::string _slop_str = tmp_str;
        compute = "select((FLOAT)(" + _slop_str + "f)*in,in,in>=(FLOAT4)((FLOAT)0))";
    }
    else
    {
        TLOG_ERR("error in ocl_relu::ocl_relu() \n");
    }
    build_options.emplace(" -DOPERATOR=" + compute);
    leaky_relu_kernel = engine->build_kernel("unary", "unary", build_options);
    max_work_group_size = engine->get_max_work_group_size(leaky_relu_kernel);
}

void ocl_relu::pre_run()
{
    struct graph* ir_graph = ir_node->graph;

    int ir_tensor_idx_input = ir_node->input_tensors[0];
    int ir_tensor_idx_output = ir_node->output_tensors[0];
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_tensor_idx_input);

    uint64_t handle_input = engine->get_gpu_mem_by_idx(ir_tensor_idx_input);
    uint64_t handle_output = engine->get_gpu_mem_by_idx(ir_tensor_idx_output);

    int output_channel = output_tensor->dims[1];
    int output_height = output_tensor->dims[2];
    int output_width = output_tensor->dims[3];
    int output_channel_block = UP_DIV(output_channel, 4);
    global_work_size = {
        (uint32_t)output_channel_block, (uint32_t)output_width, (uint32_t)output_height};

    uint32_t idx = 0;
    leaky_relu_kernel.setArg(idx++, global_work_size[0]);
    leaky_relu_kernel.setArg(idx++, global_work_size[1]);
    leaky_relu_kernel.setArg(idx++, global_work_size[2]);
    leaky_relu_kernel.setArg(idx++, *(cl::Image*)handle_input);
    leaky_relu_kernel.setArg(idx, *(cl::Image*)handle_output);
    local_work_size = find_local_group_3d(global_work_size, max_work_group_size, engine, leaky_relu_kernel, ir_node->name);
}

class ocl_relu_creator : public ocl_node_creator
{
public:
    ocl_node* creator(OCLEngine* engine, struct node* ir_node) override
    {
        return new ocl_relu(engine, ir_node);
    }
};

REGISTER_OCL_OP(OP_RELU, ocl_relu_creator)
REGISTER_OCL_OP(OP_RELU6, ocl_relu_creator)
REGISTER_OCL_OP(OP_RELU1, ocl_relu_creator)
