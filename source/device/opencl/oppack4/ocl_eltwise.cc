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

#include "ocl_eltwise.hpp"
extern "C" {
#include "eltwise_param.h"
}

//ELT_PROD,
//ELT_PROD_SCALAR,
//ELT_SUM,
//ELT_SUM_SCALAR,
//ELT_SUB,
//ELT_SUB_SCALAR,
//ELT_MAX,
//ELT_RSQRT,
//ELT_MIN_SCALAR,
//ELT_LAST,
//ELT_DIV,
//ELT_LOG,
//ELT_EXP,
//ELT_SQRT,
//ELT_FLOOR,
//ELT_SQUARE,
//ELT_POW,
//ELT_POWER,

ocl_eltwise::ocl_eltwise(OCLEngine* engine, struct node* ir_node)
    : ocl_node(engine, ir_node)
{
    auto param = (eltwise_param*)ir_node->op.param_mem;
    std::set<std::string> build_options;
    std::string operator_str;
    switch (param->type)
    {
    case ELT_PROD:
        operator_str = "in0+in1";
        break;
    case ELT_PROD_SCALAR:
        break;
    case ELT_SUM:
        operator_str = "in0+in1";
        break;
    case ELT_SUM_SCALAR:
        break;
    case ELT_SUB:
        operator_str = "in0-in1";
        break;
    case ELT_SUB_SCALAR:
        break;
    case ELT_MAX:
        operator_str = "in0>in1?in0:in1";
        break;
    case ELT_RSQRT:
        break;
    case ELT_MIN_SCALAR:
        break;
    case ELT_LAST:
        break;
    case ELT_DIV:
        break;
    case ELT_LOG:
        break;
    case ELT_EXP:
        break;
    case ELT_SQRT:
        break;
    case ELT_FLOOR:
        break;
    case ELT_SQUARE:
        break;
    case ELT_POW:
        break;
    case ELT_POWER:
        break;
    default:
        TLOG_ERR("do not support type:%d", param->type);
        break;
    }
    if (operator_str.empty())
    {
        TLOG_ERR("todo support type:%d", param->type);
    }
    build_options.emplace("-DOPERATOR=" + operator_str);
    elt_kernel = engine->build_kernel("binary", "binary", build_options);
    max_work_group_size = engine->get_max_work_group_size(elt_kernel);
}
void ocl_eltwise::pre_run()
{
    int full_count[2] = {1, 1};

    auto output_tensor = get_ir_graph_tensor(ir_node->graph, ir_node->output_tensors[0]);
    global_work_size = {(uint32_t)UP_DIV(output_tensor->dims[1], 4) * output_tensor->dims[3], (uint32_t)output_tensor->dims[2]};
    uint64_t handle_input = engine->get_gpu_mem_by_idx(ir_node->input_tensors[0]);
    uint64_t handle_input1 = engine->get_gpu_mem_by_idx(ir_node->input_tensors[1]);
    uint64_t handle_output = engine->get_gpu_mem_by_idx(ir_node->output_tensors[0]);
    int shape[] = {output_tensor->dims[0], output_tensor->dims[2], output_tensor->dims[3], UP_DIV(output_tensor->dims[1], 4)};
    uint32_t idx = 0;
    elt_kernel.setArg(idx++, global_work_size[0]);
    elt_kernel.setArg(idx++, global_work_size[1]);
    elt_kernel.setArg(idx++, *(cl::Image*)handle_input);
    elt_kernel.setArg(idx++, *(cl::Image*)handle_input1);
    elt_kernel.setArg(idx++, *(cl::Image*)handle_output);
    elt_kernel.setArg(idx++, shape);
    elt_kernel.setArg(idx, full_count);

    local_work_size = find_local_group_2d(global_work_size, max_work_group_size, engine, elt_kernel, ir_node->name);
}
void ocl_eltwise::run(struct subgraph* subgraph)
{
#ifdef OPENCL_PROFILE_TIME
    cl::Event event;
    run_node_2d(global_work_size, local_work_size, elt_kernel, &event);
    TLOG_ERR("cost: %d %s \n", (int)engine->get_cost_time(&event), ir_node->name);
#else
    run_node_2d(global_work_size, local_work_size, elt_kernel);
#endif
#ifdef OPENCL_DEBUG_DATA
    debug_data();
#endif
}

class ocl_elewise_creator : public ocl_node_creator
{
    ocl_node* creator(OCLEngine* engine, struct node* ir_node) override
    {
        return new ocl_eltwise(engine, ir_node);
    }
};

REGISTER_OCL_OP(OP_ELTWISE, ocl_elewise_creator);