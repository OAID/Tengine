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

#include "ocl_pool.hpp"
#include "ocl_executor.hpp"
ocl_pool::ocl_pool(OCLEngine* engine, struct node* ir_node)
    : ocl_node(engine, ir_node)
{
    struct pool_param* param = (struct pool_param*)ir_node->op.param_mem;
    this->pooling_param = param;
    std::set<std::string> build_option;
    if (this->pooling_param->pool_method == POOL_AVG){
        build_option.emplace("-DPOOL_AVG");
    }
    pooling_kernel = engine->build_kernel("pooling", "pooling", build_option);
    max_work_group_size = (int)engine->get_max_work_group_size(pooling_kernel);
}

void ocl_pool::pre_run()
{
    struct graph* ir_graph = ir_node->graph;

    int ir_tensor_idx_input = ir_node->input_tensors[0];
    int ir_tensor_idx_output = ir_node->output_tensors[0];

    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_tensor_idx_input);
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_tensor_idx_output);

    uint64_t handle_input = engine->get_gpu_mem_by_idx(ir_tensor_idx_input);
    uint64_t handle_output = engine->get_gpu_mem_by_idx(ir_tensor_idx_output);

    int global0 = UP_DIV(output_tensor->dims[1], 4);
    int global1 = output_tensor->dims[3];
    int global2 = output_tensor->dims[2];
    global_work_size = {(uint32_t)global0, (uint32_t)global1, (uint32_t)global2};
    //local_work_size = pool_local_work_size(global_work_size, max_work_group_size);

    int input_shape[2] = {input_tensor->dims[2], input_tensor->dims[3]};
    int padding_shape[2] = {pooling_param->pad_h0, pooling_param->pad_w0};
    int stride_shape[2] = {pooling_param->stride_h, pooling_param->stride_w};
    int kernel_shape[2] = {pooling_param->kernel_h, pooling_param->kernel_w};

    uint32_t idx = 0;
    pooling_kernel.setArg(idx++, global_work_size[0]);
    pooling_kernel.setArg(idx++, global_work_size[1]);
    pooling_kernel.setArg(idx++, global_work_size[2]);
    pooling_kernel.setArg(idx++, *(cl::Image*)handle_input);
    pooling_kernel.setArg(idx++, sizeof(input_shape), input_shape);
    pooling_kernel.setArg(idx++, output_tensor->dims[2]);
    pooling_kernel.setArg(idx++, sizeof(padding_shape), padding_shape);
    pooling_kernel.setArg(idx++, sizeof(stride_shape), stride_shape);
    pooling_kernel.setArg(idx++, sizeof(kernel_shape), kernel_shape);
    pooling_kernel.setArg(idx++, *(cl::Image*)handle_output);
    local_work_size = find_local_group_3d(global_work_size, max_work_group_size, engine, pooling_kernel, ir_node->name);
}

void ocl_pool::run(struct subgraph* subgraph)
{
#ifdef OPENCL_PROFILE_TIME
    cl::Event event;
    run_node_3d(global_work_size, local_work_size, pooling_kernel, &event);
    TLOG_ERR("cost: %d %s \n", (int)engine->get_cost_time(&event), ir_node->name);
//
//    debug_data();
#else
    run_node_3d(global_work_size, local_work_size, pooling_kernel);
#endif
}

std::vector<uint32_t> ocl_pool::pool_local_work_size(const std::vector<uint32_t>& gws, const uint32_t max_work_group_size)
{
    std::vector<uint32_t> lws(3, 0);
    auto max_work_item_size = engine->get_max_work_item_sizes();
    uint32_t device_compute_unit = engine->gpu_compute_unit;
    int coreNum = device_compute_unit;
    for (int i = 0, totalSizeNow = 1; i < gws.size(); ++i)
    {
        int remain = gws[i] % coreNum, groupSize = gws[i] / coreNum;
        if (remain == 0)
        {
            lws[i] = groupSize;
        }
        else
        {
            while (groupSize)
            {
                int remain = gws[i] % groupSize;
                if (remain == 0 && (i > 0 || groupSize <= max_work_group_size))
                {
                    lws[i] = groupSize;
                    break;
                }
                --groupSize;
            }
        }
        int limit = std::min<uint32_t>(max_work_group_size / totalSizeNow, max_work_item_size[i]);
        lws[i] = std::max<uint32_t>(std::min<uint32_t>(lws[i], limit), 1);
        totalSizeNow *= lws[i];
    }
    return lws;
}

class ocl_pool_creator : public ocl_node_creator
{
public:
    ocl_node* creator(OCLEngine* engine, struct node* ir_node) override
    {
        //pre decide which conv way
        return new ocl_pool(engine, ir_node);
    }
};

REGISTER_OCL_OP(OP_POOL, ocl_pool_creator);
