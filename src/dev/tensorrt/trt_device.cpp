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
 * Copyright (c) 2021, Open AI Lab
 * Author: lswang@openailab.com
 */

extern "C"
{
#include "vector.h"
#include "nn_device.h"
#include "tengine_ir.h"
#include "tengine_log.h"
#include "tengine_errno.h"
#include "dev_allocator.h"
#include "tengine_c_api.h"
}

#include "trt_device.hpp"

#include "trt_limit.hpp"
#include "trt_graph.hpp"


extern "C"
{
int trt_describe(struct dev_allocator* allocator, struct vector* allowed_ops, struct vector* blocked_ops, struct vector* precision);
int trt_evaluation(struct dev_allocator* allocator, struct subgraph* sub_graph, struct vector* evolution_tensors, struct vector* evolution_nodes);
int trt_allocate(struct dev_allocator* allocator, struct subgraph* sub_graph);
int trt_release(struct dev_allocator* allocator, struct subgraph* sub_graph);
}


int trt_describe(struct dev_allocator* allocator, struct vector* allowed_ops, struct vector* blocked_ops, struct vector* precision)
{
    (void)allocator;

    for (int op_type : trt_supported_ops)
    {
        push_vector_data(allowed_ops, &op_type);
    }

    for (int i = 0, j = 0; i < OP_BUILTIN_LAST; i++)
    {
        int op_type = trt_supported_ops[j];
        if (op_type != i)
        {
            push_vector_data(blocked_ops, &i);
        }
        else
        {
            if (j < sizeof(trt_supported_ops) / sizeof(trt_supported_ops[0]))
                j++;
        }
    }

    int precision_var = TENGINE_DT_UINT8;
    push_vector_data(precision, &precision_var);
    precision_var = TENGINE_DT_FP16;
    push_vector_data(precision, &precision_var);
    precision_var = TENGINE_DT_FP32;
    push_vector_data(precision, &precision_var);

    return 0;
}


int trt_evaluation(struct dev_allocator* allocator, struct subgraph* sub_graph, struct vector* evolution_tensors, struct vector* evolution_nodes)
{
    // nothing to do with tensorrt
    (void)allocator;
    (void)sub_graph;
    (void)evolution_tensors;
    (void)evolution_nodes;

    return 0;
}


int trt_allocate(struct dev_allocator* allocator, struct subgraph* sub_graph)
{
    if (nullptr == allocator)
    {
        set_tengine_errno(EBADSLT);
        return -1;
    }

    if (!strcmp(TRT_DEV_NAME, allocator->name))
    {
        set_tengine_errno(EBADSLT);
        return -1;
    }

    /* set the correct input wait count: INPUT tensor is always ready */
    sub_graph->input_wait_count = 0;

    for (int i = 0; i < sub_graph->input_num; i++)
    {
        struct ir_tensor* tensor = get_ir_graph_tensor(sub_graph->graph, sub_graph->input_tensor_list[i]);

        if (tensor->tensor_type == TENSOR_TYPE_VAR)
            sub_graph->input_wait_count++;
    }

    return 0;
}


int trt_release(struct dev_allocator* allocator, struct subgraph* sub_graph)
{
    (void)sub_graph;

    if (nullptr == allocator || !strcmp(TRT_DEV_NAME, allocator->name))
    {
        return -1;
    }

    return 0;
}


extern "C"
{
static struct trt_device nvtrt_dev = {
    .base = {
            .name = TRT_DEV_NAME,
            .init = trt_dev_init,
            .prerun = trt_dev_prerun,
            .run = trt_dev_run,
            .postrun = trt_dev_postrun,
            .async_run = nullptr,
            .async_wait = nullptr,
            .release = trt_dev_release,
            .release_exec_graph = nullptr,},
        .load_graph = nullptr,
        .load_ir_graph = nullptr,
        .unload_graph = nullptr,
};


static struct dev_allocator trt_allocator = {
    .name = TRT_DEV_NAME,
    .describe = trt_describe,
    .evaluation = trt_evaluation,
    .allocate = trt_allocate,
    .release = trt_release,
};


int register_trt_device(void)
{
    TLOG_INFO("Tengine plugin device %s is registered.\n", nvtrt_dev.base.name);
    return register_nn_device(&nvtrt_dev.base);
}



#ifdef STANDLONE_MODE
void register_trt_allocator(void)
#else
static void register_trt_allocator(void)
#endif
{
    TLOG_INFO("Tengine plugin allocator %s is registered.\n", trt_allocator.name);
    init_allocator_registry(&trt_allocator);
}


#ifndef STANDLONE_MODE
REGISTER_NN_DEVICE(&nvtrt_dev.base);
REGISTER_DEV_ALLOCATOR(register_trt_allocator);
#endif
}
