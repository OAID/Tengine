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

#include "acl_device.hpp"

#include "acl_limit.hpp"
#include "acl_graph.hpp"


extern "C"
{
int acl_describe(struct dev_allocator* allocator, struct vector* allowed_ops, struct vector* blocked_ops, struct vector* precision);
int acl_evaluation(struct dev_allocator* allocator, struct subgraph* sub_graph, struct vector* evolution_tensors, struct vector* evolution_nodes);
int acl_allocate(struct dev_allocator* allocator, struct subgraph* sub_graph);
int acl_release(struct dev_allocator* allocator, struct subgraph* sub_graph);
}


int acl_describe(struct dev_allocator* allocator, struct vector* allowed_ops, struct vector* blocked_ops, struct vector* precision)
{
    ( void )allocator;

    for (int op_type : acl_supported_ops)
    {
        push_vector_data(allowed_ops, &op_type);
    }

    for (int i = 0, j = 0; i < OP_BUILTIN_LAST; i++)
    {
        int op_type = acl_supported_ops[j];
        if (op_type != i)
        {
            push_vector_data(blocked_ops, &i);
        }
        else
        {
            if (j < (sizeof(acl_supported_ops) / sizeof(acl_supported_ops[0])))
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

int acl_evaluation(struct dev_allocator* allocator, struct subgraph* sub_graph, struct vector* evolution_tensors, struct vector* evolution_nodes)
{
    // nothing to do with tensorrt
    ( void )allocator;
    ( void )sub_graph;
    ( void )evolution_tensors;
    ( void )evolution_nodes;

    return 0;
}

int acl_allocate(struct dev_allocator* allocator, struct subgraph* sub_graph)
{
    if (nullptr == allocator)
    {
        set_tengine_errno(EBADSLT);
        return -1;
    }

    if (!strcmp(ACL_DEV_NAME, allocator->name))
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

int acl_release(struct dev_allocator* allocator, struct subgraph* sub_graph)
{
    (void)sub_graph;

    if (nullptr == allocator || !strcmp(ACL_DEV_NAME, allocator->name))
    {
        return -1;
    }

    return 0;
}

extern "C"
{
static struct acl_device acl_dev = {
    .base = {
            .name = ACL_DEV_NAME,
            .init = acl_dev_init,
            .prerun = acl_dev_prerun,
            .run = acl_dev_run,
            .postrun = acl_dev_postrun,
            .async_run = nullptr,
            .async_wait = nullptr,
            .release = acl_dev_release,
            .release_exec_graph = nullptr,},
        .load_graph = nullptr,
        .load_ir_graph = nullptr,
        .unload_graph = nullptr,
};

static struct dev_allocator acl_allocator = {
    .name = ACL_DEV_NAME,
    .describe = acl_describe,
    .evaluation = acl_evaluation,
    .allocate = acl_allocate,
    .release = acl_release,
};


int register_acl_device(void)
{
    TLOG_INFO("Tengine plugin device %s is registered.\n", acl_dev.base.name);
    return register_nn_device(&acl_dev.base);
}

#ifdef STANDLONE_MODE
void register_acl_allocator(void)
#else
static void register_acl_allocator(void)
#endif
{
    TLOG_INFO("Tengine plugin allocator %s is registered.\n", acl_allocator.name);
    init_allocator_registry(&acl_allocator);
}

#ifndef STANDLONE_MODE
REGISTER_NN_DEVICE(&acl_dev.base);
REGISTER_DEV_ALLOCATOR(register_acl_allocator);
#endif
}
