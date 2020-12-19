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
 * Copyright (c) 2020, OPEN AI LAB
 * Author: lswang@openailab.com
 */

#include "cpu_allocator.h"

#include <string.h>

#include "vector.h"
#include "tengine_ir.h"
#include "tengine_op.h"
#include "tengine_errno.h"
#include "tengine_log.h"


static int cpu_allocate(struct dev_allocator* allocator, struct subgraph* sub_graph)
{
    if (!strcmp(CPU_DEV_NAME, allocator->name))
    {
        #if MACOS

        #else
        set_tengine_errno(EBADSLT);
        #endif
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


static int cpu_describe(struct dev_allocator* allocator, struct vector* allowed_ops, struct vector* blocked_ops, struct vector* precision)
{
    if (NULL == allocator) return -1;
    if (!strcmp(allocator->name, CPU_DEV_NAME)) return -1;

    if (NULL == allowed_ops)
    {
        TLOG_ERR("Error: Allowed op list pointer is NULL\n");
    }
    if (NULL == blocked_ops)
    {
        TLOG_ERR("Error: Allowed op list pointer is NULL\n");
    }

    for (int i = OP_GENERIC + 1; i < OP_BUILTIN_LAST - 1; i++)
    {
        push_vector_data(allowed_ops, &i);
    }

    int precision_var = TENGINE_DT_FP32;
    push_vector_data(precision, &precision_var);
    precision_var = TENGINE_DT_FP16;
    push_vector_data(precision, &precision_var);
    precision_var = TENGINE_DT_UINT8;
    push_vector_data(precision, &precision_var);

    return 0;
}


static int cpu_evaluation(struct dev_allocator* allocator, struct subgraph* sub_graph, struct vector* tensor, struct vector* node)
{
    if (NULL == allocator) return -1;
    if (!strcmp(allocator->name, CPU_DEV_NAME)) return -1;

    (void)sub_graph;
    (void)tensor;
    (void)node;

    return 0;
}


static int cpu_release(struct dev_allocator* allocator, struct subgraph* sub_graph)
{
    if (NULL == allocator) return -1;
    if (!strcmp(allocator->name, CPU_DEV_NAME)) return -1;

    (void)sub_graph;

    return 0;
}


static struct dev_allocator cpu_allocator = {
    .name = CPU_DEV_NAME,
    .describe = cpu_describe,
    .evaluation = cpu_evaluation,
    .allocate = cpu_allocate,
    .release = cpu_release,
};


#ifndef STANDLONE_MODE
REGISTER_DEV_ALLOCATOR(register_cpu_allocator);
#endif


#ifdef STANDLONE_MODE
void register_cpu_allocator(void)
#else
static void register_cpu_allocator(void)
#endif
{
    init_allocator_registry(&cpu_allocator);
}
