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
 * Author: haitao@openailab.com
 */

#include <stdio.h>

#include "sys_port.h"
#include "tengine_errno.h"
#include "tengine_log.h"
#include "vector.h"
#include "tengine_ir.h"
#include "tengine_op.h"
#include "cpu_node_ops.h"

static struct vector** builtin_ops_registry;
static struct vector* custom_ops_registry;

struct custom_reg_entry
{
    int op_type;
    struct node_ops* node_ops;
};

static int init_builtin_ops_registry(void)
{
    int alloc_num = 0;

    builtin_ops_registry = ( struct vector** )sys_malloc(sizeof(void*) * OP_BUILTIN_LAST);

    if (builtin_ops_registry == NULL)
        return -1;

    for (int i = 0; i < OP_BUILTIN_LAST; i++)
    {
        builtin_ops_registry[i] = create_vector(sizeof(struct node_ops*), NULL);

        if (builtin_ops_registry[i] == NULL)
        {
            alloc_num = i;
            goto error;
        }
    }

    return 0;

error:
    for (int i = 0; i < alloc_num; i++)
    {
        release_vector(builtin_ops_registry[i]);
    }

    sys_free(builtin_ops_registry);

    builtin_ops_registry = NULL;

    return -1;
}

static void release_builtin_ops_registry(void)
{
    for (int i = 0; i < OP_BUILTIN_LAST; i++)
    {
        release_vector(builtin_ops_registry[i]);
    }

    sys_free(builtin_ops_registry);
}

int register_builtin_node_ops(int op_type, struct node_ops* node_ops)
{
    if (op_type < OP_GENERIC || op_type >= OP_BUILTIN_LAST)
        return -1;

    struct vector* ops_vector = builtin_ops_registry[op_type];

    if (push_vector_data(ops_vector, &node_ops) < 0)
        return -1;

    return 0;
}

int unregister_builtin_node_ops(int op_type, struct node_ops* node_ops)
{
    if (op_type < OP_GENERIC || op_type >= OP_BUILTIN_LAST)
        return -1;

    struct vector* ops_vector = builtin_ops_registry[op_type];

    if (remove_vector_data(ops_vector, &node_ops) < 0)
        return -1;

    return 0;
}

static inline struct node_ops* find_builtin_node_ops(struct exec_graph* exec_graph, struct ir_node* ir_node)
{
    int op_type = ir_node->op.op_type;

    struct vector* ops_vector = builtin_ops_registry[op_type];

    int num = get_vector_num(ops_vector);

    int max_score = 0;
    struct node_ops* selected_ops = NULL;

    for (int i = 0; i < num; i++)
    {
        struct node_ops* node_ops = *( struct node_ops** )get_vector_data(ops_vector, i);

        int score = node_ops->score(node_ops, exec_graph, ir_node);

        if (score > max_score)
        {
            selected_ops = node_ops;
            max_score = score;
        }
    }

    return selected_ops;
}

static int init_custom_ops_registry(void)
{
    custom_ops_registry = create_vector(sizeof(struct custom_reg_entry), NULL);

    if (custom_ops_registry == NULL)
        return -1;

    return 0;
}

static void release_custom_ops_registry(void)
{
    release_vector(custom_ops_registry);
}

int register_custom_node_ops(int op_type, struct node_ops* node_ops)
{
    if (op_type <= OP_BUILTIN_LAST)
        return -1;

    int n = get_vector_num(custom_ops_registry);

    for (int i = 0; i < n; i++)
    {
        struct custom_reg_entry* entry = ( struct custom_reg_entry* )get_vector_data(custom_ops_registry, i);

        if (entry->op_type == op_type)
        {
            TLOG_ERR("custom op %d already has registered node ops\n", op_type);
            return -1;
        }
    }

    struct custom_reg_entry e;

    e.op_type = op_type;
    e.node_ops = node_ops;

    if (push_vector_data(custom_ops_registry, &e) < 0)
        return -1;

    return 0;
}

int unregister_custom_node_ops(int op_type, struct node_ops* node_ops)
{
    if (op_type <= OP_BUILTIN_LAST)
        return -1;

    int n = get_vector_num(custom_ops_registry);

    for (int i = 0; i < n; i++)
    {
        struct custom_reg_entry* entry = ( struct custom_reg_entry* )get_vector_data(custom_ops_registry, i);

        if (entry->op_type == op_type && entry->node_ops == node_ops)
        {
            remove_vector_by_idx(custom_ops_registry, i);
            return 0;
        }
    }

    return -1;
}

static inline struct node_ops* find_custom_node_ops(struct exec_graph* exec_graph, struct ir_node* ir_node)
{
    int op_type = ir_node->op.op_type;
    int n = get_vector_num(custom_ops_registry);

    for (int i = 0; i < n; i++)
    {
        struct custom_reg_entry* entry = ( struct custom_reg_entry* )get_vector_data(custom_ops_registry, i);

        if (entry->op_type == op_type)
            return entry->node_ops;
    }

    return NULL;
}

struct node_ops* find_node_ops(struct exec_graph* exec_graph, struct ir_node* ir_node)
{
    int op_type = ir_node->op.op_type;

    if (op_type <= OP_BUILTIN_LAST)
        return find_builtin_node_ops(exec_graph, ir_node);
    else
        return find_custom_node_ops(exec_graph, ir_node);
}

int init_cpu_node_ops_registry(void)
{
    if (init_builtin_ops_registry() < 0)
        return -1;

    if (init_custom_ops_registry() < 0)
        return -1;

    return 0;
}

void release_cpu_node_ops_registry(void)
{
    release_builtin_ops_registry();
    release_custom_ops_registry();
}
