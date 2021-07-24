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
 * Author: haitao@openailab.com
 */

#include "cpu_graph.h"

#include "cpu_node.h"
#include "cpu_pool.h"
#include "cpu_module.h"

#include "defines.h"
#include "utility/sys_port.h"
#include "utility/vector.h"
#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "graph/subgraph.h"
#include "utility/utils.h"
#include "utility/log.h"
#include "serializer/serializer.h"

static struct exec_graph* new_exec_graph(void)
{
    struct exec_graph* exec_graph = (struct exec_graph*)sys_malloc(sizeof(struct exec_graph));

    if (exec_graph == NULL)
        return NULL;

    exec_graph->exec_node_list = create_vector(sizeof(struct exec_node), NULL);

    if (exec_graph->exec_node_list == NULL)
    {
        sys_free(exec_graph);
        return NULL;
    }

    exec_graph->shared_mem = NULL;
    exec_graph->shared_mem_size = 0;
    exec_graph->mem_pool = NULL;

    exec_graph->shared_pack4_mem = NULL;
    exec_graph->shared_pack4_mem_size = 0;

    return exec_graph;
}

void release_exec_graph(void* exec_graph)
{
    struct exec_graph* graph = (struct exec_graph*)exec_graph;

    int node_num = get_vector_num(graph->exec_node_list);

    for (int i = 0; i < node_num; i++)
    {
        struct exec_node* exec_node = (struct exec_node*)get_vector_data(graph->exec_node_list, i);
        struct node_ops* node_ops = exec_node->node_ops;

        release_exec_node(graph, exec_node, node_ops);
    }

    free_exec_graph_mem(graph);

    release_vector(graph->exec_node_list);

    sys_free(graph);
}

struct exec_graph* create_exec_graph(struct subgraph* subgraph, int num_thread, int mode, size_t cpu_affinity)
{
    /* generate exec_graph */
    int node_num = subgraph->node_num;
    struct graph* ir_graph = subgraph->graph;
    struct exec_graph* exec_graph = new_exec_graph();
    struct cpu_device* dev = (struct cpu_device*)subgraph->device;

    if (exec_graph == NULL)
    {
        return NULL;
    }

    exec_graph->dev = dev;
    exec_graph->num_thread = num_thread;
    exec_graph->cpu_affinity = cpu_affinity;
    exec_graph->mode = mode;

    for (int i = 0; i < node_num; i++)
    {
        struct node* ir_node = get_ir_graph_node(ir_graph, subgraph->node_list[i]);

        // fprintf(stderr, "prerun: %d, %s\n", ir_node->op.op_type, ir_node->name);

        if (ir_node->op.type == OP_CONST || ir_node->op.type == OP_INPUT)
            continue;

        struct node_ops* node_ops = find_node_ops(exec_graph, ir_node);

        if (node_ops == NULL)
        {
            TLOG_ERR("Tengine: Device(%s) failed to find node ops for node(id: %d, type: %s, name: %s).\n",
                     dev->base.name, ir_node->index, get_op_name_from_type(ir_node->op.type), ir_node->name);
            goto error;
        }

        struct exec_node exec_node;

        if (init_exec_node(exec_graph, &exec_node, ir_node, node_ops) < 0)
        {
            TLOG_ERR("Tengine: Device(%s) failed to init exec node for node(id: %d, type: %s, name: %s).\n",
                     dev->base.name, ir_node->index, get_op_name_from_type(ir_node->op.type), ir_node->name);
            goto error;
        }

        push_vector_data(exec_graph->exec_node_list, &exec_node);
    }

    return exec_graph;

error:
    release_exec_graph(exec_graph);
    return NULL;
}

int prerun_exec_graph(struct exec_graph* exec_graph)
{
    int node_num = get_vector_num(exec_graph->exec_node_list);

    for (int i = 0; i < node_num; i++)
    {
        struct exec_node* exec_node = (struct exec_node*)get_vector_data(exec_graph->exec_node_list, i);
        struct node_ops* node_ops = exec_node->node_ops;

        if (node_ops->prerun && node_ops->prerun(node_ops, exec_node, exec_graph) < 0)
        {
            TLOG_ERR("%s: failed to prerun node %d\n", exec_graph->dev->base.name, exec_node->ir_node->index);
            return -1;
        }
    }

    return 0;
}
