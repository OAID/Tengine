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
 * Revised: lswang@openailab.com
 */

#include "graph/node.h"

#include "utility/sys_port.h"
#include "graph/tensor.h"
#include "graph/graph.h"
#include "operator/op.h"
#include "module/module.h"
#include "utility/log.h"
#include "utility/utils.h"

#include <string.h>

#define TENGINE_DEFAULT_LAYOUT TENGINE_LAYOUT_NCHW

static void init_ir_node(ir_node_t* ir_node, int op_type, int op_version, int node_index)
{
    ir_node->index = node_index;
    ir_node->dynamic_shape = 0;
    ir_node->input_num = 0;
    ir_node->output_num = 0;
    ir_node->node_type = TE_NODE_TYPE_INTER;
    ir_node->input_tensors = NULL;
    ir_node->output_tensors = NULL;
    ir_node->name = NULL;
    ir_node->op.type = op_type;
    ir_node->op.version = op_version;
    ir_node->op.same_shape = 1;
    ir_node->op.param_size = 0;
    ir_node->op.param_mem = NULL;
    ir_node->op.infer_shape = NULL;
    ir_node->subgraph_idx = -1;
}

ir_node_t* create_ir_node(struct graph* ir_graph, const char* node_name, int op_type, int op_version)
{
    ir_node_t* node = (ir_node_t*)sys_malloc(sizeof(ir_node_t));
    if (NULL == node)
    {
        return NULL;
    }

    init_ir_node(node, op_type, op_version, ir_graph->node_num);

    // check if any op param should be set
    ir_method_t* method = find_op_method(op_type, op_version);
    if (!(NULL != method && NULL != method->init && 0 == method->init(&node->op)))
    {
        sys_free(node);
        return NULL;
    }

    ir_node_t** new_node_list = (ir_node_t**)sys_realloc(ir_graph->node_list, sizeof(ir_node_t*) * (ir_graph->node_num + 1));

    if (NULL == new_node_list)
    {
        return NULL;
    }

    node->graph = ir_graph;

    if (NULL != node_name)
    {
        node->name = strdup(node_name);
    }

    new_node_list[ir_graph->node_num] = node;

    ir_graph->node_list = new_node_list;
    ir_graph->node_num++;

    return node;
}

void destroy_ir_node(struct graph* ir_graph, ir_node_t* ir_node)
{
    if (NULL != ir_node->name)
    {
        sys_free(ir_node->name);
        ir_node->name = NULL;
    }

    if (0 < ir_node->input_num)
    {
        sys_free(ir_node->input_tensors);
        ir_node->input_tensors = NULL;
    }

    if (0 < ir_node->output_num)
    {
        sys_free(ir_node->output_tensors);
        ir_node->output_tensors = NULL;
    }

    ir_method_t* method = find_op_method(ir_node->op.type, ir_node->op.version);

    if (NULL != method && NULL != method->release)
    {
        method->release(&ir_node->op);
    }

    sys_free(ir_node);
}

char* create_ir_node_name_from_index(int index)
{
    char* name = (char*)sys_malloc(16);
    if (NULL == name)
    {
        return NULL;
    }

    return name;
}

int get_ir_node_index_from_name(struct graph* ir_graph, const char* node_name)
{
    ir_node_t* ir_node;

    // first: try to get idx from suffix
    const char* p = strrchr(node_name, '_');
    if (p)
    {
        int idx = atoi(++p);

        if (idx >= 0 && idx < ir_graph->node_num)
        {
            ir_node = ir_graph->node_list[idx];

            if (NULL != ir_node->name && 0 == strcmp(ir_node->name, node_name))
            {
                return idx;
            }
        }
    }

    // second: search all nodes to compare name
    for (int i = 0; i < ir_graph->node_num; i++)
    {
        ir_node = ir_graph->node_list[i];

        if (ir_node->name && !strcmp(ir_node->name, node_name))
        {
            return i;
        }
    }

    return -1;
}

int set_ir_node_input_tensor(ir_node_t* node, int input_idx, ir_tensor_t* tensor)
{
    if (input_idx >= node->input_num)
    {
        int16_t* new_tensor = (int16_t*)sys_realloc(node->input_tensors, sizeof(int16_t) * (input_idx + 1));

        if (NULL == new_tensor)
        {
            return -1;
        }

        for (int i = node->input_num; i < input_idx + 1; i++)
        {
            new_tensor[i] = -1;
        }

        node->input_tensors = (uint16_t*)new_tensor;
        node->input_num = input_idx + 1;
    }

    node->input_tensors[input_idx] = tensor->index;
    if (set_ir_tensor_consumer(tensor, node->index) < 0)
    {
        return -1;
    }
    return 0;
}

int set_ir_node_output_tensor(ir_node_t* node, int output_idx, ir_tensor_t* tensor)
{
    if (output_idx >= node->output_num)
    {
        uint16_t* new_tensor = (uint16_t*)sys_realloc(node->output_tensors, sizeof(int16_t) * (output_idx + 1));

        for (int i = node->output_num; i < output_idx + 1; i++)
        {
            new_tensor[i] = -1;
        }

        node->output_tensors = new_tensor;
        node->output_num = output_idx + 1;
    }

    node->output_tensors[output_idx] = tensor->index;
    tensor->producer = node->index;

    return 0;
}

void dump_ir_node(struct graph* ir_graph, ir_node_t* ir_node)
{
    if (NULL != ir_node->name)
    {
        TLOG_INFO("\nnode: %d op: %s name: %s\n", ir_node->index, get_op_name_from_type(ir_node->op.type), ir_node->name);
    }
    else
    {
        TLOG_INFO("\nnode: %d op: %s name: node_%d\n", ir_node->index, get_op_name_from_type(ir_node->op.type), ir_node->index);
    }

    if (0 < ir_node->input_num)
    {
        TLOG_INFO("\tinput tensors: %d\n", ir_node->input_num);
    }

    for (int i = 0; i < ir_node->input_num; i++)
    {
        ir_tensor_t* ir_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[i]);

        TLOG_INFO("\t    %d: [id: %d] ", i, ir_tensor->index);

        dump_ir_tensor(ir_graph, ir_tensor);
    }

    if (0 < ir_node->output_num)
        TLOG_INFO("\toutput tensors: %d\n", ir_node->output_num);

    for (int i = 0; i < ir_node->output_num; i++)
    {
        ir_tensor_t* ir_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[i]);

        TLOG_INFO("\t    %d: [id: %d] ", i, ir_tensor->index);

        dump_ir_tensor(ir_graph, ir_tensor);
    }
}
