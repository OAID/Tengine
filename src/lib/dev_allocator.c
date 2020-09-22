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
#include <string.h>

#include "sys_port.h"
#include "vector.h"
#include "tengine_ir.h"
#include "tengine_exec.h"
#include "dev_allocator.h"

static int allocate(struct dev_allocator* allocator, struct ir_graph* ir_graph)
{
    struct subgraph* subgraph = ( struct subgraph* )sys_malloc(sizeof(struct subgraph));

    init_subgraph(ir_graph, subgraph, 0);

    subgraph->node_num = ir_graph->node_num;
    subgraph->node_list = ( uint16_t* )sys_malloc(sizeof(uint16_t) * ir_graph->node_num);

    for (int i = 0; i < subgraph->node_num; i++)
        subgraph->node_list[i] = ir_graph->node_list[i]->idx;

    if (ir_graph->nn_dev)
        subgraph->nn_dev = ir_graph->nn_dev;
    else
        subgraph->nn_dev = ir_graph->exec_attr->exec_context->def_dev;

    /* subgraph will record the input tensors and output tensors, instead of nodes */

    for (int i = 0; i < ir_graph->input_num; i++)
    {
        struct ir_node* node = get_ir_graph_node(ir_graph, ir_graph->input_nodes[i]);

        if (node->input_num)
        {
            for (int j = 0; j < node->input_num; j++)
            {
                struct ir_tensor* tensor = get_ir_graph_tensor(ir_graph, node->input_tensors[j]);

                if (tensor->tensor_type == TENSOR_TYPE_INPUT || tensor->tensor_type == TENSOR_TYPE_VAR)
                {
                    subgraph->input_num++;
                    subgraph->input_tensor_list =
                        sys_realloc(subgraph->input_tensor_list, subgraph->input_num * sizeof(uint16_t));
                    subgraph->input_tensor_list[subgraph->input_num - 1] = tensor->idx;
                }
            }
        }
        else
        {
            for (int j = 0; j < node->output_num; j++)
            {
                struct ir_tensor* tensor = get_ir_graph_tensor(ir_graph, node->output_tensors[j]);

                if (tensor->tensor_type != TENSOR_TYPE_INPUT)
                    continue;

                subgraph->input_num++;
                subgraph->input_tensor_list =
                    sys_realloc(subgraph->input_tensor_list, subgraph->input_num * sizeof(uint16_t));
                subgraph->input_tensor_list[subgraph->input_num - 1] = tensor->idx;
            }
        }
    }

    for (int i = 0; i < ir_graph->output_num; i++)
    {
        struct ir_node* node = get_ir_graph_node(ir_graph, ir_graph->output_nodes[i]);

        for (int j = 0; j < node->output_num; j++)
        {
            struct ir_tensor* tensor = get_ir_graph_tensor(ir_graph, node->output_tensors[j]);

            if (tensor->consumer_num == 0)
            {
                subgraph->output_num++;
                subgraph->output_tensor_list =
                    sys_realloc(subgraph->output_tensor_list, subgraph->output_num * sizeof(uint16_t));
                subgraph->output_tensor_list[subgraph->output_num - 1] = tensor->idx;
            }
        }
    }

    /* strip out duplicated input tensors */
    uint16_t* real_inputs = ( uint16_t* )sys_malloc(subgraph->input_num * sizeof(uint16_t));
    int real_input_num = 1;

    real_inputs[0] = subgraph->input_tensor_list[0];

    for (int i = 1; i < subgraph->input_num; i++)
    {
        int idx = subgraph->input_tensor_list[i];
        int j;

        for (j = 0; j < real_input_num; j++)
        {
            if (idx == real_inputs[j])
                break;
        }

        if (j < real_input_num)
            continue;

        real_inputs[real_input_num] = idx;
        real_input_num++;
    }

    sys_free(subgraph->input_tensor_list);

    subgraph->input_num = real_input_num;
    subgraph->input_tensor_list = real_inputs;

    /* set the correct input wait count: INPUT tensor is always ready */
    subgraph->input_wait_count = 0;

    for (int i = 0; i < subgraph->input_num; i++)
    {
        struct ir_tensor* tensor = get_ir_graph_tensor(ir_graph, subgraph->input_tensor_list[i]);

        if (tensor->tensor_type == TENSOR_TYPE_VAR)
            subgraph->input_wait_count++;
    }

    /* attached to graph */

    push_vector_data(ir_graph->subgraph_list, &subgraph);

    return 0;
}

static struct dev_allocator single_allocator = {.allocate = allocate};

struct dev_allocator* get_default_dev_allocator(void)
{
    return &single_allocator;
}
