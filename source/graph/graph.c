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

#include "graph/graph.h"

#include "defines.h"
#include "utility/sys_port.h"
#include "utility/vector.h"
#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/subgraph.h"
#include "executer/executer.h"
#include "serializer/serializer.h"
#include "utility/utils.h"
#include "utility/log.h"

#include <string.h>

ir_graph_t* create_ir_graph(struct context* context)
{
    ir_graph_t* ir_graph = (ir_graph_t*)sys_malloc(sizeof(ir_graph_t));
    if (NULL == ir_graph)
    {
        return NULL;
    }

    ir_graph->attribute = (struct attribute*)sys_malloc(sizeof(struct attribute));

    init_ir_graph(ir_graph, context);

    return ir_graph;
}

void init_ir_graph(ir_graph_t* graph, struct context* context)
{
    graph->tensor_list = NULL;
    graph->node_list = NULL;
    graph->input_nodes = NULL;
    graph->output_nodes = NULL;

    graph->tensor_num = 0;
    graph->node_num = 0;
    graph->input_num = 0;
    graph->output_num = 0;

    graph->subgraph_list = create_vector(sizeof(struct subgraph*), NULL);

    graph->graph_layout = TENGINE_LAYOUT_NCHW;
    graph->model_layout = TENGINE_LAYOUT_NCHW;
    graph->model_format = MODEL_FORMAT_TENGINE;

    graph->serializer = NULL;
    graph->serializer_privacy = NULL;

    graph->device = NULL;
    graph->device_privacy = NULL;

    graph->status = GRAPH_STAT_CREATED;

    init_attribute(graph->attribute, context);
}

void destroy_ir_graph(ir_graph_t* graph)
{
    //!< 1, destroy subgraph
    if (NULL != graph->subgraph_list)
    {
        const int subgraph_num = get_vector_num(graph->subgraph_list);

        for (int i = 0; i < subgraph_num; i++)
        {
            struct subgraph* subgraph = *(struct subgraph**)get_vector_data(graph->subgraph_list, i);
            release_ir_subgraph(graph, subgraph);
        }

        release_vector(graph->subgraph_list);
    }

    //!< 2, destroy serializer
    struct serializer* serializer = graph->serializer;
    if (NULL != serializer && serializer->unload_graph)
    {
        serializer->unload_graph(serializer, graph, graph->serializer_privacy, graph->device_privacy);
    }

    //!< 3, destroy tensors
    for (int i = 0; i < graph->tensor_num; i++)
    {
        destroy_ir_tensor(graph, graph->tensor_list[i]);
    }

    //!< 4, destroy nodes
    for (int i = 0; i < graph->node_num; i++)
    {
        destroy_ir_node(graph, graph->node_list[i]);
    }

    sys_free(graph->tensor_list);
    sys_free(graph->node_list);
    sys_free(graph->input_nodes);
    sys_free(graph->output_nodes);

    if (NULL != graph->attribute)
    {
        destroy_attribute(graph, graph->attribute);
    }

    sys_free(graph);
}

int set_ir_graph_input_node(ir_graph_t* graph, int16_t input_nodes[], int input_number)
{
    if (0 >= input_number)
    {
        return -1;
    }

    int16_t* new_input_nodes = (int16_t*)sys_malloc(input_number * sizeof(int16_t));
    if (NULL == new_input_nodes)
    {
        return -1;
    }

    if (NULL != graph->input_nodes)
    {
        sys_free(graph->input_nodes);
        graph->input_nodes = NULL;
    }

    graph->input_nodes = new_input_nodes;
    graph->input_num = input_number;

    for (int i = 0; i < input_number; i++)
    {
        ir_node_t* node = get_ir_graph_node(graph, input_nodes[i]);
        node->node_type = TE_NODE_TYPE_INPUT;
        graph->input_nodes[i] = input_nodes[i];
    }

    return 0;
}

int set_ir_graph_output_node(ir_graph_t* graph, int16_t output_nodes[], int output_number)
{
    if (0 >= output_number)
    {
        return -1;
    }

    int16_t* new_output_nodes = (int16_t*)sys_malloc(output_number * sizeof(int16_t));
    if (NULL == new_output_nodes)
    {
        return -1;
    }

    if (NULL != graph->output_nodes)
    {
        sys_free(graph->output_nodes);
        graph->output_nodes = NULL;
    }

    graph->output_nodes = new_output_nodes;
    graph->output_num = output_number;

    for (int i = 0; i < output_number; i++)
    {
        ir_node_t* node = get_ir_graph_node(graph, output_nodes[i]);
        node->node_type = TE_NODE_TYPE_OUTPUT;

        graph->output_nodes[i] = output_nodes[i];
    }

    return 0;
}

struct tensor* get_ir_graph_tensor(ir_graph_t* graph, int index)
{
    return graph->tensor_list[index];
}

struct node* get_ir_graph_node(ir_graph_t* graph, int index)
{
    return graph->node_list[index];
}

struct subgraph* get_ir_graph_subgraph(ir_graph_t* graph, int index)
{
    return *(struct subgraph**)get_vector_data(graph->subgraph_list, index);
}

int infer_ir_graph_shape(ir_graph_t* graph)
{
    const int node_num = graph->node_num;

    for (int i = 0; i < node_num; i++)
    {
        ir_node_t* node = get_ir_graph_node(graph, i);
        ir_op_t* op = &node->op;

        if (node->input_num == 0)
            continue;

        if (node->dynamic_shape)
        {
            // populate the dynamic_shape
            int output_num = node->output_num;

            for (int j = 0; j < output_num; j++)
            {
                ir_tensor_t* tensor = get_ir_graph_tensor(graph, j);

                for (int l = 0; l < tensor->consumer_num; l++)
                {
                    ir_node_t* child_node = get_ir_graph_node(graph, l);
                    child_node->dynamic_shape = 1;
                }
            }

            continue;
        }

        if (0 != op->same_shape)
        {
            ir_tensor_t* input = get_ir_graph_tensor(graph, node->input_tensors[0]);
            ir_tensor_t* output = get_ir_graph_tensor(graph, node->output_tensors[0]);

            output->dim_num = input->dim_num;
            output->elem_num = input->elem_num;

            memcpy(output->dims, input->dims, sizeof(int32_t) * input->dim_num);
        }
        else
        {
            if (0 != op->infer_shape(node))
            {
                TLOG_ERR("Tengine FATAL: Infer node(id: %d, op: %s) shape failed.\n", node->index,
                         get_op_name_from_type(node->op.type));
                return -1;
            }
        }

        for (int j = 0; j < node->output_num; j++)
        {
            ir_tensor_t* tensor = get_ir_graph_tensor(graph, node->output_tensors[j]);

            tensor->reshaped = 0;
        }
    }

    return 0;
}

void dump_ir_graph(ir_graph_t* graph)
{
    TLOG_INFO("graph node_num %u tensor_num: %u  subgraph_num: %u\n", graph->node_num, graph->tensor_num,
              get_vector_num(graph->subgraph_list));

    TLOG_INFO("graph layout: %s model layout: %s model_format: %s\n", get_tensor_layout_string(graph->graph_layout),
              get_tensor_layout_string(graph->model_layout), get_model_format_string(graph->model_format));

    for (int i = 0; i < graph->node_num; i++)
    {
        dump_ir_node(graph, graph->node_list[i]);
    }

    TLOG_INFO("\ngraph inputs: %u\n", graph->input_num);

    for (int i = 0; i < graph->input_num; i++)
    {
        ir_node_t* node = get_ir_graph_node(graph, graph->input_nodes[i]);

        if (node->name)
        {
            TLOG_INFO("\t%s\n", node->name);
        }
        else
        {
            TLOG_INFO("\tnode_%d\n", node->index);
        }
    }

    TLOG_INFO("graph outputs: %u\n", graph->output_num);

    for (int i = 0; i < graph->output_num; i++)
    {
        ir_node_t* node = get_ir_graph_node(graph, graph->output_nodes[i]);

        if (node->name)
        {
            TLOG_INFO("\t%s\n", node->name);
        }
        else
        {
            TLOG_INFO("\tnode_%d\n", node->index);
        }
    }
}
