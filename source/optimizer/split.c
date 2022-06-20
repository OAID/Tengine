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
 * Author: lswang@openailab.com
 */

#include "optimizer/helper.h"

#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "graph/subgraph.h"
#include "device/device.h"
#include "executer/executer.h"
#include "module/module.h"
#include "utility/sys_port.h"
#include "utility/vector.h"
#include "utility/log.h"

#include <string.h>

#define MODEL_COMPLEX_COUNT 3

int check_sub_info(struct graph* ir_graph)
{
    int subgraph_num = get_vector_num(ir_graph->subgraph_list);
    if (0 == subgraph_num)
    {
        return 0;
    }

    return -1;
}

int tensor_in_precision(const struct tensor* tensor, struct vector* allowed_precision)
{
    int count = get_vector_num(allowed_precision);
    for (int i = 0; i < count; i++)
    {
        const int* const precision = (const int* const)get_vector_data(allowed_precision, i);
        if (*precision == (int)tensor->data_type || tensor->quant_param_num > 0)
        {
            return 0;
        }
    }

    return -1;
}

int node_in_precision(const struct graph* ir_graph, uint16_t node_id, struct vector* allowed_precision)
{
    if (node_id > ir_graph->node_num)
    {
        return -1;
    }

    const ir_node_t* ir_node = ir_graph->node_list[node_id];

    for (int8_t i = 0; i < ir_node->output_num; i++)
    {
        uint16_t index = ir_node->output_tensors[i];
        const struct tensor* tensor = ir_graph->tensor_list[index];

        if (TENSOR_TYPE_VAR == tensor->tensor_type || TENSOR_TYPE_INPUT == tensor->tensor_type)
        {
            const int in_precision = tensor_in_precision(tensor, allowed_precision);

            if (0 == in_precision)
            {
                return 0;
            }
        }
        else
        {
            return 0;
        }
    }

    return -1;
}

int node_in_list(const struct graph* ir_graph, struct vector* ops_list, const uint16_t node_id)
{
    if (NULL == ir_graph || NULL == ops_list)
    {
        return -1;
    }

    const uint16_t node_op_type = ir_graph->node_list[node_id]->op.type;

    for (int i = 0; i < get_vector_num(ops_list); i++)
    {
        int* loop_op = (int*)get_vector_data(ops_list, i);
        if (node_op_type == *loop_op)
        {
            return 0;
        }
    }

    return -1;
}

struct vector* get_graph_blocked_nodes(const struct graph* ir_graph, struct vector* blocked_ops, struct vector* allowed_precision)
{
    struct vector* blocked_nodes_list = create_vector(sizeof(uint16_t), NULL);

    for (uint16_t i = 0; i < ir_graph->node_num; i++)
    {
        int is_blocked_op = node_in_list(ir_graph, blocked_ops, i);
        int is_allowed_precision = node_in_precision(ir_graph, i, allowed_precision);
        if (0 == is_blocked_op || 0 != is_allowed_precision)
        {
            push_vector_data(blocked_nodes_list, &i);
            continue;
        }
    }

    return blocked_nodes_list;
}

// policy has some issue, must be fixed
void split_graph_node_to_sub_graph(struct graph* ir_graph, struct vector* allowed_ops, struct vector* blocked_ops, struct vector* allowed_precision)
{
    // get unsupported nodes
    struct vector* blocked_nodes_list = get_graph_blocked_nodes(ir_graph, blocked_ops, allowed_precision);
    const int blocked_nodes_count = get_vector_num(blocked_nodes_list);
    //sort_nodes(blocked_nodes_list);

    if (blocked_nodes_count != 0)
    {
        // from the last unsupported node to collecting all sub graphs
        // scan from back to front
        for (int i = blocked_nodes_count - 1; i >= 0; i--)
        {
            // start node id (the blocked one)
            uint16_t first_node_id = *((uint16_t*)get_vector_data(blocked_nodes_list, i));
            // end node id (not including its self; the next blocked one, or the last one)
            uint16_t last_node_id = ir_graph->node_num;
            if (i < blocked_nodes_count - 1)
            {
                last_node_id = *((uint16_t*)get_vector_data(blocked_nodes_list, i + 1));
            }

            int children_nodes_is_complicated = 0;

            // scan if these nodes is complicated to be solved
            for (uint16_t j = first_node_id; j < last_node_id; j++)
            {
                if (0 == node_in_list(ir_graph, allowed_ops, j))
                {
                    const uint16_t node_op_type = ir_graph->node_list[j]->op.type;

                    if (OP_FC == node_op_type)
                    {
                        children_nodes_is_complicated += MODEL_COMPLEX_COUNT;
                    }
                    else
                    {
                        children_nodes_is_complicated++;
                    }
                }
            }

            if (children_nodes_is_complicated < MODEL_COMPLEX_COUNT) // directly add these nodes to sub graph list
            {
                struct subgraph* sub_graph = (struct subgraph*)sys_malloc(sizeof(struct subgraph));
                init_ir_subgraph((struct graph*)ir_graph, sub_graph, 0);

                // not including the last one
                sub_graph->node_num = last_node_id - first_node_id;
                sub_graph->node_list = (uint16_t*)sys_malloc(sizeof(uint16_t) * sub_graph->node_num);

                for (uint16_t j = 0; j < sub_graph->node_num; j++)
                {
                    sub_graph->node_list[j] = j + first_node_id;
                }

                sub_graph->device = find_default_device();

                push_vector_data(ir_graph->subgraph_list, &sub_graph);
            }
            else
            {
                // add special device running nodes
                struct subgraph* sub_device_graph = (struct subgraph*)sys_malloc(sizeof(struct subgraph));
                init_ir_subgraph((struct graph*)ir_graph, sub_device_graph, 0);

                sub_device_graph->node_num = last_node_id - (first_node_id + 1);
                sub_device_graph->node_list = (uint16_t*)sys_malloc(sizeof(uint16_t) * sub_device_graph->node_num);

                for (uint16_t j = 0; j < sub_device_graph->node_num; j++)
                {
                    sub_device_graph->node_list[j] = j + first_node_id + 1;
                }

                struct device* nn_dev = ir_graph->attribute->context->device;
                sub_device_graph->device = nn_dev;

                push_vector_data(ir_graph->subgraph_list, &sub_device_graph);

                // ---------------

                // add cpu running nodes
                struct subgraph* sub_cpu_graph = (struct subgraph*)sys_malloc(sizeof(struct subgraph));
                init_ir_subgraph((struct graph*)ir_graph, sub_cpu_graph, 0);

                sub_cpu_graph->node_num = 1;
                sub_cpu_graph->node_list = (uint16_t*)sys_malloc(sizeof(uint16_t) * sub_cpu_graph->node_num);
                sub_cpu_graph->node_list[0] = first_node_id;

                sub_cpu_graph->device = find_default_device();

                push_vector_data(ir_graph->subgraph_list, &sub_cpu_graph);
            }
        }
    }

    // add main sub graph
    struct subgraph* sub_graph = (struct subgraph*)sys_malloc(sizeof(struct subgraph));
    init_ir_subgraph((struct graph*)ir_graph, sub_graph, 0);

    uint16_t stop_node_id;
    if (blocked_nodes_count == 0)
    {
        stop_node_id = ir_graph->node_num;
    }
    else
    {
        stop_node_id = *((uint16_t*)get_vector_data((struct vector*)blocked_nodes_list, 0));
    }

    sub_graph->node_num = stop_node_id;
    sub_graph->node_list = (uint16_t*)sys_malloc(sizeof(uint16_t) * sub_graph->node_num);

    for (uint16_t i = 0; i < stop_node_id; i++)
    {
        sub_graph->node_list[i] = i;
    }

    // main nodes should be running at cpu
    struct device* nn_dev = NULL;
    if (NULL != ir_graph->attribute->context->device)
    {
        nn_dev = ir_graph->attribute->context->device;
    }
    else
    {
        nn_dev = find_default_device();
    }
    sub_graph->device = nn_dev;

    push_vector_data(ir_graph->subgraph_list, &sub_graph);

    release_vector(blocked_nodes_list);

    // optimize the sub graphs
    while (1)
    {
        int same_sub_graph_found = 0;
        int sub_graphs_count = get_vector_num(ir_graph->subgraph_list);
        for (int i = 1; i < sub_graphs_count; i++)
        {
            struct subgraph* last_sub_graph = *(struct subgraph**)get_vector_data(ir_graph->subgraph_list, (sub_graphs_count - 1) - (i - 1));
            struct subgraph* current_sub_graph = *(struct subgraph**)get_vector_data(ir_graph->subgraph_list, (sub_graphs_count - 1) - i);

            if (current_sub_graph->device == last_sub_graph->device)
            {
                uint16_t* node_list = (uint16_t*)sys_malloc(sizeof(uint16_t) * (last_sub_graph->node_num + current_sub_graph->node_num));

                for (int j = 0; j < last_sub_graph->node_num; j++)
                {
                    node_list[j] = last_sub_graph->node_list[j];
                }

                for (int j = 0; j < current_sub_graph->node_num; j++)
                {
                    node_list[j + last_sub_graph->node_num] = current_sub_graph->node_list[j];
                }

                last_sub_graph->node_num += current_sub_graph->node_num;
                sys_free(last_sub_graph->node_list);
                last_sub_graph->node_list = node_list;

                remove_vector_via_index(ir_graph->subgraph_list, (sub_graphs_count - 1) - i);

                same_sub_graph_found = 1;
                break;
            }
        }

        if (!same_sub_graph_found)
            break;
    }
}

void generate_sub_graph_io(struct graph* ir_graph)
{
    int sub_graph_count = get_vector_num(ir_graph->subgraph_list);
    for (int index = 0; index < sub_graph_count; index++)
    {
        struct subgraph* sub_graph = *(struct subgraph**)get_vector_data(ir_graph->subgraph_list, index);

        uint16_t random_input_id = 0;
        uint16_t random_output_id = 0;

        for (int i = 0; i < sub_graph->node_num; i++)
        {
            uint16_t node_id = sub_graph->node_list[i];
            struct node* ir_node = ir_graph->node_list[node_id];
            if (ir_node->input_num > 0)
            {
                struct tensor* tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);

                if (tensor->tensor_type == TENSOR_TYPE_INPUT || tensor->tensor_type == TENSOR_TYPE_VAR)
                {
                    random_input_id = tensor->index;
                    break;
                }
            }
        }

        for (int i = 0; i < sub_graph->node_num; i++)
        {
            uint16_t node_id = sub_graph->node_list[i];
            struct node* ir_node = ir_graph->node_list[node_id];
            if (ir_node->output_num > 0)
            {
                struct tensor* tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
                random_output_id = tensor->index;
                break;
            }
        }

        uint16_t min_input_tensor_id = random_input_id;
        uint16_t max_input_tensor_id = random_input_id;
        uint16_t min_output_tensor_id = random_output_id;
        uint16_t max_output_tensor_id = random_output_id;

        for (int i = 0; i < sub_graph->node_num; i++)
        {
            struct node* ir_node = ir_graph->node_list[sub_graph->node_list[i]];

            for (int k = 0; k < ir_node->input_num; k++)
            {
                struct tensor* tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[k]);

                if (tensor->tensor_type == TENSOR_TYPE_INPUT || tensor->tensor_type == TENSOR_TYPE_VAR)
                {
                    if (tensor->index < min_input_tensor_id)
                        min_input_tensor_id = tensor->index;

                    if (tensor->index > max_input_tensor_id)
                        max_input_tensor_id = tensor->index;
                }
            }

            for (int k = 0; k < ir_node->output_num; k++)
            {
                struct tensor* tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[k]);

                if (tensor->tensor_type != TENSOR_TYPE_INPUT)
                {
                    if (tensor->index < min_output_tensor_id)
                        min_output_tensor_id = tensor->index;

                    if (tensor->index > max_output_tensor_id)
                        max_output_tensor_id = tensor->index;
                }
                else
                {
                    if (tensor->index < min_input_tensor_id)
                        min_input_tensor_id = tensor->index;

                    if (tensor->index > max_input_tensor_id)
                        max_input_tensor_id = tensor->index;
                }
            }
        }

        uint16_t* input_tensors = (uint16_t*)malloc(sizeof(uint16_t) * (max_input_tensor_id - min_input_tensor_id + 1));
        uint16_t* output_tensors = (uint16_t*)malloc(sizeof(uint16_t) * (max_output_tensor_id - min_output_tensor_id + 1));

        memset(input_tensors, 0, sizeof(uint16_t) * (max_input_tensor_id - min_input_tensor_id + 1));
        memset(output_tensors, 0, sizeof(uint16_t) * (max_output_tensor_id - min_output_tensor_id + 1));

        for (int j = 0; j < sub_graph->node_num; j++)
        {
            struct node* ir_node = ir_graph->node_list[sub_graph->node_list[j]];

            for (int k = 0; k < ir_node->input_num; k++)
            {
                struct tensor* tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[k]);

                if (tensor->tensor_type == TENSOR_TYPE_INPUT || tensor->tensor_type == TENSOR_TYPE_VAR)
                {
                    input_tensors[tensor->index - min_input_tensor_id]++;
                }
            }

            for (int k = 0; k < ir_node->output_num; k++)
            {
                struct tensor* tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[k]);

                if (tensor->tensor_type != TENSOR_TYPE_INPUT)
                {
                    if (tensor->tensor_type != TENSOR_TYPE_CONST)
                    {
                        output_tensors[tensor->index - min_output_tensor_id]++;
                    }
                }
                else
                {
                    input_tensors[tensor->index - min_input_tensor_id]++;
                }
            }
        }

        /*fprintf(stdout, "\tAll input node mask: [ ");
        for (int i = 0; i < (max_input_tensor_id - min_input_tensor_id) + 1; i++)
        {
            if (input_tensors[i] > 0)
                fprintf(stdout, "%d ", min_input_tensor_id + i);
        }
        fprintf(stdout, "]\n");
        fflush(stdout);

        fprintf(stdout, "\tAll output node mask: [ ");
        for (int i = 0; i < (max_output_tensor_id - min_output_tensor_id) + 1; i++)
        {
            if (output_tensors[i] > 0)
                fprintf(stdout, "%d ", min_output_tensor_id + i);
        }
        fprintf(stdout, "]\n");
        fflush(stdout);*/

        uint16_t search_start = min_input_tensor_id > min_output_tensor_id ? min_input_tensor_id : min_output_tensor_id;
        uint16_t search_end = max_input_tensor_id < max_output_tensor_id ? max_input_tensor_id : max_output_tensor_id;

        for (int i = 0; i < (search_end - search_start) + 1; i++)
        {
            int input_offset = (search_start - min_input_tensor_id) + i;
            int output_offset = (search_start - min_output_tensor_id) + i;

            int input_flag = input_tensors[input_offset];
            int output_flag = output_tensors[output_offset];

            if (input_flag > 0 && output_flag > 0)
            {
                input_tensors[input_offset] = 0;
                output_tensors[output_offset] = 0;
            }
        }

        /*fprintf(stdout, "\tNew input node mask: [ ");
        for (int i = 0; i < max_input_tensor_id - min_input_tensor_id + 1; i++)
        {
            if (input_tensors[i] > 0)
                fprintf(stdout, "%d ", min_input_tensor_id + i);
        }
        fprintf(stdout, "]\n");
        fflush(stdout);

        fprintf(stdout, "\tNew output node mask: [ ");
        for (int i = 0; i < max_output_tensor_id - min_output_tensor_id + 1; i++)
        {
            if (output_tensors[i] > 0)
                fprintf(stdout, "%d ", min_output_tensor_id + i);
        }
        fprintf(stdout, "]\n");
        fflush(stdout);*/

        sub_graph->input_num = 0;
        for (int j = 0; j < max_input_tensor_id - min_input_tensor_id + 1; j++)
        {
            if (input_tensors[j] > 0)
            {
                sub_graph->input_num++;
            }
        }

        sub_graph->output_num = 0;
        for (int j = 0; j < max_output_tensor_id - min_output_tensor_id + 1; j++)
        {
            if (output_tensors[j] > 0)
            {
                sub_graph->output_num++;
            }
        }

        sub_graph->input_tensor_list = (uint16_t*)sys_malloc(sizeof(uint16_t) * sub_graph->input_num);
        sub_graph->output_tensor_list = (uint16_t*)sys_malloc(sizeof(uint16_t) * sub_graph->output_num);

        uint16_t input_tensor_count = 0;
        for (int j = 0; j < max_input_tensor_id - min_input_tensor_id + 1; j++)
        {
            if (input_tensors[j] > 0)
            {
                sub_graph->input_tensor_list[input_tensor_count] = min_input_tensor_id + j;
                input_tensor_count++;
            }
        }

        uint16_t output_tensor_count = 0;
        for (int j = 0; j < max_output_tensor_id - min_output_tensor_id + 1; j++)
        {
            if (output_tensors[j] > 0)
            {
                sub_graph->output_tensor_list[output_tensor_count] = min_output_tensor_id + j;
                output_tensor_count++;
            }
        }

        sys_free(input_tensors);
        sys_free(output_tensors);
    }
}

void add_sub_graph_to_ir_graph(struct graph* ir_graph)
{
    const int sub_graphs_count = get_vector_num(ir_graph->subgraph_list);

    // reverse sub graphs to ir_graph
    for (int i = 0; i < sub_graphs_count / 2; i++)
    {
        struct subgraph* sub_graph_front = *(struct subgraph**)get_vector_data(ir_graph->subgraph_list, i);
        struct subgraph* sub_graph_back = *(struct subgraph**)get_vector_data(ir_graph->subgraph_list, (sub_graphs_count - 1) - i);

        struct subgraph* mid_temp = (struct subgraph*)sys_malloc(sizeof(struct subgraph));

        memcpy(mid_temp, sub_graph_back, sizeof(struct subgraph));
        memcpy(sub_graph_back, sub_graph_front, sizeof(struct subgraph));
        memcpy(sub_graph_front, mid_temp, sizeof(struct subgraph));

        sys_free(mid_temp);
    }

    // fill the sub graph id
    for (int i = 0; i < sub_graphs_count; i++)
    {
        struct subgraph* sub_graph = *(struct subgraph**)get_vector_data(ir_graph->subgraph_list, i);
        sub_graph->index = i;

        for (int j = 0; j < sub_graph->node_num; j++)
        {
            ir_graph->node_list[sub_graph->node_list[j]]->subgraph_idx = i;
        }
    }

    // find no-output input in current sub graph
    for (int i = 1; i < sub_graphs_count; i++)
    {
        struct subgraph* sub_graph = *(struct subgraph**)get_vector_data(ir_graph->subgraph_list, i);
        for (int j = 0; j < sub_graph->input_num; j++)
        {
            struct tensor* ir_tensor = ir_graph->tensor_list[sub_graph->input_tensor_list[j]];

            if (ir_tensor->tensor_type != TENSOR_TYPE_INPUT)
            {
                uint16_t node_id = ir_tensor->producer;
                uint8_t sub_graph_id = ir_graph->node_list[node_id]->subgraph_idx;

                struct subgraph* target_sub_graph = *(struct subgraph**)get_vector_data(ir_graph->subgraph_list, sub_graph_id);

                int tensor_mask_as_out_flag = 0;
                for (int k = 0; k < target_sub_graph->output_num; k++)
                {
                    if (target_sub_graph->output_tensor_list[k] == ir_tensor->index)
                    {
                        tensor_mask_as_out_flag = 1;
                        break;
                    }
                }

                if (!tensor_mask_as_out_flag)
                {
                    uint16_t* new_output_tensor_list = (uint16_t*)sys_malloc(sizeof(uint16_t) * (target_sub_graph->output_num + 1));

                    memcpy(new_output_tensor_list, target_sub_graph->output_tensor_list, sizeof(uint16_t) * target_sub_graph->output_num);
                    new_output_tensor_list[target_sub_graph->output_num] = ir_tensor->index;

                    sys_free(target_sub_graph->output_tensor_list);
                    target_sub_graph->output_tensor_list = new_output_tensor_list;
                    target_sub_graph->output_num += 1;
                }
            }
        }
    }

    // find graph output in every sub graph
    for (int i = 0; i < sub_graphs_count; i++)
    {
        struct subgraph* sub_graph = *(struct subgraph**)get_vector_data(ir_graph->subgraph_list, i);
        for (int j = 0; j < sub_graph->node_num; j++)
        {
            for (int k = 0; k < ir_graph->output_num; k++)
            {
                if (sub_graph->node_list[j] == ir_graph->output_nodes[k])
                {
                    struct node* ir_node = ir_graph->node_list[ir_graph->output_nodes[k]];
                    for (int q = 0; q < ir_node->output_num; q++)
                    {
                        struct tensor* ir_tensor = ir_graph->tensor_list[ir_node->output_tensors[q]];
                        int tensor_mask_as_out_flag = 0;
                        for (int p = 0; p < sub_graph->output_num; p++)
                        {
                            if (sub_graph->output_tensor_list[p] == ir_node->output_tensors[q])
                            {
                                tensor_mask_as_out_flag = 1;
                                break;
                            }
                        }
                        if (!tensor_mask_as_out_flag)
                        {
                            uint16_t* new_output_tensor_list = (uint16_t*)sys_malloc(sizeof(uint16_t) * (sub_graph->output_num + 1));

                            memcpy(new_output_tensor_list, sub_graph->output_tensor_list, sizeof(uint16_t) * sub_graph->output_num);
                            new_output_tensor_list[sub_graph->output_num] = ir_tensor->index;

                            sys_free(sub_graph->output_tensor_list);
                            sub_graph->output_tensor_list = new_output_tensor_list;
                            sub_graph->output_num += 1;
                        }
                    }
                }
            }
        }
        // for (int j = 0; j < sub_graph->output_num; j++)
        // {
        //     fprintf(stdout, "sub_graph->output_tensor_list[%d]:%d\n", j, sub_graph->output_tensor_list[j]);
        // }
    }

    /*
    // get all input and output tensors
    struct vector* input_tensors = create_vector(sizeof(uint16_t), NULL);
    struct vector* output_tensors = create_vector(sizeof(uint16_t), NULL);

    // add all io tensors
    for (int i = 0; i < sub_graphs_count; i++)
    {
        struct subgraph* sub_graph = get_vector_data(*(struct subgraph**)ir_graph->subgraph_list, i);
        for (int j = 0; j < sub_graph->input_num; j++)
        {
            uint16_t idx = sub_graph->input_tensor_list[j];
            push_vector_data(input_tensors, &idx);
        }

        for (int j = 0; j < sub_graph->output_num; j++)
        {
            uint16_t idx = sub_graph->output_tensor_list[j];
            push_vector_data(output_tensors, &idx);
        }
    }

    // remove same tensors
    while(1)
    {
        int has_same_tensor_flag = 0;
        int input_tensor_count = get_vector_num(input_tensors);
        for (int i = 1; i < input_tensor_count; i++)
        {
            uint16_t* current_idx = get_vector_data(input_tensors, i);
            uint16_t* before_idx = get_vector_data(input_tensors, i - 1);

            if (*current_idx == *before_idx)
            {
                has_same_tensor_flag = 1;
                remove_vector_via_index(input_tensors, i);
                break;
            }
        }

        if (!has_same_tensor_flag)
            break;
    }
    while(1)
    {
        int has_same_tensor_flag = 0;
        int output_tensor_count = get_vector_num(output_tensors);
        for (int i = 1; i < output_tensor_count; i++)
        {
            uint16_t* current_idx = get_vector_data(output_tensors, i);
            uint16_t* before_idx = get_vector_data(output_tensors, i - 1);

            if (*current_idx == *before_idx)
            {
                has_same_tensor_flag = 1;
                remove_vector_via_index(output_tensors, i);
                break;
            }
        }

        if (!has_same_tensor_flag)
            break;
    }

    // remove inside io tensors
    while(1)
    {
        int find_same_flag = 0;
        int input_tensor_index = 0, output_tensor_index = 0;
        for (int i = 0; i < get_vector_num(input_tensors); i++)
        {
            uint16_t* input_tensor_idx = get_vector_data(input_tensors, i);
            for (int j = 0; j < get_vector_num(output_tensors); j++)
            {
                uint16_t* output_tensor_idx = get_vector_data(output_tensors, j);

                if (*output_tensor_idx == *input_tensor_idx)
                {
                    input_tensor_index = i;
                    output_tensor_index = j;
                    find_same_flag = 1;
                    break;
                }
            }
            if (find_same_flag)
                break;
        }

        if (find_same_flag)
        {
            remove_vector_via_index(input_tensors, input_tensor_index);
            remove_vector_via_index(output_tensors, output_tensor_index);
        }
        else
        {
            break;
        }
    }

    // fill ir_graph input and output
    ir_graph->input_num = get_vector_num(input_tensors);
    ir_graph->output_num = get_vector_num(output_tensors);

    fprintf(stdout, "Network graph input tensors: [ ");
    for (int i = 0; i < get_vector_num(input_tensors); i++)
    {
        uint16_t* idx = get_vector_data(input_tensors, i);
        fprintf(stdout, "%d ", *idx);
    }
    fprintf(stdout, "].\n");

    fprintf(stdout, "Network graph output tensors: [ ");
    for (int i = 0; i < get_vector_num(output_tensors); i++)
    {
        uint16_t* idx = get_vector_data(output_tensors, i);
        fprintf(stdout, "%d ", *idx);
    }
    fprintf(stdout, "].\n");
    */

    // fill the wait count
    for (int i = 0; i < sub_graphs_count; i++)
    {
        struct subgraph* sub_graph = *(struct subgraph**)get_vector_data(ir_graph->subgraph_list, i);
        sub_graph->input_wait_count = 0;

        for (int j = 0; j < sub_graph->input_num; j++)
        {
            struct tensor* tensor = ir_graph->tensor_list[sub_graph->input_tensor_list[j]];

            if (tensor->tensor_type == TENSOR_TYPE_VAR)
                sub_graph->input_wait_count++;
        }
    }
}

void dump_sub_graph(struct subgraph* sub_graph)
{
    TLOG_INFO("Sub graph[%d]: {%8s } has %d nodes, %d input tensors, %d output tensors.\n", sub_graph->index, sub_graph->device->name, sub_graph->node_num, sub_graph->input_num, sub_graph->output_num);
    TLOG_INFO("\tSub nodes: [ ");

    for (int j = 0; j < sub_graph->node_num - 1; j++)
    {
        int node_id = sub_graph->node_list[j];
        TLOG_INFO("%d, ", node_id);
    }
    TLOG_INFO("%d ].\n", sub_graph->node_list[sub_graph->node_num - 1]);

    TLOG_INFO("\tSub input tensors: [ ");
    for (int j = 0; j < sub_graph->input_num - 1; j++)
    {
        int tensor_id = sub_graph->input_tensor_list[j];
        TLOG_INFO("%d, ", tensor_id);
    }
    TLOG_INFO("%d ].\n", sub_graph->input_tensor_list[sub_graph->input_num - 1]);

    TLOG_INFO("\tSub output tensors: [ ");
    for (int j = 0; j < sub_graph->output_num - 1; j++)
    {
        int tensor_id = sub_graph->output_tensor_list[j];
        TLOG_INFO("%d, ", tensor_id);
    }
    TLOG_INFO("%d ].\n", sub_graph->output_tensor_list[sub_graph->output_num - 1]);
}
