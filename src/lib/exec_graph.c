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

#include "sys_port.h"
#include "tengine_ir.h"
#include "tengine_op.h"

#include "tengine_utils.h"
#include "tengine_log.h"

#include "exec_graph.h"
#include "nn_device.h"

#include "cast_param.h"

#include <math.h>

#define MODEL_COMPLEX_COUNT 3


typedef struct infer_step
{
    uint16_t index;
    uint16_t node_num;
    uint16_t* nodes;
} infer_step_t;


int node_in_sub_graph(const struct subgraph* sub_graph, const uint16_t* node_id)
{
    if (NULL == sub_graph || NULL == node_id)
    {
        return 0;
    }

    for (uint16_t i = 0; i < sub_graph->node_num; i++)
    {
        uint16_t current_node_id = sub_graph->node_list[i];
        if (*node_id == current_node_id)
        {
            return 1;
        }
    }

    return 0;
}


int find_children_nodes(const struct subgraph* sub_graph, const uint16_t* node_id, struct vector* children_nodes)
{
    if (node_in_sub_graph(sub_graph, node_id))
    {
        struct ir_node* ir_node = sub_graph->graph->node_list[*node_id];
        for (uint8_t i = 0; i < ir_node->output_num; i++)
        {
            uint16_t tensor_id = ir_node->output_tensors[i];
            struct ir_tensor* ir_tensor = sub_graph->graph->tensor_list[tensor_id];

            for (uint8_t j = 0; j < ir_tensor->consumer_num; j++)
            {
                uint16_t consumer_tensor_id = ir_tensor->consumer[j];
                struct ir_tensor* consumer_tensor = sub_graph->graph->tensor_list[consumer_tensor_id];
                uint16_t consumer_node_id = consumer_tensor->producer;

                if (node_in_sub_graph(sub_graph, &consumer_node_id))
                {
                    push_vector_data(children_nodes, &consumer_node_id);
                }
            }
        }
    }

    return 0;
}


int find_all_children_nodes(const struct subgraph* sub_graph, const uint16_t* node_id, struct vector* children_nodes)
{
    struct vector* last_children_node = create_vector(sizeof(uint16_t), NULL);
    find_children_nodes(sub_graph, node_id, last_children_node);

    while (1)
    {
        struct vector* current_children_node = create_vector(sizeof(uint16_t), NULL);

        // loop for save and search new children nodes
        for (int i = 0; i < get_vector_num(last_children_node); i++)
        {
            uint16_t* current_node_id = get_vector_data(last_children_node, i);
            push_vector_data(children_nodes, current_node_id);

            find_children_nodes(sub_graph, current_node_id, current_children_node);
        }

        int current_nodes_num = get_vector_num(current_children_node);

        if (current_nodes_num > 0)
        {
            release_vector(last_children_node);
            last_children_node = create_vector(sizeof(uint16_t), NULL);

            for (int i = 0; i < current_nodes_num; i++)
            {
                uint16_t* current_node_id = get_vector_data(current_children_node, i);
                push_vector_data(last_children_node, current_node_id);
            }

            release_vector(current_children_node);
        }
        else
        {
            release_vector(current_children_node);
            break;
        }
    }

    release_vector(last_children_node);

    return 0;
}


int move_one_step(const struct ir_graph* ir_graph, const infer_step_t* current_step, infer_step_t* next_step)
{
    next_step->index = current_step->index + 1;
    next_step->node_num = 0;
    next_step->nodes = NULL;

    int count = 0;

    for (uint16_t i = 0; i < current_step->node_num; i++)
    {
        struct ir_node* node = ir_graph->node_list[i];
        next_step->node_num += node->output_num;
        sys_realloc(next_step->nodes, sizeof(uint16_t) * next_step->node_num);

        for (int j = 0; j < node->output_num; j++)
        {
            uint16_t ir_tensor_id = node->output_tensors[j];
            struct ir_tensor* ir_tensor = ir_graph->tensor_list[ir_tensor_id];

            next_step->nodes[count] = ir_tensor->producer;
            count++;
        }
    }

    if (next_step->node_num != count + 1)
    {
        TLOG_ERR("Error: Next step count is not equ to current loop(%d v.s. %d).\n", next_step->node_num, count + 1);
        return -1;
    }

    return 0;
}


int parser_input_step(struct ir_graph* ir_graph, infer_step_t* first_step)
{
    first_step->index = 0;
    first_step->node_num = ir_graph->input_num;
    first_step->nodes = sys_malloc(sizeof(uint16_t) * first_step->node_num);

    for (uint16_t i = 0; i < ir_graph->input_num; i++)
    {
        uint16_t node_id = ir_graph->input_nodes[i];
        first_step->nodes[i] = node_id;
    }

    return 0;
}


int walk_through_graph(struct ir_graph* ir_graph, struct vector* roadmap)
{
    infer_step_t next_step = { 0, 0, NULL };
    parser_input_step(ir_graph, &next_step);

    if (get_vector_num(roadmap) == 0)
    {
        TLOG_ERR("Error: Input node of graph is 0.\n");
        return -1;
    }

    push_vector_data(roadmap, &next_step);

    while (1)
    {
        int step_count = get_vector_num(roadmap);
        infer_step_t* current_step = get_vector_data(roadmap, step_count - 1);

        int ret = move_one_step(ir_graph, current_step, &next_step);
        if (0 != ret)
        {
            TLOG_ERR("Error: Get next step failed[%d].\n", ret);
            return -1;
        }

        if (next_step.node_num > 0)
        {
            push_vector_data(roadmap, &next_step);
        }
        else
        {
            break;
        }
    }


    return 0;
}


void dump_walk_roadmap(const struct ir_graph* ir_graph, const struct vector* roadmap)
{

}




// more like Counting Sort Algorithm
void sort_nodes(struct vector* nodes)
{
    int nodes_count = get_vector_num(nodes);

    if (nodes_count <= 1)
        return;

    uint16_t max, min;
    uint16_t* tmp;

    tmp = get_vector_data(nodes, 0);
    min = *tmp;
    max = *tmp;

    for (int i = 0; i < nodes_count; i++)
    {
        tmp = get_vector_data(nodes, i);
        if (max < *tmp) max = *tmp;
        if (min > *tmp) min = *tmp;
    }

    uint16_t* tmp_space = (uint16_t*)sys_malloc(sizeof(uint16_t) * (max - min + 1));
    memset(tmp_space, 0, sizeof(uint16_t) * (max - min + 1));

    for (int i = 0; i < nodes_count; i++)
    {
        tmp = get_vector_data(nodes, i);

        uint16_t pos = *tmp - min;
        tmp_space[pos] = 1;
    }

    uint16_t index = 0;
    for (int i = 0; i < (max - min + 1); i++)
    {
        if (tmp_space[i] > 0)
        {
            uint16_t val = min + i;
            set_vector_data(nodes, index, &val);
            index++;
        }
    }

    sys_free(tmp_space);
}


int node_in_list(const struct ir_graph* ir_graph, struct vector* ops_list, const uint16_t node_id)
{
    if (NULL == ir_graph || NULL == ops_list)
    {
        return 0;
    }

    const uint16_t node_op_type = ir_graph->node_list[node_id]->op.op_type;

    for (int i = 0; i < get_vector_num(ops_list); i++)
    {
        int* loop_op = get_vector_data(ops_list, i);
        if (node_op_type == *loop_op)
            return 1;
    }

    return 0;
}


struct vector* get_graph_blocked_nodes(const struct ir_graph* ir_graph, struct vector* blocked_ops)
{
    struct vector* blocked_nodes_list = create_vector(sizeof(uint16_t), NULL);

    for (uint16_t i = 0; i < ir_graph->node_num; i++)
    {
        if (node_in_list(ir_graph, blocked_ops, i))
        {
            push_vector_data(blocked_nodes_list, &i);
        }
    }

    return blocked_nodes_list;
}


// policy has some issue, must be fixed
void split_graph_node_to_sub_graph(struct ir_graph* ir_graph, struct vector* allowed_ops, struct vector* blocked_ops)
{
    // get unsupported nodes
    struct vector* blocked_nodes_list = get_graph_blocked_nodes(ir_graph, blocked_ops);
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
                if (node_in_list(ir_graph, allowed_ops, j))
                {
                    const uint16_t node_op_type = ir_graph->node_list[j]->op.op_type;

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

            if (children_nodes_is_complicated < MODEL_COMPLEX_COUNT)   // directly add these nodes to sub graph list
            {
                struct subgraph* sub_graph = (struct subgraph*)sys_malloc(sizeof(struct subgraph));
                init_subgraph((struct ir_graph*)ir_graph, sub_graph, 0);

                // not including the last one
                sub_graph->node_num = last_node_id - first_node_id;
                sub_graph->node_list = (uint16_t*)sys_malloc(sizeof(uint16_t) * sub_graph->node_num);

                for (uint16_t j = 0; j < sub_graph->node_num; j++)
                {
                    sub_graph->node_list[j] = j + first_node_id;
                }

                struct nn_device* nn_dev = ir_graph->exec_attr->exec_context->def_dev;
                sub_graph->nn_dev = nn_dev;

                push_vector_data(ir_graph->subgraph_list, &sub_graph);
            }
            else
            {
                // add special device running nodes
                struct subgraph* sub_device_graph = (struct subgraph*)sys_malloc(sizeof(struct subgraph));
                init_subgraph((struct ir_graph*)ir_graph, sub_device_graph, 0);

                sub_device_graph->node_num = last_node_id - (first_node_id + 1);
                sub_device_graph->node_list = (uint16_t*)sys_malloc(sizeof(uint16_t) * sub_device_graph->node_num);

                for (uint16_t j = 0; j < sub_device_graph->node_num; j++)
                {
                    sub_device_graph->node_list[j] = j + first_node_id + 1;
                }

                struct nn_device* nn_dev = *(struct nn_device**)get_vector_data(ir_graph->exec_attr->exec_context->dev_list, 0);
                sub_device_graph->nn_dev = nn_dev;

                push_vector_data(ir_graph->subgraph_list, &sub_device_graph);

                // ---------------

                // add cpu running nodes
                struct subgraph* sub_cpu_graph = (struct subgraph*)sys_malloc(sizeof(struct subgraph));
                init_subgraph((struct ir_graph*)ir_graph, sub_cpu_graph, 0);

                sub_cpu_graph->node_num = 1;
                sub_cpu_graph->node_list = (uint16_t*)sys_malloc(sizeof(uint16_t) * sub_cpu_graph->node_num);
                sub_cpu_graph->node_list[0] = first_node_id;

                sub_cpu_graph->nn_dev = ir_graph->exec_attr->exec_context->def_dev;

                push_vector_data(ir_graph->subgraph_list, &sub_cpu_graph);
            }
        }
    }

    // add main sub graph
    struct subgraph* sub_graph = (struct subgraph*)sys_malloc(sizeof(struct subgraph));
    init_subgraph((struct ir_graph*)ir_graph, sub_graph, 0);

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
    struct nn_device* nn_dev = NULL;
    if (NULL != ir_graph->exec_attr->exec_context->dev_list && get_vector_num(ir_graph->exec_attr->exec_context->dev_list) > 0)
    {
        nn_dev = *(struct nn_device**)get_vector_data(ir_graph->exec_attr->exec_context->dev_list, 0);
    }
    else
    {
        nn_dev = get_default_nn_device();
    }
    sub_graph->nn_dev = nn_dev;

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

            if (current_sub_graph->nn_dev == last_sub_graph->nn_dev)
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

                remove_vector_by_idx(ir_graph->subgraph_list, (sub_graphs_count - 1) - i);

                same_sub_graph_found = 1;
                break;
            }
        }

        if (!same_sub_graph_found)
            break;
    }
}


void generate_sub_graph_io(struct ir_graph* ir_graph)
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
            struct ir_node* ir_node = ir_graph->node_list[node_id];
            if (ir_node->input_num > 0)
            {
                struct ir_tensor* tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);

                if (tensor->tensor_type == TENSOR_TYPE_INPUT || tensor->tensor_type == TENSOR_TYPE_VAR)
                {
                    random_input_id = tensor->idx;
                    break;
                }
            }
        }

        for (int i = 0; i < sub_graph->node_num; i++)
        {
            uint16_t node_id = sub_graph->node_list[i];
            struct ir_node* ir_node = ir_graph->node_list[node_id];
            if (ir_node->output_num > 0)
            {
                struct ir_tensor* tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
                random_output_id = tensor->idx;
                break;
            }
        }

        uint16_t min_input_tensor_id = random_input_id;
        uint16_t max_input_tensor_id = random_input_id;
        uint16_t min_output_tensor_id = random_output_id;
        uint16_t max_output_tensor_id = random_output_id;

        for (int i = 0; i < sub_graph->node_num; i++)
        {
            struct ir_node* ir_node = ir_graph->node_list[sub_graph->node_list[i]];

            for (int k = 0; k < ir_node->input_num; k++)
            {
                struct ir_tensor* tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[k]);

                if (tensor->tensor_type == TENSOR_TYPE_INPUT || tensor->tensor_type == TENSOR_TYPE_VAR)
                {
                    if (tensor->idx < min_input_tensor_id)
                        min_input_tensor_id = tensor->idx;

                    if (tensor->idx > max_input_tensor_id)
                        max_input_tensor_id = tensor->idx;
                }
            }

            for (int k = 0; k < ir_node->output_num; k++)
            {
                struct ir_tensor* tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[k]);

                if (tensor->tensor_type != TENSOR_TYPE_INPUT)
                {
                    if (tensor->idx < min_output_tensor_id)
                        min_output_tensor_id = tensor->idx;

                    if (tensor->idx > max_output_tensor_id)
                        max_output_tensor_id = tensor->idx;
                }
                else
                {
                    if (tensor->idx < min_input_tensor_id)
                        min_input_tensor_id = tensor->idx;

                    if (tensor->idx > max_input_tensor_id)
                        max_input_tensor_id = tensor->idx;
                }
            }
        }

        uint16_t* input_tensors = (uint16_t*)malloc(sizeof(uint16_t) * (max_input_tensor_id - min_input_tensor_id + 1));
        uint16_t* output_tensors = (uint16_t*)malloc(sizeof(uint16_t) * (max_output_tensor_id - min_output_tensor_id + 1));

        memset(input_tensors, 0, sizeof(uint16_t) * (max_input_tensor_id - min_input_tensor_id + 1));
        memset(output_tensors, 0, sizeof(uint16_t) * (max_output_tensor_id - min_output_tensor_id + 1));

        for (int j = 0; j < sub_graph->node_num; j++)
        {
            struct ir_node* ir_node = ir_graph->node_list[sub_graph->node_list[j]];

            for (int k = 0; k < ir_node->input_num; k++)
            {
                struct ir_tensor* tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[k]);

                if (tensor->tensor_type == TENSOR_TYPE_INPUT || tensor->tensor_type == TENSOR_TYPE_VAR)
                {
                    input_tensors[tensor->idx - min_input_tensor_id]++;
                }
            }

            for (int k = 0; k < ir_node->output_num; k++)
            {
                struct ir_tensor* tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[k]);

                if (tensor->tensor_type != TENSOR_TYPE_INPUT)
                {
                    if (tensor->tensor_type != TENSOR_TYPE_CONST)
                    {
                        output_tensors[tensor->idx - min_output_tensor_id]++;
                    }
                }
                else
                {
                    input_tensors[tensor->idx - min_input_tensor_id]++;
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


void add_sub_graph_to_ir_graph(struct ir_graph* ir_graph)
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
        sub_graph->idx = i;

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
            struct ir_tensor* ir_tensor = ir_graph->tensor_list[sub_graph->input_tensor_list[j]];

            if (ir_tensor->tensor_type != TENSOR_TYPE_INPUT)
            {
                uint16_t node_id = ir_tensor->producer;
                uint8_t sub_graph_id = ir_graph->node_list[node_id]->subgraph_idx;

                struct subgraph* target_sub_graph = *(struct subgraph**)get_vector_data(ir_graph->subgraph_list, sub_graph_id);

                int tensor_mask_as_out_flag = 0;
                for (int k = 0; k < target_sub_graph->output_num; k++)
                {
                    if (target_sub_graph->output_tensor_list[k] == ir_tensor->idx)
                    {
                        tensor_mask_as_out_flag = 1;
                        break;
                    }
                }

                if (!tensor_mask_as_out_flag)
                {
                    uint16_t* new_output_tensor_list = sys_malloc(sizeof(uint16_t) * (target_sub_graph->output_num + 1));
                    memcpy(new_output_tensor_list, target_sub_graph->output_tensor_list, sizeof(uint16_t) * target_sub_graph->output_num);
                    new_output_tensor_list[target_sub_graph->output_num] = ir_tensor->idx;

                    sys_free(target_sub_graph->output_tensor_list);
                    target_sub_graph->output_tensor_list = new_output_tensor_list;
                    target_sub_graph->output_num += 1;
                }
            }
        }
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
                remove_vector_by_idx(input_tensors, i);
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
                remove_vector_by_idx(output_tensors, i);
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
            remove_vector_by_idx(input_tensors, input_tensor_index);
            remove_vector_by_idx(output_tensors, output_tensor_index);
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
            struct ir_tensor* tensor = ir_graph->tensor_list[sub_graph->input_tensor_list[j]];

            if (tensor->tensor_type == TENSOR_TYPE_VAR)
                sub_graph->input_wait_count++;
        }
    }
}


void dump_sub_graph(struct subgraph* sub_graph)
{
    TLOG_INFO("Sub graph[%d]: {%8s } has %d nodes, %d input tensors, %d output tensors.\n", sub_graph->idx, sub_graph->nn_dev->name, sub_graph->node_num, sub_graph->input_num, sub_graph->output_num);
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


int split_graph(struct ir_graph* ir_graph)
{
    struct nn_device* default_device = get_default_nn_device();
    struct dev_allocator* exec_allocator = ir_graph->exec_attr->exec_context->dev_allocator;

    struct vector* allowed_ops = create_vector(sizeof(int), NULL);
    struct vector* blocked_ops = create_vector(sizeof(int), NULL);
    struct vector* precision = create_vector(sizeof(int), NULL);

    if (NULL != default_device && NULL != exec_allocator && 0 != strcmp(default_device->name, exec_allocator->name))
    {
        exec_allocator->describe(exec_allocator, allowed_ops, blocked_ops, precision);
    }

    split_graph_node_to_sub_graph(ir_graph, allowed_ops, blocked_ops);

    release_vector(allowed_ops);
    release_vector(blocked_ops);
    release_vector(precision);

    //
    generate_sub_graph_io(ir_graph);
    add_sub_graph_to_ir_graph(ir_graph);

    // add node sub graph id
    for (int i = 0; i < (uint16_t)get_vector_num(ir_graph->subgraph_list); i++)
    {
        struct subgraph* sub_graph = *(struct subgraph**)get_vector_data(ir_graph->subgraph_list, i);
        sub_graph->idx = i;

        for (uint16_t j = 0; j < sub_graph->node_num; j++)
        {
            uint16_t node_id = sub_graph->node_list[j];
            struct ir_node* ir_node = get_ir_graph_node(ir_graph, node_id);
            ir_node->subgraph_idx = sub_graph->idx;
        }
    }

    return 0;
}


int adapt_converted_tensor(struct subgraph* sub_graph, struct ir_node* original_node, struct ir_tensor* original_tensor, struct ir_node* converted_node, struct ir_tensor* converted_tensor)
{
    (void)sub_graph;
    (void)original_node;

    // copy shape
    converted_tensor->dim_num = original_tensor->dim_num;
    memcpy(converted_tensor->dims, original_tensor->dims, sizeof(int) * original_tensor->dim_num);
    converted_tensor->elem_num = original_tensor->elem_num;

    // copy layout
    converted_tensor->layout = original_tensor->layout;

    // add this original tensor consumer to converted tensor
    converted_tensor->consumer_num++;
    if (MAX_CONSUMER_NUM < converted_tensor->consumer_num)
        sys_realloc(converted_tensor->consumer, sizeof(int16_t) * converted_tensor->consumer_num);
    converted_tensor->consumer[converted_tensor->consumer_num - 1] = original_node->idx;

    // record useful consumer
    struct vector* tensor_consumer_list = create_vector(sizeof(uint16_t), NULL);
    for (uint8_t i = 0; i < original_tensor->consumer_num; i++)
    {
        uint16_t consumer_node_id = original_tensor->consumer[i];

        if (consumer_node_id != original_node->idx)
        {
            push_vector_data(tensor_consumer_list, &consumer_node_id);
        }
    }

    // record convert node
    push_vector_data(tensor_consumer_list, &converted_node->idx);

    // set original tensor consumer
    original_tensor->consumer_num = get_vector_num(tensor_consumer_list);
    for (int i = 0; i < original_tensor->consumer_num; i++)
    {
        uint16_t* consumer_node_id = get_vector_data(tensor_consumer_list, i);
        original_tensor->consumer[i] = *consumer_node_id;
    }

    release_vector(tensor_consumer_list);

    // set converted tensor producer
    converted_tensor->producer = converted_node->idx;

    // for debug
    //converted_tensor->data = sys_malloc(converted_tensor->elem_size * converted_tensor->elem_num);

    return 0;
}


int adapt_converted_node(struct subgraph* sub_graph, struct ir_node* original_node, struct ir_tensor* original_tensor, struct ir_node* convert_node, struct ir_tensor* converted_tensor)
{
    if (NULL == convert_node->input_tensors || NULL == convert_node->output_tensors)
    {
        convert_node->subgraph_idx = sub_graph->idx;

        // set node param
        struct cast_param* cast_param = ( struct cast_param* )convert_node->op.param_mem;
        cast_param->type_from = original_tensor->data_type;
        cast_param->type_to = converted_tensor->data_type;

        // set cast node input & output
        convert_node->input_num = 1;
        convert_node->input_tensors = (int16_t*)sys_malloc(sizeof(int16_t) * 1);
        set_ir_node_input_tensor(convert_node, 0, original_tensor);

        convert_node->output_num = 1;
        convert_node->output_tensors = (int16_t*)sys_malloc(sizeof(int16_t) * 1);
        set_ir_node_output_tensor(convert_node, 0, converted_tensor);

        // change original node input tensor to converted tensor
        for (uint8_t i = 0; i < original_node->input_num; i++)
            if (original_tensor->idx == original_node->input_tensors[i])
                original_node->input_tensors[i] = converted_tensor->idx;

        // allocate new memory for adding node
        uint16_t* new_node_list = sys_malloc((int)sizeof(uint16_t) * (sub_graph->node_num + 1));

        // insert & copy
        new_node_list[0] = convert_node->idx;
        for (uint16_t i = 0; i < sub_graph->node_num; i++)
        {
            new_node_list[i + 1] = sub_graph->node_list[i];
        }

        // add to sub graph node count
        sub_graph->node_num += 1;

        // free original node list
        sys_free(sub_graph->node_list);

        sub_graph->node_list = new_node_list;
    }

    return 0;
}


int remap_node_input_tensor(struct subgraph* sub_graph, struct ir_tensor* original_tensor, struct ir_tensor* converted_tensor)
{
    for (uint8_t i = 0; i < original_tensor->consumer_num; i++)
    {
        uint16_t consumer_node_id = original_tensor->consumer[i];
        struct ir_node* consumer_node = get_ir_graph_node(sub_graph->graph, consumer_node_id);

        if (consumer_node->subgraph_idx == sub_graph->idx)
        {
            for (uint8_t j = 0; j < consumer_node->input_num; j++)
            {
                if (original_tensor->idx == consumer_node->input_tensors[j])
                {
                    consumer_node->input_tensors[j] = converted_tensor->idx;
                }
            }
        }
    }

    return 0;
}


int add_transmuted_adapter_node_and_tensor(struct subgraph* sub_graph, struct ir_tensor* ir_tensor_transmuted, int precision)
{
    struct ir_tensor* ir_tensor_original = get_ir_graph_tensor(sub_graph->graph, ir_tensor_transmuted->idx);

    if (TENSOR_TYPE_VAR == ir_tensor_transmuted->tensor_type || TENSOR_TYPE_INPUT == ir_tensor_transmuted->tensor_type)
    {
        ir_tensor_original->data_type = ir_tensor_transmuted->data_type;

        for (uint8_t i = 0; i < ir_tensor_original->consumer_num; i++)
        {
            uint16_t consumer_node_id = ir_tensor_original->consumer[i];
            struct ir_node* consumer_node = get_ir_graph_node(sub_graph->graph, consumer_node_id);

            if (consumer_node->subgraph_idx != sub_graph->idx && 0 != strcmp(sub_graph->nn_dev->name, get_default_device()))
            {
                struct subgraph* sub_graph_related = get_ir_graph_subgraph(sub_graph->graph, consumer_node->subgraph_idx);

                struct ir_tensor* ir_tensor_converted = create_ir_tensor(sub_graph->graph, NULL, precision);
                struct ir_node* convert_node = create_ir_node(sub_graph->graph, NULL, OP_CAST, 0);

                adapt_converted_tensor(sub_graph_related, consumer_node, ir_tensor_original, convert_node, ir_tensor_converted);
                adapt_converted_node(sub_graph_related, consumer_node, ir_tensor_original, convert_node, ir_tensor_converted);

                break;
            }
        }
    }

    return 0;
}


int check_tensor_has_cast_node_and_tensor(struct subgraph* sub_graph, struct ir_tensor* input_tensor, struct ir_node* converted_node, struct ir_tensor* converted_tensor)
{
    for (uint8_t i = 0; i < input_tensor->consumer_num; i++)
    {
        uint16_t consumer_node_id = input_tensor->consumer[i];
        struct ir_node* consumer_node = get_ir_graph_node(sub_graph->graph, consumer_node_id);

        if (sub_graph->idx == consumer_node->subgraph_idx && OP_CAST == consumer_node->op.op_type)
        {
            if (NULL != converted_node)
                converted_node = consumer_node;

            if (NULL != converted_tensor)
            {
                uint16_t converted_tensor_id = consumer_node->output_tensors[0];
                converted_tensor = get_ir_graph_tensor(sub_graph->graph, converted_tensor_id);
            }

            return 1;
        }
    }

    converted_node = NULL;
    converted_tensor = NULL;

    return 0;
}


int deal_with_sub_graph_evolution(struct subgraph* sub_graph, struct dev_allocator* exec_allocator, int precision)
{
    if (0 != strcmp(sub_graph->nn_dev->name, get_default_device() ) )
    {
        TLOG_INFO("Optimizing sub graph(id: %d, dev: %s).\n", sub_graph->idx, sub_graph->nn_dev->name);

        struct vector* evolution_tensors = create_vector(sizeof(struct ir_tensor), NULL);
        struct vector* evolution_nodes = create_vector(sizeof(struct ir_node), NULL);

        exec_allocator->evaluation(exec_allocator, sub_graph, evolution_tensors, evolution_nodes);

        TLOG_INFO("Evolution tensors count %d.\n", get_vector_num(evolution_tensors));
        TLOG_INFO("Evolution nodes count %d.\n", get_vector_num(evolution_nodes));

        for (int m = 0; m < get_vector_num(evolution_tensors); m++)
        {
            struct ir_tensor* ir_tensor = get_vector_data(evolution_tensors, m);

            add_transmuted_adapter_node_and_tensor(sub_graph, ir_tensor, precision);
        }

        release_vector(evolution_tensors);
        release_vector(evolution_nodes);
    }
}


int add_cast_node_and_tensor_for_input(struct subgraph* sub_graph, struct ir_tensor* input_tensor, int precision)
{
    struct vector* consumer_node_list = create_vector(sizeof(uint16_t), NULL);

    for (uint8_t i = 0; i < input_tensor->consumer_num; i++)
    {
        uint16_t consumer_node_id = input_tensor->consumer[i];
        struct ir_node* consumer_node = get_ir_graph_node(sub_graph->graph, consumer_node_id);

        if (sub_graph->idx == consumer_node->subgraph_idx)
        {
            push_vector_data(consumer_node_list, &consumer_node_id);
        }
    }

    int consumer_node_count = get_vector_num(consumer_node_list);

    if (consumer_node_count > 0)
    {
        // add a name
        char* convert_name = sys_malloc(sizeof(char) * (strlen(input_tensor->name) + strlen("_cast") + 1));
        memset(convert_name, 0, sizeof(char) * (strlen(input_tensor->name) + strlen("_cast") + 1));
        sprintf(convert_name, "%s_cast", input_tensor->name);

        struct ir_tensor* input_tensor_converted = create_ir_tensor(sub_graph->graph, convert_name, precision);

        struct ir_node* input_tensor_producer_node = get_ir_graph_node(sub_graph->graph, input_tensor->producer);

        char* convert_node_name = sys_malloc(sizeof(char) * (strlen(input_tensor_producer_node->name) + strlen("_cast") + 1));
        memset(convert_name, 0, sizeof(char) * (strlen(input_tensor_producer_node->name) + strlen("_cast") + 1));
        sprintf(convert_name, "%s_cast", input_tensor_producer_node->name);

        struct ir_node* convert_node = create_ir_node(sub_graph->graph, convert_name, OP_CAST, 0);

        for (int i = 0; i < consumer_node_count; i++)
        {
            uint16_t* consumer_node_id = get_vector_data(consumer_node_list, i);
            struct ir_node* consumer_node = get_ir_graph_node(sub_graph->graph, *consumer_node_id);

            adapt_converted_tensor(sub_graph, consumer_node, input_tensor, convert_node, input_tensor_converted);
            adapt_converted_node(sub_graph, consumer_node, input_tensor, convert_node, input_tensor_converted);
        }

        sys_free(convert_name);
    }

    release_vector(consumer_node_list);

    return 0;
}


int optimize_graph(struct ir_graph* ir_graph, int precision)
{
    const int sub_graph_count = get_vector_num(ir_graph->subgraph_list);
    int is_heterogeneous_computing = 0;

    struct dev_allocator* exec_allocator = ir_graph->exec_attr->exec_context->dev_allocator;
    if (NULL != exec_allocator && 0 != strcmp(exec_allocator->name, get_default_device()) && sub_graph_count > 1)
    {
        is_heterogeneous_computing = 1;
    }
    /*if (heterogeneous_computing)
    {
        for (int i = 0; i < sub_graph_count; i++)
        {
            struct subgraph* sub_graph = *(struct subgraph**)get_vector_data(ir_graph->subgraph_list, i);
            deal_with_sub_graph_evolution(sub_graph, exec_allocator, precision);
        }
    }*/

    if (is_heterogeneous_computing)
    {
        // for each input tensor, check precision and insert needed cast node & tensor
        for (int i = 0; i < sub_graph_count; i++)
        {
            struct subgraph* sub_graph = *(struct subgraph**)get_vector_data(ir_graph->subgraph_list, i);

            if (0 == strcmp(sub_graph->nn_dev->name, get_default_device()))
            {
                for (uint8_t j = 0; j < sub_graph->input_num; j++)
                {
                    uint16_t input_tensor_id = sub_graph->input_tensor_list[j];
                    struct ir_tensor* input_tensor = get_ir_graph_tensor(ir_graph, input_tensor_id);

                    int input_tensor_has_cast = check_tensor_has_cast_node_and_tensor(sub_graph, input_tensor, NULL, NULL);

                    // has a cast, broadcast to the related other
                    if (0 == input_tensor_has_cast)
                    {
                        add_cast_node_and_tensor_for_input(sub_graph, input_tensor, precision);
                    }
                }
            }
        }

        dump_graph(ir_graph);

        // deal with mix
        for (int i = 0; i < sub_graph_count; i++)
        {
            struct subgraph* sub_graph = *(struct subgraph**)get_vector_data(ir_graph->subgraph_list, i);

            // change CPU tensor type
            if (0 == strcmp(sub_graph->nn_dev->name, get_default_device()))
            {
                for (uint16_t j = 0; j < sub_graph->node_num; j++)
                {
                    uint16_t node_id = sub_graph->node_list[j];
                    struct ir_node* node = get_ir_graph_node(ir_graph, node_id);


                    if (OP_CAST != node->op.op_type)
                    {

                        for (uint8_t k = 0; k < node->output_num; k++)
                        {
                            uint16_t tensor_id = node->output_tensors[k];
                            struct ir_tensor* tensor = get_ir_graph_tensor(ir_graph, tensor_id);

                            if (precision != tensor->data_type)
                            {
                                if (TENSOR_TYPE_CONST == tensor->tensor_type)
                                {
                                    TLOG_INFO("Subgraph[%d] node[%d]: ", i, node_id);
                                    TLOG_INFO("Type [%d] -> [%d]; Size [%d] -> [%d]. Scale: %.6f, Zp: %d.", tensor->data_type, precision, tensor->elem_size, data_type_size(precision), tensor->scale, tensor->zero_point);

                                    if (TENGINE_DT_UINT8 == tensor->data_type && TENGINE_DT_FP32 == precision)
                                    {
                                        float* cast_data = (float*)sys_malloc(data_type_size(precision) * tensor->elem_num);
                                        uint8_t* original_data = (uint8_t*)tensor->data;

                                        for (int p = 0; p < tensor->elem_num; p++)
                                        {
                                            cast_data[p] = (float)((int)original_data[p] - tensor->zero_point) * tensor->scale;
                                        }

                                        //sys_free(tensor->data);
                                        tensor->data = cast_data;
                                    }

                                    if (TENGINE_DT_FP32 == tensor->data_type && TENGINE_DT_UINT8 == precision)
                                    {
                                        uint8_t* cast_data = (uint8_t*)sys_malloc(data_type_size(precision) * tensor->elem_num);
                                        float* original_data = (float*)tensor->data;

                                        for (int p = 0; p < tensor->elem_num; p++)
                                        {
                                            int val = (int)round((original_data[p] / tensor->scale) + (float)tensor->zero_point);

                                            if (val > 255) { val = 255; };
                                            if (val <   0) { val =   0; };

                                            cast_data[p] = (uint8_t)val;
                                        }

                                        //sys_free(tensor->data);
                                        tensor->data = cast_data;
                                    }


                                    tensor->data_type = precision;
                                    tensor->elem_size = data_type_size(precision);

                                    TLOG_INFO("\n");
                                }
                                else
                                {
                                    TLOG_INFO("Subgraph[%d] node[%d]: ", i, node_id);
                                    TLOG_INFO("Type [%d] -> [%d]; Size [%d] -> [%d]. Scale: %.6f, Zp: %d.", tensor->data_type, precision, tensor->elem_size, data_type_size(precision), tensor->scale, tensor->zero_point);

                                    tensor->data_type = precision;
                                    tensor->elem_size = data_type_size(precision);

                                    TLOG_INFO("\n");
                                }
                            }
                        }
                    }
                }
            }
            else
            {
                // deal with conv bias
                for (uint16_t j = 0; j < sub_graph->node_num; j++)
                {
                    uint16_t node_id = sub_graph->node_list[j];
                    struct ir_node* node = get_ir_graph_node(ir_graph, node_id);

                    if (OP_CONV == node->op.op_type)
                    {
                        for (uint8_t k = 0; k < node->input_num; k++)
                        {
                            uint16_t tensor_id = node->input_tensors[k];
                            struct ir_tensor* tensor = get_ir_graph_tensor(ir_graph, tensor_id);

                            if (TENGINE_DT_FP32 == tensor->data_type)
                            {
                                TLOG_INFO("Subgraph[%d] node[%d]: ", i, node_id);

                                if (TENSOR_TYPE_CONST == tensor->tensor_type && 2 == k)
                                {
                                    TLOG_INFO("Type [%d] -> [%d]; Size [%d] -> [%d]. Scale: %.6f, Zp: %d.", tensor->data_type, TENGINE_DT_INT32, tensor->elem_size, data_type_size(TENGINE_DT_INT32), tensor->scale, tensor->zero_point);

                                    int32_t* cast_data = (int32_t*)sys_malloc(sizeof(int32_t) * tensor->elem_num);
                                    float* original_data = (float*)tensor->data;

                                    for (int p = 0; p < tensor->elem_num; p++)
                                    {
                                        cast_data[p] = (int32_t)round(original_data[p] / tensor->scale);
                                    }

                                    //sys_free(tensor->data);
                                    tensor->data = cast_data;

                                    tensor->data_type = TENGINE_DT_INT32;
                                    tensor->elem_size = data_type_size(TENGINE_DT_INT32);
                                }
                                else
                                {
                                    if (TENSOR_TYPE_CONST == tensor->tensor_type)
                                    {
                                        TLOG_INFO("Type [%d] -> [%d]; Size [%d] -> [%d]. Scale: %.6f, Zp: %d.", tensor->data_type, TENGINE_DT_UINT8, tensor->elem_size, data_type_size(TENGINE_DT_UINT8), tensor->scale, tensor->zero_point);

                                        uint8_t* cast_data = (uint8_t*)sys_malloc(sizeof(uint8_t) * tensor->elem_num);
                                        float* original_data = (float*)tensor->data;

                                        for (int p = 0; p < tensor->elem_num; p++)
                                        {
                                            int val = (int)round((original_data[p] / tensor->scale) + (float)tensor->zero_point);

                                            if (val > 255) { val = 255; };
                                            if (val <   0) { val =   0; };

                                            cast_data[p] = (uint8_t)val;
                                        }

                                        //sys_free(tensor->data);
                                        tensor->data = cast_data;
                                    }

                                    tensor->data_type = TENGINE_DT_UINT8;
                                    tensor->elem_size = data_type_size(TENGINE_DT_UINT8);
                                }

                                TLOG_INFO("\n");
                            }
                        }
                    }
                }

                // other
                for (uint16_t j = 0; j < sub_graph->node_num; j++)
                {
                    uint16_t node_id = sub_graph->node_list[j];
                    struct ir_node* node = get_ir_graph_node(ir_graph, node_id);

                    TLOG_INFO("Subgraph[%d] node[%d]: ", i, node_id);

                    if (OP_CAST != node->op.op_type)
                    {
                        for (uint8_t k = 0; k < node->output_num; k++)
                        {
                            uint16_t tensor_id = node->output_tensors[k];
                            struct ir_tensor* tensor = get_ir_graph_tensor(ir_graph, tensor_id);

                            if (TENGINE_DT_FP32 == tensor->data_type)
                            {
                                if (TENSOR_TYPE_CONST == tensor->tensor_type)
                                {
                                    if (OP_CONV == node->op.op_type && 3 == k)
                                    {
                                        TLOG_INFO("Type [%d] -> [%d]; Size [%d] -> [%d]. Scale: %.6f, Zp: %d.", tensor->data_type, TENGINE_DT_INT32, tensor->elem_size, data_type_size(TENGINE_DT_INT32), tensor->scale, tensor->zero_point);

                                        int32_t* cast_data = (int32_t*)sys_malloc(sizeof(int32_t) * tensor->elem_num);
                                        float* original_data = (float*)tensor->data;

                                        for (int p = 0; p < tensor->elem_num; p++)
                                        {
                                            cast_data[p] = (int32_t)round(original_data[p] / tensor->scale);
                                        }

                                        //sys_free(tensor->data);
                                        tensor->data = cast_data;

                                        tensor->data_type = TENGINE_DT_INT32;
                                        tensor->elem_size = data_type_size(TENGINE_DT_INT32);
                                    }
                                    else
                                    {
                                        TLOG_INFO("Type [%d] -> [%d]; Size [%d] -> [%d]. Scale: %.6f, Zp: %d.", tensor->data_type, TENGINE_DT_UINT8, tensor->elem_size, data_type_size(TENGINE_DT_UINT8), tensor->scale, tensor->zero_point);

                                        uint8_t* cast_data = (uint8_t*)sys_malloc(data_type_size(TENGINE_DT_UINT8) * tensor->elem_num);
                                        float* original_data = (float*)tensor->data;

                                        for (int p = 0; p < tensor->elem_num; p++)
                                        {
                                            int val = (int)round((original_data[p] / tensor->scale) + (float)tensor->zero_point);

                                            if (val > 255) { val = 255; };
                                            if (val <   0) { val =   0; };

                                            cast_data[p] = (uint8_t)val;
                                        }

                                        //sys_free(tensor->data);
                                        tensor->data = cast_data;

                                        tensor->data_type = TENGINE_DT_UINT8;
                                        tensor->elem_size = data_type_size(TENGINE_DT_UINT8);
                                    }
                                }
                                else
                                {
                                    TLOG_INFO("Type [%d] -> [%d]; Size [%d] -> [%d]. Scale: %.6f, Zp: %d.", tensor->data_type, TENGINE_DT_UINT8, tensor->elem_size, data_type_size(TENGINE_DT_UINT8), tensor->scale, tensor->zero_point);

                                    tensor->data_type = TENGINE_DT_UINT8;
                                    tensor->elem_size = data_type_size(TENGINE_DT_UINT8);
                                }
                            }
                        }

                        TLOG_INFO("\n");
                    }
                }
            }
        }
    }

    // dump graph
    TLOG_INFO("Total sub_graph: %d.\n", sub_graph_count);
    for (int i = 0; i < sub_graph_count; i++)
    {
        struct subgraph* sub_graph = *(struct subgraph**)get_vector_data(ir_graph->subgraph_list, i);
        dump_sub_graph(sub_graph);
    }

    dump_graph(ir_graph);

    return 0;
}
