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

#include <stdio.h>
#include <string.h>

#include "sys_port.h"
#include "vector.h"
#include "tengine_op.h"
#include "tengine_ir.h"
#include "tengine_exec.h"

#include "acl_allocator.h"

const int acl_white_list[] = {OP_CONST, OP_INPUT, OP_BATCHNORM, OP_CONV, 
                              // OP_CONCAT,
                              OP_DROPOUT, OP_ELTWISE, OP_FC, OP_POOL, OP_RELU,
                              OP_RELU6, OP_RESIZE, OP_SOFTMAX};
const int acl_allow_list[] = { OP_CONV, OP_FC, OP_POOL };

int node_in_white_list(const struct ir_graph* ir_graph, const uint16_t node_id)
{
    int block_list_size = sizeof(acl_white_list) / sizeof(acl_white_list[0]);

    const uint16_t node_op_type = ir_graph->node_list[node_id]->op.op_type;

    
    for (int i = 0; i < block_list_size; i++)
    {
        if (node_op_type == acl_white_list[i])
            return 1;
    }

    return 0;
}

int node_in_allow_list(const struct ir_graph* ir_graph, const uint16_t node_id)
{
    int allow_list_size = sizeof(acl_allow_list) / sizeof(acl_allow_list[0]);

    const uint16_t node_op_type = ir_graph->node_list[node_id]->op.op_type;

    for (int i = 0; i < allow_list_size; i++)
    {
        if (node_op_type == acl_allow_list[i])
            return 1;
    }

    return 0;
}

#ifdef PARTITION_VERSION_2
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
        if (max < *tmp)
            max = *tmp;
        if (min > *tmp)
            min = *tmp;
    }

    uint16_t* tmp_space = ( uint16_t* )sys_malloc(sizeof(uint16_t) * (max - min + 1));
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

// more like Counting Sort Algorithm
void sort_io_array(uint16_t* tensor_array, int size)
{
    if (size <= 1)
        return;

    uint16_t max, min;
    uint16_t tmp;

    tmp = tensor_array[0];
    min = tmp;
    max = tmp;

    for (int i = 0; i < size; i++)
    {
        tmp = tensor_array[i];
        if (max < tmp)
            max = tmp;
        if (min > tmp)
            min = tmp;
    }

    uint16_t* tmp_space = ( uint16_t* )sys_malloc(sizeof(uint16_t) * (max - min + 1));
    memset(tmp_space, 0, sizeof(uint16_t) * (max - min + 1));

    for (int i = 0; i < size; i++)
    {
        tmp = tensor_array[i];

        uint16_t pos = tmp - min;
        tmp_space[pos] = 1;
    }

    uint16_t index = 0;
    for (int i = 0; i < (max - min + 1); i++)
    {
        if (tmp_space[i] > 0)
        {
            uint16_t val = min + i;
            tensor_array[index] = val;
            index++;
        }
    }

    sys_free(tmp_space);
}

// reverse judege calculation (search up)
// if not complex enough, change support attr to cpu 
// return change or not
int reverse_judge_calculation(struct ir_graph* ir_graph, int search_node_list[], int node_id)
{
    struct vector* bfs_queue = create_vector(sizeof(int), NULL);
    struct vector* tmp_node_id = create_vector(sizeof(int), NULL);
    push_vector_data(bfs_queue, &node_id);
    int first_node_support_attr = -1; // -2 for cpu, -1 for acl
    int count = 0; // num of find node
    if(node_in_allow_list(ir_graph, node_id))
        count++;
    int queue_num = get_vector_num(bfs_queue);
    while(queue_num != 0)
    {
        int queue_front = *(int*)get_vector_data(bfs_queue, 0); // queue.front
        remove_vector_by_idx(bfs_queue, 0);                     // queue.pop

        // searching all consumer nodes of all output tensors
        struct ir_node* cur_node = get_ir_graph_node(ir_graph, queue_front);
        int input_num = cur_node->input_num;
        for (int i = 0; i < input_num; i++)
        {
            struct ir_tensor* cur_input_tensor = get_ir_graph_tensor(ir_graph, cur_node->input_tensors[i]);
            int pre_node_id = cur_input_tensor->producer;

            // push condition judge
            // if supported attr no equal
            if(search_node_list[pre_node_id] != first_node_support_attr)
                continue;
                
            // if the node has allocated
            if(search_node_list[pre_node_id] > 0)
                continue;
            // if the node has pre_node which unallocated
            // todo

            // push node to queue and set subgraph_idx and add count
            push_vector_data(bfs_queue, &pre_node_id);
            push_vector_data(tmp_node_id, &pre_node_id);
            if(node_in_allow_list(ir_graph, node_id))
            {
                count++;
                if(count > 3)
                    return 0;
            }
        }
        queue_num = get_vector_num(bfs_queue);
    }
    for (int i = 0; i < get_vector_num(tmp_node_id); i++)
    {
        int node_id = *(int*)get_vector_data(tmp_node_id, i);
        search_node_list[node_id] = -2;
    }
    return 1;
}

// judege calculation (search down)
// if not complex enough, change support attr to cpu 
// return change or not
int judge_calculation(struct ir_graph* ir_graph, int search_node_list[], int node_id)
{
    struct vector* bfs_queue = create_vector(sizeof(int), NULL);
    struct vector* tmp_node_id = create_vector(sizeof(int), NULL);
    push_vector_data(bfs_queue, &node_id);
    int first_node_support_attr = -1; // -2 for cpu, -1 for acl
    int count = 0; // num of find node
    if(node_in_allow_list(ir_graph, node_id))
        count++;
    int queue_num = get_vector_num(bfs_queue);
    while(queue_num != 0)
    {
        int queue_front = *(int*)get_vector_data(bfs_queue, 0); // queue.front
        remove_vector_by_idx(bfs_queue, 0);                     // queue.pop

        // searching all consumer nodes of all output tensors
        struct ir_node* cur_node = get_ir_graph_node(ir_graph, queue_front);
        int output_num = cur_node->output_num;
        for (int i = 0; i < output_num; i++)
        {
            struct ir_tensor* cur_output_tensor = get_ir_graph_tensor(ir_graph, cur_node->output_tensors[i]);
            for (int j = 0; j < cur_output_tensor->consumer_num; j++)
            {
                int later_node_id = cur_output_tensor->consumer[j];

                // push condition judge
                // if supported attr no equal
                if(search_node_list[later_node_id] != first_node_support_attr)
                    continue;
                    
                // if the node has allocated
                if(search_node_list[later_node_id] > 0)
                    continue;
                // if the node has pre_node which unallocated
                // todo

                // push node to queue and set subgraph_idx and add count
                push_vector_data(bfs_queue, &later_node_id);
                push_vector_data(tmp_node_id, &later_node_id);
                if(node_in_allow_list(ir_graph, node_id))
                {
                    count++;
                    if(count > 3)
                        return 0;
                }
            }
        }
        queue_num = get_vector_num(bfs_queue);
    }
    for (int i = 0; i < get_vector_num(tmp_node_id); i++)
    {
        int node_id = *(int*)get_vector_data(tmp_node_id, i);
        search_node_list[node_id] = -2;
    }
    return 1;
}

int reverse_bfs_from_incomplete_node(struct ir_graph* ir_graph, int search_node_list[], int subgraph_idx, int node_id)
{
    struct vector* bfs_queue = create_vector(sizeof(int), NULL);
    push_vector_data(bfs_queue, &node_id);
    int first_node_support_attr = search_node_list[node_id]; // -2 for cpu, -1 for acl
    search_node_list[node_id] = subgraph_idx;
    int count = 1; // num of find node
    int queue_num = get_vector_num(bfs_queue);
    while(queue_num != 0)
    {
        int queue_front = *(int*)get_vector_data(bfs_queue, 0); // queue.front
        remove_vector_by_idx(bfs_queue, 0);                     // queue.pop

        // searching all producer nodes of all input tensors
        struct ir_node* cur_node = get_ir_graph_node(ir_graph, queue_front);
        int input_num = cur_node->input_num;
        for (int i = 0; i < input_num; i++)
        {
            struct ir_tensor* cur_input_tensor = get_ir_graph_tensor(ir_graph, cur_node->input_tensors[i]);

            int pre_node_id = cur_input_tensor->producer;

            // push condition judge
            // if the node has allocated
            if(search_node_list[pre_node_id] >= 0)
                continue;
            // if supported attr no equal
            if(search_node_list[pre_node_id] != first_node_support_attr)
            {
                if(first_node_support_attr == -2)
                {
                    // judge acl calculation
                    int is_change = 0;
                    is_change = reverse_judge_calculation(ir_graph, search_node_list, pre_node_id);

                    if(is_change)
                        search_node_list[pre_node_id] = -2;
                    else
                        continue;
                }
                else
                    continue;
            }
            
            // push node to queue and set subgraph_idx and add count
            push_vector_data(bfs_queue, &pre_node_id);
            search_node_list[pre_node_id] = subgraph_idx;
            count++;

            // set node const tensor producer node subgraph_idx and add count
            struct ir_node* pre_node = get_ir_graph_node(ir_graph, pre_node_id);
            for (int k = 0; k < pre_node->input_num; k++)
            {
                struct ir_tensor* pre_node_input_tensor = get_ir_graph_tensor(ir_graph, pre_node->input_tensors[k]);
                if(pre_node_input_tensor->tensor_type == TENSOR_TYPE_CONST)
                {
                    int pre_node_id = pre_node_input_tensor->producer;
                    search_node_list[pre_node_id] = subgraph_idx;
                    count++;
                }
            }
        }
        queue_num = get_vector_num(bfs_queue);
    }
    return count;
}

int bfs_from_first_node_id(struct ir_graph* ir_graph, int search_node_list[], int subgraph_idx, int node_id)
{
    struct vector* bfs_queue = create_vector(sizeof(int), NULL);
    push_vector_data(bfs_queue, &node_id);
    int first_node_support_attr = search_node_list[node_id]; // -2 for cpu, -1 for acl
    search_node_list[node_id] = subgraph_idx;
    int count = 1; // num of find node
    int queue_num = get_vector_num(bfs_queue);
    while(queue_num != 0)
    {
        int queue_front = *(int*)get_vector_data(bfs_queue, 0); // queue.front
        remove_vector_by_idx(bfs_queue, 0);                     // queue.pop

        // searching all consumer nodes of all output tensors
        struct ir_node* cur_node = get_ir_graph_node(ir_graph, queue_front);
        int output_num = cur_node->output_num;
        for (int i = 0; i < output_num; i++)
        {
            struct ir_tensor* cur_output_tensor = get_ir_graph_tensor(ir_graph, cur_node->output_tensors[i]);
            for (int j = 0; j < cur_output_tensor->consumer_num; j++)
            {
                int later_node_id = cur_output_tensor->consumer[j];

                // push condition judge
                // if the node has allocated
                if(search_node_list[later_node_id] >= 0)
                    continue;
                // if supported attr no equal
                if(search_node_list[later_node_id] != first_node_support_attr)
                {
                    if(first_node_support_attr == -2)
                    {
                        // judge acl calculation
                        int is_change = 0;
                        is_change = judge_calculation(ir_graph, search_node_list, later_node_id);

                        if(is_change)
                            search_node_list[later_node_id] = -2;
                        else
                            continue;
                    }
                    else
                        continue;
                }
                    
                // if the node has other pre_nodes which unallocated
                int complete_flag = 1;
                struct ir_node* later_node = get_ir_graph_node(ir_graph, later_node_id);
                for (int k = 0; k < later_node->input_num; k++)
                {
                    struct ir_tensor* later_node_input_tensor = get_ir_graph_tensor(ir_graph, later_node->input_tensors[k]);
                    if(later_node_input_tensor->tensor_type == TENSOR_TYPE_CONST)
                        continue;
                    int later_pre_node_id = later_node_input_tensor->producer;
                    if(search_node_list[later_pre_node_id] < 0)
                    {
                        complete_flag = 0;
                        break;
                    }
                }

                // if(!complete_flag)
                //     continue;

                // reverse bfs test
                if(!complete_flag)
                {
                    count += reverse_bfs_from_incomplete_node(ir_graph, search_node_list, subgraph_idx, later_node_id);
                    push_vector_data(bfs_queue, &later_node_id);
                    continue;
                }
                
                // push node to queue and set subgraph_idx and add count
                push_vector_data(bfs_queue, &later_node_id);
                search_node_list[later_node_id] = subgraph_idx;
                count++;

                // set node const tensor producer node subgraph_idx and add count
                for (int k = 0; k < later_node->input_num; k++)
                {
                    struct ir_tensor* later_node_input_tensor = get_ir_graph_tensor(ir_graph, later_node->input_tensors[k]);
                    if(later_node_input_tensor->tensor_type == TENSOR_TYPE_CONST)
                    {
                        int pre_node_id = later_node_input_tensor->producer;
                        search_node_list[pre_node_id] = subgraph_idx;
                        count++;
                    }
                }
            }
        }
        queue_num = get_vector_num(bfs_queue);
    }
    return count;
}

int find_children_nodes(struct ir_graph* ir_graph, int search_node_list[], int subgraph_idx)
{
    int find_node_num = 0;
    int node_id = -1;

    // find first node 
    if(subgraph_idx == 0) // from ir graph find input node
    {
        if(ir_graph->input_num == 1)
        {
            struct ir_node* node = get_ir_graph_node(ir_graph, ir_graph->input_nodes[0]);
            node_id = node->idx;
        }
        else
            return -1;
    }
    else  // from unallocated node list find first optimal node
    {
        for (int i = 0; i < ir_graph->node_num; i++)
        {
            if(search_node_list[i] < 0)
            {
                struct ir_node* tmp_node = get_ir_graph_node(ir_graph, i);
                int tmp_input_num = tmp_node->input_num;
                int all_pre_node_allocated = 1;
                for (int j = 0; j < tmp_input_num; j++)
                {
                    struct ir_tensor* tmp_input_tensor = get_ir_graph_tensor(ir_graph, tmp_node->input_tensors[j]);
                    int pre_node_id = tmp_input_tensor->producer;
                    if(search_node_list[pre_node_id] < 0)
                    {
                        all_pre_node_allocated = 0;
                        break;
                    }
                }
                if(all_pre_node_allocated)
                {
                    node_id = i;
                    break;
                }
            }
        }
        
    }
    if(node_id == -1)
        return -1;

    // printf("first node id:%d\n",node_id);
    // bfs
    find_node_num = bfs_from_first_node_id(ir_graph, search_node_list, subgraph_idx, node_id);
    
    return find_node_num;
}

void split_graph_node_to_sub_graph(struct ir_graph* ir_graph)
{
    /* 
        search_node_list
        <0   : unallocated
             -1 : supported
             -2 : unsupported
        >=0  : allocated
             idx : subgraph_idx
    */
    int search_node_list[ir_graph->node_num];
    memset(search_node_list, -1, sizeof(int) * ir_graph->node_num);
    for (uint16_t i = 0; i < ir_graph->node_num; i++)
    {
        if (!node_in_white_list(ir_graph, i))
            search_node_list[i] = -2;
    }
    
    int allocated_num = 0;
    int subgraph_idx = 0;
    while(allocated_num < ir_graph->node_num)
    {
        int cur_allo_num = 0;
        cur_allo_num = find_children_nodes(ir_graph, search_node_list, subgraph_idx);
        if(cur_allo_num < 0)
            continue;
        allocated_num += cur_allo_num;
        subgraph_idx++;
        // printf("allocated_num:%d\n", allocated_num);
    }
    if(allocated_num != ir_graph->node_num)
        printf("WARNING!!! allocated_num != ir_graph->node_num\n");

    // test
    struct nn_device* test_dev = (struct nn_device*)sys_malloc(sizeof(struct nn_device));
    memcpy(test_dev, get_default_nn_device(), sizeof(struct nn_device));
    test_dev->name = "test";

    int subgraph_num = subgraph_idx;
    for (int i = 0; i < subgraph_num; i++)
    {
        int node_num = 0;
        int allow_num = 0;
        struct vector* tmp_node_list = create_vector(sizeof(int), NULL);
        for (int j = 0; j < ir_graph->node_num; j++)
        {
            if(search_node_list[j] == i)
            {
                node_num++;
                push_vector_data(tmp_node_list, &j);
                if(node_in_allow_list(ir_graph, j));
                    allow_num++;
            }
        }

        struct subgraph* sub_graph = ( struct subgraph* )sys_malloc(sizeof(struct subgraph));
        init_subgraph(( struct ir_graph* )ir_graph, sub_graph, i);
        sub_graph->node_num = node_num;
        sub_graph->node_list = ( uint16_t* )sys_malloc(sizeof(uint16_t) * node_num);

        for (uint16_t k = 0; k < sub_graph->node_num; k++)
        {
            int tmp = *(int*)get_vector_data(tmp_node_list, k);
            sub_graph->node_list[k] = (uint16_t)tmp;
        }

        uint16_t first_node_id = sub_graph->node_list[0];
        if (!node_in_white_list(ir_graph, first_node_id)) // cpu
        {
            sub_graph->nn_dev = get_default_nn_device(); 
        }
        else // acl
        {
            if(allow_num < 3)
                sub_graph->nn_dev = get_default_nn_device(); 
            else
                // sub_graph->nn_dev = test_dev;
                sub_graph->nn_dev = ir_graph->nn_dev;
        }
        push_vector_data(ir_graph->subgraph_list, &sub_graph);
        release_vector(tmp_node_list);
    }

        // optimize the sub graphs
    while (1)
    {
        int same_sub_graph_found = 0;
        int sub_graphs_count = get_vector_num(ir_graph->subgraph_list);
        for (int i = 1; i < sub_graphs_count; i++)
        {
            struct subgraph* last_sub_graph = get_ir_graph_subgraph(ir_graph, (sub_graphs_count - 1) - (i - 1));
            struct subgraph* current_sub_graph = get_ir_graph_subgraph(ir_graph, (sub_graphs_count - 1) - i);

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
        struct subgraph* sub_graph = *( struct subgraph** )get_vector_data(ir_graph->subgraph_list, index);

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

        uint16_t* input_tensors =
            ( uint16_t* )malloc(sizeof(uint16_t) * (max_input_tensor_id - min_input_tensor_id + 1));
        uint16_t* output_tensors =
            ( uint16_t* )malloc(sizeof(uint16_t) * (max_output_tensor_id - min_output_tensor_id + 1));

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

        sub_graph->input_tensor_list = ( uint16_t* )sys_malloc(sizeof(uint16_t) * sub_graph->input_num);
        sub_graph->output_tensor_list = ( uint16_t* )sys_malloc(sizeof(uint16_t)* (sub_graph->output_num + 1));
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
        for (uint16_t j = 0; j < max_output_tensor_id - min_output_tensor_id + 1; j++)
        {
            if (output_tensors[j] > 0)
            {
                sub_graph->output_tensor_list[output_tensor_count] = min_output_tensor_id + j;
                output_tensor_count++;
            }
        }
    }

    // fill the sub graph id
    for (int i = 0; i < sub_graph_count; i++)
    {
        struct subgraph* sub_graph = *( struct subgraph** )get_vector_data(ir_graph->subgraph_list, i);
        sub_graph->idx = i;

        for (int j = 0; j < sub_graph->node_num; j++)
        {
            ir_graph->node_list[sub_graph->node_list[j]]->subgraph_idx = i;
        }
    }

    // find no-output input in current sub graph
    for (int i = 1; i < sub_graph_count; i++)
    {
        struct subgraph* sub_graph = *( struct subgraph** )get_vector_data(ir_graph->subgraph_list, i);
        for (int j = 0; j < sub_graph->input_num; j++)
        {
            struct ir_tensor* ir_tensor = ir_graph->tensor_list[sub_graph->input_tensor_list[j]];

            if (ir_tensor->tensor_type != TENSOR_TYPE_INPUT)
            {
                uint16_t node_id = ir_tensor->producer;
                uint8_t sub_graph_id = ir_graph->node_list[node_id]->subgraph_idx;

                struct subgraph* target_sub_graph =
                    *( struct subgraph** )get_vector_data(ir_graph->subgraph_list, sub_graph_id);

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
                    target_sub_graph->output_num += 1;
                    sys_realloc(target_sub_graph->output_tensor_list, sizeof(uint16_t) * target_sub_graph->output_num);
                    target_sub_graph->output_tensor_list[target_sub_graph->output_num - 1] = ir_tensor->idx;
                }
            }
        }
    }

    // sort tensor io
    for (int i = 0; i < sub_graph_count; i++)
    {
        if(i == sub_graph_count - 1)
            continue;
        struct subgraph* subgraph = *(struct subgraph**)get_vector_data(ir_graph->subgraph_list, i);
        // int input_num = subgraph->input_num;
        // sort_io_array(subgraph->input_tensor_list, input_num);
        int output_num = subgraph->output_num;
        sort_io_array(subgraph->output_tensor_list, output_num);
    }
    
}

void dump_sub_graph0(struct ir_graph* ir_graph)
{
    const int sub_graph_count = get_vector_num(ir_graph->subgraph_list);

    fprintf(stdout, "Total sub_graph: %d.\n", sub_graph_count);
    for (int i = 0; i < sub_graph_count; i++)
    {
        struct subgraph* sub_graph = *( struct subgraph** )get_vector_data(ir_graph->subgraph_list, i);

        fprintf(stdout, "Sub graph[%d]: {%8s } has %d nodes, %d input tensors, %d output tensors.\n", sub_graph->idx,
                sub_graph->nn_dev->name, sub_graph->node_num, sub_graph->input_num, sub_graph->output_num);
        fprintf(stdout, "\tSub nodes: [ ");
        for (int j = 0; j < sub_graph->node_num - 1; j++)
        {
            int node_id = sub_graph->node_list[j];
            fprintf(stdout, "%d, ", node_id);
        }
        fprintf(stdout, "%d ].\n", sub_graph->node_list[sub_graph->node_num - 1]);
        fflush(stdout);

        fprintf(stdout, "\tSub input  tensors: [ ");
        for (int j = 0; j < sub_graph->input_num - 1; j++)
        {
            int tensor_id = sub_graph->input_tensor_list[j];
            fprintf(stdout, "%d, ", tensor_id);
        }
        fprintf(stdout, "%d ].\n", sub_graph->input_tensor_list[sub_graph->input_num - 1]);
        fflush(stdout);

        fprintf(stdout, "\tSub output tensors: [ ");
        for (int j = 0; j < sub_graph->output_num - 1; j++)
        {
            int tensor_id = sub_graph->output_tensor_list[j];
            fprintf(stdout, "%d, ", tensor_id);

        }
        fprintf(stdout, "%d ].\n\n", sub_graph->output_tensor_list[sub_graph->output_num - 1]);
        fflush(stdout);
    }
}

void dump_sub_graph(struct ir_graph* ir_graph)
{
    int subgraph_num = get_vector_num(ir_graph->subgraph_list);
    for (int i = 0; i < subgraph_num; i++)
    {
        struct subgraph* subgraph = *(struct subgraph**)get_vector_data(ir_graph->subgraph_list, i);
        printf("subgraph[%2d] dev[%7s] node:[ ",i,subgraph->nn_dev->name);
        for (int j = 0; j < subgraph->node_num; j++)
        {
            printf("%d ",subgraph->node_list[j]);
        }
        printf("]");

        printf(" in tensor:[ ");
        int input_num = subgraph->input_num;
        for (int j = 0; j < input_num; j++)
        {
            printf("%d ",subgraph->input_tensor_list[j]);
        }
        printf("]");
        
        printf(" out tensor:[ ");
        int output_num = subgraph->output_num;
        for (int j = 0; j < output_num; j++)
        {
            printf("%d ",subgraph->output_tensor_list[j]);
        }
        printf("]\n");
    }
}

static int acl_allocate(struct dev_allocator* allocator, struct ir_graph* ir_graph)
{
    printf("run into acl allocate!!!\n");

    split_graph_node_to_sub_graph(ir_graph);
    generate_sub_graph_io(ir_graph);
    // dump_sub_graph(ir_graph);

    return 0;
}
#endif

// PARTITION VERSION 1
struct vector* get_graph_blocked_nodes(const struct ir_graph* ir_graph)
{
    struct vector* blocked_nodes_list = create_vector(sizeof(uint16_t), NULL);
    printf("blocked_node list[ ");
    for (uint16_t i = 0; i < ir_graph->node_num; i++)
    {
        if (!node_in_white_list(ir_graph, i))
        {
            push_vector_data(blocked_nodes_list, &i);
            printf("%d ", i);
            // printf("%s ", get_op_name(ir_graph->node_list[i]->op.op_type));
        }
    }
    printf(" ]\n");
    return blocked_nodes_list;
}


struct vector* find_children_node(const struct ir_graph* ir_graph, struct vector* candidate_nodes, const struct vector* selected_nodes)
{
    struct vector* related_tensor_list = create_vector(sizeof(uint16_t), NULL);

    for (int i = 0; i < get_vector_num((struct vector*)selected_nodes); i++)
    {
        int16_t* selected_node_id = (int16_t*)get_vector_data((struct vector*)selected_nodes, i);

        // get each node output tensors index
        struct ir_node* ir_node = ir_graph->node_list[*selected_node_id];
        for (uint16_t tensor_index = 0; tensor_index < ir_node->output_num; tensor_index++)
        {
            uint16_t tensor_id = ir_node->output_tensors[tensor_index];
            push_vector_data(related_tensor_list, &tensor_id);
        }
    }

    // to record related child nodes via each node input tensor id
    struct vector* related_node_list = create_vector(sizeof(uint16_t), NULL);
    for (int node_index = 0; node_index < get_vector_num((struct vector*)candidate_nodes); node_index++)
    {
        int16_t* current_node_id = (int16_t*)get_vector_data((struct vector*)candidate_nodes, node_index);
        struct ir_node* current_ir_node = ir_graph->node_list[*current_node_id];

        for (int16_t input_index = 0; input_index < current_ir_node->input_num; input_index++)
        {
            int16_t node_input_tensor_id = current_ir_node->input_tensors[input_index];

            for (int tensor_index = 0; tensor_index < get_vector_num(related_tensor_list); tensor_index++)
            {
                const uint16_t* tensor_id = (uint16_t*)get_vector_data(related_tensor_list, tensor_index);
                if (*tensor_id == node_input_tensor_id)
                {
                    push_vector_data(related_node_list, current_node_id);
                }
            }
        }
    }

    release_vector(related_tensor_list);

    // remove related nodes from candidate nodes
    for (int i = 0; i < get_vector_num(related_node_list); i++)
    {
        int16_t* related_node_id = (int16_t*)get_vector_data(related_node_list, i);

        for (int j = 0; j < get_vector_num(candidate_nodes); j++)
        {
            int16_t* candidate_node_id = (int16_t*)get_vector_data(candidate_nodes, j);

            if (*candidate_node_id == *related_node_id)
            {
                remove_vector_by_idx(candidate_nodes, j);
                break;
            }
        }
    }

    return related_node_list;
}


// policy has some issue, must be fixed
int split_graph_node_to_sub_graph(struct ir_graph* ir_graph)
{
    // get unsupported nodes
    struct vector* blocked_nodes_list = get_graph_blocked_nodes(ir_graph);
    const int blocked_nodes_count = get_vector_num((struct vector*)blocked_nodes_list);
    //sort_nodes(blocked_nodes_list);
    // struct vector* sub_graphs_list = create_vector(sizeof(struct subgraph), NULL);
    if (blocked_nodes_count == 0)
        // return NULL;
    {
        struct subgraph* sub_graph = (struct subgraph*)sys_malloc(sizeof(struct subgraph));
        init_subgraph((struct ir_graph*)ir_graph, sub_graph, 0);

        // not including the last one
        sub_graph->node_num = ir_graph->node_num;
        sub_graph->node_list = (uint16_t*)sys_malloc(sizeof(uint16_t) * sub_graph->node_num);

        for (uint16_t j = 0; j < sub_graph->node_num; j++)
        {
            sub_graph->node_list[j] = j;
        }

        sub_graph->nn_dev = ir_graph->nn_dev;

        push_vector_data(ir_graph->subgraph_list, &sub_graph);
        return 0;
    }

    // prepare sub graph list
    // struct vector* sub_graphs_list = create_vector(sizeof(struct subgraph), NULL);

    // from the last unsupported node to collecting all sub graphs
    // scan from back to front
    int subgraph_idx = 0;
    for (int i = blocked_nodes_count - 1; i >= 0; i--)
    {

        // start node id (the blocked one)
        uint16_t first_node_id = *((uint16_t*)get_vector_data((struct vector*)blocked_nodes_list, i));
        // end node id (not including its self; the next blocked one, or the last one)
        uint16_t last_node_id = ir_graph->node_num;
        if (i < blocked_nodes_count - 1)
        {
            last_node_id = *((uint16_t*)get_vector_data((struct vector*)blocked_nodes_list, i + 1));
        }

        int children_nodes_is_complicated = 0;

        // scan if these nodes is complicated to be solved
        for (uint16_t j = first_node_id; j < last_node_id; j++)
        {
            if (node_in_allow_list(ir_graph, j))
            {
                children_nodes_is_complicated++;
            }
        }

        if (children_nodes_is_complicated < 3)   // directly add these nodes to sub graph list
        {
            struct subgraph* sub_graph = (struct subgraph*)sys_malloc(sizeof(struct subgraph));
            init_subgraph((struct ir_graph*)ir_graph, sub_graph, subgraph_idx++);

            // not including the last one
            sub_graph->node_num = last_node_id - first_node_id;
            sub_graph->node_list = (uint16_t*)sys_malloc(sizeof(uint16_t) * sub_graph->node_num);

            for (uint16_t j = 0; j < sub_graph->node_num; j++)
            {
                sub_graph->node_list[j] = j + first_node_id;
            }

            // sub_graph->nn_dev = ir_graph->exec_attr->exec_context->def_dev;
            sub_graph->nn_dev = get_default_nn_device();
            push_vector_data(ir_graph->subgraph_list, &sub_graph);
        }
        else
        {
            // add acl some
            struct subgraph* sub_acl_graph = (struct subgraph*)sys_malloc(sizeof(struct subgraph));
            init_subgraph((struct ir_graph*)ir_graph, sub_acl_graph, subgraph_idx++);

            sub_acl_graph->node_num = last_node_id - (first_node_id + 1);
            sub_acl_graph->node_list = (uint16_t*)sys_malloc(sizeof(uint16_t) * sub_acl_graph->node_num);

            for (uint16_t j = 0; j < sub_acl_graph->node_num; j++)
            {
                sub_acl_graph->node_list[j] = j + first_node_id + 1;
            }

            sub_acl_graph->nn_dev = ir_graph->nn_dev;
            push_vector_data(ir_graph->subgraph_list, &sub_acl_graph);

            // ---------------

            // add cpu one
            struct subgraph* sub_cpu_graph = (struct subgraph*)sys_malloc(sizeof(struct subgraph));
            init_subgraph((struct ir_graph*)ir_graph, sub_cpu_graph, subgraph_idx++);

            sub_cpu_graph->node_num = 1;
            sub_cpu_graph->node_list = (uint16_t*)sys_malloc(sizeof(uint16_t) * sub_cpu_graph->node_num);
            sub_cpu_graph->node_list[0] = first_node_id;

            //sub_cpu_graph->nn_dev = ir_graph->exec_attr->exec_context->def_dev;
            sub_cpu_graph->nn_dev = get_default_nn_device();
            push_vector_data(ir_graph->subgraph_list, &sub_cpu_graph);
        }
    }

    // add main sub graph
    struct subgraph* sub_graph = (struct subgraph*)sys_malloc(sizeof(struct subgraph));
    init_subgraph((struct ir_graph*)ir_graph, sub_graph, 0);

    uint16_t stop_node_id = *((uint16_t*)get_vector_data((struct vector*)blocked_nodes_list, 0));

    sub_graph->node_num = stop_node_id;
    sub_graph->node_list = (uint16_t*)sys_malloc(sizeof(uint16_t) * sub_graph->node_num);

    for (uint16_t i = 0; i < stop_node_id; i++)
    {
        sub_graph->node_list[i] = i;
    }

    sub_graph->nn_dev = ir_graph->nn_dev;
    push_vector_data(ir_graph->subgraph_list, &sub_graph);
    

    release_vector(blocked_nodes_list);

    // optimize the sub graphs
    while (1)
    {
        int same_sub_graph_found = 0;
        int sub_graphs_count = get_vector_num(ir_graph->subgraph_list);
        for (int i = 1; i < sub_graphs_count; i++)
        {
            struct subgraph* last_sub_graph = get_ir_graph_subgraph(ir_graph, (sub_graphs_count - 1) - (i - 1));
            struct subgraph* current_sub_graph = get_ir_graph_subgraph(ir_graph, (sub_graphs_count - 1) - i);

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

    return 0;
}


/* 
    start: subgraph's first node idx
    end: subgraph's last node idx
    subgraph_idx: index
*/
static int partition_graph(struct ir_graph* ir_graph, struct subgraph* subgraph, int start, int end, int subgraph_idx)
{
    // set node's subgraph_id
    for (int i = start; i <= end; i++)
    {
        struct ir_node* node = get_ir_graph_node(ir_graph, i);
        node->subgraph_idx = subgraph_idx;
    }
    subgraph->idx = subgraph_idx;
    
    /* subgraph will record the input tensors and output tensors, instead of nodes */

    /* find all input tensors and output tensors of this subgraph */
    if(0 == start) // first subgraph
    {
        for(int i = 0; i < ir_graph->input_num; i++) 
        {
            struct ir_node* node = get_ir_graph_node(ir_graph, ir_graph->input_nodes[i]);

            if(node->input_num)
            {
                for(int j = 0; j < node->input_num; j++)
                {
                    struct ir_tensor* tensor = get_ir_graph_tensor(ir_graph, node->input_tensors[j]);

                    if(tensor->tensor_type == TENSOR_TYPE_INPUT || tensor->tensor_type == TENSOR_TYPE_VAR)
                    {
                        subgraph->input_num++;
                        subgraph->input_tensor_list =
                            (uint16_t*)sys_realloc(subgraph->input_tensor_list, subgraph->input_num * sizeof(uint16_t));
                        subgraph->input_tensor_list[subgraph->input_num - 1] = tensor->idx;
                    }
                }
            }
            else
            {
                for(int j = 0; j < node->output_num; j++)
                {
                    struct ir_tensor* tensor = get_ir_graph_tensor(ir_graph, node->output_tensors[j]);

                    if(tensor->tensor_type != TENSOR_TYPE_INPUT)
                        continue;

                    subgraph->input_num++;
                    subgraph->input_tensor_list =
                        (uint16_t*)sys_realloc(subgraph->input_tensor_list, subgraph->input_num * sizeof(uint16_t));
                    subgraph->input_tensor_list[subgraph->input_num - 1] = tensor->idx;
                }
            }
        }
    }
    
    else
    {
        int all_tensors[ir_graph->tensor_num];
        memset(all_tensors, 0, sizeof(all_tensors));

        // find all tensor
        for (int i = start; i <= end; i++)
        {
            struct ir_node* node = get_ir_graph_node(ir_graph, i);
            for (int j = 0; j < node->input_num; j++)
            {
                struct ir_tensor* input_tensor = get_ir_graph_tensor(ir_graph, node->input_tensors[j]);
                if(input_tensor->tensor_type == TENSOR_TYPE_CONST)
                    continue;
                int tensor_id = get_tensor_idx_from_name(ir_graph, input_tensor->name);
                all_tensors[tensor_id]++;
            }
            for (int k = 0; k < node->output_num; k++)
            {
                struct ir_tensor* output_tensor = get_ir_graph_tensor(ir_graph, node->output_tensors[k]);
                if(output_tensor->tensor_type == TENSOR_TYPE_CONST)
                    continue;
                int tensor_id = get_tensor_idx_from_name(ir_graph, output_tensor->name);
                all_tensors[tensor_id] -= 10;
            }
        }

        // find input tensor
        for (int i = 0; i < ir_graph->tensor_num; i++)
        {
            if(all_tensors[i] > 0)
            {
                subgraph->input_num++;
                subgraph->input_tensor_list =
                    (uint16_t*)sys_realloc(subgraph->input_tensor_list, subgraph->input_num * sizeof(uint16_t));
                subgraph->input_tensor_list[subgraph->input_num - 1] = i;
                struct ir_tensor* tensor = get_ir_graph_tensor(ir_graph, i);
            }
        }
    }
    
    // find output tensors
    if(ir_graph->node_num - 1 == end) // last subgraph
    {
        for(int i = 0; i < ir_graph->output_num; i++)
        {
            struct ir_node* node = get_ir_graph_node(ir_graph, ir_graph->output_nodes[i]);

            for(int j = 0; j < node->output_num; j++)
            {
                struct ir_tensor* tensor = get_ir_graph_tensor(ir_graph, node->output_tensors[j]);

                if(tensor->consumer_num == 0)
                {
                    subgraph->output_num++;
                    subgraph->output_tensor_list =
                        (uint16_t*)sys_realloc(subgraph->output_tensor_list, subgraph->output_num * sizeof(uint16_t));
                    subgraph->output_tensor_list[subgraph->output_num - 1] = tensor->idx;
                }
            }
        }
    }
    else 
    {
        int all_tensors[ir_graph->tensor_num];
        memset(all_tensors, 0, sizeof(all_tensors));

        // find all tensor
        for (int i = start; i <= end; i++)
        {
            struct ir_node* node = get_ir_graph_node(ir_graph, i);
            for (int j = 0; j < node->input_num; j++)
            {
                struct ir_tensor* input_tensor = get_ir_graph_tensor(ir_graph, node->input_tensors[j]);
                if(input_tensor->tensor_type == TENSOR_TYPE_CONST)
                    continue;
                int tensor_id = get_tensor_idx_from_name(ir_graph, input_tensor->name);
                all_tensors[tensor_id] -= 10;
            }
            for (int k = 0; k < node->output_num; k++)
            {
                struct ir_tensor* output_tensor = get_ir_graph_tensor(ir_graph, node->output_tensors[k]);
                if(output_tensor->tensor_type == TENSOR_TYPE_CONST)
                    continue;
                int tensor_id = get_tensor_idx_from_name(ir_graph, output_tensor->name);
                all_tensors[tensor_id]++;
            }
        }

        // find next subgraph input tensor which same as this subgraph output tensor
        for (int i = 0; i < ir_graph->tensor_num; i++)
        {
            if(all_tensors[i] > 0)
            {
                subgraph->output_num++;
                subgraph->output_tensor_list =
                    (uint16_t*)sys_realloc(subgraph->output_tensor_list, subgraph->output_num * sizeof(uint16_t));
                subgraph->output_tensor_list[subgraph->output_num - 1] = i;
            }
        }
        
    }
    
    /* strip out duplicated input tensors */
    uint16_t* real_inputs = ( uint16_t* )sys_malloc(subgraph->input_num * sizeof(uint16_t));
    int real_input_num = 1;

    real_inputs[0] = subgraph->input_tensor_list[0];

    for(int i = 1; i < subgraph->input_num; i++)
    {
        int idx = subgraph->input_tensor_list[i];
        int j;

        for(j = 0; j < real_input_num; j++)
        {
            if(idx == real_inputs[j])
                break;
        }

        if(j < real_input_num)
            continue;

        real_inputs[real_input_num] = idx;
        real_input_num++;
    }

    sys_free(subgraph->input_tensor_list);

    subgraph->input_num = real_input_num;
    subgraph->input_tensor_list = real_inputs;

    /* set the correct input wait count: INPUT tensor is always ready */
    subgraph->input_wait_count = 0;

    for(int i = 0; i < subgraph->input_num; i++)
    {
        struct ir_tensor* tensor = get_ir_graph_tensor(ir_graph, subgraph->input_tensor_list[i]);
        if(tensor->tensor_type == TENSOR_TYPE_VAR)
            subgraph->input_wait_count++;
    }

    return 0;
}


int find_omit_output_tensor(struct ir_graph* ir_graph)
{
    int subgraph_num = get_vector_num(ir_graph->subgraph_list);
    for (int i = 1; i < subgraph_num; i++)
    {
        struct subgraph* subgraph = get_ir_graph_subgraph(ir_graph, i);
        int input_num = subgraph->input_num;
        for (int j = 0; j < input_num; j++)
        {
            struct ir_tensor* input_tensor = get_ir_graph_tensor(ir_graph, subgraph->input_tensor_list[j]);
            if(input_tensor->tensor_type == TENSOR_TYPE_INPUT)
                continue;
            int pre_node_id = input_tensor->producer;
            int pre_subgraph_id = get_ir_graph_node(ir_graph, pre_node_id)->subgraph_idx;
            struct subgraph* pre_subgraph = get_ir_graph_subgraph(ir_graph, pre_subgraph_id);
            int tensor_has_not_output = 1;
            for (int k = 0; k < pre_subgraph->output_num; k++)
            {
                if(subgraph->input_tensor_list[j] == pre_subgraph->output_tensor_list[k])
                {
                    tensor_has_not_output = 0;
                    break;
                }
            }
            if(tensor_has_not_output)
            {
                pre_subgraph->output_num++;
                pre_subgraph->output_tensor_list =
                    (uint16_t*)sys_realloc(pre_subgraph->output_tensor_list, pre_subgraph->output_num * sizeof(uint16_t));
                pre_subgraph->output_tensor_list[pre_subgraph->output_num - 1] = subgraph->input_tensor_list[j];
            }
        }
        
    }
    return 0;
}

int generate_subgraph_io(struct ir_graph* ir_graph)
{
    const int sub_graph_count = get_vector_num(ir_graph->subgraph_list);

    int subgraph_idx = 0;
    for (int i = 0; i < sub_graph_count; i++)
    {
        struct subgraph* sub_graph =get_ir_graph_subgraph(ir_graph, i);
        int start = sub_graph->node_list[0];
        int end = sub_graph->node_list[sub_graph->node_num - 1];
        // printf("start:%d, end:%d\n",start, end);
        if(partition_graph(ir_graph, sub_graph, start, end, subgraph_idx++) < 0)
            return -1;
    }

    if(find_omit_output_tensor(ir_graph) < 0)
        return -1;
    return 0;
}

void dump_subgraph(struct ir_graph* ir_graph)
{ 
    const int sub_graph_count = get_vector_num(ir_graph->subgraph_list);

    fprintf(stdout, "Total sub_graph: %d.\n", sub_graph_count);

    for (int i = 0; i < sub_graph_count; i++)
    {
        struct subgraph* sub_graph =get_ir_graph_subgraph(ir_graph, i);

        fprintf(stdout, "Sub nodes: {%8s }[ ", sub_graph->nn_dev->name);
        for (int j = 0; j < sub_graph->node_num - 1; j++)
        {
            int node_id = sub_graph->node_list[j];
            fprintf(stdout, "%d, ", node_id);
        }
        fprintf(stdout, "%d ].\n", sub_graph->node_list[sub_graph->node_num - 1]);
    }
    fflush(stdout);

    for (int i = 0; i < sub_graph_count; i++)
    {
        struct subgraph* sub_graph =get_ir_graph_subgraph(ir_graph, i);
        printf("subgraphg:%d, addr:%p, node_num:%d, node list 0:%d\n",i, sub_graph, sub_graph->node_num, sub_graph->node_list[0]);
        printf("subgraph nn dev:%s\n",sub_graph->nn_dev->name);
        for (int j = 0; j < sub_graph->input_num; j++)
        {
            struct ir_tensor* input_tensor = get_ir_graph_tensor(ir_graph, sub_graph->input_tensor_list[j]);
            printf("input tensor name: %30s, tensor idx:%d\n",input_tensor->name, input_tensor->idx);
        }

        for (int k = 0; k < sub_graph->output_num; k++)
        {
            struct ir_tensor* output_tensor = get_ir_graph_tensor(ir_graph, sub_graph->output_tensor_list[k]);
            printf("output tensor name:%30s, tensor idx:%d\n",output_tensor->name, output_tensor->idx);
        }
        
    }
}

static int acl_allocate(struct dev_allocator* allocator, struct ir_graph* ir_graph)
{
    // printf("run into acl allocate!!!\n");
    fprintf(stdout, "ACL initialized\n");

    if(split_graph_node_to_sub_graph(ir_graph) < 0)
        return -1;
    
    if(generate_subgraph_io(ir_graph) < 0)
        return -1;

    // dump_subgraph(ir_graph);
    return 0;
}

static struct dev_allocator acl_allocator = {
    .name = "ACL",
    .allocate = acl_allocate
};


#ifdef STANDLONE_MODE
struct dev_allocator* get_acl_allocator(void)
{
    return &acl_allocator;
}
#else
static void register_acl_allocator(void)
{
    init_allocator_registry(&acl_allocator);
}

REGISTER_DEV_ALLOCATOR(register_acl_allocator);
#endif
