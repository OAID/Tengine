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

extern "C"
{
    #include "sys_port.h"
    #include "vector.h"
    #include "tengine_op.h"
    #include "tengine_ir.h"
    #include "tengine_exec.h"
    #include "tengine_utils.h"
}
#include "tengine_log.h"
#include "vk_allocator.hpp"

const int vk_white_list[] = { OP_CONST, OP_INPUT, OP_CONV, OP_POOL, OP_FC, OP_FLATTEN, OP_RELU, OP_DROPOUT, OP_ELTWISE, OP_PERMUTE, OP_CONCAT, OP_RESHAPE, OP_SOFTMAX, OP_PRIORBOX, OP_INTERP, OP_CROP, OP_UPSAMPLE};
const int vk_allow_list[] = { OP_CONV, OP_POOL, OP_FC, OP_FLATTEN, OP_RELU, OP_DROPOUT, OP_ELTWISE, OP_PERMUTE, OP_CONCAT, OP_RESHAPE, OP_SOFTMAX, OP_PRIORBOX, OP_INTERP, OP_CROP, OP_UPSAMPLE};
// OP_SOFTMAX, OP_PRIORBOX, OP_UPSAMPLE


int node_in_white_list(const struct ir_graph* ir_graph, const uint16_t node_id)
{
    int block_list_size = sizeof(vk_white_list) / sizeof(vk_white_list[0]);

    const uint16_t node_op_type = ir_graph->node_list[node_id]->op.op_type;

    
    for (int i = 0; i < block_list_size; i++)
    {
        if (node_op_type == vk_white_list[i])
            return 1;
    }

    return 0;
}


int node_in_allow_list(const struct ir_graph* ir_graph, const uint16_t node_id)
{
    int allow_list_size = sizeof(vk_allow_list) / sizeof(vk_allow_list[0]);

    const uint16_t node_op_type = ir_graph->node_list[node_id]->op.op_type;

    for (int i = 0; i < allow_list_size; i++)
    {
        if (node_op_type == vk_allow_list[i])
            return 1;
    }

    return 0;
}


struct vector* get_graph_blocked_nodes(const struct ir_graph* ir_graph)
{
    struct vector* blocked_nodes_list = create_vector(sizeof(uint16_t), NULL);
    TLOG_INFO("blocked_node list[ ");
    for (uint16_t i = 0; i < ir_graph->node_num; i++)
    {
        if (!node_in_white_list(ir_graph, i))
        {
            push_vector_data(blocked_nodes_list, &i);
            TLOG_INFO("%d ", i);
            TLOG_INFO("%s ", get_op_name(ir_graph->node_list[i]->op.op_type));
        }
    }
    TLOG_INFO(" ]\n");
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
            // add vk some
            struct subgraph* sub_vk_graph = (struct subgraph*)sys_malloc(sizeof(struct subgraph));
            init_subgraph((struct ir_graph*)ir_graph, sub_vk_graph, subgraph_idx++);

            sub_vk_graph->node_num = last_node_id - (first_node_id + 1);
            sub_vk_graph->node_list = (uint16_t*)sys_malloc(sizeof(uint16_t) * sub_vk_graph->node_num);

            for (uint16_t j = 0; j < sub_vk_graph->node_num; j++)
            {
                sub_vk_graph->node_list[j] = j + first_node_id + 1;
            }

            sub_vk_graph->nn_dev = ir_graph->nn_dev;
            push_vector_data(ir_graph->subgraph_list, &sub_vk_graph);

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
        // TLOG_INFO("start:%d, end:%d\n",start, end);
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

    TLOG_INFO("Total sub_graph: %d.\n", sub_graph_count);

    for (int i = 0; i < sub_graph_count; i++)
    {
        struct subgraph* sub_graph =get_ir_graph_subgraph(ir_graph, i);

        TLOG_INFO("Sub nodes: {%8s }[ ", sub_graph->nn_dev->name);
        for (int j = 0; j < sub_graph->node_num - 1; j++)
        {
            int node_id = sub_graph->node_list[j];
            TLOG_INFO("%d, ", node_id);
        }
        TLOG_INFO("%d ].\n", sub_graph->node_list[sub_graph->node_num - 1]);
    }
    fflush(stdout);

    for (int i = 0; i < sub_graph_count; i++)
    {
        struct subgraph* sub_graph =get_ir_graph_subgraph(ir_graph, i);
        TLOG_INFO("subgraphg:%d, addr:%p, node_num:%d, node list 0:%d\n",i, sub_graph, sub_graph->node_num, sub_graph->node_list[0]);
        TLOG_INFO("subgraph nn dev:%s\n",sub_graph->nn_dev->name);
        for (int j = 0; j < sub_graph->input_num; j++)
        {
            struct ir_tensor* input_tensor = get_ir_graph_tensor(ir_graph, sub_graph->input_tensor_list[j]);
            TLOG_INFO("input tensor name: %30s, tensor idx:%d\n",input_tensor->name, input_tensor->idx);
        }

        for (int k = 0; k < sub_graph->output_num; k++)
        {
            struct ir_tensor* output_tensor = get_ir_graph_tensor(ir_graph, sub_graph->output_tensor_list[k]);
            TLOG_INFO("output tensor name:%30s, tensor idx:%d\n",output_tensor->name, output_tensor->idx);
        }
        
    }
}

static int vk_allocate(struct dev_allocator* allocator, struct ir_graph* ir_graph)
{
    TLOG_INFO("run into vk allocate!!!\n");

    if(split_graph_node_to_sub_graph(ir_graph) < 0)
        return -1;
    // dump_subgraph(ir_graph);
    if(generate_subgraph_io(ir_graph) < 0)
        return -1;

    dump_subgraph(ir_graph);
    return 0;
}

static struct dev_allocator vk_allocator = {
    .name = "VK",
    .allocate = vk_allocate
};


#ifdef STANDLONE_MODE
struct dev_allocator* get_vk_allocator(void)
{
    return &vk_allocator;
}
#else
REGISTER_DEV_ALLOCATOR(register_vk_allocator);
static void register_vk_allocator(void)
{
    TLOG_INFO("start to run register vk allocator\n");
    init_allocator_registry(&vk_allocator);
}
#endif
