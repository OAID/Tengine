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

#include "optimizer/estimation.h"

#include "defines.h"
#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "graph/subgraph.h"
#include "operator/op.h"
#include "optimizer/helper.h"
#include "utility/vector.h"
#include "utility/sys_port.h"

#ifdef TENGINE_ENABLE_ENV_VAR
#include <stdlib.h>
#endif

void init_memory_block(memory_block_t* memory_block, uint16_t index)
{
    if (NULL != memory_block)
    {
        memory_block->index = index;
        memory_block->size = 0;
        memory_block->tensor_count = 0;
        memory_block->tensor_list = NULL;
        memory_block->tensor_index = 0;
        memory_block->inuse = 0;
    }
}

memory_block_t* find_unused_memory_block(struct vector* memory_blocks)
{
    int memory_blocks_count = get_vector_num(memory_blocks);
    for (int i = 0; i < memory_blocks_count; i++)
    {
        memory_block_t* memory_block = (memory_block_t*)get_vector_data(memory_blocks, i);
        if (0 == memory_block->inuse)
        {
            return memory_block;
        }
    }

    return NULL;
}

memory_block_t* get_usable_memory_block(struct vector* memory_blocks)
{
    memory_block_t* memory_block = find_unused_memory_block(memory_blocks);

    if (NULL == memory_block)
    {
        int memory_blocks_count = get_vector_num(memory_blocks);
        memory_block_t new_memory_block;
        init_memory_block(&new_memory_block, memory_blocks_count);

        push_vector_data(memory_blocks, &new_memory_block);

        memory_blocks_count = get_vector_num(memory_blocks);
        memory_block = (memory_block_t*)get_vector_data(memory_blocks, memory_blocks_count - 1);
    }

    return memory_block;
}

int mark_memory_block_with_tensor(ir_graph_t* graph, memory_block_t* memory_block, uint16_t index)
{
    ir_tensor_t* tensor = get_ir_graph_tensor(graph, index);

    memory_block->tensor_count += 1;
    memory_block->tensor_list = (uint16_t*)sys_realloc(memory_block->tensor_list, memory_block->tensor_count * sizeof(uint16_t));
    memory_block->inuse = 1;

    uint32_t tensor_buffer_size = tensor->elem_num * tensor->elem_size;
    if (tensor_buffer_size > memory_block->size)
    {
        memory_block->size = tensor_buffer_size;
        memory_block->tensor_index = index;
    }

    return 0;
}

int estimate_subgraph_memory_blocks(struct subgraph* subgraph, struct vector* memory_blocks)
{
    if (NULL == subgraph || NULL == memory_blocks)
    {
        return -1;
    }

    for (uint16_t i = 0; i < subgraph->node_num; i++)
    {
        uint16_t node_index = subgraph->node_list[i];
        ir_node_t* node = get_ir_graph_node(subgraph->graph, node_index);

        if (OP_CONST != node->op.type)
        {
            for (uint8_t j = 0; j < node->output_num; j++)
            {
                uint16_t index = node->output_tensors[j];

                memory_block_t* memory_block = get_usable_memory_block(memory_blocks);
                if (NULL != memory_block)
                {
                    int ret = mark_memory_block_with_tensor(subgraph->graph, memory_block, index);
                    if (0 != ret)
                    {
                        return -1;
                    }
                }
                else
                {
                    return -1;
                }
            }
        }
    }

    return 0;
}
