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

#pragma once

#include <stdint.h>

struct subgraph;
struct vector;

/*!
 * @struct memory_block_t
 * @brief  Memory block structure
 */
typedef struct memory_block
{
    uint16_t index;        //!< the index of a memory_block
    uint32_t size;         //!< final estimated memory size
    uint16_t tensor_count; //!< referenced tensor count
    uint16_t* tensor_list; //!< referenced tensor list
    uint16_t tensor_index; //!< referenced tensor index, which is largest one
    uint8_t inuse;         //!< flag mark if this block is inuse
} memory_block_t;

/*!
 * @brief  Init memory block with index.
 *
 * @param [in]  memory_block: specific memory_block.
 * @param [in]  index: index of this specific memory_block.
 */
void init_memory_block(memory_block_t* memory_block, uint16_t index);

/*!
 * @brief put all output tensors in each node of the subgraph into the memory blocks.
 *
 * @param [in]  subgraph: specific working subgraph.
 * @param [in]  memory_blocks: estimated memory_blocks.
 *
 * @return statue value, 0 success, other value failure.
 */
int estimate_subgraph_memory_blocks(struct subgraph* subgraph, struct vector* memory_blocks);
