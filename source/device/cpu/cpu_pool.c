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

#include "cpu_pool.h"

#include "cpu_node.h"
#include "cpu_graph.h"

#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "graph/subgraph.h"
#include "utility/sys_port.h"
#include "utility/vector.h"
#include "utility/log.h"

struct mem_record
{
    struct tensor* ir_tensor;
    int used;
    int block_id;
};

static int find_inplace_input(struct exec_node* exec_node, int output_slot, struct node* ir_node, struct graph* ir_graph)
{
    if (exec_node->inplace_map_num == 0)
        return -1;

    uint8_t* inplace_map;

    if (exec_node->inplace_map_num > 2)
        inplace_map = exec_node->inplace_map_ptr;
    else
        inplace_map = exec_node->inplace_map;

    int i;
    for (i = 0; i < 2 * exec_node->inplace_map_num; i += 2)
    {
        if (inplace_map[i] == output_slot)
            break;
    }

    /* no map */
    if (i == 2 * exec_node->inplace_map_num)
        return -1;

    int input_slot = inplace_map[i + 1];

    struct tensor* tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[input_slot]);

    if (tensor->consumer_num > 1)
        return -1;

    return input_slot;
}

static int find_tensor_mem_list(struct vector* tensor_mem_list, const struct tensor* ir_tensor)
{
    int rec_number = get_vector_num(tensor_mem_list);

    for (int i = 0; i < rec_number; i++)
    {
        struct mem_record* rec = (struct mem_record*)get_vector_data(tensor_mem_list, i);

        if (rec->ir_tensor == ir_tensor)
            return i;
    }

    return -1;
}

void free_exec_graph_mem(struct exec_graph* graph)
{
    /* free the shared memory */
    if (graph->shared_mem)
    {
        sys_free(graph->shared_mem);
        graph->shared_mem = NULL;
        graph->shared_mem_size = 0;
    }
    /* free the shared pack4 memory */
    if (graph->shared_pack4_mem)
    {
        sys_free(graph->shared_pack4_mem);
        graph->shared_pack4_mem = NULL;
        graph->shared_pack4_mem_size = 0;
    }

    /* free the mem pool */
    if (graph->mem_pool)
    {
        release_mem_pool(graph->mem_pool);
        graph->mem_pool = NULL;
    }
}

static void mem_pool_dump(struct mem_pool* mem_pool)
{
    int block_number = get_vector_num(mem_pool->block_list);

    TLOG_INFO("Tengine: Block number: %d align size: %d\n", block_number, mem_pool->align_size);

    for (int i = 0; i < block_number; i++)
    {
        struct mem_block_entry* entry = (struct mem_block_entry*)get_vector_data(mem_pool->block_list, i);

        TLOG_INFO("Tengine: %d: %p (%d) used: %d free: %d\n", i, entry->addr, entry->block_size, entry->alloc_count,
                  entry->free_count);
    }
}

static void* mem_pool_get_mem_block(struct mem_pool* mem_pool, int block_id)
{
    struct mem_block_entry* entry = (struct mem_block_entry*)get_vector_data(mem_pool->block_list, block_id);

    size_t addr = (size_t)(entry->addr);
    size_t aligned_addr = (addr + 4 + mem_pool->align_size) & (~(mem_pool->align_size - 1));

    return (void*)aligned_addr;
}

static int mem_pool_get_backend_mem(struct mem_pool* mem_pool)
{
    int block_num = get_vector_num(mem_pool->block_list);

    for (int i = 0; i < block_num; i++)
    {
        struct mem_block_entry* entry = (struct mem_block_entry*)get_vector_data(mem_pool->block_list, i);

        entry->block_size = entry->max_req_size + mem_pool->align_size + 128;

        entry->addr = sys_malloc(entry->block_size);

        if (entry->addr == NULL)
            return -1;
    }

    return 0;
}

static int mem_pool_allocate(struct mem_pool* mem_pool, int size)
{
    int block_num = get_vector_num(mem_pool->block_list);
    ;

    for (int i = 0; i < block_num; i++)
    {
        struct mem_block_entry* entry = (struct mem_block_entry*)get_vector_data(mem_pool->block_list, i);

        if (entry->free_count != entry->alloc_count)
            continue;

        /* TODO: use the best match alg */

        entry->alloc_count++;

        if (entry->max_req_size < size)
            entry->max_req_size = size;

        return i;
    }

    /* create new block */

    struct mem_block_entry e;

    e.addr = NULL;
    e.max_req_size = size;
    e.alloc_count = 1;
    e.free_count = 0;

    push_vector_data(mem_pool->block_list, &e);

    return block_num;
}

static void mem_pool_free(struct mem_pool* mem_pool, int block_id)
{
    struct mem_block_entry* block = (struct mem_block_entry*)get_vector_data(mem_pool->block_list, block_id);

    block->free_count++;
}

void release_mem_pool(struct mem_pool* mem_pool)
{
    if (mem_pool->block_list != NULL)
    {
        int block_num = get_vector_num(mem_pool->block_list);

        for (int i = 0; i < block_num; i++)
        {
            struct mem_block_entry* entry = (struct mem_block_entry*)get_vector_data(mem_pool->block_list, i);

            sys_free(entry->addr);
        }

        release_vector(mem_pool->block_list);
    }

    sys_free(mem_pool);
}

static struct mem_pool* create_mem_pool(void)
{
    struct mem_pool* mem_pool = (struct mem_pool*)sys_malloc(sizeof(struct mem_pool));

    if (mem_pool == NULL)
        return NULL;

    mem_pool->align_size = 16;
    mem_pool->block_list = create_vector(sizeof(struct mem_block_entry), NULL);

    if (mem_pool->block_list == NULL)
        goto error;

    mem_pool->allocate = mem_pool_allocate;
    mem_pool->free = mem_pool_free;
    mem_pool->dump = mem_pool_dump;
    mem_pool->get_backend_mem = mem_pool_get_backend_mem;
    mem_pool->get_mem_block = mem_pool_get_mem_block;

    return mem_pool;

error:

    release_mem_pool(mem_pool);

    return NULL;
}

int alloc_exec_graph_mem(struct exec_graph* exec_graph)
{
    struct mem_pool* mem_pool;
    int max_shared_mem_size = 0;
    int max_shared_pack4_mem_size = 0;

    int node_num = get_vector_num(exec_graph->exec_node_list);

    struct vector* tensor_mem_list = create_vector(sizeof(struct mem_record), NULL);

    if (tensor_mem_list == NULL)
        return -1;

    mem_pool = create_mem_pool();

    if (mem_pool == NULL)
        return -1;

    exec_graph->mem_pool = mem_pool;

    for (int i = 0; i < node_num; i++)
    {
        struct exec_node* exec_node = (struct exec_node*)get_vector_data(exec_graph->exec_node_list, i);
        struct node* ir_node = exec_node->ir_node;
        struct graph* ir_graph = ir_node->graph;

        int8_t* block_id;

        if (exec_node->output_num > 4)
            block_id = exec_node->block_id_ptr;
        else
            block_id = exec_node->block_id;

        for (int j = 0; j < ir_node->output_num; j++)
        {
            struct tensor* ir_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[j]);

            if (ir_tensor->data != NULL)
                continue;

            int inplace_input = find_inplace_input(exec_node, j, ir_node, ir_graph);

            if (inplace_input >= 0)
            {
                struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[inplace_input]);

                int idx = find_tensor_mem_list(tensor_mem_list, input_tensor);

                /* if the input is from outside buffer, input_r should be NULL */
                if (idx < 0)
                    continue;

                struct mem_record* input_r = (struct mem_record*)get_vector_data(tensor_mem_list, idx);

                input_r->ir_tensor = ir_tensor;
                input_r->used = ir_tensor->consumer_num;
                block_id[j] = INPLACE_BLOCK_FLAG | inplace_input;
                continue;
            }

            /* allocate mem from pool */
            int mem_size = ir_tensor->elem_size * ir_tensor->elem_num;

            struct mem_record r;

            r.ir_tensor = ir_tensor;
            r.block_id = mem_pool->allocate(mem_pool, mem_size);
            r.used = ir_tensor->consumer_num;

            block_id[j] = r.block_id;

            push_vector_data(tensor_mem_list, &r);
        }

        /* clear input tensor count */
        for (int j = 0; j < ir_node->input_num; j++)
        {
            struct tensor* ir_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[j]);

            if (ir_tensor->data != NULL)
                continue;

            int idx = find_tensor_mem_list(tensor_mem_list, ir_tensor);

            if (idx < 0)
                continue;

            struct mem_record* input_r = (struct mem_record*)get_vector_data(tensor_mem_list, idx);

            input_r->used--;

            if (input_r->used == 0)
            {
                mem_pool->free(mem_pool, input_r->block_id);
                remove_vector_via_index(tensor_mem_list, idx);
            }
        }

        /* handle shared mem */
        if (exec_node->shared_mem_size > max_shared_mem_size)
            max_shared_mem_size = exec_node->shared_mem_size;
        if (exec_node->shared_pack4_mem_size > max_shared_pack4_mem_size)
            max_shared_pack4_mem_size = exec_node->shared_pack4_mem_size;
    }

    TLOG_DEBUG("Tengine: Final tensor_mem_list number: %d\n", get_vector_num(tensor_mem_list));

    release_vector(tensor_mem_list);

    exec_graph->shared_mem_size = max_shared_mem_size;
    exec_graph->shared_pack4_mem_size = max_shared_pack4_mem_size;

    if (max_shared_mem_size > 0)
    {
        exec_graph->shared_mem = sys_malloc(max_shared_mem_size);

        if (exec_graph->shared_mem == NULL)
        {
            TLOG_ERR("Tengine: Cannot allocate shared memory. size=%d\n", max_shared_mem_size);
            return -1;
        }
    }
    if (max_shared_pack4_mem_size > 0)
    {
        exec_graph->shared_pack4_mem = sys_malloc(max_shared_pack4_mem_size);

        if (exec_graph->shared_pack4_mem == NULL)
        {
            TLOG_ERR("Tengine: Cannot allocate shared pack4 memory. size=%d\n", max_shared_pack4_mem_size);
            return -1;
        }
    }

    TLOG_DEBUG("Tengine: Shared memory: %p size=%d\n", exec_graph->shared_mem, max_shared_mem_size);
    TLOG_DEBUG("Tengine: Shared pack4 memory: %p size=%d\n", exec_graph->shared_pack4_mem, max_shared_pack4_mem_size);

    if (mem_pool->get_backend_mem(mem_pool) < 0)
    {
        TLOG_ERR("Tengine: Cannot allocate enough memory from backend\n");
        return -1;
    }

    mem_pool->dump(mem_pool);

    /* now, the real allocate */
    for (int i = 0; i < node_num; i++)
    {
        struct exec_node* exec_node = (struct exec_node*)get_vector_data(exec_graph->exec_node_list, i);
        struct node* ir_node = exec_node->ir_node;
        struct graph* ir_graph = ir_node->graph;
        struct mem_pool* local_mem_pool = exec_graph->mem_pool;

        int8_t* block_id;

        if (exec_node->output_num > 4)
            block_id = exec_node->block_id_ptr;
        else
            block_id = exec_node->block_id;

        for (int j = 0; j < ir_node->output_num; j++)
        {
            struct tensor* ir_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[j]);

            if (block_id[j] < 0)
                continue;

            if (block_id[j] & INPLACE_BLOCK_FLAG)
            {
                int input_idx = block_id[j] & (INPLACE_BLOCK_FLAG - 1);

                struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[input_idx]);
                ir_tensor->data = input_tensor->data;
                ir_tensor->free_host_mem = 0;
                ir_tensor->internal_allocated = MEM_POOL_ALLOCATED;
            }
            else
            {
                ir_tensor->data = local_mem_pool->get_mem_block(local_mem_pool, block_id[j]);
                // ir_tensor->data = sys_malloc(ir_tensor->elem_size * ir_tensor->elem_num);
                ir_tensor->free_host_mem = 0;
                ir_tensor->internal_allocated = MEM_POOL_ALLOCATED;
            }
        }
    }

    return 0;
}
