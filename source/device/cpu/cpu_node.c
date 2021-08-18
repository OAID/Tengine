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
 */

#include "cpu_node.h"

#include "cpu_module.h"
#include "graph/node.h"
#include "utility/sys_port.h"

int init_exec_node(struct exec_graph* exec_graph, struct exec_node* exec_node, struct node* ir_node, struct node_ops* node_ops)
{
    exec_node->ir_node = ir_node;
    exec_node->node_ops = node_ops;
    exec_node->ops_priv = NULL;
    exec_node->inplace_map_num = 0;
    exec_node->inplace_map_ptr = NULL;
    exec_node->shared_mem_size = 0;
    exec_node->shared_pack4_mem_size = 0;
    exec_node->output_num = ir_node->output_num;

    int8_t* block_id = exec_node->block_id;

    if (exec_node->output_num > 4)
    {
        exec_node->block_id_ptr = (int8_t*)sys_malloc(sizeof(int8_t) * exec_node->output_num);
        block_id = exec_node->block_id_ptr;
    }

    for (int i = 0; i < exec_node->output_num; i++)
        block_id[i] = -1;

    if (node_ops->init_node && node_ops->init_node(node_ops, exec_node, exec_graph) < 0)
        return -1;

    return 0;
}

void release_exec_node(struct exec_graph* exec_graph, struct exec_node* exec_node, struct node_ops* node_ops)
{
    if (node_ops->release_node)
        node_ops->release_node(node_ops, exec_node, exec_graph);

    if (exec_node->inplace_map_num > 2)
        sys_free(exec_node->inplace_map_ptr);

    if (exec_node->output_num > 4)
        sys_free(exec_node->block_id_ptr);
}
