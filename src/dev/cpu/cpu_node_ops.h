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

#ifndef __CPU_NODE_OPS_H__
#define __CPU_NODE_OPS_H__

#include "module.h"
#include "cpu_device.h"

#define OPS_SCORE_STATIC 10000
#define OPS_SCORE_BEST 8000
#define OPS_SCORE_PREFER 6000
#define OPS_SCORE_CANDO 4000
#define OPS_SCORE_NOTSUP 2000

struct node_ops
{
    int (*prerun)(struct node_ops*, struct exec_node*, struct exec_graph*);
    int (*run)(struct node_ops*, struct exec_node*, struct exec_graph*);
    int (*reshape)(struct node_ops*, struct exec_node*, struct exec_graph*);
    int (*postrun)(struct node_ops*, struct exec_node*, struct exec_graph*);

    /* for private data of each node */
    /* init node is called when a node_ops is bound to an exec_node,
       just before prerun() is called.
       so that the node_ops can set the shared_mem needed in this function,
       as well as set the ops_priv in exec_node.
    */
    int (*init_node)(struct node_ops*, struct exec_node*, struct exec_graph*);

    /* release node is called after postrun() is called */
    int (*release_node)(struct node_ops*, struct exec_node*, struct exec_graph*);

    /* score */
    int (*score)(struct node_ops*, struct exec_graph*, struct ir_node*);
};

int init_cpu_node_ops_registry(void);
void release_cpu_node_ops_registry(void);

int register_builtin_node_ops(int op_type, struct node_ops* node_ops);
int unregister_builtin_node_ops(int op_type, struct node_ops* node_ops);

int register_custom_node_ops(int op_type, struct node_ops* node_ops);
int unregister_custom_node_ops(int op_type, struct node_ops* node_ops);

struct node_ops* find_node_ops(struct exec_graph* exec_graph, struct ir_node* ir_node);

#define AUTO_REGISTER_OPS(reg_func) REGISTER_MODULE_INIT(MOD_OP_LEVEL, #reg_func, reg_func)
#define AUTO_UNREGISTER_OPS(unreg_func) REGISTER_MODULE_EXIT(MOD_OP_LEVEL, #unreg_func, unreg_func);

#endif
