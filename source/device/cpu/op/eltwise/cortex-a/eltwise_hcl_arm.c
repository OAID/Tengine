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
 * Author: qtang@openailab.com
 */

#include "eltwise_hcl_arm.h"

#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "utility/sys_port.h"
#include "utility/float.h"
#include "utility/log.h"
#include "device/cpu/cpu_node.h"
#include "device/cpu/cpu_graph.h"
#include "device/cpu/cpu_module.h"


static int init_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    return 0;
}

static int release_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    return 0;
}

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* ir_node = exec_node->ir_node;
    struct graph* ir_graph = ir_node->graph;
    struct tensor* input_tensor0;
    struct tensor* output_tensor;

    input_tensor0 = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    struct eltwise_param* eltwise_param = ( struct eltwise_param* )ir_node->op.param_mem;

    struct tensor* input_tensor1 = NULL;
    if (ir_node->input_num > 1)
    {
        input_tensor1 = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    }

    int ret = perf_eltwise_fp32(output_tensor, input_tensor0, input_tensor1, eltwise_param, exec_graph->num_thread);

    return ret;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct node* exec_node)
{
    struct node* ir_node = exec_node;
    struct graph* ir_graph = ir_node->graph;
    struct tensor* input_tensor_0;
    struct tensor* input_tensor_1;

    if (ir_node->input_num != 2)
        return 0;

    input_tensor_0 = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    input_tensor_1 = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    struct eltwise_param* eltwise_param = ( struct eltwise_param* )ir_node->op.param_mem;

    if (input_tensor_0->data_type != TENGINE_DT_FP32 || ir_graph->graph_layout != TENGINE_LAYOUT_NCHW)
        return 0;
    if (eltwise_param->type != ELT_SUM || input_tensor_0->elem_num != input_tensor_1->elem_num)
        return 0;

    return OPS_SCORE_BEST;
}

static struct node_ops hcl_node_ops = {.prerun = NULL,
                                       .run = run,
                                       .reshape = NULL,
                                       .postrun = NULL,
                                       .init_node = init_node,
                                       .release_node = release_node,
                                       .score = score};

int register_eltwise_hcl_arm_op()
{
    return register_builtin_node_ops(OP_ELTWISE, &hcl_node_ops);
}

int unregister_eltwise_hcl_arm_op()
{
    return unregister_builtin_node_ops(OP_ELTWISE, &hcl_node_ops);
}
