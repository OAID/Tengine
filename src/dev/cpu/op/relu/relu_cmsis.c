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

#include "arm_math.h"
#include "sys_port.h"
#include "module.h"
#include "tengine_errno.h"
#include "tengine_log.h"
#include "tengine_ir.h"
#include "cpu_node_ops.h"
#include "tengine_op.h"

/**
 * @brief Q7 RELU function
 * @param[in,out]   data        pointer to input
 * @param[in]       size        number of elements
 * @return none.
 *
 * @details
 *
 * Optimized relu with QSUB instructions.
 *
 */

void arm_relu_q7(q7_t* data, uint16_t size);

static int init_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    exec_node->inplace_map[0] = 0;
    exec_node->inplace_map[1] = 0;
    exec_node->inplace_map_num = 1;

    return 0;
}

static int release_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    exec_node->inplace_map_num = 0;

    return 0;
}

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct ir_node* ir_node = exec_node->ir_node;
    struct ir_graph* ir_graph = ir_node->graph;
    struct ir_tensor* input_tensor;
    struct ir_tensor* output_tensor;

    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    if (input_tensor->data != output_tensor->data)
    {
        TLOG_ERR("input and output are not the same mem\n");
        set_tengine_errno(EFAULT);
        return -1;
    }

    arm_relu_q7(input_tensor->data, input_tensor->elem_num);

    return 0;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct ir_node* exec_node)
{
    return OPS_SCORE_BEST;
}

static struct node_ops cmsis_node_ops = {.prerun = NULL,
                                         .run = run,
                                         .reshape = NULL,
                                         .postrun = NULL,
                                         .init_node = init_node,
                                         .release_node = release_node,
                                         .score = score};

static int reg_relu_cmsis_ops(void* arg)
{
    return register_builtin_node_ops(OP_RELU, &cmsis_node_ops);
}

static int unreg_relu_cmsis_ops(void* arg)
{
    return unregister_builtin_node_ops(OP_RELU, &cmsis_node_ops);
}

AUTO_REGISTER_OPS(reg_relu_cmsis_ops);
AUTO_UNREGISTER_OPS(unreg_relu_cmsis_ops);
