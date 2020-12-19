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
 * @brief Q7 softmax function
 * @param[in]       vec_in      pointer to input vector
 * @param[in]       dim_vec     input vector dimention
 * @param[out]      p_out       pointer to output vector
 * @return none.
 *
 */

static int reshape(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct ir_node* ir_node = exec_node->ir_node;
    struct ir_graph* ir_graph = ir_node->graph;
    struct ir_tensor* input_tensor;
    struct ir_tensor* output_tensor;
    int ret = 0;

    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    if (input_tensor->dims[0] != output_tensor->dims[0] || input_tensor->dims[1] != output_tensor->dims[1] || input_tensor->dims[2] != output_tensor->dims[2] ||
        input_tensor->dims[3] != output_tensor->dims[3])
    ret = set_ir_tensor_shape(output_tensor, input_tensor->dims, input_tensor->dim_num);

    return ret;
}

void arm_softmax_q7(const q7_t* vec_in, const uint16_t dim_vec, q7_t* p_out);

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct ir_node* ir_node = exec_node->ir_node;
    struct ir_graph* ir_graph = ir_node->graph;
    struct ir_tensor* input_tensor;
    struct ir_tensor* output_tensor;

    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    arm_softmax_q7(input_tensor->data, input_tensor->elem_num, output_tensor->data);

    return 0;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct ir_node* exec_node)
{
    return OPS_SCORE_BEST;
}

static struct node_ops cmsis_node_ops = {.prerun = NULL,
                                         .run = run,
                                         .reshape = reshape,
                                         .postrun = NULL,
                                         .init_node = NULL,
                                         .release_node = NULL,
                                         .score = score};

static int reg_softmax_cmsis_ops(void* arg)
{
    return register_builtin_node_ops(OP_SOFTMAX, &cmsis_node_ops);
}

static int unreg_softmax_cmsis_ops(void* arg)
{
    return unregister_builtin_node_ops(OP_SOFTMAX, &cmsis_node_ops);
}

AUTO_REGISTER_OPS(reg_softmax_cmsis_ops);
AUTO_UNREGISTER_OPS(unreg_softmax_cmsis_ops);
