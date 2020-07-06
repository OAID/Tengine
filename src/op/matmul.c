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
 * Author: bhu@openailab.com
 */

#include <stdio.h>
#include <assert.h>
#include "sys_port.h"
#include "tengine_ir.h"
#include "tengine_errno.h"
#include "tengine_log.h"
#include "tengine_op.h"

static int infer_shape(struct ir_node* node)
{
    struct ir_graph* graph = node->graph;
    struct ir_tensor* input0 = get_ir_graph_tensor(graph, node->input_tensors[0]);
    struct ir_tensor* input1 = get_ir_graph_tensor(graph, node->input_tensors[1]);
    struct ir_tensor* output = get_ir_graph_tensor(graph, node->output_tensors[0]);

    int output_number;

    if (input1->dim_num != input0->dim_num)
    {
        TLOG_ERR("dim's size of inputs must be qual for operator matmul\n");
        set_tengine_errno(EFAULT);
        return -1;
    }

    set_ir_tensor_shape(output, input0->dims, input0->dim_num);

    return 0;
}

static int init_op(struct ir_op* op)
{
    op->same_shape = 0;
    op->infer_shape = infer_shape;
    return 0;
}

static int register_matmul_op(void* arg)
{
    struct op_method m;

    m.op_version = 1;
    m.init_op = init_op;
    m.release_op = NULL;
    m.access_param_entry = NULL;

    return register_op(OP_MATMUL, OP_MATMUL_NAME, &m);
}

static int unregister_matmul_op(void* arg)
{
    return unregister_op(OP_MATMUL, 1);
}

AUTO_REGISTER_OP(register_matmul_op);
AUTO_UNREGISTER_OP(unregister_matmul_op);
