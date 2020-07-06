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
 * Author: qli@openailab.com
 */
#include <stdio.h>
#include <assert.h>
#include "sys_port.h"
#include "tengine_ir.h"
#include "tengine_errno.h"
#include "tengine_log.h"
#include "tengine_op.h"
#include "parameter.h"
#include "gemm_param.h"

DEFINE_PARM_PARSE_ENTRY(gemm_param, alpha, beta, transA, transB);

static int infer_shape(struct ir_node* node)
{
    struct ir_graph* graph = node->graph;
    struct ir_tensor* input = get_ir_graph_tensor(graph, node->input_tensors[0]);
    struct ir_tensor* output = get_ir_graph_tensor(graph, node->output_tensors[0]);
    struct ir_tensor* weight = get_ir_graph_tensor(graph, node->input_tensors[1]);

    struct gemm_param* gemm_param = ( struct gemm_param* )(node->op.param_mem);

    int dims[2];
    if (gemm_param->transA)
        dims[0] = input->dims[1];
    else
        dims[0] = input->dims[0];

    if (gemm_param->transB)
        dims[1] = weight->dims[0];
    else
        dims[1] = weight->dims[1];

    set_ir_tensor_shape(output, dims, 2);

    return 0;
}
static int init_op(struct ir_op* op)
{
    struct gemm_param* gemm_param = ( struct gemm_param* )sys_malloc(sizeof(struct gemm_param));

    if (gemm_param == NULL)
    {
        set_tengine_errno(ENOMEM);
        return -1;
    }
    op->param_mem = gemm_param;
    op->param_size = sizeof(struct gemm_param);
    op->same_shape = 0;
    op->infer_shape = infer_shape;

    return 0;
}

static void release_op(struct ir_op* op)
{
    sys_free(op->param_mem);
}
static int register_gemm_op(void* arg)
{
    struct op_method m;

    m.op_version = 1;
    m.init_op = init_op;
    m.release_op = release_op;
    m.access_param_entry = access_param_entry;

    return register_op(OP_GEMM, OP_GEMM_NAME, &m);
}

static int unregister_gemm_op(void* arg)
{
    sys_free(GET_PARAM_PARSE_MAP(gemm_param));
    return unregister_op(OP_GEMM, 1);
}

AUTO_REGISTER_OP(register_gemm_op);
AUTO_UNREGISTER_OP(unregister_gemm_op);
