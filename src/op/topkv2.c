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

#include <stdio.h>
#include <assert.h>
#include <stdbool.h>

#include "sys_port.h"
#include "tengine_ir.h"
#include "tengine_errno.h"
#include "tengine_log.h"
#include "tengine_op.h"
#include "parameter.h"
#include "topkv2_param.h"

DEFINE_PARM_PARSE_ENTRY(topkv2_param, k, sorted);

static int infer_shape(struct ir_node* node)
{
    struct ir_graph* ir_graph = node->graph;
    struct ir_tensor* input = get_ir_graph_tensor(ir_graph, node->input_tensors[0]);
    struct ir_tensor* output = get_ir_graph_tensor(ir_graph, node->output_tensors[0]);
    struct ir_tensor* output1 = get_ir_graph_tensor(ir_graph, node->output_tensors[1]);
    struct topkv2_param* topkv2_param = ( struct topkv2_param* )node->op.param_mem;

    int in_size = input->dim_num;
    int* in_dim = ( int* )sys_malloc((in_size) * sizeof(int));
    if (topkv2_param->k > input->dims[in_size - 1])
    {
        set_tengine_errno(ENOENT);
        return false;
    }
    for (int i = 0; i < in_size - 1; i++)
    {
        in_dim[i] = input->dims[i];
    }
    in_dim[in_size - 1] = topkv2_param->k;
    set_ir_tensor_shape(output, in_dim, in_size);

    set_ir_tensor_shape(output1, in_dim, in_size);

    sys_free(in_dim);
    return 0;
}

static int init_op(struct ir_op* op)
{
    struct topkv2_param* topkv2_param = ( struct topkv2_param* )sys_malloc(sizeof(struct topkv2_param));

    if (topkv2_param == NULL)
    {
        set_tengine_errno(ENOMEM);
        return -1;
    }

    topkv2_param->k = 1;
    topkv2_param->sorted = false;

    op->param_mem = topkv2_param;
    op->param_size = sizeof(struct topkv2_param);
    op->same_shape = 0;
    op->infer_shape = infer_shape;

    return 0;
}

static void release_op(struct ir_op* op)
{
    sys_free(op->param_mem);
}

static int register_topkv2_op(void* arg)
{
    struct op_method m;

    m.op_version = 1;
    m.init_op = init_op;
    m.release_op = release_op;
    m.access_param_entry = access_param_entry;

    return register_op(OP_TOPKV2, OP_TOPKV2_NAME, &m);
}

static int unregister_topkv2_op(void* arg)
{
    sys_free(GET_PARAM_PARSE_MAP(topkv2_param));
    return unregister_op(OP_TOPKV2, 1);
}

AUTO_REGISTER_OP(register_topkv2_op);
AUTO_UNREGISTER_OP(unregister_topkv2_op);
