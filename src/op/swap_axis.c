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
#include "swap_axis_param.h"

DEFINE_PARM_PARSE_ENTRY(swap_axis_param, dim_0, dim_1);

static int infer_shape(struct ir_node* node)
{
    struct ir_graph* ir_graph = node->graph;
    struct ir_tensor* input = get_ir_graph_tensor(ir_graph, node->input_tensors[0]);
    struct ir_tensor* output = get_ir_graph_tensor(ir_graph, node->output_tensors[0]);
    struct swap_axis_param* swap_axis_param = ( struct swap_axis_param* )node->op.param_mem;

    if (swap_axis_param->dim_0 == swap_axis_param->dim_1)
    {
        return -1;
    }
    int in_size = 1;
    for (int i = 0; i < input->dim_num; i++)
    {
        in_size *= input->dims[i];
    }
    int out_size = 1;
    for (int i = 0; i < output->dim_num; i++)
    {
        out_size *= output->dims[i];
    }
    if (in_size != 1 || out_size != 1)
        return -1;

    if (swap_axis_param->dim_0 >= in_size || swap_axis_param->dim_1 >= in_size)
        return -1;

    int* newdim = ( int* )sys_malloc(in_size * sizeof(int));
    for (int i = 0; i < in_size; i++)
    {
        newdim[i] = input->dims[i];
    }
    newdim[swap_axis_param->dim_0] = input->dims[swap_axis_param->dim_1];
    newdim[swap_axis_param->dim_1] = input->dims[swap_axis_param->dim_0];
    set_ir_tensor_shape(output, newdim, in_size);

    sys_free(newdim);
    return 0;
}

static int init_op(struct ir_op* op)
{
    struct swap_axis_param* swap_axis_param = ( struct swap_axis_param* )sys_malloc(sizeof(struct swap_axis_param));

    if (swap_axis_param == NULL)
    {
        set_tengine_errno(ENOMEM);
        return -1;
    }

    swap_axis_param->dim_0 = 0;
    swap_axis_param->dim_1 = 1;

    op->param_mem = swap_axis_param;
    op->param_size = sizeof(struct swap_axis_param);
    op->same_shape = 0;
    op->infer_shape = infer_shape;

    return 0;
}

static void release_op(struct ir_op* op)
{
    sys_free(op->param_mem);
}

static int register_swap_axis_op(void* arg)
{
    struct op_method m;

    m.op_version = 1;
    m.init_op = init_op;
    m.release_op = release_op;
    m.access_param_entry = access_param_entry;

    return register_op(OP_SWAP_AXIS, OP_SWAP_AXIS_NAME, &m);
}

static int unregister_swap_axis_op(void* arg)
{
    sys_free(GET_PARAM_PARSE_MAP(swap_axis_param));
    return unregister_op(OP_SWAP_AXIS, 1);
}

AUTO_REGISTER_OP(register_swap_axis_op);
AUTO_UNREGISTER_OP(unregister_swap_axis_op);
