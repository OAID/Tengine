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
#include <stdbool.h>

#include "sys_port.h"
#include "tengine_ir.h"
#include "tengine_errno.h"
#include "tengine_log.h"
#include "tengine_op.h"
#include "parameter.h"
#include "reducel2_param.h"

DEFINE_PARM_PARSE_ENTRY(reducel2_param, axis, keepdim);

static int infer_shape(struct ir_node* node)
{
    struct ir_graph* ir_graph = node->graph;
    struct ir_tensor* input = get_ir_graph_tensor(ir_graph, node->input_tensors[0]);
    struct ir_tensor* output = get_ir_graph_tensor(ir_graph, node->output_tensors[0]);
    struct reducel2_param* reducel2_param = ( struct reducel2_param* )node->op.param_mem;

    int kd = reducel2_param->keepdim;
    int axis = reducel2_param->axis;

    int* out_dim = ( int* )sys_malloc(input->dim_num * sizeof(int));

    if (axis < 0)
        axis = axis + input->dim_num;

    for (unsigned int i = 0; i < input->dim_num && i < ( unsigned int )axis; i++)
    {
        out_dim[i] = input->dims[i];
    }

    if (kd == 1)
    {
        for (unsigned int i = axis; i < input->dim_num; i++)
        {
            out_dim[i] = 1;
        }
    }

    set_ir_tensor_shape(output, out_dim, input->dim_num);

    sys_free(out_dim);

    return 0;
}

static int init_op(struct ir_op* op)
{
    struct reducel2_param* reducel2_param = ( struct reducel2_param* )sys_malloc(sizeof(struct reducel2_param));

    if (reducel2_param == NULL)
    {
        set_tengine_errno(ENOMEM);
        return -1;
    }

    reducel2_param->axis = 0;
    reducel2_param->keepdim = 1;

    op->param_mem = reducel2_param;
    op->param_size = sizeof(struct reducel2_param);
    op->same_shape = 0;
    op->infer_shape = infer_shape;

    return 0;
}

static void release_op(struct ir_op* op)
{
    sys_free(op->param_mem);
    sys_free(GET_PARAM_PARSE_MAP(reducel2_param));
}

static int register_reducel2_op(void* arg)
{
    struct op_method m;

    m.op_version = 1;
    m.init_op = init_op;
    m.release_op = release_op;
    m.access_param_entry = access_param_entry;

    return register_op(OP_REDUCEL2, OP_REDUCEL2_NAME, &m);
}

static int unregister_reducel2_op(void* arg)
{
    return unregister_op(OP_REDUCEL2, 1);
}

AUTO_REGISTER_OP(register_reducel2_op);
AUTO_UNREGISTER_OP(unregister_reducel2_op);
