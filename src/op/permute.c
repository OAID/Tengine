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

#include "sys_port.h"
#include "tengine_ir.h"
#include "tengine_errno.h"
#include "tengine_log.h"
#include "tengine_op.h"
#include "parameter.h"
#include "permute_param.h"

DEFINE_PARM_PARSE_ENTRY(permute_param, flag, order0, order1, order2, order3);

static int infer_shape(struct ir_node* node)
{
    struct ir_graph* graph = node->graph;
    struct ir_tensor* input = get_ir_graph_tensor(graph, node->input_tensors[0]);
    struct ir_tensor* output = get_ir_graph_tensor(graph, node->output_tensors[0]);
    permute_param_t* param = ( struct permute_param* )(node->op.param_mem);

    int dims[MAX_SHAPE_DIM_NUM] = {0};
    int dim_size = input->dim_num;

    if ((param->order0 == 0) && (param->order1 == 2) && (param->order2 == 3) && (param->order3 == 1))
    {
        dims[0] = input->dims[0];
        dims[1] = input->dims[2];
        dims[2] = input->dims[3];
        dims[3] = input->dims[1];

        output->layout = TENGINE_LAYOUT_NHWC;
    }
    else if ((param->order0 == 1) && (param->order1 == 0) && (param->order2 == 2) && dim_size == 3)
    {
        dims[0] = input->dims[1];
        dims[1] = input->dims[0];
        dims[2] = input->dims[2];
    }
    else
    {
        return -1;
    }

    set_ir_tensor_shape(output, dims, dim_size);

    return 0;
}

static int init_op(struct ir_op* op)
{
    struct permute_param* permute_param = ( struct permute_param* )sys_malloc(sizeof(struct permute_param));

    if (permute_param == NULL)
    {
        set_tengine_errno(ENOMEM);
        return -1;
    }

    /*set the param default value */
    permute_param->flag = 0;
    permute_param->order0 = 0;
    permute_param->order1 = 1;
    permute_param->order2 = 2;
    permute_param->order3 = 3;
    op->param_mem = permute_param;
    op->param_size = sizeof(struct permute_param);
    op->same_shape = 0;
    op->infer_shape = infer_shape;

    return 0;
}

static void release_op(struct ir_op* op)
{
    sys_free(op->param_mem);
}

static int register_permute_op(void* arg)
{
    struct op_method m;

    m.op_version = 1;
    m.init_op = init_op;
    m.release_op = release_op;
    m.access_param_entry = access_param_entry;

    return register_op(OP_PERMUTE, OP_PERMUTE_NAME, &m);
}

static int unregister_permute_op(void* arg)
{
    sys_free(GET_PARAM_PARSE_MAP(permute_param));
    return unregister_op(OP_PERMUTE, 1);
}

AUTO_REGISTER_OP(register_permute_op);
AUTO_UNREGISTER_OP(unregister_permute_op);
