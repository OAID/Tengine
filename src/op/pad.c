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
 * Author: sqfu@openailab.com
 */

#include <stdio.h>
#include <assert.h>

#include "sys_port.h"
#include "tengine_ir.h"
#include "tengine_errno.h"
#include "tengine_log.h"
#include "tengine_op.h"
#include "parameter.h"
#include "pad_param.h"

DEFINE_PARM_PARSE_ENTRY(pad_param, mode, pad_0_h, pad_0_w, pad_1_h, pad_1_w, pad_2_h, pad_2_w, pad_3_h, pad_3_w, value);

static int infer_shape(struct ir_node* node)
{
    struct ir_graph* graph = node->graph;
    struct ir_tensor* input = get_ir_graph_tensor(graph, node->input_tensors[0]);
    struct ir_tensor* output = get_ir_graph_tensor(graph, node->output_tensors[0]);

    struct pad_param* pad_param = ( struct pad_param* )(node->op.param_mem);

    int dims[MAX_SHAPE_DIM_NUM] = {0};
    if (pad_param->pad_0_h != -1 && pad_param->pad_0_w != -1 && pad_param->pad_1_h != -1 && pad_param->pad_1_w != -1 &&
        pad_param->pad_2_h != -1 && pad_param->pad_2_w != -1 && pad_param->pad_3_h != -1 && pad_param->pad_3_w != -1)
    {
        dims[0] = input->dims[0] + pad_param->pad_0_h + pad_param->pad_0_w;
        ;
        dims[1] = input->dims[1] + pad_param->pad_1_h + pad_param->pad_1_w;
        ;
        dims[2] = input->dims[2] + pad_param->pad_2_h + pad_param->pad_2_w;
        ;
        dims[3] = input->dims[3] + pad_param->pad_3_h + pad_param->pad_3_w;
        ;
    }
    else
    {
        return 0;
    }

    set_ir_tensor_shape(output, dims, input->dim_num);

    return 0;
}
static int init_op(struct ir_op* op)
{
    struct pad_param* pad_param = ( struct pad_param* )sys_malloc(sizeof(struct pad_param));

    if (pad_param == NULL)
    {
        set_tengine_errno(ENOMEM);
        return -1;
    }

    pad_param->mode = 0;
    pad_param->pad_0_h = -1;    // n
    pad_param->pad_0_w = -1;
    pad_param->pad_1_h = -1;    // c
    pad_param->pad_1_w = -1;
    pad_param->pad_2_h = -1;    // h
    pad_param->pad_2_w = -1;
    pad_param->pad_3_h = -1;    // w
    pad_param->pad_3_w = -1;
    pad_param->value = 0;

    /*set the param default value */
    op->param_mem = pad_param;
    op->param_size = sizeof(struct pad_param);
    op->same_shape = 0;
    op->infer_shape = infer_shape;

    return 0;
}

static void release_op(struct ir_op* op)
{
    sys_free(op->param_mem);
}

static int register_pad_op(void* arg)
{
    struct op_method m;

    m.op_version = 1;
    m.init_op = init_op;
    m.release_op = release_op;
    m.access_param_entry = access_param_entry;

    return register_op(OP_PAD, OP_PAD_NAME, &m);
}

static int unregister_pad_op(void* arg)
{
    sys_free(GET_PARAM_PARSE_MAP(pad_param));
    return unregister_op(OP_PAD, 1);
}

AUTO_REGISTER_OP(register_pad_op);
AUTO_UNREGISTER_OP(unregister_pad_op);
