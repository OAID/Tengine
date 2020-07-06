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
#include "parameter.h"
#include "batchtospacend_param.h"

DEFINE_PARM_PARSE_ENTRY(batchtospacend_param, dilation_x, dilation_y, crop_top, crop_bottom, crop_left, crop_right);

static int infer_shape(struct ir_node* node)
{
    struct ir_graph* graph = node->graph;
    struct ir_tensor* input = get_ir_graph_tensor(graph, node->input_tensors[0]);
    struct ir_tensor* output = get_ir_graph_tensor(graph, node->output_tensors[0]);

    struct batchtospacend_param* batchtospacend_param = ( struct batchtospacend_param* )(node->op.param_mem);

    int out_dim[4];

    out_dim[0] = input->dims[0] / (batchtospacend_param->dilation_x * batchtospacend_param->dilation_y);
    out_dim[1] = input->dims[1] * batchtospacend_param->dilation_y - batchtospacend_param->crop_top -
                 batchtospacend_param->crop_bottom;
    out_dim[2] = input->dims[2] * batchtospacend_param->dilation_x - batchtospacend_param->crop_left -
                 batchtospacend_param->crop_right;
    out_dim[3] = input->dims[3];

    set_ir_tensor_shape(output, out_dim, 4);

    return 0;
}

static int init_op(struct ir_op* op)
{
    struct batchtospacend_param* batchtospacend_param =
        ( struct batchtospacend_param* )sys_malloc(sizeof(struct batchtospacend_param));

    if (batchtospacend_param == NULL)
    {
        set_tengine_errno(ENOMEM);
        return -1;
    }

    /*set the param default value */
    batchtospacend_param->dilation_x = 0;
    batchtospacend_param->dilation_y = 0;
    batchtospacend_param->crop_top = 0;
    batchtospacend_param->crop_bottom = 0;
    batchtospacend_param->crop_left = 0;
    batchtospacend_param->crop_right = 0;

    op->param_mem = batchtospacend_param;
    op->param_size = sizeof(struct batchtospacend_param);
    op->same_shape = 0;
    op->infer_shape = infer_shape;

    return 0;
}

static void release_op(struct ir_op* op)
{
    sys_free(op->param_mem);
}

static int register_batchtospacend_op(void* arg)
{
    struct op_method m;

    m.op_version = 1;
    m.init_op = init_op;
    m.release_op = release_op;
    m.access_param_entry = access_param_entry;

    return register_op(OP_BATCHTOSPACEND, OP_BATCHTOSPACEND_NAME, &m);
}

static int unregister_batchtospacend_op(void* arg)
{
    sys_free(GET_PARAM_PARSE_MAP(batchtospacend_param));
    return unregister_op(OP_BATCHTOSPACEND, 1);
}

AUTO_REGISTER_OP(register_batchtospacend_op);
AUTO_UNREGISTER_OP(unregister_batchtospacend_op);
