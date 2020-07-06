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
#include "spacetobatchnd_param.h"

DEFINE_PARM_PARSE_ENTRY(spacetobatchnd_param, dilation_x, dilation_y, pad_top, pad_bottom, pad_left, pad_right);

static int infer_shape(struct ir_node* node)
{
    struct ir_graph* graph = node->graph;
    struct ir_tensor* input = get_ir_graph_tensor(graph, node->input_tensors[0]);
    struct ir_tensor* output = get_ir_graph_tensor(graph, node->output_tensors[0]);

    struct spacetobatchnd_param* spacetobatchnd_param = ( struct spacetobatchnd_param* )(node->op.param_mem);

    int out_dim[4];

    out_dim[0] = input->dims[0] * (spacetobatchnd_param->dilation_x) * (spacetobatchnd_param->dilation_y);
    out_dim[1] = (input->dims[1] + spacetobatchnd_param->pad_top + spacetobatchnd_param->pad_bottom) /
                 spacetobatchnd_param->dilation_y;
    out_dim[2] = (input->dims[2] + spacetobatchnd_param->pad_left + spacetobatchnd_param->pad_right) /
                 spacetobatchnd_param->dilation_x;
    out_dim[3] = input->dims[3];

    set_ir_tensor_shape(output, out_dim, 4);

    return 0;
}

static int init_op(struct ir_op* op)
{
    struct spacetobatchnd_param* spacetobatchnd_param =
        ( struct spacetobatchnd_param* )sys_malloc(sizeof(struct spacetobatchnd_param));

    if (spacetobatchnd_param == NULL)
    {
        set_tengine_errno(ENOMEM);
        return -1;
    }

    /*set the param default value */
    spacetobatchnd_param->dilation_x = 0;
    spacetobatchnd_param->dilation_y = 0;
    spacetobatchnd_param->pad_top = 0;
    spacetobatchnd_param->pad_bottom = 0;
    spacetobatchnd_param->pad_left = 0;
    spacetobatchnd_param->pad_right = 0;

    op->param_mem = spacetobatchnd_param;
    op->param_size = sizeof(struct spacetobatchnd_param);
    op->same_shape = 0;
    op->infer_shape = infer_shape;

    return 0;
}

static void release_op(struct ir_op* op)
{
    sys_free(op->param_mem);
}

static int register_spacetobatchnd_op(void* arg)
{
    struct op_method m;

    m.op_version = 1;
    m.init_op = init_op;
    m.release_op = release_op;
    m.access_param_entry = access_param_entry;

    return register_op(OP_SPACETOBATCHND, OP_SPACETOBATCHND_NAME, &m);
}

static int unregister_spacetobatchnd_op(void* arg)
{
    sys_free(GET_PARAM_PARSE_MAP(spacetobatchnd_param));
    return unregister_op(OP_SPACETOBATCHND, 1);
}

AUTO_REGISTER_OP(register_spacetobatchnd_op);
AUTO_UNREGISTER_OP(unregister_spacetobatchnd_op);
