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
#include "interp_param.h"

DEFINE_PARM_PARSE_ENTRY(interp_param, resize_type, output_height, output_width, height_scale, width_scale);

static int infer_shape(struct ir_node* node)
{
    struct ir_graph* graph = node->graph;
    struct ir_tensor* input = get_ir_graph_tensor(graph, node->input_tensors[0]);
    struct ir_tensor* output = get_ir_graph_tensor(graph, node->output_tensors[0]);
    int in_n = input->dims[0];
    int in_c = input->dims[1];
    int in_h = input->dims[2];
    int in_w = input->dims[3];

    struct interp_param* param = ( struct interp_param* )(node->op.param_mem);

    if (param == NULL)
    {
        set_tengine_errno(ENOMEM);
        return -1;
    }

    if (param->height_scale != 0 && param->width_scale != 0)
    {
        param->output_height = in_h * param->height_scale;
        param->output_width = in_w * param->width_scale;
    }
    else
    {
        param->height_scale = (float )param->output_height / (float )in_h;
        param->width_scale = (float )param->output_width / (float )in_w;
    }

    int dim[4] = {0};

    dim[0] = in_n;
    dim[1] = in_c;
    dim[2] = param->output_height;
    dim[3] = param->output_width;

    set_ir_tensor_shape(output, dim, 4);

    return 0;
}

static int init_op(struct ir_op* op)
{
    struct interp_param* interp_param = ( struct interp_param* )sys_malloc(sizeof(struct interp_param));

    if (interp_param == NULL)
    {
        set_tengine_errno(ENOMEM);
        return -1;
    }

    /*set the param default value */
    interp_param->resize_type = 1;
    interp_param->output_height = 0;
    interp_param->output_width = 0;
    interp_param->height_scale = 1.f;
    interp_param->width_scale = 1.f;

    op->param_mem = interp_param;
    op->param_size = sizeof(struct interp_param);
    op->same_shape = 0;
    op->infer_shape = infer_shape;

    return 0;
}

static void release_op(struct ir_op* op)
{
    sys_free(op->param_mem);
}

static int register_interp_op(void* arg)
{
    struct op_method m;

    m.op_version = 1;
    m.init_op = init_op;
    m.release_op = release_op;
    m.access_param_entry = access_param_entry;

    return register_op(OP_INTERP, OP_INTERP_NAME, &m);
}

static int unregister_interp_op(void* arg)
{
    sys_free(GET_PARAM_PARSE_MAP(interp_param));
    return unregister_op(OP_INTERP, 1);
}

AUTO_REGISTER_OP(register_interp_op);
AUTO_UNREGISTER_OP(unregister_interp_op);
