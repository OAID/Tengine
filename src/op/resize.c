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
 * Author: qtang@openailab.com
 */

#include <stdio.h>
#include <assert.h>

#include "sys_port.h"
#include "tengine_ir.h"
#include "tengine_errno.h"
#include "tengine_log.h"
#include "tengine_op.h"
#include "parameter.h"
#include "resize_param.h"

DEFINE_PARM_PARSE_ENTRY(resize_param, scale_h, scale_w, type);

static int infer_shape(struct ir_node* node)
{
    struct ir_graph* graph = node->graph;
    struct ir_tensor* input = get_ir_graph_tensor(graph, node->input_tensors[0]);
    struct ir_tensor* output = get_ir_graph_tensor(graph, node->output_tensors[0]);
    struct resize_param* resize_param = ( struct resize_param* )(node->op.param_mem);

    int dims[4];
    dims[0] = input->dims[0];
    if (graph->graph_layout == TENGINE_LAYOUT_NCHW)
    {
        dims[1] = input->dims[1];
        dims[2] = ( int )(input->dims[2] * resize_param->scale_h);
        dims[3] = ( int )(input->dims[3] * resize_param->scale_w);
    }
    else if (graph->graph_layout == TENGINE_LAYOUT_NHWC)
    {
        dims[1] = ( int )(input->dims[1] * resize_param->scale_h);
        dims[2] = ( int )(input->dims[2] * resize_param->scale_w);
        dims[3] = input->dims[3];
    }
    else
    {
        TLOG_ERR("resizeolution infer shape: unknown graph layout: %d\n", graph->graph_layout);
        set_tengine_errno(EFAULT);
        return -1;
    }

    set_ir_tensor_shape(output, dims, 4);

    return 0;
}

static int init_op(struct ir_op* op)
{
    struct resize_param* resize_param = ( struct resize_param* )sys_malloc(sizeof(struct resize_param));

    if (resize_param == NULL)
    {
        set_tengine_errno(ENOMEM);
        return -1;
    }

    /*set the param default value */
    resize_param->scale_h = 1.0;
    resize_param->scale_w = 1.0;
    resize_param->type = 0;

    op->param_mem = resize_param;
    op->param_size = sizeof(struct resize_param);
    op->same_shape = 0;
    op->infer_shape = infer_shape;

    return 0;
}

static void release_op(struct ir_op* op)
{
    sys_free(op->param_mem);
}

static int register_resize_op(void* arg)
{
    struct op_method m;

    m.op_version = 1;
    m.init_op = init_op;
    m.release_op = release_op;
    m.access_param_entry = access_param_entry;

    return register_op(OP_RESIZE, OP_RESIZE_NAME, &m);
}

static int unregister_resize_op(void* arg)
{
    sys_free(GET_PARAM_PARSE_MAP(resize_param));
    return unregister_op(OP_RESIZE, 1);
}

AUTO_REGISTER_OP(register_resize_op);
AUTO_UNREGISTER_OP(unregister_resize_op);
