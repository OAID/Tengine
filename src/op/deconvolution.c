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
#include "deconv_param.h"

DEFINE_PARM_PARSE_ENTRY(deconv_param, num_output, kernel_h, kernel_w, stride_h, stride_w, pad_h0, pad_w0, pad_h1,
                        pad_w1, dilation_h, dilation_w, group);

static int infer_shape(struct ir_node* node)
{
    struct ir_graph* graph = node->graph;
    struct ir_tensor* input = get_ir_graph_tensor(graph, node->input_tensors[0]);
    struct ir_tensor* output = get_ir_graph_tensor(graph, node->output_tensors[0]);

    struct deconv_param* deconv_param = ( struct deconv_param* )(node->op.param_mem);

    int n = input->dims[0];
    int h, w;

    if (graph->graph_layout == TENGINE_LAYOUT_NCHW)
    {
        h = input->dims[2];
        w = input->dims[3];
    }
    else if (graph->graph_layout == TENGINE_LAYOUT_NHWC)
    {
        h = input->dims[1];
        w = input->dims[2];
    }
    else
    {
        TLOG_ERR("deconvolution infer shape: unknown graph layout: %d\n", graph->graph_layout);
        set_tengine_errno(EFAULT);
        return -1;
    }

    int out_c = deconv_param->num_output;

    // fprintf(stderr, "   out c: %d  \n", out_c);

    int kernel_extent_w = deconv_param->dilation_w * (deconv_param->kernel_w - 1) + 1;
    int kernel_extent_h = deconv_param->dilation_h * (deconv_param->kernel_h - 1) + 1;

    int output_h = (h - 1) * deconv_param->stride_h + kernel_extent_h - deconv_param->pad_h0 - deconv_param->pad_h1 + deconv_param->output_pad_h0;
    int output_w = (w - 1) * deconv_param->stride_w + kernel_extent_w - deconv_param->pad_w0 - deconv_param->pad_w1 + deconv_param->output_pad_w0;

    int dims[4];

    dims[0] = n;

    if (graph->graph_layout == TENGINE_LAYOUT_NCHW)
    {
        dims[1] = out_c;
        dims[2] = output_h;
        dims[3] = output_w;
    }
    else if (graph->graph_layout == TENGINE_LAYOUT_NHWC)
    {
        dims[1] = output_h;
        dims[2] = output_w;
        dims[3] = out_c;
    }
    else
    {
        TLOG_ERR("deconvolution infer shape: unknown graph layout: %d\n", graph->graph_layout);
        set_tengine_errno(EFAULT);
        return -1;
    }

    set_ir_tensor_shape(output, dims, 4);

    return 0;
}

static int init_op(struct ir_op* op)
{
    struct deconv_param* deconv_param = ( struct deconv_param* )sys_malloc(sizeof(struct deconv_param));

    if (deconv_param == NULL)
    {
        set_tengine_errno(ENOMEM);
        return -1;
    }

    /*set the param default value */
    deconv_param->num_output = 1;
    deconv_param->kernel_h = 1;
    deconv_param->kernel_w = 1;
    deconv_param->stride_h = 1;
    deconv_param->stride_w = 1;
    deconv_param->pad_h0 = 0;
    deconv_param->pad_h1 = 0;
    deconv_param->pad_w0 = 0;
    deconv_param->pad_w1 = 0;
    deconv_param->dilation_h = 1;
    deconv_param->dilation_w = 1;
    deconv_param->group = 1;
    deconv_param->activation = -1;
    deconv_param->output_pad_h0 = 0;
    deconv_param->output_pad_w0 = 0;

    op->param_mem = deconv_param;
    op->param_size = sizeof(struct deconv_param);
    op->same_shape = 0;
    op->infer_shape = infer_shape;

    return 0;
}

static void release_op(struct ir_op* op)
{
    sys_free(op->param_mem);
}

static int register_deconv_op(void* arg)
{
    struct op_method m;

    m.op_version = 1;
    m.init_op = init_op;
    m.release_op = release_op;
    m.access_param_entry = access_param_entry;

    return register_op(OP_DECONV, OP_DECONV_NAME, &m);
}

static int unregister_deconv_op(void* arg)
{
    sys_free(GET_PARAM_PARSE_MAP(deconv_param));
    return unregister_op(OP_DECONV, 1);
}

AUTO_REGISTER_OP(register_deconv_op);
AUTO_UNREGISTER_OP(unregister_deconv_op);
