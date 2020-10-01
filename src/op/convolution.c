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
#include "convolution_param.h"

DEFINE_PARM_PARSE_ENTRY(conv_param, kernel_h, kernel_w, stride_h, stride_w, pad_h0, pad_h1, pad_w0, pad_w1, dilation_h,
                        dilation_w, input_channel, output_channel, group);

static int infer_shape(struct ir_node* node)
{
    struct ir_graph* graph = node->graph;
    struct ir_tensor* input = get_ir_graph_tensor(graph, node->input_tensors[0]);
    struct ir_tensor* output = get_ir_graph_tensor(graph, node->output_tensors[0]);

    struct conv_param* conv_param = ( struct conv_param* )(node->op.param_mem);

    int n = input->dims[0];
    int h, w;

    if (conv_param->kernel_w == 0)
    {
        conv_param->kernel_w = 1;
        conv_param->pad_w0 = 0;
        conv_param->pad_w1 = 0;
    }

    if (conv_param->kernel_h == 0)
        conv_param->kernel_h = 1;
    if (conv_param->stride_w == 0)
        conv_param->stride_w = 1;
    if (conv_param->stride_h == 0)
        conv_param->stride_h = 1;

    if (graph->graph_layout == TENGINE_LAYOUT_NCHW)
    {
        conv_param->input_channel = input->dims[1];
        h = input->dims[2];
        w = input->dims[3];
    }
    else if (graph->graph_layout == TENGINE_LAYOUT_NHWC)
    {
        h = input->dims[1];
        w = input->dims[2];
        conv_param->input_channel = input->dims[1];
    }
    else
    {
        TLOG_ERR("convolution infer shape: unknown graph layout: %d\n", graph->graph_layout);
        set_tengine_errno(EFAULT);
        return -1;
    }

    int out_c = conv_param->output_channel;
    int out_h, out_w;

    /* handle the same padding case, which pad_h0 and pad_h1 is -1 (SAME_UPPER)
        -2 (SAME_LOWER) */

    if (conv_param->pad_h0 < 0)
    {
        out_h = (h - 1) / conv_param->stride_h + 1;

        int total_len = (out_h - 1) * conv_param->stride_h + conv_param->kernel_h;

        int pad_num = total_len - h;

        if (conv_param->pad_h0 == -1)
        {
            conv_param->pad_h0 = pad_num / 2;
            conv_param->pad_h1 = pad_num - pad_num / 2;
        }
        else
        {
            conv_param->pad_h1 = pad_num / 2;
            conv_param->pad_h0 = pad_num - pad_num / 2;
        }
    }
    else
        out_h =
            (h - conv_param->dilation_h * (conv_param->kernel_h - 1) - 1 + conv_param->pad_h0 + conv_param->pad_h1) /
                conv_param->stride_h +
            1;

    if (conv_param->pad_w0 < 0)
    {
        out_w = (w - 1) / conv_param->stride_w + 1;

        int total_len = (out_w - 1) * conv_param->stride_w + conv_param->kernel_w;

        int pad_num = total_len - w;

        if (conv_param->pad_w0 == -1)
        {
            conv_param->pad_w0 = pad_num / 2;
            conv_param->pad_w1 = pad_num - pad_num / 2;
        }
        else
        {
            conv_param->pad_w1 = pad_num / 2;
            conv_param->pad_w0 = pad_num - pad_num / 2;
        }
    }
    else
        out_w =
            (w - conv_param->dilation_w * (conv_param->kernel_w - 1) - 1 + conv_param->pad_w0 + conv_param->pad_w1) /
                conv_param->stride_w +
            1;

    int dims[4];

    dims[0] = n;

    if (graph->graph_layout == TENGINE_LAYOUT_NCHW)
    {
        dims[1] = out_c;
        dims[2] = out_h;
        dims[3] = out_w;
    }
    else
    {
        dims[1] = out_h;
        dims[2] = out_w;
        dims[3] = out_c;
    }

    for (int i=0; i<4; i++)
    {
        if (dims[i] == 0)
            dims[i] = 1;
    }

    set_ir_tensor_shape(output, dims, 4);

    return 0;
}

static int init_op(struct ir_op* op)
{
    struct conv_param* conv_param = ( struct conv_param* )sys_malloc(sizeof(struct conv_param));

    if (conv_param == NULL)
    {
        set_tengine_errno(ENOMEM);
        return -1;
    }

    /* set the param default value */
    conv_param->kernel_h = 1;
    conv_param->kernel_w = 1;
    conv_param->stride_h = 1;
    conv_param->stride_w = 1;
    conv_param->pad_h0 = 0;
    conv_param->pad_h1 = 0;
    conv_param->pad_w0 = 0;
    conv_param->pad_w1 = 0;
    conv_param->dilation_h = 1;
    conv_param->dilation_w = 1;
    conv_param->input_channel = 64;
    conv_param->output_channel = 64;
    conv_param->group = 1;
    conv_param->activation = -1;

    op->param_mem = conv_param;
    op->param_size = sizeof(struct conv_param);
    op->same_shape = 0;
    op->infer_shape = infer_shape;

    return 0;
}

static void release_op(struct ir_op* op)
{
    sys_free(op->param_mem);
}

static int register_conv_op(void* arg)
{
    struct op_method m;

    m.op_version = 1;
    m.init_op = init_op;
    m.release_op = release_op;
    m.access_param_entry = access_param_entry;

    return register_op(OP_CONV, OP_CONV_NAME, &m);
}

static int unregister_conv_op(void* arg)
{
    sys_free(GET_PARAM_PARSE_MAP(conv_param));
    return unregister_op(OP_CONV, 1);
}

AUTO_REGISTER_OP(register_conv_op);
AUTO_UNREGISTER_OP(unregister_conv_op);
