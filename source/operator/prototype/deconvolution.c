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
 * Copyright (c) 2021, OPEN AI LAB
 * Author: haitao@openailab.com
 */

#include "deconv_param.h"

#include "api/c_api.h"
#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "module/module.h"
#include "utility/sys_port.h"
#include "utility/log.h"


static int infer_shape(struct node* node)
{
    struct graph* graph = node->graph;
    struct tensor* input = get_ir_graph_tensor(graph, node->input_tensors[0]);
    struct tensor* output = get_ir_graph_tensor(graph, node->output_tensors[0]);

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
        return -1;
    }

    int out_c = deconv_param->num_output;

    // TLOG_ERR("   out c: %d  \n", out_c);

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
        return -1;
    }

    set_ir_tensor_shape(output, dims, 4);

    return 0;
}

static int init_op(struct op* op)
{
    struct deconv_param* deconv_param = ( struct deconv_param* )sys_malloc(sizeof(struct deconv_param));

    if (deconv_param == NULL)
    {
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

static void release_op(struct op* op)
{
    sys_free(op->param_mem);
}

int register_deconvolution_op()
{
    struct method m;

    m.version = 1;
    m.init = init_op;
    m.release = release_op;


    return register_op(OP_DECONV, OP_DECONV_NAME, &m);
}

int unregister_deconvolution_op()
{
    return unregister_op(OP_DECONV, 1);
}
