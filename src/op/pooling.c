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
#include "pooling_param.h"

DEFINE_PARM_PARSE_ENTRY(pool_param, pool_method, kernel_h, kernel_w, stride_h, stride_w, pad_h0, pad_h1, pad_w0, pad_w1,
                        caffe_flavor, funct);

static int infer_shape(struct ir_node* node)
{
    struct ir_graph* ir_graph = node->graph;
    struct ir_tensor* input = get_ir_graph_tensor(ir_graph, node->input_tensors[0]);
    struct ir_tensor* output = get_ir_graph_tensor(ir_graph, node->output_tensors[0]);
    struct pool_param* pool_param = ( struct pool_param* )node->op.param_mem;

    int batch, channel, input_h, input_w, output_h, output_w;

    batch = input->dims[0];
    if (ir_graph->graph_layout == TENGINE_LAYOUT_NCHW)
    {
        channel = input->dims[1];
        input_h = input->dims[2];
        input_w = input->dims[3];
    }
    else
    {
        channel = input->dims[3];
        input_h = input->dims[1];
        input_w = input->dims[2];
    }

    if (pool_param->kernel_h == input_h && pool_param->kernel_w == input_w)
        pool_param->global = 1;

    if (pool_param->global)
    {
        pool_param->pad_h0 = 0;
        pool_param->pad_h1 = 0;
        pool_param->pad_w0 = 0;
        pool_param->pad_w1 = 0;

        pool_param->kernel_h = input_h;
        pool_param->kernel_w = input_w;
        pool_param->pad_h0 = pool_param->pad_h1 = pool_param->pad_w0 = pool_param->pad_w1 = 0;
        pool_param->stride_h = pool_param->stride_w = 1;

        output_h = 1;
        output_w = 1;
    }
    else
    {
        int caffe = pool_param->caffe_flavor & ~(COUNT_INCLUDE_PAD_MSK);
        output_h = calc_output_size(input_h, pool_param->kernel_h, pool_param->stride_h, pool_param->pad_h0_org,
                                    pool_param->caffe_flavor);
        output_w = calc_output_size(input_w, pool_param->kernel_w, pool_param->stride_w, pool_param->pad_w0_org,
                                    pool_param->caffe_flavor);
        if (2 != caffe)
        {
            calc_real_pads(output_h, input_h, pool_param->kernel_h, pool_param->stride_h, pool_param->pad_h0_org,
                           &pool_param->pad_h0, &pool_param->pad_h1);
            calc_real_pads(output_w, input_w, pool_param->kernel_w, pool_param->stride_w, pool_param->pad_w0_org,
                           &pool_param->pad_w0, &pool_param->pad_w1);
        }
        else
        {
            int pad_w0 = pool_param->pad_w0_org;
            int pad_h0 = pool_param->pad_h0_org;
            pool_param->pad_w0 = pad_w0 / 2;
            pool_param->pad_h0 = pad_h0 / 2;
            pool_param->pad_w1 = pad_w0 - pad_w0 / 2;
            pool_param->pad_h1 = pad_h0 - pad_h0 / 2;
        }
    }

    int dims[4];

    dims[0] = batch;

    if (ir_graph->graph_layout == TENGINE_LAYOUT_NCHW)
    {
        dims[1] = channel;
        dims[2] = output_h;
        dims[3] = output_w;
    }
    else
    {
        dims[1] = output_h;
        dims[2] = output_w;
        dims[3] = channel;
    }

    set_ir_tensor_shape(output, dims, 4);

    return 0;
}

static int init_op(struct ir_op* op)
{
    struct pool_param* pool_param = ( struct pool_param* )sys_malloc(sizeof(struct pool_param));

    if (pool_param == NULL)
    {
        set_tengine_errno(ENOMEM);
        return -1;
    }

    pool_param->pool_method = POOL_MAX;
    pool_param->global = 0;
    pool_param->kernel_h = 2;
    pool_param->kernel_w = 2;
    pool_param->stride_h = 2;
    pool_param->stride_w = 2;
    pool_param->pad_h0 = 0;
    pool_param->pad_h1 = 0;
    pool_param->pad_w0 = 0;
    pool_param->pad_h1 = 0;
    pool_param->pad_h0_org = 0;
    pool_param->pad_h1_org = 0;
    pool_param->pad_w0_org = 0;
    pool_param->pad_h1_org = 0;
    pool_param->caffe_flavor = 0;
    pool_param->funct = NULL;

    op->param_mem = pool_param;
    op->param_size = sizeof(struct pool_param);
    op->same_shape = 0;
    op->infer_shape = infer_shape;

    return 0;
}

static void release_op(struct ir_op* op)
{
    sys_free(op->param_mem);
}

static int register_pool_op(void* arg)
{
    struct op_method m;

    m.op_version = 1;
    m.init_op = init_op;
    m.release_op = release_op;
    m.access_param_entry = access_param_entry;

    return register_op(OP_POOL, OP_POOL_NAME, &m);
}

static int unregister_pool_op(void* arg)
{
    sys_free(GET_PARAM_PARSE_MAP(pool_param));
    return unregister_op(OP_POOL, 1);
}

AUTO_REGISTER_OP(register_pool_op);
AUTO_UNREGISTER_OP(unregister_pool_op);
