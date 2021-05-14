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

#include "pooling_param.h"

#include "api/c_api.h"
#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "module/module.h"
#include "utility/sys_port.h"


static int infer_shape(ir_node_t* node)
{
    ir_graph_t* ir_graph = node->graph;
    ir_tensor_t* input = get_ir_graph_tensor(ir_graph, node->input_tensors[0]);
    ir_tensor_t* output = get_ir_graph_tensor(ir_graph, node->output_tensors[0]);
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


static int init_op(ir_op_t* op)
{
    struct pool_param* pool_param = ( struct pool_param* )sys_malloc(sizeof(struct pool_param));

    if (pool_param == NULL)
    {
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
    pool_param->pad_w1 = 0;
    pool_param->pad_h0_org = 0;
    pool_param->pad_h1_org = 0;
    pool_param->pad_w0_org = 0;
    pool_param->pad_w1_org = 0;
    pool_param->caffe_flavor = 0;
    pool_param->funct = NULL;

    op->param_mem = pool_param;
    op->param_size = sizeof(struct pool_param);
    op->same_shape = 0;
    op->infer_shape = infer_shape;

    return 0;
}


static void release_op(ir_op_t* op)
{
    sys_free(op->param_mem);
}


int register_pooling_op()
{
    ir_method_t m;

    m.version = 1;
    m.init = init_op;
    m.release = release_op;

    return register_op(OP_POOL, OP_POOL_NAME, &m);
}


int unregister_pooling_op()
{
    return unregister_op(OP_POOL, 1);
}
