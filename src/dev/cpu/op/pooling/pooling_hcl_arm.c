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

#include "sys_port.h"
#include "module.h"
#include "tengine_ir.h"
#include "../../cpu_node_ops.h"
#include "tengine_op.h"
#include "pooling_param.h"
#include "pooling_hcl_arm.h"

#define POOL_K2S2 1
#define POOL_K3S2 2
#define POOL_K3S1 3

static int prerun(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct ir_node* ir_node = exec_node->ir_node;
    struct ir_graph* ir_graph = ir_node->graph;
    struct ir_tensor* input_tensor;
    struct ir_tensor* output_tensor;

    struct pool_param* pool_param = ( struct pool_param* )ir_node->op.param_mem;

    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    pooling_kernel_perf_prerun(input_tensor, output_tensor, pool_param);

    return 0;
}

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct ir_node* ir_node = exec_node->ir_node;
    struct ir_graph* ir_graph = ir_node->graph;
    struct ir_tensor* input_tensor;
    struct ir_tensor* output_tensor;

    struct pool_param* pool_param = ( struct pool_param* )ir_node->op.param_mem;

    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    pooling_kernel_perf_run(input_tensor, output_tensor, pool_param, exec_graph->num_thread);

    return 0;
}

static int postrun(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    return 0;
}

static int init_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* dev)
{
    return 0;
}

static int release_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* dev)
{
    return 0;
}

static int reshape(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct ir_node* ir_node = exec_node->ir_node;
    struct ir_graph* ir_graph = ir_node->graph;
    struct ir_tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct ir_tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    struct pool_param* pool_param = ( struct pool_param* )ir_node->op.param_mem;

    int batch, channel, input_h, input_w, output_h, output_w;
    int ret = 0;

    batch = input_tensor->dims[0];
    if (ir_graph->graph_layout == TENGINE_LAYOUT_NCHW)
    {
        channel = input_tensor->dims[1];
        input_h = input_tensor->dims[2];
        input_w = input_tensor->dims[3];
    }
    else
    {
        channel = input_tensor->dims[3];
        input_h = input_tensor->dims[1];
        input_w = input_tensor->dims[2];
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
        if (output_tensor->dims[1] != channel || output_tensor->dims[2] != output_h ||
            output_tensor->dims[3] != output_w)
        {
            dims[1] = channel;
            dims[2] = output_h;
            dims[3] = output_w;
            ret = set_ir_tensor_shape(output_tensor, dims, 4);
        }
    }
    else
    {
        if (output_tensor->dims[1] != output_h || output_tensor->dims[2] != output_w ||
            output_tensor->dims[3] != channel)
        {
            dims[1] = output_h;
            dims[2] = output_w;
            dims[3] = channel;
            ret = set_ir_tensor_shape(output_tensor, dims, 4);
        }
    }

    return ret;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct ir_node* exec_node)
{
    struct pool_param* pool_param = ( struct pool_param* )exec_node->op.param_mem;

    int global = pool_param->global;
    int type = pool_param->pool_method;
    int kernel_h = pool_param->kernel_h;
    int kernel_w = pool_param->kernel_w;
    int stride_h = pool_param->stride_h;
    int stride_w = pool_param->stride_w;
    int pad_h0 = pool_param->pad_h0;
    int pad_h1 = pool_param->pad_h1;
    int pad_w0 = pool_param->pad_w0;
    int pad_w1 = pool_param->pad_w1;
    int pad_tf = pool_param->pad_h0_org;    // maybe there is a bug.

    int pool_size = 0;

    struct ir_node* ir_node = exec_node;
    struct ir_graph* ir_graph = ir_node->graph;
    struct ir_tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);

    /* todo support uint8 */
    if (input_tensor->data_type != TENGINE_DT_FP32)
        return 0;

    /* filter perf global pooling case */
    if (global)
        return OPS_SCORE_BEST;
    /* filter perf general pooling case */
    else
    {
        if (stride_h == 2 && stride_w == 2)
        {
            if (kernel_h == 2 && kernel_w == 2)
                pool_size = POOL_K2S2;
            if (kernel_h == 3 && kernel_w == 3)
                pool_size = POOL_K3S2;
        }

        if (stride_h == 1 && stride_w == 1 && kernel_h == 3 && kernel_w == 3)
            pool_size = POOL_K3S1;

        /* general max pooling, k2s2, k2k2p1, k3s1p1, k3s2, k3s2p1 */
        if (type == POOL_MAX && (pad_h0 == pad_w0) && (pad_h1 == pad_w1) && pad_tf != -1)
        {
            if (pad_h0 == 0 && (pool_size == POOL_K2S2 || pool_size == POOL_K3S2))
                return OPS_SCORE_BEST;
            if (pad_h0 == 1 && (pool_size == POOL_K2S2 || pool_size == POOL_K3S2 || pool_size == POOL_K3S1))
                return OPS_SCORE_BEST;
        }

        /* general avg pooling, k2s2, k2s2p1, k3s2, k3s2p1 */
        if (type == POOL_AVG && (pad_h0 == pad_w0) && (pad_h1 == pad_w1))
        {
            if (pad_h0 == 0 && pad_h1 == 0 && (pool_size == POOL_K2S2 || pool_size == POOL_K3S2))
                return OPS_SCORE_BEST;
            if (pad_h0 == 1 && pad_h1 == 1 && (pool_size == POOL_K2S2 || pool_size == POOL_K3S2 || pool_size == POOL_K3S1))
                return OPS_SCORE_BEST;
            else if(pad_h0 == 0 && pad_h1 == 1 && (pool_size == POOL_K3S2))
                return OPS_SCORE_BEST;
        }
    }

    return 0;
}

static struct node_ops hcl_node_ops = {.prerun = prerun,
                                       .run = run,
                                       .reshape = reshape,
                                       .postrun = postrun,
                                       .init_node = init_node,
                                       .release_node = release_node,
                                       .score = score};

static int reg_pooling_hcl_ops(void* arg)
{
    return register_builtin_node_ops(OP_POOL, &hcl_node_ops);
}

static int unreg_pooling_hcl_ops(void* arg)
{
    return unregister_builtin_node_ops(OP_POOL, &hcl_node_ops);
}

AUTO_REGISTER_OPS(reg_pooling_hcl_ops);
AUTO_UNREGISTER_OPS(unreg_pooling_hcl_ops);
