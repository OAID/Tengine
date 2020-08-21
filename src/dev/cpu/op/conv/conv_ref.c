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
#include "sys_port.h"
#include "module.h"
#include "tengine_errno.h"
#include "tengine_log.h"
#include "tengine_ir.h"
#include "../../cpu_node_ops.h"
#include "tengine_op.h"
#include "convolution_param.h"
#include "conv_ref_kernel.h"

static int prerun(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct ir_node* ir_node = exec_node->ir_node;
    struct ir_graph* ir_graph = ir_node->graph;
    struct ir_tensor* input_tensor;
    struct ir_tensor* filter_tensor;
    struct ir_tensor* output_tensor;

    struct conv_priv_info* conv_priv_info = ( struct conv_priv_info* )exec_node->ops_priv;

    if (conv_kernel_set_shared_mem)
    {
        if (conv_kernel_set_shared_mem(conv_priv_info, exec_graph->shared_mem, exec_node->shared_mem_size) < 0)
        {
            TLOG_ERR("hcl conv: set shared memory failed\n");
            set_tengine_errno(EFAULT);
            return -1;
        }
    }

    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    filter_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    struct conv_param* conv_param = ( struct conv_param* )ir_node->op.param_mem;

    /* prerun now */
    if (conv_kernel_prerun(input_tensor, filter_tensor, output_tensor, conv_priv_info, conv_param) < 0)
    {
        TLOG_ERR("hcl conv prerun failed\n");
        set_tengine_errno(EFAULT);
        return -1;
    }

    return 0;
}

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct ir_node* ir_node = exec_node->ir_node;
    struct ir_graph* ir_graph = ir_node->graph;
    struct ir_tensor* input_tensor;
    struct ir_tensor* weight_tensor;
    struct ir_tensor* bias_tensor = NULL;
    struct ir_tensor* output_tensor = NULL;
    int num_thread   = exec_graph->num_thread;
    int cpu_affinity = exec_graph->cpu_affinity;

    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    weight_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    if (ir_node->input_num > 2)
    {
        bias_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[2]);
    }
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    struct conv_param* conv_param = ( struct conv_param* )ir_node->op.param_mem;
    struct conv_priv_info* conv_priv_info = ( struct conv_priv_info* )exec_node->ops_priv;

    if (conv_kernel_run(input_tensor, weight_tensor, bias_tensor, output_tensor, conv_priv_info, conv_param,
                        num_thread, cpu_affinity) < 0)
    {
        TLOG_ERR("hcl conv run failed\n");
        set_tengine_errno(EFAULT);
        return -1;
    }

    return 0;
}

static int reshape(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct ir_node* ir_node = exec_node->ir_node;
    struct ir_graph* ir_graph = ir_node->graph;
    struct ir_tensor* input_tensor;
    struct ir_tensor* output_tensor;

    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    struct conv_param* conv_param = ( struct conv_param* )ir_node->op.param_mem;

    /* dynamic get the shape of output tensor */
    int n = input_tensor->dims[0];
    int h, w;
    int ret = 0;

    if (ir_graph->graph_layout == TENGINE_LAYOUT_NCHW)
    {
        h = input_tensor->dims[2];
        w = input_tensor->dims[3];
    }
    else if (ir_graph->graph_layout == TENGINE_LAYOUT_NHWC)
    {
        h = input_tensor->dims[1];
        w = input_tensor->dims[2];
    }
    else
    {
        TLOG_ERR("convolution infer shape: unknown graph layout: %d\n", ir_graph->graph_layout);
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
    {
        out_h =
            (h - conv_param->dilation_h * (conv_param->kernel_h - 1) - 1 + conv_param->pad_h0 + conv_param->pad_h1) /
                conv_param->stride_h +
            1;
    }

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
    {
        out_w =
            (w - conv_param->dilation_w * (conv_param->kernel_w - 1) - 1 + conv_param->pad_w0 + conv_param->pad_w1) /
                conv_param->stride_w +
            1;
    }

    int dims[4];
    dims[0] = n;
    if (ir_graph->graph_layout == TENGINE_LAYOUT_NCHW)
    {
        if (output_tensor->dims[1] != out_c || output_tensor->dims[2] != out_h || output_tensor->dims[3] != out_w)
        {
            dims[1] = out_c;
            dims[2] = out_h;
            dims[3] = out_w;
            ret = set_ir_tensor_shape(output_tensor, dims, 4);
        }
    }
    else
    {
        if (output_tensor->dims[1] != out_h || output_tensor->dims[2] != out_w || output_tensor->dims[3] != out_c)
        {
            dims[1] = out_h;
            dims[2] = out_w;
            dims[3] = out_c;
            ret = set_ir_tensor_shape(output_tensor, dims, 4);
        }
    }

    return ret;
}

static int postrun(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct conv_priv_info* conv_priv_info = ( struct conv_priv_info* )exec_node->ops_priv;

    if (conv_kernel_postrun(conv_priv_info) < 0)
    {
        TLOG_ERR("hcl conv prerun failed\n");
        set_tengine_errno(EFAULT);
        return -1;
    }

    return 0;
}

static int init_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct ir_node* ir_node = exec_node->ir_node;
    struct ir_graph* ir_graph = ir_node->graph;
    struct ir_tensor* input_tensor;
    struct ir_tensor* output_tensor;

    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    struct conv_param* conv_param = ( struct conv_param* )ir_node->op.param_mem;
    struct conv_priv_info* conv_priv_info = ( struct conv_priv_info* )sys_malloc(sizeof(struct conv_priv_info));
    if (conv_priv_info == NULL)
    {
        set_tengine_errno(ENOMEM);
        return -1;
    }

    memset(conv_priv_info, 0, sizeof(struct conv_priv_info));

    /* get shared memory size */
    exec_node->ops_priv = conv_priv_info;
    exec_node->shared_mem_size = conv_kernel_get_shared_mem_size(input_tensor, output_tensor, conv_param);

    return 0;
}

static int release_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct conv_priv_info* conv_priv_info = ( struct conv_priv_info* )exec_node->ops_priv;
    sys_free(conv_priv_info);
    exec_node->ops_priv = NULL;

    return 0;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct ir_node* exec_node)
{
    return OPS_SCORE_CANDO;
}

static struct node_ops hcl_node_ops = {.prerun = prerun,
                                       .run = run,
                                       .reshape = reshape,
                                       .postrun = postrun,
                                       .init_node = init_node,
                                       .release_node = release_node,
                                       .score = score};

static int reg_conv_hcl_ops(void* arg)
{
    return register_builtin_node_ops(OP_CONV, &hcl_node_ops);
}

static int unreg_conv_hcl_ops(void* arg)
{
    unregister_builtin_node_ops(OP_CONV, &hcl_node_ops);
    return 0;
}

AUTO_REGISTER_OPS(reg_conv_hcl_ops);
AUTO_UNREGISTER_OPS(unreg_conv_hcl_ops);
