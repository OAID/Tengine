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
 * Author: haoluo@openailab.com
 * update: qtang@openailab.com
 */

#include "conv_dw_kernel_arm.h"

#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "module/module.h"
#include "operator/op.h"
#include "utility/sys_port.h"
#include "utility/log.h"
#include "device/cpu/cpu_node.h"
#include "device/cpu/cpu_graph.h"
#include "device/cpu/cpu_module.h"

#include <string.h>

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#include "cortex_a/conv_dw_kernel_fp16_arm82.h"
#endif

static int prerun(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* ir_node = exec_node->ir_node;
    struct graph* ir_graph = ir_node->graph;
    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct tensor* filter_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    struct conv_param* conv_param = ( struct conv_param* )ir_node->op.param_mem;
    struct conv_priv_info* conv_priv_info = ( struct conv_priv_info* )exec_node->ops_priv;

    /* get cpu affinity */
    conv_priv_info->cpu_type = exec_graph->cpu_affinity;

    /* fp32 prerun */
    if (exec_graph->mode == TENGINE_MODE_FP32 || exec_graph->mode == TENGINE_MODE_UINT8)
    {
        /* do prerun */
        if (conv_dw_prerun(input_tensor, filter_tensor, output_tensor, conv_priv_info, conv_param) < 0)
        {
            TLOG_ERR("hcl conv dw prerun failed\n");

            return -1;
        }
    }
        /* int8 prerun */
    else if (exec_graph->mode == TENGINE_MODE_INT8)
    {
        /* do prerun */
        if (conv_dw_int8_prerun(input_tensor, filter_tensor, output_tensor, conv_priv_info, conv_param) < 0)
        {
            TLOG_ERR("hcl conv dw int8 prerun failed\n");

            return -1;
        }
    }

    return 0;
}

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* ir_node = exec_node->ir_node;
    struct graph* ir_graph = ir_node->graph;
    struct tensor* input_tensor;
    struct tensor* weight_tensor;
    struct tensor* bias_tensor = NULL;
    struct tensor* output_tensor = NULL;
    int num_thread = exec_graph->num_thread;
    int cpu_affinity = exec_graph->cpu_affinity;

    /* set the input data and shape again, in case of reshape or dynamic shape */
    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    weight_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    if (ir_node->input_num > 2)
        bias_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[2]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    struct conv_param* conv_param = ( struct conv_param* )ir_node->op.param_mem;
    struct conv_priv_info* conv_priv_info = ( struct conv_priv_info* )exec_node->ops_priv;

    /* fp32 run */
    if (exec_graph->mode == TENGINE_MODE_FP32)
    {
        if (conv_dw_run(input_tensor, weight_tensor, bias_tensor, output_tensor, conv_priv_info, conv_param, num_thread, cpu_affinity) < 0)
        {
            TLOG_ERR("hcl conv run failed\n");

            return -1;
        }
    }
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        else if (exec_graph->mode == TENGINE_MODE_FP16)
    {
        if (conv_dw_fp16_run(input_tensor, weight_tensor, bias_tensor, output_tensor, conv_param, num_thread, cpu_affinity) < 0)
        {
            TLOG_ERR("hcl conv fp16 run failed\n");
            set_tengine_errno(EFAULT);
            return -1;
        }
    }
#endif
        /* int8 run */
    else if (exec_graph->mode == TENGINE_MODE_INT8)
    {
        if (conv_dw_int8_run(input_tensor, weight_tensor, bias_tensor, output_tensor, conv_priv_info, conv_param, num_thread, cpu_affinity) < 0)
        {
            TLOG_ERR("hcl conv dw int8 run failed\n");

            return -1;
        }
    }
    else
    {
        TLOG_ERR("Tengine work node not support %d\n", exec_graph->mode);
        return -1;
    }

    return 0;
}

static int postrun(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct conv_priv_info* conv_priv_info = ( struct conv_priv_info* )exec_node->ops_priv;

    /* fp32 postrun */
    if (exec_graph->mode == TENGINE_MODE_FP32 || exec_graph->mode == TENGINE_MODE_UINT8)
    {
        if (conv_dw_postrun(conv_priv_info) < 0)
        {
            TLOG_ERR("hcl conv postrun failed\n");

            return -1;
        }
    }
        /* int8 postrun */
    else if (exec_graph->mode == TENGINE_MODE_INT8)
    {
        if (conv_dw_int8_postrun(conv_priv_info) < 0)
        {
            TLOG_ERR("hcl conv dw int8 postrun failed\n");

            return -1;
        }
    }

    return 0;
}


static int init_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* ir_node = exec_node->ir_node;
    struct conv_param* conv_param = ( struct conv_param* )ir_node->op.param_mem;

    /* init the private info data of convolution op */
    struct conv_priv_info* conv_priv_info = ( struct conv_priv_info* )sys_malloc(sizeof(struct conv_priv_info));
    if (conv_priv_info == NULL)
    {

        return -1;
    }
    memset(conv_priv_info, 0, sizeof(struct conv_priv_info));
    exec_node->ops_priv = conv_priv_info;
    return 0;
}

static int release_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct conv_priv_info* conv_priv_info = ( struct conv_priv_info* )exec_node->ops_priv;
    sys_free(conv_priv_info);
    exec_node->ops_priv = NULL;
    return 0;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct node* exec_node)
{
    struct conv_param* param = ( struct conv_param* )exec_node->op.param_mem;
    struct node* ir_node = exec_node;
    struct graph* ir_graph = ir_node->graph;

    struct tensor* input_tensor;
    struct tensor* output_tensor;

    int group = param->group;
    int kernel_h = param->kernel_h;
    int kernel_w = param->kernel_w;
    int stride_h = param->stride_h;
    int stride_w = param->stride_w;
    int dilation_h = param->dilation_h;
    int dilation_w = param->dilation_w;
    int pad_h0 = param->pad_h0;
    int pad_w0 = param->pad_w0;
    int pad_h1 = param->pad_h1;
    int pad_w1 = param->pad_w1;

    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    int in_c = input_tensor->dims[1] / group;
    int out_c = output_tensor->dims[1] / group;

    /* todo support uint8 */
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    if (input_tensor->data_type != TENGINE_DT_FP32 && input_tensor->data_type != TENGINE_DT_FP16)
        return 0;
#else
    if (input_tensor->data_type != TENGINE_DT_FP32 && input_tensor->data_type != TENGINE_DT_INT8)
        return 0;
#endif
    if (kernel_h == 7 && kernel_w == 7 && stride_h == 1 && stride_w == 1)    // this is a bug, todo fix it.
        return 0;

    if (kernel_h == 2 && kernel_w == 2)    // this is a bug, todo fix it.
        return 0;

    if (dilation_h != 1 || dilation_w != 1)
        return 0;

    if (param->group > 1 && in_c == 1 && out_c == 1 && pad_h0 == pad_h1 && pad_w0 == pad_w1) // caution this, todo fix.
        return OPS_SCORE_BEST;
    else
        return 0;
}

static struct node_ops hcl_node_ops = {.prerun = prerun,
        .run = run,
        .reshape = NULL,
        .postrun = postrun,
        .init_node = init_node,
        .release_node = release_node,
        .score = score
};

int register_conv_dw_hcl_arm_op()
{
    return register_builtin_node_ops(OP_CONV, &hcl_node_ops);
}

int unregister_conv_dw_hcl_arm_op()
{
    unregister_builtin_node_ops(OP_CONV, &hcl_node_ops);
    return 0;
}
