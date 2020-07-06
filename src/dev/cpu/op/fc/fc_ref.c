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
#include "tengine_errno.h"
#include "tengine_log.h"
#include "tengine_ir.h"
#include "../../cpu_node_ops.h"
#include "tengine_op.h"
#include "fc_param.h"
#include <math.h>

struct fc_data
{
    int need_trans;
    int batch;    // N
    int out_number;    // OUT
    int hidden;    // hidden
    int zero[3];    // input, kernel, output
    float scale[3];    // input, kernel, output
};

static int ref_fc_fp32(const float* input, float* output, const float* weight, const float* bias, struct fc_data* param)
{
    int batch = param->batch;
    int hidden = param->hidden;
    int out_number = param->out_number;

    int n, i, j;
    for (n = 0; n < batch; n++)
    {
        for (i = 0; i < out_number; i++)
        {
            float tmp = bias ? bias[i] : 0.0;
            for (j = 0; j < hidden; j++)
            {
                if (param->need_trans == 0)
                    tmp += input[n * hidden + j] * weight[i * hidden + j];
                else
                    tmp += input[n * hidden + j] * weight[i + j * out_number];
            }
            output[n * out_number + i] = tmp;
        }
    }
    return 0;
}

static int init_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct fc_data* op_param = ( struct fc_data* )sys_malloc(sizeof(struct fc_data));
    memset(op_param, 0, sizeof(struct fc_data));
    exec_node->ops_priv = op_param;
    return 0;
}

static int release_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    sys_free(exec_node->ops_priv);
    return 0;
}

static int prerun(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct ir_node* ir_node = exec_node->ir_node;
    struct ir_graph* ir_graph = ir_node->graph;
    struct ir_tensor* input_tensor;
    struct ir_tensor* weight_tensor;
    struct ir_tensor* output_tensor;

    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    weight_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    struct fc_param* param = ( struct fc_param* )ir_node->op.param_mem;
    struct fc_data* op_param = ( struct fc_data* )exec_node->ops_priv;

    if (ir_graph->graph_layout == TENGINE_LAYOUT_NCHW)
    {
        int hidden = input_tensor->dims[1];
        if (input_tensor->dim_num > 2)
            hidden = hidden * input_tensor->dims[2];
        if (input_tensor->dim_num > 3)
            hidden = hidden * input_tensor->dims[3];
        op_param->hidden = hidden;
    }
    else
    {
        int hidden = 0;
        if (input_tensor->dim_num == 2)
            hidden = input_tensor->dims[1];
        if (input_tensor->dim_num == 3)
            hidden = input_tensor->dims[1] * input_tensor->dims[2];
        if (input_tensor->dim_num == 4)
            hidden = input_tensor->dims[1] * input_tensor->dims[2] * input_tensor->dims[3];
        op_param->hidden = hidden;
    }
    op_param->batch = input_tensor->dims[0];
    op_param->out_number = param->num_output;

    int weight_out = weight_tensor->dims[0];

    if (weight_out == op_param->out_number)
        op_param->need_trans = 0;
    else
        op_param->need_trans = 1;

    return 0;
}

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct ir_node* ir_node = exec_node->ir_node;
    struct ir_graph* ir_graph = ir_node->graph;
    struct ir_tensor* input_tensor;
    struct ir_tensor* weight_tensor;
    struct ir_tensor* bias_tensor;
    struct ir_tensor* output_tensor;

    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    weight_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    struct fc_param* param = ( struct fc_param* )ir_node->op.param_mem;
    struct fc_data* op_param = ( struct fc_data* )exec_node->ops_priv;

    const void* input_data = input_tensor->data;
    void* weight_data = weight_tensor->data;
    void* output_data = output_tensor->data;

    void* bias_data = NULL;
    if (ir_node->input_num > 2)
    {
        bias_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[2]);
        bias_data = bias_tensor->data;
    }
    if (ref_fc_fp32(input_data, output_data, weight_data, bias_data, op_param) < 0)
        return -1;

    return 0;
}

static int reshape(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct ir_node* node = exec_node->ir_node;
    struct ir_graph* graph = node->graph;
    struct ir_tensor* input = get_ir_graph_tensor(graph, node->input_tensors[0]);
    struct ir_tensor* weight = get_ir_graph_tensor(graph, node->input_tensors[1]);
    struct ir_tensor* output = get_ir_graph_tensor(graph, node->output_tensors[0]);

    int dim[4];

    int n = weight->dims[0];
    int k = weight->dims[1];

    int m = input->dims[0];
    int input_k = input->dims[1];

    if (input->dim_num == 2)
    {
        dim[0] = m;
        dim[1] = n;
    }
    else if (input->dim_num == 3)
    {
        input_k *= input->dims[2];
        if (graph->graph_layout == TENGINE_LAYOUT_NHWC)
        {
            dim[0] = m;
            dim[1] = 1;
            dim[2] = n;
        }
        else
        {
            dim[0] = m;
            dim[1] = n;
            dim[2] = 1;
        }
    }
    else if (input->dim_num == 4)
    {
        input_k *= input->dims[2] * input->dims[3];
        if (graph->graph_layout == TENGINE_LAYOUT_NHWC)
        {
            dim[0] = m;
            dim[1] = 1;
            dim[2] = 1;
            dim[3] = n;
        }
        else
        {
            dim[0] = m;
            dim[1] = n;
            dim[2] = 1;
            dim[3] = 1;
        }
    }
    else
        return -1;

    if (k != input_k)
    {
        TLOG_ERR("fc: input tensor and weight tensor shape does not match, hidden_number: %d\n", k);
        set_tengine_errno(EFAULT);
        return -1;
    }

    int ret = set_ir_tensor_shape(output, dim, input->dim_num);

    return ret;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct ir_node* exec_node)
{
    return OPS_SCORE_CANDO;
}

static struct node_ops hcl_node_ops = {.prerun = prerun,
                                       .run = run,
                                       .reshape = reshape,
                                       .postrun = NULL,
                                       .init_node = init_node,
                                       .release_node = release_node,
                                       .score = score};

static int reg_fc_hcl_ops(void* arg)
{
    return register_builtin_node_ops(OP_FC, &hcl_node_ops);
}

static int unreg_fc_hcl_ops(void* arg)
{
    return unregister_builtin_node_ops(OP_FC, &hcl_node_ops);
}

AUTO_REGISTER_OPS(reg_fc_hcl_ops);
AUTO_UNREGISTER_OPS(unreg_fc_hcl_ops);
