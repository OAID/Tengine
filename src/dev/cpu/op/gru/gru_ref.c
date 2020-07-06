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

#include <math.h>
#include <stdbool.h>
#include "sys_port.h"
#include "module.h"
#include "tengine_errno.h"
#include "tengine_log.h"
#include "tengine_ir.h"
#include "../../cpu_node_ops.h"
#include "tengine_op.h"
#include "gru_param.h"
#include "vector.h"
#include "gru_kernel_ref.h"
#include "string.h"

int ref_gru_fp32(float* input, float* output, struct gru_param_ref* param)
{
    float* init_h = ( float* )malloc(param->batch_size * param->hidden_size * sizeof(float));

    if (param->init_h_data)
    {
        for (int i = 0; i < param->batch_size; i++)
        {
            memcpy(init_h + i * param->hidden_size, param->init_h_data, param->hidden_size * sizeof(float));
        }
    }
    else
    {
        memset(init_h, 0x0, sizeof(param->batch_size * param->hidden_size * sizeof(float)));
    }
    for (int i = 0; i < param->seq_lens; i++)
    {
        const float* seq_input = input + i * param->batch_size * param->input_size;
        if (!do_GRU_step(seq_input, init_h, param->kernel, param->bias, param->candidate_kernel, param->candidate_bias,
                         param->batch_size, param->input_size, param->hidden_size, param->mxnet_flag))
        {
            return -1;
        }

        if (i + param->output_len >= param->seq_lens)
        {
            memcpy(output, init_h, param->batch_size * param->hidden_size * sizeof(float));
            output += param->batch_size * param->hidden_size;
        }
    }
    free(init_h);
    return 0;
    return 0;
}
static int init_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    // exec_node->inplace_map[0] = 0;
    // exec_node->inplace_map[1] = 0;
    // exec_node->inplace_map_num = 1;
    struct gru_priv_info* gru_priv_info = ( struct gru_priv_info* )sys_malloc(sizeof(struct gru_priv_info));

    if (gru_priv_info == NULL)
    {
        set_tengine_errno(ENOMEM);
        return -1;
    }

    memset(gru_priv_info, 0, sizeof(struct gru_priv_info));

    /* get shared memory size */
    exec_node->ops_priv = gru_priv_info;

    return 0;
}

static int release_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    // exec_node->inplace_map_num = 0;
    struct gru_priv_info* gru_priv_info = ( struct gru_priv_info* )exec_node->ops_priv;

    sys_free(gru_priv_info);

    exec_node->ops_priv = NULL;

    return 0;
}

static int prerun(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct ir_node* ir_node = exec_node->ir_node;
    struct ir_graph* ir_graph = ir_node->graph;
    int in_num = ir_node->input_num;
    struct gru_priv_info* gru_priv_info = ( struct gru_priv_info* )exec_node->ops_priv;
    for (int i = 0; i < in_num; i++)
    {
        struct ir_tensor* tmp_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[i]);
        char* name = tmp_tensor->name;
        if (strstr(name, "gates/kernel") != NULL)
        {
            gru_priv_info->kernel_tensor = tmp_tensor;
        }
        if (strstr(name, "init_h") != NULL)
        {
            gru_priv_info->init_h_tensor = tmp_tensor;
        }
        if (strstr(name, "gates/bias") != NULL)
        {
            gru_priv_info->bias_tensor = tmp_tensor;
        }
        if (strstr(name, "candidate/kernel") != NULL)
        {
            gru_priv_info->candidate_kernel_tensor = tmp_tensor;
        }
        if (strstr(name, "candidate/bias") != NULL)
        {
            gru_priv_info->candidate_bias_tensor = tmp_tensor;
        }
        if (strstr(name, "i2h_weight") != NULL)
        {
            gru_priv_info->kernel_tensor = tmp_tensor;
        }
        if (strstr(name, "i2h_bias") != NULL)
        {
            gru_priv_info->bias_tensor = tmp_tensor;
        }
        if (strstr(name, "h2h_weight") != NULL)
        {
            gru_priv_info->candidate_kernel_tensor = tmp_tensor;
        }
        if (strstr(name, "h2h_bias") != NULL)
        {
            gru_priv_info->candidate_bias_tensor = tmp_tensor;
        }
        if (strstr(name, "parameters") != NULL)
        {
            gru_priv_info->fused_kernel_tensor = tmp_tensor;
        }
    }

    return 0;
}

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct ir_node* ir_node = exec_node->ir_node;
    gru_param_t* _param = ( struct gru_param* )(ir_node->op.param_mem);
    struct ir_graph* ir_graph = ir_node->graph;
    struct ir_tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct ir_tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    struct gru_param_ref op_param;

    int hidden_size = _param->hidden_size;
    int input_size = 0;

    int seq_lens = input_tensor->dims[1];
    int batch_size = input_tensor->dims[0];
    int output_len = _param->output_len;
    int mxnet_flag = _param->mxnet_flag;

    if (mxnet_flag == 1)
    {
        seq_lens = input_tensor->dims[0];
        batch_size = input_tensor->dims[1];
        input_size = input_tensor->dims[2];
    }
    else
    {
        input_size = _param->input_size;
    }
    float* output = ( float* )output_tensor->data;
    float* input = ( float* )input_tensor->data;
    float* init_h = ( float* )malloc(batch_size * hidden_size * sizeof(float));

    struct gru_priv_info* gru_priv_info = ( struct gru_priv_info* )exec_node->ops_priv;

    struct ir_tensor* init_h_data_tensor = gru_priv_info->init_h_tensor;
    float* init_h_data = ( float* )init_h_data_tensor->data;
    if (init_h_data_tensor->data)
    {
        for (int i = 0; i < batch_size; i++)
        {
            memcpy(init_h + i * hidden_size, ( float* )init_h_data_tensor->data, hidden_size * sizeof(float));
        }
    }
    else
    {
        memset(init_h, 0x0, sizeof(batch_size * hidden_size * sizeof(float)));
    }
    float* kernel = NULL;
    float* bias = NULL;
    float* fused_kernel = NULL;
    float* candidate_kernel = NULL;
    float* candidate_bias = NULL;

    if (gru_priv_info->kernel_tensor)
    {
        struct ir_tensor* kernel_tensor = gru_priv_info->kernel_tensor;
        kernel = ( float* )kernel_tensor->data;
    }
    if (gru_priv_info->bias_tensor)
    {
        struct ir_tensor* bias_tensor = gru_priv_info->bias_tensor;
        bias = ( float* )bias_tensor->data;
    }

    if (gru_priv_info->candidate_kernel_tensor)
    {
        struct ir_tensor* candidate_kernel_tensor = gru_priv_info->candidate_kernel_tensor;
        candidate_kernel = ( float* )candidate_kernel_tensor->data;
    }

    if (gru_priv_info->candidate_bias_tensor)
    {
        struct ir_tensor* candidate_bias_tensor = gru_priv_info->candidate_bias_tensor;
        candidate_bias = ( float* )candidate_bias_tensor->data;
    }

    // int bsize=2*cell_size*4;

    if (gru_priv_info->fused_kernel_tensor)
    {
        struct ir_tensor* fused_kernel_tensor = gru_priv_info->fused_kernel_tensor;
        fused_kernel = ( float* )fused_kernel_tensor->data;
        // int kernel_size = fused_kernel_tensor->elem_size / sizeof(float);
        kernel = fused_kernel;
        candidate_kernel = kernel + input_size * hidden_size * 3;
        bias = candidate_kernel + hidden_size * hidden_size * 3;
        candidate_bias = bias + hidden_size * 3;
    }

    op_param.init_h_data = init_h_data;
    op_param.bias = bias;
    op_param.kernel = kernel;
    op_param.candidate_kernel = candidate_kernel;
    op_param.candidate_bias = candidate_bias;
    op_param.fused_kernel = fused_kernel;
    op_param.seq_lens = seq_lens;
    op_param.batch_size = batch_size;
    op_param.input_size = input_size;
    op_param.output_len = output_len;
    op_param.hidden_size = hidden_size;
    op_param.mxnet_flag = mxnet_flag;
    if (ref_gru_fp32(input, output, &op_param) < 0)
    {
        return -1;
    }

    return 0;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct ir_node* exec_node)
{
    return OPS_SCORE_BEST;
}

static struct node_ops gru_node_ops = {.prerun = prerun,
                                       .run = run,
                                       .reshape = NULL,
                                       .postrun = NULL,
                                       .init_node = init_node,
                                       .release_node = release_node,
                                       .score = score};

static int reg_gru_ops(void* arg)
{
    return register_builtin_node_ops(OP_GRU, &gru_node_ops);
}

static int unreg_gru_ops(void* arg)
{
    return unregister_builtin_node_ops(OP_GRU, &gru_node_ops);
}

AUTO_REGISTER_OPS(reg_gru_ops);
AUTO_UNREGISTER_OPS(unreg_gru_ops);
