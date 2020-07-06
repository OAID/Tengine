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
#include "lstm_param.h"
#include "vector.h"
#include "lstm_kernel_ref.h"
#include "string.h"

int ref_lstm_fp32(float* input, float* output, struct lstm_param_ref* param)
{
    float* init_h = ( float* )malloc(param->batch_size * param->hidden_size * sizeof(float));

    float* init_c = ( float* )malloc(param->batch_size * param->cell_size * sizeof(float));

    if (param->init_h_data)
    {
        for (int i = 0; i < param->batch_size; i++)
        {
            memcpy(init_h + i * param->hidden_size, param->init_h_data, param->hidden_size * sizeof(float));
            memcpy(init_c + i * param->cell_size, param->init_c_data, param->cell_size * sizeof(float));
        }
    }
    else
    {
        memset(init_h, 0x0, sizeof(param->batch_size * param->hidden_size * sizeof(float)));
        memset(init_c, 0x0, sizeof(param->batch_size * param->cell_size * sizeof(float)));
    }
    for (int i = 0; i < param->seq_lens; i++)
    {
        const float* seq_input = input + i * param->batch_size * param->input_size;

        if (!do_LSTM_step(seq_input, init_h, init_c, param->kernel, param->bias, param->h2h_kernel, param->h2h_bias,
                          param->w_f_data, param->w_i_data, param->w_o_data, param->projection, param->forget_bias,
                          param->batch_size, param->input_size, param->hidden_size, param->cell_size,
                          param->mxnet_flag))
            return false;

        if (i + param->output_len >= param->seq_lens)
        {
            memcpy(output, init_h, param->hidden_size * param->batch_size * sizeof(float));
            output += param->batch_size * param->hidden_size;
        }
    }
    free(init_h);
    free(init_c);
    return 0;
}
static int init_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    // exec_node->inplace_map[0] = 0;
    // exec_node->inplace_map[1] = 0;
    // exec_node->inplace_map_num = 1;
    struct lstm_priv_info* lstm_priv_info = ( struct lstm_priv_info* )sys_malloc(sizeof(struct lstm_priv_info));

    if (lstm_priv_info == NULL)
    {
        set_tengine_errno(ENOMEM);
        return -1;
    }

    memset(lstm_priv_info, 0, sizeof(struct lstm_priv_info));

    /* get shared memory size */
    exec_node->ops_priv = lstm_priv_info;

    return 0;
}

static int release_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    // exec_node->inplace_map_num = 0;
    struct lstm_priv_info* lstm_priv_info = ( struct lstm_priv_info* )exec_node->ops_priv;

    sys_free(lstm_priv_info);

    exec_node->ops_priv = NULL;

    return 0;
}

static int prerun(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct ir_node* ir_node = exec_node->ir_node;
    struct ir_graph* ir_graph = ir_node->graph;
    int in_num = ir_node->input_num;
    struct lstm_priv_info* lstm_priv_info = ( struct lstm_priv_info* )exec_node->ops_priv;
    for (int i = 0; i < in_num; i++)
    {
        struct ir_tensor* tmp_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[i]);
        char* name = tmp_tensor->name;
        if (strstr(name, "kernel") != NULL && strstr(name, "projection"))
        {
            lstm_priv_info->kernel_tensor = tmp_tensor;
        }
        if (strstr(name, "init_c") != NULL)
        {
            lstm_priv_info->init_c_tensor = tmp_tensor;
        }
        if (strstr(name, "init_h") != NULL)
        {
            lstm_priv_info->init_h_tensor = tmp_tensor;
        }
        if (strstr(name, "bias") != NULL)
        {
            lstm_priv_info->bias_tensor = tmp_tensor;
        }
        if (strstr(name, "w_f_diag") != NULL)
        {
            lstm_priv_info->w_f_tensor = tmp_tensor;
        }
        if (strstr(name, "w_o_diag") != NULL)
        {
            lstm_priv_info->w_o_tensor = tmp_tensor;
        }
        if (strstr(name, "w_i_diag") != NULL)
        {
            lstm_priv_info->w_i_tensor = tmp_tensor;
        }
        if (strstr(name, "projection") != NULL)
        {
            lstm_priv_info->proj_tensor = tmp_tensor;
        }
        if (strstr(name, "i2h_weight") != NULL)
        {
            lstm_priv_info->kernel_tensor = tmp_tensor;
        }
        if (strstr(name, "i2h_bias") != NULL)
        {
            lstm_priv_info->bias_tensor = tmp_tensor;
        }
        if (strstr(name, "h2h_weight") != NULL)
        {
            lstm_priv_info->h2h_kernel_tensor = tmp_tensor;
        }
        if (strstr(name, "h2h_bias") != NULL)
        {
            lstm_priv_info->h2h_bias_tensor = tmp_tensor;
        }
        if (strstr(name, "parameters") != NULL)
        {
            lstm_priv_info->fused_kernel_tensor = tmp_tensor;
        }
    }

    return 0;
}

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct ir_node* ir_node = exec_node->ir_node;
    lstm_param_t* _param = ( struct lstm_param* )(ir_node->op.param_mem);
    struct ir_graph* ir_graph = ir_node->graph;
    struct ir_tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct ir_tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    struct lstm_param_ref op_param;

    float forget_bias = _param->forget_bias;
    bool has_peephole = _param->has_peephole;
    bool has_projection = _param->has_projection;

    int hidden_size = _param->hidden_size;
    int cell_size = _param->cell_size;
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

    struct lstm_priv_info* lstm_priv_info = ( struct lstm_priv_info* )exec_node->ops_priv;

    float* init_c = ( float* )malloc(batch_size * cell_size * sizeof(float));

    if (init_c == NULL)
    {
        free(init_h);
        set_tengine_errno(ENOMEM);
        return false;
    }
    struct ir_tensor* init_h_data_tensor = lstm_priv_info->init_h_tensor;
    struct ir_tensor* init_c_data_tensor = lstm_priv_info->init_c_tensor;
    float* init_h_data = ( float* )init_h_data_tensor->data;
    float* init_c_data = ( float* )init_c_data_tensor->data;
    if (init_h_data_tensor->data)
    {
        for (int i = 0; i < batch_size; i++)
        {
            memcpy(init_h + i * hidden_size, ( float* )init_h_data_tensor->data, hidden_size * sizeof(float));
            memcpy(init_c + i * cell_size, ( float* )init_c_data_tensor->data, cell_size * sizeof(float));
        }
    }
    else
    {
        memset(init_h, 0x0, sizeof(batch_size * hidden_size * sizeof(float)));
        memset(init_c, 0x0, sizeof(batch_size * cell_size * sizeof(float)));
    }
    float* kernel = NULL;
    float* bias = NULL;
    float* w_f_data = NULL;
    float* w_i_data = NULL;
    float* w_o_data = NULL;
    float* projection = NULL;
    float* h2h_kernel = NULL;
    float* h2h_bias = NULL;
    float* fused_kernel = NULL;
    if (lstm_priv_info->kernel_tensor)
    {
        struct ir_tensor* kernel_tensor = lstm_priv_info->kernel_tensor;
        kernel = ( float* )kernel_tensor->data;
    }

    if (lstm_priv_info->bias_tensor)
    {
        struct ir_tensor* bias_tensor = lstm_priv_info->bias_tensor;
        bias = ( float* )bias_tensor->data;
    }

    if (lstm_priv_info->h2h_kernel_tensor)
    {
        struct ir_tensor* h2h_kernel_tensor = lstm_priv_info->h2h_kernel_tensor;
        h2h_kernel = ( float* )h2h_kernel_tensor->data;
    }

    if (lstm_priv_info->h2h_bias_tensor)
    {
        struct ir_tensor* h2h_bias_tensor = lstm_priv_info->h2h_bias_tensor;
        h2h_bias = ( float* )h2h_bias_tensor->data;
    }

    if (has_peephole)
    {
        struct ir_tensor* w_f_tensor = lstm_priv_info->w_f_tensor;
        struct ir_tensor* w_i_tensor = lstm_priv_info->w_i_tensor;
        struct ir_tensor* w_o_tensor = lstm_priv_info->w_o_tensor;
        w_f_data = ( float* )w_f_tensor->data;
        w_i_data = ( float* )w_i_tensor->data;
        w_o_data = ( float* )w_o_tensor->data;
    }
    // int bsize=2*cell_size*4;

    if (lstm_priv_info->fused_kernel_tensor)
    {
        struct ir_tensor* fused_kernel_tensor = lstm_priv_info->fused_kernel_tensor;
        fused_kernel = ( float* )fused_kernel_tensor->data;
        int kernel_size = fused_kernel_tensor->elem_size / sizeof(float);
        kernel = fused_kernel;
        h2h_kernel = kernel + input_size * hidden_size * 4;
        bias = kernel + kernel_size - hidden_size * 4 * 2;
        h2h_bias = bias + hidden_size * 4;
    }
    if (has_projection)
    {
        struct ir_tensor* proj_tensor = lstm_priv_info->proj_tensor;
        projection = ( float* )proj_tensor->data;
    }

    op_param.init_h_data = init_h_data;
    op_param.init_c_data = init_c_data;
    op_param.bias = bias;
    op_param.forget_bias = forget_bias;
    op_param.kernel = kernel;
    op_param.w_f_data = w_f_data;
    op_param.w_i_data = w_i_data;
    op_param.w_o_data = w_o_data;
    op_param.projection = projection;
    op_param.h2h_kernel = h2h_kernel;
    op_param.h2h_bias = h2h_bias;
    op_param.fused_kernel = fused_kernel;
    op_param.seq_lens = seq_lens;
    op_param.batch_size = batch_size;
    op_param.input_size = input_size;
    op_param.output_len = output_len;
    op_param.hidden_size = hidden_size;
    op_param.cell_size = cell_size;
    op_param.mxnet_flag = mxnet_flag;
    if (ref_lstm_fp32(input, output, &op_param) < 0)
    {
        return -1;
    }

    return 0;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct ir_node* exec_node)
{
    return OPS_SCORE_BEST;
}

static struct node_ops lstm_node_ops = {.prerun = prerun,
                                        .run = run,
                                        .reshape = NULL,
                                        .postrun = NULL,
                                        .init_node = init_node,
                                        .release_node = release_node,
                                        .score = score};

static int reg_lstm_ops(void* arg)
{
    return register_builtin_node_ops(OP_LSTM, &lstm_node_ops);
}

static int unreg_lstm_ops(void* arg)
{
    return unregister_builtin_node_ops(OP_LSTM, &lstm_node_ops);
}

AUTO_REGISTER_OPS(reg_lstm_ops);
AUTO_UNREGISTER_OPS(unreg_lstm_ops);
