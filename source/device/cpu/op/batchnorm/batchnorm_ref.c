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
 * Author: bhu@openailab.com
 * Update: hhchen@openailab.com
 */

#include "batchnorm_param.h"

#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "utility/sys_port.h"
#include "utility/log.h"
#include "device/cpu/cpu_node.h"
#include "device/cpu/cpu_graph.h"
#include "device/cpu/cpu_module.h"

#include <stdbool.h>
#include <string.h>
#include <math.h>


struct ref_batchnorm_param
{
    int input_n;
    int input_h;
    int input_w;
    int input_c;
    int layout;
    bool iscaffe;
    float* scale_mean;
    float* scale_var_inv;
    float* gamma;
    float* beta;
    float in_scale;
    int in_zero;
    float out_scale;
    int out_zero;
};


static int ref_batchnorm_uint8(struct tensor* input_tensor, struct tensor* output_tensor, const struct ref_batchnorm_param* param, int num_thread)
{
    float* scale_mean = param->scale_mean;
    float* scale_var_inv = param->scale_var_inv;
    float* gamma = param->gamma;
    float* beta = param->beta;

    int img_size = param->input_c * param->input_h * param->input_w;
    int total_size = img_size * param->input_n;

    // dequant
    uint8_t* input_uint8 = input_tensor->data;
    uint8_t* output_uint8 = output_tensor->data;
    float input_scale = input_tensor->scale;
    float output_scale = output_tensor->scale;
    int32_t input_zero = input_tensor->zero_point;
    int32_t output_zero = output_tensor->zero_point;

    float* data_fp32 = (float*) sys_malloc(total_size * sizeof(float));
    for(int i = 0; i < total_size; i++)
        data_fp32[i] = ((float) input_uint8[i] - (float)input_zero) * input_scale;

    for (int n = 0; n < param->input_n; ++n)
    {
#pragma omp parallel for num_threads(num_thread)
        for (int h = 0; h < param->input_h; ++h)
        {
            for (int w = 0; w < param->input_w; ++w)
            {
                for (int c = 0; c < param->input_c; ++c)
                {
                    float s_mean = scale_mean[c];
                    float s_var = scale_var_inv[c];
                    float s_val1 = s_mean;
                    float s_val2 = s_var;
                    if (!param->iscaffe)
                    {
                        float s_gamma = gamma[c];
                        float s_beta = beta[c];
                        s_val1 = s_beta + s_gamma * s_mean;
                        s_val2 = s_gamma * s_var;
                    }
                    int offset = 0;
                    if (TENGINE_LAYOUT_NCHW == param->layout)
                    {
                        offset = n * img_size + c * param->input_h * param->input_w + h * param->input_w + w;
                    }
                    else
                    {
                        offset = n * img_size + h * param->input_w * param->input_c + w * param->input_c + c;
                    }
                    data_fp32[offset] = data_fp32[offset] * s_val2 + s_val1;
                }
            }
        }
    }

    // quant
    for(int i=0; i<total_size; i++)
    {
        int udata = (int)roundf(data_fp32[i] / output_scale + output_zero);
        if (udata > 255)
            udata = 255;
        else if (udata < 0)
            udata = 0;
        output_uint8[i] = udata;
    }

    return 0;
}


static int ref_batchnorm_fp32(float* input, float* output, const struct ref_batchnorm_param* param, int num_thread)
{
    float* scale_mean = param->scale_mean;
    float* scale_var_inv = param->scale_var_inv;
    float* gamma = param->gamma;
    float* beta = param->beta;

    int img_size = param->input_c * param->input_h * param->input_w;

    for (int n = 0; n < param->input_n; ++n)
    {
#pragma omp parallel for num_threads(num_thread)
        for (int h = 0; h < param->input_h; ++h)
        {
            for (int w = 0; w < param->input_w; ++w)
            {
                for (int c = 0; c < param->input_c; ++c)
                {
                    float s_mean = scale_mean[c];
                    float s_var = scale_var_inv[c];
                    float s_val1 = s_mean;
                    float s_val2 = s_var;
                    if (!param->iscaffe)
                    {
                        float s_gamma = gamma[c];
                        float s_beta = beta[c];
                        s_val1 = s_beta + s_gamma * s_mean;
                        s_val2 = s_gamma * s_var;
                    }
                    int offset = 0;
                    if (TENGINE_LAYOUT_NCHW == param->layout)
                    {
                        offset = n * img_size + c * param->input_h * param->input_w + h * param->input_w + w;
                    }
                    else
                    {
                        offset = n * img_size + h * param->input_w * param->input_c + w * param->input_c + c;
                    }
                    output[offset] = input[offset] * s_val2 + s_val1;
                }
            }
        }
    }
    return 0;
}

static int init_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct ref_batchnorm_param* batchnorm_op_param =
        ( struct ref_batchnorm_param* )sys_malloc(sizeof(struct ref_batchnorm_param));
    memset(batchnorm_op_param, 0, sizeof(struct ref_batchnorm_param));
    exec_node->ops_priv = batchnorm_op_param;
    return 0;
}

static int release_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    sys_free(exec_node->ops_priv);
    return 0;
}

static int prerun(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* ir_node = exec_node->ir_node;
    struct graph* ir_graph = ir_node->graph;
    struct tensor* output_tensor;
    const struct tensor* input_tensor;
    int channel_num;
    // struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    const struct tensor* mean_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[3]);
    const struct tensor* var_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[4]);
    ;

    struct ref_batchnorm_param* op_param = ( struct ref_batchnorm_param* )exec_node->ops_priv;
    struct batchnorm_param* batchnorm_param = ( struct batchnorm_param* )ir_node->op.param_mem;

    if (ir_graph->graph_layout == TENGINE_LAYOUT_NCHW)
    {
        channel_num = input_tensor->dims[1];
    }
    else if (ir_graph->graph_layout == TENGINE_LAYOUT_NHWC)
    {
        channel_num = input_tensor->dims[3];
    }

    float* scale_mean = ( float* )sys_malloc(channel_num * sizeof(float));
    float* scale_var_inv = ( float* )sys_malloc(channel_num * sizeof(float));
    const float* mean = ( const float* )mean_tensor->data;
    const float* var = ( const float* )var_tensor->data;

    float rescale_factor;
    float eps = batchnorm_param->eps;

    rescale_factor = batchnorm_param->rescale_factor ? 1 / batchnorm_param->rescale_factor : 0;

    for (int c = 0; c < channel_num; c++)
    {
        float tmp = sqrtf(var[c] * rescale_factor + eps);
        scale_var_inv[c] = ( float )(1.f / tmp);
        tmp = rescale_factor * scale_var_inv[c];
        scale_mean[c] = ( float )(-mean[c] * tmp);
    }
    float* gamma = NULL;
    float* beta = NULL;
    if (!batchnorm_param->caffe_flavor)
    {
        const struct tensor* gamma_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
        const struct tensor* beta_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[2]);
        gamma = ( float* )gamma_tensor->data;
        beta = ( float* )beta_tensor->data;
    }
    int layout = ir_graph->graph_layout;
    op_param->iscaffe = batchnorm_param->caffe_flavor;
    op_param->scale_mean = scale_mean;
    op_param->scale_var_inv = scale_var_inv;
    op_param->gamma = gamma;
    op_param->beta = beta;
    op_param->layout = layout;

    return 0;
}

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* ir_node = exec_node->ir_node;
    struct graph* ir_graph = ir_node->graph;
    struct tensor* input_tensor;
    struct tensor* output_tensor;

    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    struct ref_batchnorm_param* batchnorm_op_param = ( struct ref_batchnorm_param* )exec_node->ops_priv;
    void* out_data = output_tensor->data;
    void* input = input_tensor->data;

    if (TENGINE_LAYOUT_NCHW == ir_graph->graph_layout)
    {
        if (4 == input_tensor->dim_num)
        {
            batchnorm_op_param->input_n = input_tensor->dims[0];
            batchnorm_op_param->input_c = input_tensor->dims[1];
            batchnorm_op_param->input_h = input_tensor->dims[2];
            batchnorm_op_param->input_w = input_tensor->dims[3];
        }
        else if (3 == input_tensor->dim_num)
        {
            batchnorm_op_param->input_n = input_tensor->dims[0];
            batchnorm_op_param->input_c = input_tensor->dims[1];
            batchnorm_op_param->input_w = input_tensor->dims[2];
            batchnorm_op_param->input_h = 1;
        }
        else
        {
            return false;
        }
    }
    else
    {
        if (4 == input_tensor->dim_num)
        {
            batchnorm_op_param->input_n = input_tensor->dims[0];
            batchnorm_op_param->input_c = input_tensor->dims[3];
            batchnorm_op_param->input_h = input_tensor->dims[1];
            batchnorm_op_param->input_w = input_tensor->dims[2];
        }
        else if (3 == input_tensor->dim_num)
        {
            batchnorm_op_param->input_n = input_tensor->dims[0];
            batchnorm_op_param->input_c = input_tensor->dims[2];
            batchnorm_op_param->input_w = input_tensor->dims[1];
            batchnorm_op_param->input_h = 1;
        }
        else
        {
            return false;
        }
    }
    
    int ret = -1;
    if (input_tensor->data_type == TENGINE_DT_FP32)
        ret = ref_batchnorm_fp32(input, out_data, batchnorm_op_param, exec_graph->num_thread);
    else if (input_tensor->data_type == TENGINE_DT_UINT8)
        ret = ref_batchnorm_uint8(input_tensor, output_tensor, batchnorm_op_param, exec_graph->num_thread);

    return ret;
}

static int postrun(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct ref_batchnorm_param* batchnorm_op_param = ( struct ref_batchnorm_param* )exec_node->ops_priv;

    sys_free(batchnorm_op_param->scale_mean);
    sys_free(batchnorm_op_param->scale_var_inv);

    return 0;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct node* exec_node)
{
    return OPS_SCORE_CANDO;
}

static struct node_ops hcl_node_ops = {.prerun = prerun,
                                       .run = run,
                                       .reshape = NULL,
                                       .postrun = postrun,
                                       .init_node = init_node,
                                       .release_node = release_node,
                                       .score = score};

int register_batchnorm_ref_op()
{
    return register_builtin_node_ops(OP_BATCHNORM, &hcl_node_ops);
}

int unregister_batchnorm_ref_op()
{
    return unregister_builtin_node_ops(OP_BATCHNORM, &hcl_node_ops);
}
