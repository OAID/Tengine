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
 * Author: hhchen@openailab.com
 */

#include "sys_port.h"
#include "module.h"
#include "tengine_errno.h"
#include "tengine_log.h"
#include "tengine_ir.h"
#include "../../cpu_node_ops.h"
#include "tengine_op.h"
#include "selu_param.h"
#include <math.h>

int ref_selu_fp32(struct ir_tensor* output_tensor, struct ir_tensor* input_tensor, struct selu_param* selu_param,
                  int num_thread)
{
    float* data = ( float* )input_tensor->data;
    float* out_data = ( float* )output_tensor->data;
    float alpha = selu_param->alpha;
    float lambda = selu_param->lambda;
    float alpha_lambda = alpha * lambda;

    int chan_num = input_tensor->dims[0] * input_tensor->dims[1];
    int chan_size = input_tensor->dims[2] * input_tensor->dims[3];

#pragma omp parallel for num_threads(num_thread)
    for (int i = 0; i < chan_num; i++)
    {
        int offset = i * chan_size;
        float* input_data = ( float* )input_tensor->data + i * chan_size;
        float* output_data = ( float* )output_tensor->data + i * chan_size;

        for (int i = 0; i < chan_size; i++)
        {
            if (input_data[i] < 0.f)
                output_data[i] = (exp(input_data[i]) - 1.f) * alpha_lambda;
            else
                output_data[i] = input_data[i] * lambda;
        }
    }

    return 0;
}

int ref_selu_uint8(struct ir_tensor* output_tensor, struct ir_tensor* input_tensor, struct selu_param* selu_param,
                  int num_thread)
{
    /* dequant */
    uint8_t* input_uint8 = input_tensor->data;
    uint8_t* output_uint8 = output_tensor->data;
    float input_scale = input_tensor->scale;
    float output_scale = output_tensor->scale;
    int32_t input_zero = input_tensor->zero_point;
    int32_t output_zero = output_tensor->zero_point;
    int input_size = input_tensor->elem_num;
    int output_size = output_tensor->elem_num;

    float* input_data = ( float* )sys_malloc(input_size * sizeof(float));
    float* output_data = ( float* )sys_malloc(output_size * sizeof(float));

    for (int i = 0; i < input_size; i++)
    {
        input_data[i] = (( float )input_uint8[i] - ( float )input_zero) * input_scale;
    }

    float alpha = selu_param->alpha;
    float lambda = selu_param->lambda;
    float alpha_lambda = alpha * lambda;

    int chan_num = input_tensor->dims[0] * input_tensor->dims[1];
    int chan_size = input_tensor->dims[2] * input_tensor->dims[3];

#pragma omp parallel for num_threads(num_thread)
    for (int i = 0; i < chan_num; i++)
    {
        int offset = i * chan_size;
        float* input_data = ( float* )input_tensor->data + i * chan_size;
        float* output_data = ( float* )output_tensor->data + i * chan_size;

        for (int i = 0; i < chan_size; i++)
        {
            if (input_data[i] < 0.f)
                output_data[i] = (exp(input_data[i]) - 1.f) * alpha_lambda;
            else
                output_data[i] = input_data[i] * lambda;
        }
    }

    /* quant */
    for (int i = 0; i < output_size; i++)
    {
        int udata = round(output_data[i] / output_scale + output_zero);
        if (udata > 255)
            udata = 255;
        else if (udata < 0)
            udata = 0;
        output_uint8[i] = udata;
    }

    sys_free(input_data);
    sys_free(output_data);

    return 0;
}

static int init_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    return 0;
}

static int release_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    return 0;
}

static int prerun(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    return 0;
}

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct ir_node* ir_node = exec_node->ir_node;
    struct ir_graph* ir_graph = ir_node->graph;
    struct ir_tensor* input_tensor;
    struct ir_tensor* output_tensor;

    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    struct selu_param* selu_param = ( struct selu_param* )ir_node->op.param_mem;

    int num_thread = exec_graph->num_thread;

	int ret = -1;
    if (input_tensor->data_type == TENGINE_DT_FP32)
        ret = ref_selu_fp32(output_tensor, input_tensor, selu_param, num_thread);
    else if(input_tensor->data_type == TENGINE_DT_UINT8)
        ret = ref_selu_uint8(output_tensor, input_tensor, selu_param, num_thread);

    return ret;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct ir_node* exec_node)
{
    struct ir_node* ir_node = exec_node;
    struct ir_graph* ir_graph = ir_node->graph;
    struct ir_tensor* input_tensor;

    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);

    if (input_tensor->data_type != TENGINE_DT_FP32 || input_tensor->layout != TENGINE_LAYOUT_NCHW)
        return 0;

    return OPS_SCORE_CANDO;
}

static struct node_ops hcl_node_ops = {.prerun = prerun,
                                       .run = run,
                                       .reshape = NULL,
                                       .postrun = NULL,
                                       .init_node = init_node,
                                       .release_node = release_node,
                                       .score = score};

static int reg_selu_hcl_ops(void* arg)
{
    return register_builtin_node_ops(OP_SELU, &hcl_node_ops);
}

static int unreg_selu_hcl_ops(void* arg)
{
    return unregister_builtin_node_ops(OP_SELU, &hcl_node_ops);
}

AUTO_REGISTER_OPS(reg_selu_hcl_ops);
AUTO_UNREGISTER_OPS(unreg_selu_hcl_ops);
