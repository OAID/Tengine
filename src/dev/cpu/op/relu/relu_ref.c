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
#include "sys_port.h"
#include "module.h"
#include "tengine_ir.h"
#include "../../cpu_node_ops.h"
#include "tengine_op.h"
#include "relu_param.h"
#include "compiler_fp16.h"

static int ref_relu_fp32(struct ir_tensor* input_tensor, struct ir_tensor* output_tensor, float negative_slope,
                         int num_thread)
{
    int batch = input_tensor->dims[0];
    int channels = input_tensor->dims[1];
    int h = input_tensor->dims[2];
    int w = input_tensor->dims[3];

    int size = h * w;
    int c_step = h * w;
    int batch_step = channels * c_step;

    float* input_data = input_tensor->data;
    float* out_data = output_tensor->data;

    if (negative_slope == 0)
    {
        for (int n = 0; n < batch; n++)
        {
#pragma omp parallel for num_threads(num_thread)
            for (int q = 0; q < channels; q++)
            {
                float* src = input_data + batch_step * n + c_step * q;
                float* dst = out_data + batch_step * n + c_step * q;

                for (int i = 0; i < size; i++)
                {
                    if (src[i] < 0)
                        dst[i] = 0;
                    else
                        dst[i] = src[i];
                }
            }
        }
    }
    else
    {
        for (int n = 0; n < batch; n++)
        {
#pragma omp parallel for num_threads(num_thread)
            for (int q = 0; q < channels; q++)
            {
                float* src = input_data + batch_step * n + c_step * q;
                float* dst = out_data + batch_step * n + c_step * q;

                for (int i = 0; i < size; i++)
                {
                    if (src[i] < 0)
                        dst[i] = src[i] * negative_slope;
                    else
                        dst[i] = src[i];
                }
            }
        }
    }

    return 0;
}

static int ref_relu_fp16(struct ir_tensor* input_tensor, struct ir_tensor* output_tensor, float negative_slope,
                         int num_thread)
{
    int batch = input_tensor->dims[0];
    int channels = input_tensor->dims[1];
    int h = input_tensor->dims[2];
    int w = input_tensor->dims[3];

    int size = h * w;
    int c_step = h * w;
    int batch_step = channels * c_step;
    int total_size = batch * batch_step;

    /* cost fp16 to fp32 */
    __fp16* input_fp16 = input_tensor->data;
    __fp16* output_fp16 = output_tensor->data;
    float* input_fp32 = (float*)sys_malloc(total_size * sizeof(float));

    for(int i=0; i<total_size; i++)
    {
        input_fp32[i] = fp16_to_fp32(input_fp16[i]);
    }

    /* process */
    if (negative_slope == 0)
    {
        for (int n = 0; n < batch; n++)
        {
//#pragma omp parallel for num_threads(num_thread)
            for (int q = 0; q < channels; q++)
            {
                float* src = input_fp32 + batch_step * n + c_step * q;
                float* dst = input_fp32 + batch_step * n + c_step * q;

                for (int i = 0; i < size; i++)
                {
                    if (src[i] < 0)
                        dst[i] = 0;
                    else
                        dst[i] = src[i];
                }
            }
        }
    }
    else
    {
        for (int n = 0; n < batch; n++)
        {
//#pragma omp parallel for num_threads(num_thread)
            for (int q = 0; q < channels; q++)
            {
                float* src = input_fp32 + batch_step * n + c_step * q;
                float* dst = input_fp32 + batch_step * n + c_step * q;

                for (int i = 0; i < size; i++)
                {
                    if (src[i] < 0)
                        dst[i] = src[i] * negative_slope;
                    else
                        dst[i] = src[i];
                }
            }
        }
    }

    /* cost fp32 to fp16 */
    for(int i=0; i<total_size; i++)
    {
        output_fp16[i] = fp32_to_fp16(input_fp32[i]);
    }

    sys_free(input_fp32);

    return 0;
}

static int ref_relu_uint8(struct ir_tensor* input_tensor, struct ir_tensor* output_tensor, float negative_slope,
                          int num_thread)
{
    int batch = input_tensor->dims[0];
    int channels = input_tensor->dims[1];
    int h = input_tensor->dims[2];
    int w = input_tensor->dims[3];

    int size = h * w;
    int c_step = h * w;
    int batch_step = channels * c_step;
    int total_size = batch * batch_step;

    /* dequant */
    uint8_t* input_uint8 = input_tensor->data;
    uint8_t* output_uint8 = output_tensor->data;
    float input_scale = input_tensor->scale;
    float output_scale = output_tensor->scale;
    int32_t input_zero = input_tensor->zero_point;
    int32_t output_zero = output_tensor->zero_point;

    float* data_fp32 = (float*)sys_malloc(total_size * sizeof(float));

    for(int i=0; i<total_size; i++)
    {
        data_fp32[i] = ((float )input_uint8[i] - (float )input_zero) * input_scale;
    }

    /* process */
    if (negative_slope == 0)
    {
        for (int n = 0; n < batch; n++)
        {
//#pragma omp parallel for num_threads(num_thread)
            for (int q = 0; q < channels; q++)
            {
                float* src = data_fp32 + batch_step * n + c_step * q;
                float* dst = data_fp32 + batch_step * n + c_step * q;

                for (int i = 0; i < size; i++)
                {
                    if (src[i] < 0)
                        dst[i] = 0;
                    else
                        dst[i] = src[i];
                }
            }
        }
    }
    else
    {
        for (int n = 0; n < batch; n++)
        {
//#pragma omp parallel for num_threads(num_thread)
            for (int q = 0; q < channels; q++)
            {
                float* src = data_fp32 + batch_step * n + c_step * q;
                float* dst = data_fp32 + batch_step * n + c_step * q;

                for (int i = 0; i < size; i++)
                {
                    if (src[i] < 0)
                        dst[i] = src[i] * negative_slope;
                    else
                        dst[i] = src[i];
                }
            }
        }
    }

    /* quant */
    for(int i=0; i<total_size; i++)
    {
        int udata = round(data_fp32[i] / output_scale + output_zero);
        if (udata > 255)
            udata = 255;
        else if (udata < 0)
            udata = 0;
        output_uint8[i] = udata;
    }

    sys_free(data_fp32);

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

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct ir_node* ir_node = exec_node->ir_node;
    struct ir_graph* ir_graph = ir_node->graph;
    struct ir_tensor* input_tensor;
    struct ir_tensor* output_tensor;

    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    struct relu_param* relu_param = ( struct relu_param* )ir_node->op.param_mem;

    int ret = 0;
    if (input_tensor->data_type == TENGINE_DT_FP32)
        ret = ref_relu_fp32(input_tensor, output_tensor, relu_param->negative_slope, exec_graph->num_thread);
    else if (input_tensor->data_type == TENGINE_DT_FP16)
        ret = ref_relu_fp16(input_tensor, output_tensor, relu_param->negative_slope, exec_graph->num_thread);
    else if (input_tensor->data_type == TENGINE_DT_UINT8)
        ret = ref_relu_uint8(input_tensor, output_tensor, relu_param->negative_slope, exec_graph->num_thread);
    else
    {
        printf("Input data type %d not to be supported.\n", input_tensor->data_type);
        return -1;
    }

    return ret;
}

static int reshape(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct ir_node* node = exec_node->ir_node;
    struct ir_graph* ir_graph = node->graph;
    struct ir_tensor* input = get_ir_graph_tensor(ir_graph, node->input_tensors[0]);
    struct ir_tensor* output = get_ir_graph_tensor(ir_graph, node->output_tensors[0]);

    int ret = set_ir_tensor_shape(output, input->dims, input->dim_num);
    return ret;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct ir_node* exec_node)
{
    return OPS_SCORE_CANDO;
}

static struct node_ops hcl_node_ops = {.prerun = NULL,
                                       .run = run,
                                       .reshape = reshape,
                                       .postrun = NULL,
                                       .init_node = init_node,
                                       .release_node = release_node,
                                       .score = score};

static int reg_relu_hcl_ops(void* arg)
{
    return register_builtin_node_ops(OP_RELU, &hcl_node_ops);
}

static int unreg_relu_hcl_ops(void* arg)
{
    return unregister_builtin_node_ops(OP_RELU, &hcl_node_ops);
}

AUTO_REGISTER_OPS(reg_relu_hcl_ops);
AUTO_UNREGISTER_OPS(unreg_relu_hcl_ops);
