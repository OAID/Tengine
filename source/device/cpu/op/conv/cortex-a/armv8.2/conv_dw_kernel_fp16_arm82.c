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
 * Author: qtang@openailab.com
 */

#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include "compiler_fp16.h"
#include "conv_dw_kernel_fp16_arm82.h"

void dw_k3s1p1_fp16_a76(__fp16* input, __fp16* kernel, __fp16* output, long channel_number, long input_w, long input_h, __fp16* bias);
void dw_k3s1p1_fp16_relu_fused_a76(__fp16* input, __fp16* kernel, __fp16* output, long channel_number, long input_w, long input_h, __fp16* bias);
void dw_k3s1p1_fp16_relu6_fused_a76(__fp16* input, __fp16* kernel, __fp16* output, long channel_number, long input_w, long input_h, __fp16* bias);

void dw_k3s2_fp16_a76(__fp16* bias, __fp16* input, __fp16* kernel, __fp16* output, long channel_number, long input_w, long input_h, long pad0);
void dw_k3s2_fp16_relu_fused_a76(__fp16* bias, __fp16* input, __fp16* kernel, __fp16* output, long channel_number, long input_w, long input_h, long pad0);
void dw_k3s2_fp16_relu6_fused_a76(__fp16* bias, __fp16* input, __fp16* kernel, __fp16* output, long channel_number, long input_w, long input_h, long pad0);

int conv_dw_fp16_run(struct tensor* input_tensor, struct tensor* filter_tensor, struct tensor* bias_tensor,
                     struct tensor* output_tensor, struct conv_param* param, int num_thread, int cpu_affinity)
{
    /* param */
    int pads[4];
    int group = param->group;
    int kernel_h = param->kernel_h;
    int kernel_w = param->kernel_w;
    int stride_h = param->stride_h;
    int stride_w = param->stride_w;
    int dilation_h = param->dilation_h;
    int dilation_w = param->dilation_w;
    pads[0] = param->pad_h0;
    pads[1] = param->pad_w0;
    pads[2] = param->pad_h1;
    pads[3] = param->pad_w1;

    if (stride_h != stride_w)
        return -1;

    if (pads[0] != pads[1])
        return -1;

    int activation = param->activation;
    int batch = input_tensor->dims[0];
    int in_c = input_tensor->dims[1] / group;
    int in_h = input_tensor->dims[2];
    int in_w = input_tensor->dims[3];
    int input_size = in_c * in_h * in_w;

    int out_c = output_tensor->dims[1] / group;
    int out_h = output_tensor->dims[2];
    int out_w = output_tensor->dims[3];
    int output_size = out_c * out_h * out_w;

    /* buffer addr */
    __fp16* input_buf = input_tensor->data;
    __fp16* kernel_buf = filter_tensor->data;
    __fp16* output_buf = output_tensor->data;
    __fp16* bias_buf = NULL;
    if (bias_tensor)
        bias_buf = bias_tensor->data;

    for (int n = 0; n < batch; n++) // batch size
    {
        __fp16* input = input_buf + n * input_size * group;
        __fp16* output = output_buf + n * output_size * group;
        int stride = stride_h;
        int pad = pads[0];
        int channel_size = in_h * in_w;
        int channel_size_out = out_h * out_w;

        if (stride == 1 && pad == 1)
        {
            if (activation == 0)
            {
#pragma omp parallel for num_threads(num_thread)
                for (int i = 0; i < group; i++)
                {
                    __fp16* cur_input = input + i * channel_size;
                    __fp16* cur_output = output + i * channel_size_out;
                    __fp16* cur_kernel = kernel_buf + i * 9;
                    __fp16* cur_bias = NULL;
                    if (bias_buf)
                        cur_bias = bias_buf + i;
                    dw_k3s1p1_fp16_relu_fused_a76(cur_input, cur_kernel, cur_output, 1, in_w, in_h, cur_bias);
                }
            }
            else if (activation > 0)
            {
#pragma omp parallel for num_threads(num_thread)
                for (int i = 0; i < group; i++)
                {
                    __fp16* cur_input = input + i * channel_size;
                    __fp16* cur_output = output + i * channel_size_out;
                    __fp16* cur_kernel = kernel_buf + i * 9;
                    __fp16* cur_bias = NULL;
                    if (bias_buf)
                        cur_bias = bias_buf + i;
                    dw_k3s1p1_fp16_relu6_fused_a76(cur_input, cur_kernel, cur_output, 1, in_w, in_h, cur_bias);
                }
            }
            else
            {
#pragma omp parallel for num_threads(num_thread)
                for (int i = 0; i < group; i++)
                {
                    __fp16* cur_input = input + i * channel_size;
                    __fp16* cur_output = output + i * channel_size_out;
                    __fp16* cur_kernel = kernel_buf + i * 9;
                    __fp16* cur_bias = NULL;
                    if (bias_buf)
                        cur_bias = bias_buf + i;
                    dw_k3s1p1_fp16_a76(cur_input, cur_kernel, cur_output, 1, in_w, in_h, cur_bias);
                }
            }
        }
        else if (stride == 2)
        {
            if (activation == 0)
            {
#pragma omp parallel for num_threads(num_thread)
                for (int i = 0; i < group; i++)
                {
                    __fp16* cur_input = input + i * channel_size;
                    __fp16* cur_output = output + i * channel_size_out;
                    __fp16* cur_kernel = kernel_buf + i * 9;
                    __fp16* cur_bias = NULL;
                    if (bias_buf)
                        cur_bias = bias_buf + i;
                    dw_k3s2_fp16_relu_fused_a76(cur_bias, cur_input, cur_kernel, cur_output, 1, in_w, in_h, pad);
                }
            }
            else if (activation > 0)
            {
#pragma omp parallel for num_threads(num_thread)
                for (int i = 0; i < group; i++)
                {
                    __fp16* cur_input = input + i * channel_size;
                    __fp16* cur_output = output + i * channel_size_out;
                    __fp16* cur_kernel = kernel_buf + i * 9;
                    __fp16* cur_bias = NULL;
                    if (bias_buf)
                        cur_bias = bias_buf + i;
                    dw_k3s2_fp16_relu6_fused_a76(cur_bias, cur_input, cur_kernel, cur_output, 1, in_w, in_h, pad);
                }
            }
            else
            {
#pragma omp parallel for num_threads(num_thread)
                for (int i = 0; i < group; i++)
                {
                    __fp16* cur_input = input + i * channel_size;
                    __fp16* cur_output = output + i * channel_size_out;
                    __fp16* cur_kernel = kernel_buf + i * 9;
                    __fp16* cur_bias = NULL;
                    if (bias_buf)
                        cur_bias = bias_buf + i;
                    dw_k3s2_fp16_a76(cur_bias, cur_input, cur_kernel, cur_output, 1, in_w, in_h, pad);
                }
            }
        }
        else
        {
            TLOG_ERR("fp16 only support k3s1p1 or k3s2pn\n");
            return -1;
        }
    }

    return 0;
}
