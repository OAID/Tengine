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
 * Author: qwang@openailab.com
 */

#include "conv_dw_kernel_int8_arm.h"

#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>

#include "utility/sys_port.h"

#ifdef __aarch64__
void depthwise_k3s1p1_int8_a72(int8_t* input, int8_t* kernel, int8_t* out, int* bias, long out_h, long out_w,
                               long multi, long shift, long input_w, long act_min, long act_max);
void depthwise_k3s2p1_int8_a72(int8_t* input, int8_t* kernel, int8_t* out, int* bias, long out_h, long out_w,
                               long multi, long shift, long input_w, long act_min, long act_max);
#else
void depthwise_k3s1_int8(int8_t* input, int8_t* kernel, int8_t* out, int* bias, int out_h, int out_w,
                         int multi, int shift, int input_w, int act_min, int act_max);
void depthwise_k3s2_int8(int8_t* input, int8_t* kernel, int8_t* out, int* bias, int out_h, int out_w,
                         int multi, int shift, int input_w, int act_min, int act_max);
#endif

int conv_dw_int8_prerun(struct tensor* input_tensor, struct tensor* filter_tensor, struct tensor* output_tensor,
                        struct conv_priv_info* priv_info, struct conv_param* param)
{
    int batch = input_tensor->dims[0];
    int in_c = input_tensor->dims[1];
    int in_h = input_tensor->dims[2];
    int in_w = input_tensor->dims[3];

    int out_c = output_tensor->dims[1];
    int out_h = output_tensor->dims[2];
    int out_w = output_tensor->dims[3];

    priv_info->multi = (int*)sys_malloc(out_c * sizeof(int));
    priv_info->q_shift = (int*)sys_malloc(out_c * sizeof(int));

    float input_scale = input_tensor->scale;
    float* kernel_scales = filter_tensor->scale_list;
    float output_scale = output_tensor->scale;

    priv_info->activation_min = -127;
    priv_info->activation_max = 127;
    /*  set activation   */
    if (param->activation >= 0)
    {
        priv_info->activation_min = 0;
        if (param->activation == 1)
            priv_info->activation_max = round(1.0 / output_scale);
        if (param->activation == 6)
            priv_info->activation_max = round(6.0 / output_scale);

        if (priv_info->activation_max > 127)
            priv_info->activation_max = 127;
    }

    for (int i = 0; i < out_c; i++)
    {
        float kernel_scale = kernel_scales[i];
        float scale = input_scale * kernel_scale / output_scale;

        int shift;
        float q = frexp(scale, &shift);
        int fix_q = round(q * (1ll << 31));
        // TLOG_ERR("prerun: %f,%lld,%d,%d, %lld\n",q, fix_q, multi, q_shift, 1ll<<31);
        if (fix_q == (1l << 31))
        {
            fix_q /= 2;
            shift++;
        }

        priv_info->multi[i] = (int)fix_q;
        priv_info->q_shift[i] = (int)shift;
    }
    return 0;
}

int conv_dw_int8_postrun(struct conv_priv_info* priv_info)
{
    if (priv_info->multi)
    {
        sys_free(priv_info->multi);
        priv_info->multi = NULL;
    }
    if (priv_info->q_shift)
    {
        sys_free(priv_info->q_shift);
        priv_info->q_shift = NULL;
    }

    return 0;
}

void conv_dw_int8_direct(int8_t* input_buf, int8_t* weight_buf, int8_t* output_buf, int* bias, int input_h, int input_w,
                         int output_h, int output_w, int channel_num, int stride, int* pads, int* p_multi, int* p_shift,
                         int activation_min, int activation_max, int num_thread, int cpu_affinity)
{
    int channel_size = input_h * input_w;
#ifndef __aarch64__
    int8_t* input_pad = NULL;
    int input_h_pad = input_h + pads[0] + pads[2];
    int input_w_pad = input_w + pads[1] + pads[3];
    int is_pad0 = (pads[0] == 0 && pads[1] == 0 && pads[2] == 0 && pads[3] == 0);
    if (!is_pad0)
    {
        input_pad = (int8_t*)malloc(sizeof(int8_t) * channel_num * input_h_pad * input_w_pad + 128);
        memset(input_pad, 0, sizeof(int8_t) * channel_num * input_h_pad * input_w_pad + 128);
    }
#endif
#pragma omp parallel for num_threads(num_thread)
    for (int i = 0; i < channel_num; i++)
    {
        int8_t* input_tmp = NULL;
        int* bias_tmp = bias ? (bias + i) : NULL;
#ifndef __aarch64__
        if (!is_pad0)
        {
            int8_t* tmp = input_pad + i * input_h_pad * input_w_pad;
            input_tmp = tmp;
            tmp += pads[0] * input_w_pad + pads[1];
            for (int j = 0; j < input_h; j++)
            {
                memcpy(tmp, input_buf + i * channel_size + j * input_w, input_w);
                tmp += input_w_pad;
            }
        }
        else
#endif
        {
            input_tmp = input_buf + i * channel_size;
        }
        if (1 == stride)
        {
#ifdef __aarch64__
            depthwise_k3s1p1_int8_a72(input_tmp, weight_buf + 9 * i, output_buf + i * output_h * output_w, bias_tmp, output_h, output_w,
                                      p_multi[i], p_shift[i], input_w, activation_min, activation_max);
#else
            depthwise_k3s1_int8(input_tmp, weight_buf + 9 * i, output_buf + i * output_h * output_w, bias_tmp, output_h, output_w,
                                p_multi[i], p_shift[i], input_w_pad, activation_min, activation_max);
#endif
        }
        else if (2 == stride)
        {
#ifdef __aarch64__
            depthwise_k3s2p1_int8_a72(input_tmp, weight_buf + 9 * i, output_buf + i * output_h * output_w, bias_tmp, output_h, output_w,
                                      p_multi[i], p_shift[i], input_w, activation_min, activation_max);
#else
            depthwise_k3s2_int8(input_tmp, weight_buf + 9 * i, output_buf + i * output_h * output_w, bias_tmp, output_h, output_w,
                                p_multi[i], p_shift[i], input_w_pad, activation_min, activation_max);
#endif
        }
    }
#ifndef __aarch64__
    if (!is_pad0)
    {
        free(input_pad);
        input_pad = NULL;
    }
#endif
}

int conv_dw_int8_run(struct tensor* input_tensor, struct tensor* filter_tensor, struct tensor* bias_tensor,
                     struct tensor* output_tensor, struct conv_priv_info* priv_info, struct conv_param* param,
                     int num_thread, int cpu_affinity)
{
    /* param */
    int pads[4] = {0};
    int group = param->group;
    int kernel_h = param->kernel_h;
    int kernel_w = param->kernel_w;
    int stride_h = param->stride_h;
    int stride_w = param->stride_w;
    int dilation_h = param->dilation_h;
    int dilation_w = param->dilation_w;
    int act_type = param->activation;
    pads[0] = param->pad_h0;
    pads[1] = param->pad_w0;
    pads[2] = param->pad_h1;
    pads[3] = param->pad_w1;

    int batch = input_tensor->dims[0];
    int in_c = input_tensor->dims[1] / group;
    int in_h = input_tensor->dims[2];
    int in_w = input_tensor->dims[3];
    int input_size = in_c * in_h * in_w;
    int kernel_size = in_c * kernel_h * kernel_w;
    int input_image_size = input_tensor->dims[1] * input_tensor->dims[2] * input_tensor->dims[3];

    int out_c = output_tensor->dims[1] / group;
    int out_h = output_tensor->dims[2];
    int out_w = output_tensor->dims[3];
    int out_hw = out_h * out_w;
    int output_size = out_c * out_h * out_w;
    int out_c_align = ((out_c + 3) & -4);
    int output_image_size = output_tensor->dims[1] * output_tensor->dims[2] * output_tensor->dims[3];

    int activation_min = priv_info->activation_min;
    int activation_max = priv_info->activation_max;

    /* buffer addr */
    int8_t* input_buf = (int8_t*)input_tensor->data;
    int8_t* kernel_buf = (int8_t*)filter_tensor->data;
    int8_t* output_buf = (int8_t*)output_tensor->data;
    int32_t* biases_buf = NULL;
    if (bias_tensor != NULL)
    {
        biases_buf = (int32_t*)bias_tensor->data;
    }

    int* multi = priv_info->multi;
    int* q_shift = priv_info->q_shift;
    for (int n = 0; n < batch; n++) // batch size
    {
        int8_t* input = input_buf + n * input_size * group;
        int8_t* kernel = kernel_buf + n * kernel_size * group;
        int8_t* output = output_buf + n * output_size * group;
        conv_dw_int8_direct(input, kernel, output, biases_buf, in_h, in_w,
                            out_h, out_w, in_c * group, stride_h, pads, multi, q_shift,
                            activation_min, activation_max, num_thread, cpu_affinity);
    }
    return 0;
}
