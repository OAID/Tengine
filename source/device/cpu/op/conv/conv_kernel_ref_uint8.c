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
 * update：qtang@openailab.com
 */

#include "convolution_param.h"

#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "utility/sys_port.h"
#include "utility/float.h"
#include "utility/log.h"
#include "device/cpu/cpu_node.h"
#include "device/cpu/cpu_graph.h"
#include "device/cpu/cpu_module.h"

#include "conv_kernel_ref.h"

#include <math.h>


int ref_conv_uint8(struct tensor* input_tensor, struct tensor* output_tensor, struct tensor* kernel,
                          struct tensor* bias, struct conv_param* conv_param)
{
    int batch = input_tensor->dims[0];
    int group = conv_param->group;
    int input_c = conv_param->input_channel / group;
    int input_h = input_tensor->dims[2];
    int input_w = input_tensor->dims[3];
    int output_c = output_tensor->dims[1] / group;
    int output_h = output_tensor->dims[2];
    int output_w = output_tensor->dims[3];

    int kernel_size = input_c * conv_param->kernel_h * conv_param->kernel_w;
    int n, g, c, h, w, kc, kh, kw;
    int input_offset = 0;
    int kernel_offset = 0;
    int output_offset = 0;

    uint8_t* input_data = input_tensor->data;
    uint8_t* output_data = output_tensor->data;
    uint8_t* kernel_data = kernel->data;
    int32_t* bias_data = NULL;
    if (bias != NULL)
        bias_data = bias->data;

    float input_scale = input_tensor->scale;
    float kernel_scale = kernel->scale;
    float output_scale = output_tensor->scale;
    int32_t kernel_zero = kernel->zero_point;
    int32_t input_zero = input_tensor->zero_point;
    int32_t output_zero = output_tensor->zero_point;

    /* dequant input  */
    int input_size = batch * group * input_c * input_h * input_w;
    float* input_fp32 = ( float* )sys_malloc(sizeof(float) * input_size);
    for (int i = 0; i < input_size; i++)
        input_fp32[i] = (( float )input_data[i] - input_zero) * input_scale;

    /* dequant kernel  */
    int kernel_total = group * output_c * kernel_size;
    float* kernel_fp32 = ( float* )sys_malloc(sizeof(float) * kernel_total);
    for (int i = 0; i < kernel_total; i++)
        kernel_fp32[i] = (( float )kernel_data[i] - kernel_zero) * kernel_scale;

    /* dequant biases  */
    int bias_size = group * output_c;

    float* bias_fp32 = NULL;
    if (bias != NULL)
    {
        bias_fp32 = ( float* )sys_malloc(sizeof(float) * bias_size);
        for (int i = 0; i < bias_size; i++)
            bias_fp32[i] = ( float )bias_data[i] * input_scale * kernel_scale;
    }

    if (conv_param->kernel_h == 0)
        conv_param->kernel_h = 1;
    if (conv_param->kernel_w == 0)
        conv_param->kernel_w = 1;
    if (input_w == 0)
        input_w = 1;

    for (n = 0; n < batch; ++n)
    {
        for (g = 0; g < group; ++g)
        {
            for (c = 0; c < output_c; ++c)
            {
                for (h = 0; h < output_h; ++h)
                {
                    for (w = 0; w < output_w; ++w)
                    {
                        const int h_start = (h * conv_param->stride_h) - conv_param->pad_h0;
                        const int w_start = (w * conv_param->stride_w) - conv_param->pad_w0;
                        float total = 0.f;
                        if (input_tensor->layout == 0)
                        {
                            output_offset = n * group * output_c * output_h * output_w +
                                            g * output_c * output_h * output_w + c * output_h * output_w +
                                            h * output_w + w;
                        }
                        else
                        {
                            output_offset = n * group * output_c * output_h * output_w +
                                            h * output_w * group * output_c + w * group * output_c + output_c * g + c;
                        }
                        for (kc = 0; kc < input_c; ++kc)
                        {
                            for (kh = 0; kh < conv_param->kernel_h; ++kh)
                            {
                                for (kw = 0; kw < conv_param->kernel_w; ++kw)
                                {
                                    const int cur_y = h_start + conv_param->dilation_h * kh;
                                    const int cur_x = w_start + conv_param->dilation_w * kw;
                                    // If the location is outside the bounds of the input image,
                                    // use zero as a default value.
                                    if ((cur_x >= 0) && (cur_x < input_w) && (cur_y >= 0) && (cur_y < input_h))
                                    {
                                        if (input_tensor->layout == 0)
                                        {
                                            input_offset = n * group * input_c * input_h * input_w +
                                                           g * input_c * input_h * input_w + kc * input_h * input_w +
                                                           cur_y * input_w + cur_x;
                                            kernel_offset = g * output_c * kernel_size + c * kernel_size +
                                                            kc * conv_param->kernel_h * conv_param->kernel_w +
                                                            kh * conv_param->kernel_w + kw;
                                        }
                                        else
                                        {
                                            input_offset = n * group * input_c * input_h * input_w +
                                                           cur_y * input_w * input_c * group + cur_x * input_c * group +
                                                           g * input_c + kc;
                                            kernel_offset = c * group * kernel_size +
                                                            kh * conv_param->kernel_w * input_c * group +
                                                            kw * input_c * group + g * input_c + kc;
                                        }

                                        total += input_fp32[input_offset] * kernel_fp32[kernel_offset];
                                    }
                                }
                            }
                        }

                        if (bias != NULL)
                            total += bias_fp32[output_c * g + c];

                        if (conv_param->activation >= 0)
                        {
                            if (total < 0 && conv_param->activation != 1)
                            {
                                total = 0;
                            }
                            if (total > 1 && conv_param->activation == 1)
                            {
                                total = 1;
                            }
                            if (total > 6 && conv_param->activation == 6)
                            {
                                total = 6;
                            }
                            if (total < -1 && conv_param->activation == 1)
                            {
                                total = -1;
                            }
                        }

                        int out = round(total / output_scale) + output_zero;
                        if (out > 255)
                            out = 255;
                        if (out < 0)
                            out = 0;
                        output_data[output_offset] = out;
                    }
                }
            }
        }
    }

    sys_free(input_fp32);
    sys_free(kernel_fp32);
    if (bias != NULL)
        sys_free(bias_fp32);

    return 0;
}
