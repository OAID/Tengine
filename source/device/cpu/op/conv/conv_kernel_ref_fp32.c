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


int ref_conv_fp32(struct tensor* input_tensor, struct tensor* output_tensor, struct tensor* kernel,
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

    float* input_data = input_tensor->data;
    float* output_data = output_tensor->data;
    float* kernel_data = kernel->data;
    float* bias_data = NULL;
    if (bias != NULL)
        bias_data = bias->data;

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

                        output_offset = n * group * output_c * output_h * output_w +
                                        g * output_c * output_h * output_w + c * output_h * output_w +
                                        h * output_w + w;

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

                                        input_offset = n * group * input_c * input_h * input_w +
                                                        g * input_c * input_h * input_w + kc * input_h * input_w +
                                                        cur_y * input_w + cur_x;
                                        kernel_offset = g * output_c * kernel_size + c * kernel_size +
                                                        kc * conv_param->kernel_h * conv_param->kernel_w +
                                                        kh * conv_param->kernel_w + kw;

                                        total += input_data[input_offset] * kernel_data[kernel_offset];
                                    }
                                }
                            }
                        }

                        if (bias != NULL)
                            total += bias_data[output_c * g + c];

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
                        output_data[output_offset] = total;
                    }
                }
            }
        }
    }

    return 0;
}
