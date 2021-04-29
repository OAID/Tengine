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
 * Parts of the following code in this file refs to
 * https://github.com/Tencent/ncnn/blob/master/src/layer/arm/convolutiondepthwise_5x5.h
 * Tencent is pleased to support the open source community by making ncnn
 * available.
 *
 * Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
 *
 * Licensed under the BSD 3-Clause License (the "License"); you may not use this
 * file except in compliance with the License. You may obtain a copy of the
 * License at
 *
 * https://opensource.org/licenses/BSD-3-Clause
 */

/*
 * Copyright (c) 2021, OPEN AI LAB
 * Author: qtang@openailab.com
 */

#include "conv_dw_kernel_rv64.h"

#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "utility/sys_port.h"
#include "utility/float.h"
#include "utility/log.h"
#include "device/cpu/cpu_node.h"
#include "device/cpu/cpu_graph.h"
#include "device/cpu/cpu_module.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>


#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))

static void relu(float* data, int size, int activation)
{
    for (int i = 0; i < size; i++)
    {
        data[i] = max(data[i], ( float )0);

        if (activation > 0)
        {
            data[i] = min(data[i], ( float )activation);
        }
    }
}

static void pad(float* input, float* output, int in_h, int in_w, int out_h, int out_w, int top, int left, float v)
{
    float* ptr = input;
    float* outptr = output;

    int y = 0;
    // fill top
    for (; y < top; y++)
    {
        int x = 0;
        for (; x < out_w; x++)
        {
            outptr[x] = v;
        }
        outptr += out_w;
    }
    // fill center
    for (; y < (top + in_h); y++)
    {
        int x = 0;
        for (; x < left; x++)
        {
            outptr[x] = v;
        }
        if (in_w < 12)
        {
            for (; x < (left + in_w); x++)
            {
                outptr[x] = ptr[x - left];
            }
        }
        else
        {
            memcpy(outptr + left, ptr, in_w * sizeof(float));
            x += in_w;
        }
        for (; x < out_w; x++)
        {
            outptr[x] = v;
        }
        ptr += in_w;
        outptr += out_w;
    }
    // fill bottom
    for (; y < out_h; y++)
    {
        int x = 0;
        for (; x < out_w; x++)
        {
            outptr[x] = v;
        }
        outptr += out_w;
    }
}

static void convdw3x3s1(float* output, float* input, float* _kernel, float* _bias, int channel, int in_h, int in_w,
                        int out_h, int out_w, int num_thread)
{
    int w = in_w;
    int h = in_h;
    int c_step_in = w * h;

    int outw = out_w;
    int outh = out_h;
    int c_step_out = outw * outh;

    const int group = channel;
    const float* kernel = _kernel;

#pragma omp parallel for num_threads(num_thread)
    for (int g = 0; g < group; g++)
    {
        float* out = output + g * c_step_out;
        float* outptr = out;
        float* outptr2 = outptr + outw;

        const float bias0 = _bias ? _bias[g] : 0.f;
        const float* kernel0 = kernel + g * 9;

        const float* img0 = input + g * c_step_in;
        const float* r0 = img0;
        const float* r1 = img0 + w;
        const float* r2 = img0 + w * 2;
        const float* r3 = img0 + w * 3;

        const float* k0 = kernel0;
        const float* k1 = kernel0 + 3;
        const float* k2 = kernel0 + 6;

        int i = 0;
        for (; i + 1 < outh; i += 2)
        {
            int remain = outw;

            for (; remain > 0; remain--)
            {
                float sum = bias0;
                sum += r0[0] * k0[0];
                sum += r0[1] * k0[1];
                sum += r0[2] * k0[2];
                sum += r1[0] * k1[0];
                sum += r1[1] * k1[1];
                sum += r1[2] * k1[2];
                sum += r2[0] * k2[0];
                sum += r2[1] * k2[1];
                sum += r2[2] * k2[2];

                float sum2 = bias0;
                sum2 += r1[0] * k0[0];
                sum2 += r1[1] * k0[1];
                sum2 += r1[2] * k0[2];
                sum2 += r2[0] * k1[0];
                sum2 += r2[1] * k1[1];
                sum2 += r2[2] * k1[2];
                sum2 += r3[0] * k2[0];
                sum2 += r3[1] * k2[1];
                sum2 += r3[2] * k2[2];

                *outptr = sum;
                *outptr2 = sum2;

                r0++;
                r1++;
                r2++;
                r3++;
                outptr++;
                outptr2++;
            }

            r0 += 2 + w;
            r1 += 2 + w;
            r2 += 2 + w;
            r3 += 2 + w;

            outptr += outw;
            outptr2 += outw;
        }

        for (; i < outh; i++)
        {
            int remain = outw;

            for (; remain > 0; remain--)
            {
                float sum = bias0;
                sum += r0[0] * k0[0];
                sum += r0[1] * k0[1];
                sum += r0[2] * k0[2];
                sum += r1[0] * k1[0];
                sum += r1[1] * k1[1];
                sum += r1[2] * k1[2];
                sum += r2[0] * k2[0];
                sum += r2[1] * k2[1];
                sum += r2[2] * k2[2];

                *outptr = sum;

                r0++;
                r1++;
                r2++;
                outptr++;
            }

            r0 += 2;
            r1 += 2;
            r2 += 2;
        }
    }
}

static void convdw3x3s2(float* output, float* input, float* _kernel, float* _bias, int channel, int in_h, int in_w,
                        int out_h, int out_w, int num_thread)
{
    int w = in_w;
    int h = in_h;
    int c_step_in = w * h;

    int outw = out_w;
    int outh = out_h;
    int c_step_out = outw * outh;

    const int group = channel;

    const int tailstep = w - 2 * outw + w;
    const float* kernel = _kernel;

#pragma omp parallel for num_threads(num_thread)
    for (int g = 0; g < group; g++)
    {
        float* out = output + g * c_step_out;
        float* outptr = out;

        const float* kernel0 = kernel + g * 9;
        const float bias0 = _bias ? _bias[g] : 0.f;

        const float* img0 = input + g * c_step_in;
        const float* r0 = img0;
        const float* r1 = img0 + w;
        const float* r2 = img0 + w * 2;

        const float* k0 = kernel0;
        const float* k1 = kernel0 + 3;
        const float* k2 = kernel0 + 6;

        int i = 0;
        for (; i < outh; i++)
        {
            int remain = outw;
            for (; remain > 0; remain--)
            {
                float sum = bias0;
                sum += r0[0] * k0[0];
                sum += r0[1] * k0[1];
                sum += r0[2] * k0[2];
                sum += r1[0] * k1[0];
                sum += r1[1] * k1[1];
                sum += r1[2] * k1[2];
                sum += r2[0] * k2[0];
                sum += r2[1] * k2[1];
                sum += r2[2] * k2[2];

                *outptr = sum;

                r0 += 2;
                r1 += 2;
                r2 += 2;
                outptr++;
            }

            r0 += tailstep;
            r1 += tailstep;
            r2 += tailstep;
        }
    }
}

static void convdw5x5s1(float* output, float* input, float* _kernel, float* _bias, int channel, int in_h, int in_w,
                        int out_h, int out_w, int num_thread)
{
    int w = in_w;
    int h = in_h;
    int c_step_in = w * h;

    int outw = out_w;
    int outh = out_h;
    int c_step_out = outw * outh;

    const int group = channel;
    const float* kernel = _kernel;    

#pragma omp parallel for num_threads(opt.num_threads)
    for (int g = 0; g < group; g++)
    {
        float* out = output + g * c_step_out;
        float* outptr = out;
        float* outptr2 = outptr + outw;

        const float bias0 = _bias ? _bias[g] : 0.f;
        const float* kernel0 = kernel + g * 25;

        const float* img0 = input + g * c_step_in;
        const float* r0 = img0;
        const float* r1 = img0 + w;
        const float* r2 = img0 + w * 2;
        const float* r3 = img0 + w * 3;
        const float* r4 = img0 + w * 4;
        const float* r5 = img0 + w * 5;

        const float* k0 = kernel0;
        const float* k1 = kernel0 + 5;
        const float* k2 = kernel0 + 10;
        const float* k3 = kernel0 + 15;
        const float* k4 = kernel0 + 20;

        int i = 0;

        for (; i + 1 < outh; i += 2)
        {
            int remain = outw;

            for (; remain > 0; remain--)
            {
                float sum = bias0;
                float sum2 = bias0;

                sum += r0[0] * k0[0];
                sum += r0[1] * k0[1];
                sum += r0[2] * k0[2];
                sum += r0[3] * k0[3];
                sum += r0[4] * k0[4];

                sum += r1[0] * k1[0];
                sum += r1[1] * k1[1];
                sum += r1[2] * k1[2];
                sum += r1[3] * k1[3];
                sum += r1[4] * k1[4];

                sum += r2[0] * k2[0];
                sum += r2[1] * k2[1];
                sum += r2[2] * k2[2];
                sum += r2[3] * k2[3];
                sum += r2[4] * k2[4];

                sum += r3[0] * k3[0];
                sum += r3[1] * k3[1];
                sum += r3[2] * k3[2];
                sum += r3[3] * k3[3];
                sum += r3[4] * k3[4];

                sum += r4[0] * k4[0];
                sum += r4[1] * k4[1];
                sum += r4[2] * k4[2];
                sum += r4[3] * k4[3];
                sum += r4[4] * k4[4];

                sum2 += r1[0] * k0[0];
                sum2 += r1[1] * k0[1];
                sum2 += r1[2] * k0[2];
                sum2 += r1[3] * k0[3];
                sum2 += r1[4] * k0[4];

                sum2 += r2[0] * k1[0];
                sum2 += r2[1] * k1[1];
                sum2 += r2[2] * k1[2];
                sum2 += r2[3] * k1[3];
                sum2 += r2[4] * k1[4];

                sum2 += r3[0] * k2[0];
                sum2 += r3[1] * k2[1];
                sum2 += r3[2] * k2[2];
                sum2 += r3[3] * k2[3];
                sum2 += r3[4] * k2[4];

                sum2 += r4[0] * k3[0];
                sum2 += r4[1] * k3[1];
                sum2 += r4[2] * k3[2];
                sum2 += r4[3] * k3[3];
                sum2 += r4[4] * k3[4];

                sum2 += r5[0] * k4[0];
                sum2 += r5[1] * k4[1];
                sum2 += r5[2] * k4[2];
                sum2 += r5[3] * k4[3];
                sum2 += r5[4] * k4[4];

                *outptr = sum;
                *outptr2 = sum2;

                r0++;
                r1++;
                r2++;
                r3++;
                r4++;
                r5++;
                outptr++;
                outptr2++;
            }

            r0 += 4 + w;
            r1 += 4 + w;
            r2 += 4 + w;
            r3 += 4 + w;
            r4 += 4 + w;
            r5 += 4 + w;

            outptr += outw;
            outptr2 += outw;
        }

        for (; i < outh; i++)
        {
            int remain = outw;

            for (; remain > 0; remain--)
            {
                float sum = bias0;

                sum += r0[0] * k0[0];
                sum += r0[1] * k0[1];
                sum += r0[2] * k0[2];
                sum += r0[3] * k0[3];
                sum += r0[4] * k0[4];

                sum += r1[0] * k1[0];
                sum += r1[1] * k1[1];
                sum += r1[2] * k1[2];
                sum += r1[3] * k1[3];
                sum += r1[4] * k1[4];

                sum += r2[0] * k2[0];
                sum += r2[1] * k2[1];
                sum += r2[2] * k2[2];
                sum += r2[3] * k2[3];
                sum += r2[4] * k2[4];

                sum += r3[0] * k3[0];
                sum += r3[1] * k3[1];
                sum += r3[2] * k3[2];
                sum += r3[3] * k3[3];
                sum += r3[4] * k3[4];

                sum += r4[0] * k4[0];
                sum += r4[1] * k4[1];
                sum += r4[2] * k4[2];
                sum += r4[3] * k4[3];
                sum += r4[4] * k4[4];

                *outptr = sum;

                r0++;
                r1++;
                r2++;
                r3++;
                r4++;
                outptr++;
            }

            r0 += 4;
            r1 += 4;
            r2 += 4;
            r3 += 4;
            r4 += 4;
        }
    }
}

static void convdw5x5s2(float* output, float* input, float* _kernel, float* _bias, int channel, int in_h, int in_w,
                        int out_h, int out_w, int num_thread)
{
    int w = in_w;
    int h = in_h;
    int c_step_in = w * h;

    int outw = out_w;
    int outh = out_h;
    int c_step_out = outw * outh;

    const int group = channel;

    const int tailstep = w - 2 * outw + w;
    const float* kernel = _kernel;

#pragma omp parallel for num_threads(opt.num_threads)
    for (int g = 0; g < group; g++)
    {
        float* out = output + g * c_step_out;
        float* outptr = out;

        const float* kernel0 = kernel + g * 25;
        const float bias0 = _bias ? _bias[g] : 0.f;

        const float* img0 = input + g * c_step_in;
        const float* r0 = img0;
        const float* r1 = img0 + w;
        const float* r2 = img0 + w * 2;
        const float* r3 = img0 + w * 3;
        const float* r4 = img0 + w * 4;

        const float* k0 = kernel0;
        const float* k1 = kernel0 + 5;
        const float* k2 = kernel0 + 10;
        const float* k3 = kernel0 + 15;
        const float* k4 = kernel0 + 20;

        int i = 0;

        for (; i < outh; i++)
        {
            int remain = outw;

            for (; remain > 0; remain--)
            {
                float sum = bias0;

                sum += r0[0] * k0[0];
                sum += r0[1] * k0[1];
                sum += r0[2] * k0[2];
                sum += r0[3] * k0[3];
                sum += r0[4] * k0[4];

                sum += r1[0] * k1[0];
                sum += r1[1] * k1[1];
                sum += r1[2] * k1[2];
                sum += r1[3] * k1[3];
                sum += r1[4] * k1[4];

                sum += r2[0] * k2[0];
                sum += r2[1] * k2[1];
                sum += r2[2] * k2[2];
                sum += r2[3] * k2[3];
                sum += r2[4] * k2[4];

                sum += r3[0] * k3[0];
                sum += r3[1] * k3[1];
                sum += r3[2] * k3[2];
                sum += r3[3] * k3[3];
                sum += r3[4] * k3[4];

                sum += r4[0] * k4[0];
                sum += r4[1] * k4[1];
                sum += r4[2] * k4[2];
                sum += r4[3] * k4[3];
                sum += r4[4] * k4[4];

                *outptr = sum;

                r0 += 2;
                r1 += 2;
                r2 += 2;
                r3 += 2;
                r4 += 2;
                outptr++;
            }

            r0 += tailstep;
            r1 += tailstep;
            r2 += tailstep;
            r3 += tailstep;
            r4 += tailstep;
        }
    }
}

int conv_dw_run(struct tensor* input_tensor, struct tensor* weight_tensor, struct tensor* bias_tensor,
                struct tensor* output_tensor, struct conv_priv_info* conv_info, struct conv_param* param, int num_thread, int cpu_affinity)
{
    float* input = ( float* )input_tensor->data;
    float* output = ( float* )output_tensor->data;
    float* kernel = ( float* )weight_tensor->data;
    float* biases = NULL;
    if (bias_tensor)
        biases = ( float* )bias_tensor->data;

    int batch_number = input_tensor->dims[0];
    int inc = input_tensor->dims[1];
    int inh = input_tensor->dims[2];
    int inw = input_tensor->dims[3];
    int in_chw = inc * inh * inw;

    int outc = output_tensor->dims[1];
    int outh = output_tensor->dims[2];
    int outw = output_tensor->dims[3];
    int out_hw = outh * outw;
    int out_chw = out_hw * outc;

    int ksize_h = param->kernel_h;
    int ksize_w = param->kernel_w;
    int pad_w = param->pad_w0;
    int pad_h = param->pad_h0;

    int stride_w = param->stride_w;
    int stride_h = param->stride_h;
    int dilation_w = param->dilation_w;
    int dilation_h = param->dilation_h;
    int group = param->group;

    int activation = param->activation;

    /* pading */
    int inh_tmp = inh + pad_h + pad_h;
    int inw_tmp = inw + pad_w + pad_w;
    float* input_tmp = NULL;
    if (inh_tmp == inh && inw_tmp == inw)
        input_tmp = input;
    else
    {
        input_tmp = ( float* )sys_malloc(inh_tmp * inw_tmp * group * sizeof(float));
#pragma omp parallel for num_threads(num_thread)        
        for (int g = 0; g < group; g++)
        {
            float* pad_in = input + g * inh * inw;
            float* pad_out = input_tmp + g * inh_tmp * inw_tmp;
            pad(pad_in, pad_out, inh, inw, inh_tmp, inw_tmp, pad_h, pad_w, 0.f);
        }
    }

    /* process */
    for (int i = 0; i < batch_number; i++)
    {
        if (ksize_h ==3 && stride_h == 1)
            convdw3x3s1(output, input_tmp, kernel, biases, group, inh_tmp, inw_tmp, outh, outw, num_thread);
        else if  (ksize_h ==3 && stride_h == 2)
            convdw3x3s2(output, input_tmp, kernel, biases, group, inh_tmp, inw_tmp, outh, outw, num_thread);
        else if  (ksize_h ==5 && stride_h == 1)
            convdw5x5s1(output, input_tmp, kernel, biases, group, inh_tmp, inw_tmp, outh, outw, num_thread);
        else if  (ksize_h ==5 && stride_h == 2)
            convdw5x5s2(output, input_tmp, kernel, biases, group, inh_tmp, inw_tmp, outh, outw, num_thread);
        else
            TLOG_ERR("convdw %d x %d, s %d not support.\n", ksize_h, ksize_w, stride_h);
    }

    /* relu */
    if (activation >= 0)
        relu(output, batch_number * out_chw, activation);

    if (!(inh_tmp == inh && inw_tmp == inw))
        sys_free(input_tmp);

    return 0;
}
