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
 * Copyright (c) 2019, Open AI Lab
 * Author: jingyou@openailab.com
 */

#ifndef __REF_REGION_KERNEL_H__
#define __REF_REGION_KERNEL_H__

#include <stdint.h>
#include <math.h>

#include "compiler_fp16.h"

#ifdef __cplusplus
extern "C" {
#endif

struct ref_region_param
{
    int dims[4];
    int num_box;
    int num_class;
    int coords;
    int zero[2]; /* input, output */
    float scale[2]; /* input, output */
};

static int entry_index(int batch, int location, int entry, int hw, int chw, int classes)
{
    int coords = 4;
    int n = location / hw;
    int loc = location % hw;
    return batch * chw + n * hw * (coords + classes + 1) + entry * hw + loc;
}

static inline float logistic_activate(float x)
{
    return 1. / (1. + exp(-x));
}

static void logit_activate_array(float* x, const int n)
{
    int i;
    for(i = 0; i < n; ++i)
    {
        x[i] = logistic_activate(x[i]);
    }
}

static void softmax(const float* input, int n, int stride, float* output)
{
    int i;
    float sum = 0;
    float largest = input[0];
    for(i = 0; i < n; ++i)
    {
        if(input[i * stride] > largest)
            largest = input[i * stride];
    }
    for(i = 0; i < n; ++i)
    {
        float e = exp(input[i * stride] - largest);
        sum += e;
        output[i * stride] = e;
    }
    for(i = 0; i < n; ++i)
    {
        output[i * stride] /= sum;
    }
}

static void softmax_cpu(const float* input, int n, int batch, int batch_offset, int groups, int stride, float* output)
{
    int g, b;
    for(b = 0; b < batch; ++b)
    {
        for(g = 0; g < groups; ++g)
        {
            softmax(input + b * batch_offset + g, n, stride, output + b * batch_offset + g);
        }
    }
}

static int ref_region_common(const float* in_data, float* out_data, ref_region_param* param)
{
    int batch = param->dims[0];
    int hw = param->dims[2] * param->dims[3];
    int chw = param->dims[1] * hw;
    int nchw = param->dims[0] * chw;
    int num_box = param->num_box;
    int num_class = param->num_class;
    int coords = param->coords;

    memcpy(out_data, in_data, nchw * sizeof(float));

    for(int b = 0; b < batch; b++)
    {
        for(int n = 0; n < num_box; n++)
        {
            int index = entry_index(b, n * hw, 0, hw, chw, num_class);
            logit_activate_array(out_data + index, 2 * hw);
            index = entry_index(b, n * hw, coords, hw, chw, num_class);
            logit_activate_array(out_data + index, hw);
            index = entry_index(b, n * hw, coords + 1, hw, chw, num_class);
        }
    }

    int index = entry_index(0, 0, coords + 1, hw, chw, num_class);
    softmax_cpu(in_data + index, num_class, batch * num_box, chw / num_box, hw, hw, out_data + index);

    return 0;
}

typedef int (*ref_region_kernel_t)(const void* in_data, void* out_data, ref_region_param* param);

#ifdef CONFIG_KERNEL_FP32
#include "ref_region_fp32.c"
#endif

#ifdef CONFIG_KERNEL_FP16
#include "ref_region_fp16.c"
#endif

#ifdef CONFIG_KERNEL_INT8
#include "ref_region_int8.c"
#endif

#ifdef CONFIG_KERNEL_UINT8
#include "ref_region_uint8.c"
#endif

#ifdef __cplusplus
}
#endif

#endif
