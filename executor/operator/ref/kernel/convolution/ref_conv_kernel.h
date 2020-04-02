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
 * Author: haoluo@openailab.com
 */

#ifndef __REF_CONV_KERNEL_H__
#define __REF_CONV_KERNEL_H__

#include <stdint.h>

#include "compiler_fp16.h"

#ifdef __cplusplus
extern "C" {
#endif

struct op_data
{
    int in_shape[3];    // NCHW
    int out_shape[3];    // CHW
    int kernels[2];
    int strides[2];
    int dilations[2];
    int pads[2];
    int batch;
    int group;
    int activation;
    int layout;
    int zero[3];    // input, kernel, output
    float scale[2];    // input output
    float* k_scale;
};

static inline float activation(float input, int activation)
{
    if(activation >= 0)
    {
        if(input < 0 && activation != 1)
            input = 0;
        if(activation == 1 && input > 1)
            input = 1;
        if(activation == 6 && input > 6)
            input = 6;
        if(activation == 1 && input < -1)
            input = -1;
    }

    return input;
}

typedef int (*ref_conv_kernel_t)(const void* input, void* output, const void* kernel, const void* bias, op_data* param);

#ifdef CONFIG_KERNEL_FP32
#include "ref_conv_fp32.c"
#endif

#ifdef CONFIG_KERNEL_FP16
#include "ref_conv_fp16.c"
#endif

#ifdef CONFIG_KERNEL_INT8
#include "ref_conv_int8.c"
#endif

#ifdef CONFIG_KERNEL_UINT8
#include "ref_conv_uint8.c"
#endif

#ifdef __cplusplus
}
#endif

#endif
