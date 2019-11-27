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
 * Author: zpluo@openailab.com
 */

#ifndef __REF_PAD_KERNEL_H__
#define __REF_PAD_KERNEL_H__

#include <stdint.h>
#include <math.h>
#include "compiler_fp16.h"

#ifdef __cplusplus
extern "C" {
#endif

struct pad_param;

struct pad_param
{
    int mode;
    float cv_f32;
    __fp16 cv_f16;
    int8_t cv_int8;
    uint8_t cv_uint8;
    int in_size;
    int out_size;
    int in_n;
    int in_h;
    int in_w;
    int in_c;
    int out_h;
    int out_w;
    int out_n;
    int pad_0_h;
    int pad_0_w;
    int pad_1_h;
    int pad_1_w;
    int pad_2_h;
    int pad_2_w;
    int pad_3_h;
    int pad_3_w;
    float scale[2];
    int zero[2];
};

typedef int (*ref_pad_t)(void* data, void* out_data, pad_param* param);

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

#ifdef CONFIG_KERNEL_FP32
#include "ref_pad_fp32.c"
#endif

#ifdef CONFIG_KERNEL_FP16
#include "ref_pad_fp16.c"
#endif

#ifdef CONFIG_KERNEL_INT8
#include "ref_pad_int8.c"
#endif

#ifdef CONFIG_KERNEL_UINT8
#include "ref_pad_uint8.c"
#endif

#ifdef __cplusplus
}
#endif

#endif