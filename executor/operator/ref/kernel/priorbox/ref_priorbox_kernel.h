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
 * Author: ruizhang@openailab.com
 */

#ifndef __REF_PIORBOX_KERNEL_H__
#define __REF_PIORBOX_KERNEL_H__

#include <stdint.h>
#include <math.h>

#include "compiler_fp16.h"

#ifdef __cplusplus
extern "C" {
#endif

struct priorbox_ref_param
{
    int image_h;
    int image_w;
    float step_h;
    float step_w;
    int num_priors;
    float offset;
    int feature_h;
    int feature_w;
    int max_size_num;
    float* max_size;
    int min_size_num;
    float* min_size;
    int aspect_ratio_size;
    float* aspect_ratio;
    float* variance;
    int flip;
    int clip;
    int out_dim;
    int image_size;
    float in_scale[2];
    float out_scale;
    int out_zero;
};

typedef int (*ref_priorbox_kernel_t)(void* output, const priorbox_ref_param* param, int elem_size);

#ifdef CONFIG_KERNEL_FP32
#include "ref_priorbox_fp32.c"
#endif

#ifdef CONFIG_KERNEL_FP16
#include "ref_priorbox_fp16.c"
#endif

#ifdef CONFIG_KERNEL_INT8
#include "ref_priorbox_int8.c"
#endif

#ifdef CONFIG_KERNEL_UINT8
#include "ref_priorbox_uint8.c"
#endif

#ifdef __cplusplus
}
#endif

#endif
