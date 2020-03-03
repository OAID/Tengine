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
 * Author: bzhang@openailab.com
 */

#ifndef __REF_PSROIPOOLING_KERNEL_H__
#define __REF_PSROIPOOLING_KERNEL_H__

#include <stdint.h>
#include <math.h>

#include "compiler_fp16.h"

#ifdef __cplusplus
extern "C" {
#endif

struct psroipooling_ref_param
{
    int out_h;
    int out_w;
    int channel;
    int num_rois;
    int in_h;
    int in_w;
    float spatial_scale;
    float feat_scale;
    int feat_zero;
    float roi_scale;
    int roi_zero;
    float out_scale;
    int out_zero;
    int output_dim;
};
#define T_MAX(a, b) ((a) > (b) ? (a) : (b))
#define T_MIN(a, b) ((a) < (b) ? (a) : (b))
typedef int (*ref_psroipooling_kernel_t)(void* featmap, void* roi, void* output, psroipooling_ref_param* param);

#ifdef CONFIG_KERNEL_FP32
#include "ref_psroipooling_fp32.c"
#endif

#ifdef CONFIG_KERNEL_FP16
#include "ref_psroipooling_fp16.c"
#endif

#ifdef CONFIG_KERNEL_INT8
#include "ref_psroipooling_int8.c"
#endif

#ifdef CONFIG_KERNEL_UINT8
#include "ref_psroipooling_uint8.c"
#endif

#ifdef __cplusplus
}
#endif

#endif
