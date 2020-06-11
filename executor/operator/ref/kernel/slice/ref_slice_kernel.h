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

#ifndef __SLICE_KERNEL_H__
#define __SLICE_KERNEL_H__

#include <stdint.h>
#include <math.h>

#include "compiler_fp16.h"

#ifdef __cplusplus
extern "C" {
#endif

struct shape_dim
{
    int dims[4];    // for caffe
    int begins[4];    // for tf
    int sizes[4];    // for tf
};

struct slice_param
{
    int in_shape[4];    // the dim of the input
    int in_shape_3[3];
    int in_shape_2[2];
    struct shape_dim* output_shape;    // out shape
    int out_num;
    int dim_num;
    int axis;    // for caffe
    float out_scale;    // for input tensor int8
    bool iscaffe;
    bool ismxnet;
    bool isonnx;
    bool isncnn;
    int begin;
    int end;
};

typedef int (*ref_slice_t)(const int8_t* in_data, int8_t** out_data, const struct slice_param* param);

#include "ref_slice_common.c"
#ifdef CONFIG_KERNEL_FP32
#include "ref_slice_fp32.c"
#endif

#ifdef CONFIG_KERNEL_FP16
#include "ref_slice_fp16.c"
#endif

#ifdef CONFIG_KERNEL_INT8
#include "ref_slice_int8.c"
#endif

#ifdef CONFIG_KERNEL_UINT8
#include "ref_slice_uint8.c"
#endif

#ifdef __cplusplus
}
#endif

#endif
