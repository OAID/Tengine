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

#ifndef __ELTWISE_KERNEL_H__
#define __ELTWISE_KERNEL_H__

#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include "compiler_fp16.h"

#ifdef __cplusplus
extern "C" {
#endif


struct eltwise_param;

struct eltwise_param
{
    float scale[3];
    int zero[3];
};

typedef int (*eltwise_t)(void* output, void* input0, void* input1, int type, int input_count4,
            int input_chan,int input_chan_1,int input_hw,int input_hw_1, int input1_count4,
            int input_h,int input_w,int input_h_1,int input_w_1,int input_n,int input_n_1,int layout,
            int out_size,float* output_buf,eltwise_param* param);


#ifdef CONFIG_KERNEL_FP32
#include "eltwise_fp32.c"
#endif

#ifdef CONFIG_KERNEL_FP16
#include "eltwise_fp16.c" 
#endif

#ifdef CONFIG_KERNEL_INT8
#include "eltwise_int8.c"
#endif

#ifdef CONFIG_KERNEL_UINT8
#include "eltwise_uint8.c"
#endif




#ifdef __cplusplus
}
#endif

#endif