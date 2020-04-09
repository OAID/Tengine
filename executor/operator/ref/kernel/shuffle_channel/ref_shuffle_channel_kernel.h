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

#ifndef __REF_SHUFFLE_CHANNEL_H__
#define __REF_SHUFFLE_CHANNEL_H__

#include <stdint.h>
#include <string.h>
#include <math.h>
#include "compiler_fp16.h"
 
#ifdef __cplusplus
extern "C" {
#endif
struct shuffle_channel_param
{
    int group;
    int n;
    int c;
    int h;
    int w;
    size_t eletsize;
};

typedef int (*ref_shuffle_channel_t)(void* input, void* output, shuffle_channel_param* op_param);

static int ref_shuffle_channel_common(const char* in_data, char* out_data, shuffle_channel_param* op_param)
{
    int chs_per_group = op_param->c / op_param->group;
    for (int n = 0; n != op_param->n; n++)
    {
        for (int i = 0; i != op_param->group; i++)
        {
            for (int j = 0; j != chs_per_group; j++)
            {
                int src_q =n*op_param->c* op_param->h*op_param->w +(chs_per_group * i + j)*op_param->h*op_param->w*op_param->eletsize;
                int dst_q =n*op_param->c* op_param->h*op_param->w + (op_param->group * j + i)*op_param->h*op_param->w*op_param->eletsize;
                memcpy(out_data+dst_q, in_data+src_q, op_param->h*op_param->w*op_param->eletsize);
            }
        }
    }
                
    return 0;
}

#ifdef CONFIG_KERNEL_FP32
#include "ref_shuffle_channel_fp32.c"
#endif

#ifdef CONFIG_KERNEL_FP16
#include "ref_shuffle_channel_fp16.c"
#endif

#ifdef CONFIG_KERNEL_INT8
#include "ref_shuffle_channel_int8.c"
#endif

#ifdef CONFIG_KERNEL_UINT8
#include "ref_shuffle_channel_uint8.c"
#endif

#ifdef __cplusplus
}
#endif

#endif
