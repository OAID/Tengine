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
 * Author: haitao@openailab.com
 */

#ifndef __REF_SOFTMAX_OP_KERNEL_H__
#define __REF_SOFTMAX_OP_KERNEL_H__

#include <stdint.h>
#include <string.h>
#include <math.h>
#include "compiler_fp16.h"

#ifdef __cplusplus
extern "C" {
#endif

struct op_data
{
    int out_size;
    int in_size;
    int on_size;
    int i_zero;
    float i_scale;
    int o_zero;
    float o_scale;
};

static void GetMaxArray(float* input, float* array, int in_size, int on_size)
{
    float* input_ptr = ( float* )input;
    float* array_ptr = ( float* )array;
    memcpy(array_ptr,input_ptr,in_size* sizeof(float));
    for(int j = 0; j < on_size; j++)
        for(int l = 0; l < in_size; l++)
        {
            if(array_ptr[l] < input_ptr[j * in_size + l])
                array_ptr[l] = input_ptr[j * in_size + l];

        }
}

static void GetOutResult(float* input, float* output, float* array, float* sum_array, int in_size, int on_size)
{
    float* input_ptr = ( float* )input;
    float* output_ptr = ( float* )output;
    float* array_ptr = ( float* )array;
    float* sum_array_ptr = ( float* )sum_array;

    memset(sum_array, 0x0, in_size * sizeof(float));

    /* get the exp and the summary */

    for(int j = 0; j < on_size; j++)
    {
       
        for(int l = 0; l < in_size; l++)
        {
            int index = j * in_size + l;
            output_ptr[index] = exp(input_ptr[index] - array_ptr[l]);
            sum_array_ptr[l] += output_ptr[index];
           
        }
    }
     
    /* the final result */
    for(int j = 0; j < on_size; j++)
    {
        for(int l = 0; l < in_size; l++)
        {
            int index = j * in_size + l;
            output_ptr[index] /= sum_array_ptr[l];
        }
    }
}

typedef int (*ref_softmax_kernel_t)(void* input, void* output, void* max_array, void* sum_array, op_data* op_param);

#ifdef CONFIG_KERNEL_FP32
#include "ref_softmax_fp32.c"
#endif

#ifdef CONFIG_KERNEL_FP16
#include "ref_softmax_fp16.c"
#endif

#ifdef CONFIG_KERNEL_INT8
#include "ref_softmax_int8.c"
#endif

#ifdef CONFIG_KERNEL_UINT8
#include "ref_softmax_uint8.c"
#endif

#ifdef __cplusplus
}
#endif

#endif
