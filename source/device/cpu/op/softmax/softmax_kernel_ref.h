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
 * Copyright (c) 2021, OPEN AI LAB
 * Author:
 */

#ifndef __SOFTMAX_KERNEL_REF_H__
#define __SOFTMAX_KERNEL_REF_H__

#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"

#include <math.h>
#include <string.h>

static void GetMaxArray(void* input, void* array, int in_size, int on_size)
{
    float* input_ptr = (float*)input;
    float* array_ptr = (float*)array;

    memcpy(array_ptr, input_ptr, in_size * sizeof(float));

    for (int j = 0; j < on_size; j++)
    {
        for (int l = 0; l < in_size; l++)
        {
            if (array_ptr[l] < input_ptr[j * in_size + l])
                array_ptr[l] = input_ptr[j * in_size + l];
        }
    }
}

static void GetOutResult(void* input, void* output, void* array, void* sum_array, int in_size, int on_size)
{
    float* input_ptr = (float*)input;
    float* output_ptr = (float*)output;
    float* array_ptr = (float*)array;
    float* sum_array_ptr = (float*)sum_array;

    memset(sum_array, 0x0, in_size * sizeof(float));

    /* get the exp and the summary */
    for (int j = 0; j < on_size; j++)
    {
        for (int l = 0; l < in_size; l++)
        {
            int index = j * in_size + l;
            output_ptr[index] = exp(input_ptr[index] - array_ptr[l]);
            sum_array_ptr[l] += output_ptr[index];
        }
    }

    /* the final result */
    for (int j = 0; j < on_size; j++)
    {
        for (int l = 0; l < in_size; l++)
        {
            int index = j * in_size + l;
            output_ptr[index] /= sum_array_ptr[l];
        }
    }
}

int ref_softmax_fp32(struct tensor* input_tensor, struct tensor* output_tensor, int axis);

int ref_softmax_int8(struct tensor* input_tensor, struct tensor* output_tensor, int axis);

int ref_softmax_uint8(struct tensor* input_tensor, struct tensor* output_tensor, int axis);

#endif
