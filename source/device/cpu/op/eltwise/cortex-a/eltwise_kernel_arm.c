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
 * Author: haitao@openailab.com
 */

#include "eltwise_kernel_arm.h"

#include <math.h>
#include <stddef.h>

#include <arm_neon.h>

static int kernel_run(float* output, float* input0, float* input1, int type, int in_size0, int in_size1, int stride)
{
    float* out_ptr = output;
    float* in0 = input0;
    float* in1 = input1;
    int loop_time = 0;

    switch (type)
    {
    case ELT_SUM:
        if (in_size0 == 1)
        {
            float32x4_t data0 = vdupq_n_f32(in0[0]);
            for (int i = 0; i < in_size1; i = i + 4)
            {
                float32x4_t data1 = vld1q_f32(in1 + i);
                float32x4_t sum = vaddq_f32(data0, data1);
                vst1q_f32(out_ptr + i, sum);
            }
            loop_time = in_size1 / 4;
            for (int i = loop_time * 4; i < in_size1; i++)
            {
                out_ptr[i] = in1[i] + in0[0];
            }
        }
        else if (in_size1 == in_size0)
        {
            for (int i = 0; i < in_size1; i = i + 4)
            {
                float32x4_t data0 = vld1q_f32(in0 + i);
                float32x4_t data1 = vld1q_f32(in1 + i);
                float32x4_t sum = vaddq_f32(data0, data1);
                vst1q_f32(out_ptr + i, sum);
            }
            loop_time = in_size1 / 4;

            for (int i = loop_time * 4; i < in_size1; i++)
            {
                out_ptr[i] = in1[i] + in0[i];
            }
        }
        else if (in_size0 < in_size1 && in_size0 != 1)
        {
            for (int i = 0; i < in_size1; ++i)
            {
                *out_ptr++ = in1[i] + in0[i / stride];
            }
        }
        else
            return -1;
        break;

    default:
        break;
    }

    return 0;
}

int eltwise_run(struct tensor* output_tensor, struct tensor* input_tensor0, struct tensor* input_tensor1,
                struct eltwise_param* eltwise_param, int num_thread)
{
    // input
    float* input0 = (float*)input_tensor0->data;
    int in_size0 = input_tensor0->elem_num;

    float* input1 = NULL;
    int in_size1 = 0;

    if (input_tensor1)
    {
        input1 = (float*)input_tensor1->data;
        in_size1 = input_tensor1->elem_num;
    }

    struct tensor* input_tensor_tmp;
    int max_size = 0;
    int min_size = 0;
    float* main_data = NULL;
    float* scale_data = NULL;

    int batch_num = input_tensor0->dims[0];
    if (in_size0 >= in_size1)
    {
        input_tensor_tmp = input_tensor0;
        main_data = input0;
        scale_data = input1;
        max_size = in_size0 / batch_num;
        min_size = in_size1 / batch_num;
    }
    else
    {
        input_tensor_tmp = input_tensor1;
        main_data = input1;
        scale_data = input0;
        max_size = in_size1 / batch_num;
        min_size = in_size0 / batch_num;
    }

    // this version only support for input_num=2
    // int input_number=node->GetInputNum();

    // output
    float* output = (float*)output_tensor->data;
    int result = 0;

    int stride = input_tensor_tmp->dims[2] * input_tensor_tmp->dims[3];
    int channel = input_tensor_tmp->dims[1];

    //    #pragma omp parallel for num_threads(num_thread)
    for (int n = 0; n < batch_num; n++)
    {
        float* input_data0 = scale_data + n * in_size0;
        float* input_data1 = main_data + n * in_size1;

        // only support eltwise sum
        result = kernel_run(output, input_data0, input_data1, eltwise_param->type, min_size, max_size, stride);
    }
    return result;
}
