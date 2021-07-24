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
 * Author: qtang@openailab.com
 */

#ifndef __ELTWISE_HCL_ARM_H__
#define __ELTWISE_HCL_ARM_H__

#include "eltwise_param.h"

#include "graph/tensor.h"

#include <arm_neon.h>

int perf_eltwise_fp32(struct tensor* output_tensor, struct tensor* input_tensor0, struct tensor* input_tensor1,
                      struct eltwise_param* eltwise_param, int num_thread)
{
    int batch = input_tensor0->dims[0] ? input_tensor0->dims[0] : 1;
    int channel = input_tensor0->dims[1] ? input_tensor0->dims[1] : 1;
    int in_h = input_tensor0->dims[2] ? input_tensor0->dims[2] : 1;
    int in_w = input_tensor0->dims[3] ? input_tensor0->dims[3] : 1;
    int c_step = in_h * in_w;
    int b_step = channel * in_h * in_w;

    for (int n = 0; n < batch; n++)
    {
        float* input0 = (float*)input_tensor0->data + n * b_step;
        float* input1 = (float*)input_tensor1->data + n * b_step;
        float* output = (float*)output_tensor->data + n * b_step;

#pragma omp parallel for num_threads(num_thread)
        for (int q = 0; q < channel; q++)
        {
            float* input0_data = input0 + q * c_step;
            float* input1_data = input1 + q * c_step;
            float* output_data = output + q * c_step;

#if __ARM_NEON
            int nn = c_step >> 2;
            int remain = c_step - (nn << 2);
#else
            int remain = c_step;
#endif

#if __ARM_NEON
            for (; nn > 0; nn--)
            {
                float32x4_t data0 = vld1q_f32(input0_data);
                float32x4_t data1 = vld1q_f32(input1_data);
                float32x4_t sum = vaddq_f32(data0, data1);
                vst1q_f32(output_data, sum);

                input0_data += 4;
                input1_data += 4;
                output_data += 4;
            }
#endif
            for (; remain > 0; remain--)
            {
                output_data[0] = input0_data[0] + input1_data[0];

                input0_data += 1;
                input1_data += 1;
                output_data += 1;
            }
        }
    }

    return 0;
}

#endif
