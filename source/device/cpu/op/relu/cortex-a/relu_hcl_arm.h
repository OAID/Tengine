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

#ifndef _RELU_KERNEL_ARM_H_
#define _RELU_KERNEL_ARM_H_

#include "relu_param.h"

#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"

#include <arm_neon.h>

static int perf_relu_fp32(struct tensor* input_tensor, struct tensor* output_tensor, float negative_slope,
                          int num_thread)
{
    int batch = input_tensor->dims[0] ? input_tensor->dims[0] : 1;
    int channels = input_tensor->dims[1] ? input_tensor->dims[1] : 1;
    int h = output_tensor->dims[2] ? output_tensor->dims[2] : 1;
    int w = input_tensor->dims[3] ? input_tensor->dims[3] : 1;

    int size = h * w;
    int c_step = h * w;
    int b_step = channels * h * w;

    float* input_data = (float*)input_tensor->data;
    float* out_data = (float*)output_tensor->data;

    if (negative_slope == 0)
    {
        for (int n = 0; n < batch; n++)
        {
            float* input = input_data + n * b_step;
            float* output = out_data + n * b_step;
#pragma omp parallel for num_threads(num_thread)
            for (int q = 0; q < channels; q++)
            {
                float* src = input + c_step * q;
                float* dst = output + c_step * q;

#if __ARM_NEON
                int nn = size >> 2;
                int remain = size - (nn << 2);
#else
                int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
                float32x4_t _zero = vdupq_n_f32(0.f);
                for (; nn > 0; nn--)
                {
                    float32x4_t _p = vld1q_f32(src);
                    _p = vmaxq_f32(_p, _zero);
                    vst1q_f32(dst, _p);

                    src += 4;
                    dst += 4;
                }
#endif
                for (; remain > 0; remain--)
                {
                    if (src[0] < 0)
                        dst[0] = 0;
                    else
                        dst[0] = src[0];

                    src++;
                    dst++;
                }
            }
        }
    }
    else
    {
        for (int n = 0; n < batch; n++)
        {
            float* input = input_data + n * b_step;
            float* output = out_data + n * b_step;
#pragma omp parallel for num_threads(num_thread)
            for (int q = 0; q < channels; q++)
            {
                float* src = input + c_step * q;
                float* dst = output + c_step * q;

#if __ARM_NEON
                int nn = size >> 2;
                int remain = size - (nn << 2);
#else
                int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
                float32x4_t _zero = vdupq_n_f32(0.f);
                float32x4_t _slope = vdupq_n_f32(negative_slope);
                for (; nn > 0; nn--)
                {
                    float32x4_t _p = vld1q_f32(src);
                    uint32x4_t _lemask = vcleq_f32(_p, _zero);
                    float32x4_t _ps = vmulq_f32(_p, _slope);
                    _p = vbslq_f32(_lemask, _ps, _p);
                    vst1q_f32(dst, _p);

                    src += 4;
                    dst += 4;
                }
#endif
                for (; remain > 0; remain--)
                {
                    if (src[0] < 0)
                        dst[0] = src[0] * negative_slope;
                    else
                        dst[0] = src[0];

                    src++;
                    dst++;
                }
            }
        }
    }

    return 0;
}

#endif
