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
 * Author: bhu@openailab.com
 */

#include "prelu_kernel_arm.h"

#include <arm_neon.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

int prelu_kernel_run(float* input, float* output, int dim0, int dim1, int dim2, int dim3, float* slope, int layout,
                     int num_thread)
{
    int channel_size = dim1 * dim2 * dim3;
    float32x4_t _zero = vdupq_n_f32(0.f);
    if (layout == 0)
    {
        int size = dim2 * dim3;
        for (int n = 0; n < dim0; n++)
        {
#pragma omp parallel for num_threads(num_thread)
            for (int c = 0; c < dim1; c++)
            {
                float* cur_input = input + n * channel_size + c * size;
                float* cur_output = output + n * channel_size + c * size;
                float32x4_t _slope = vdupq_n_f32(slope[c]);
                for (int k = 0; k < (size & -4); k += 4)
                {
                    float32x4_t _p = vld1q_f32(cur_input);
                    // ri = ai <= bi ? 1...1:0...0
                    uint32x4_t _lemask = vcleq_f32(_p, _zero);
                    float32x4_t _ps = vmulq_f32(_p, _slope);
                    // bitwise select
                    _p = vbslq_f32(_lemask, _ps, _p);
                    vst1q_f32(cur_output, _p);
                    cur_input += 4;
                    cur_output += 4;
                }
                for (int k = size & ~3; k < size; k++)
                {
                    *cur_output = MAX(cur_input[0], 0.f) + slope[c] * MIN(cur_input[0], 0.f);
                    cur_input++;
                    cur_output++;
                }
            }
        }
    }
    else
    {
        return -1;
    }

    return 0;
}
