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

#include "relu_kernel_arm.h"

#include <math.h>
#include <arm_neon.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

static inline int relu_kernel(const int i, const int id, const void* data, const float* input, float* output,
                              const float slope)
{
    float32x4_t _zero = vdupq_n_f32(0.f);
    int step = ((int*)data)[0];
    const float* cur_input = input + id * step;
    float* cur_output = output + id * step;
    if (slope == 0)
    {
        for (int l = 0; l < (step & -4); l += 4)
        {
            float32x4_t _p = vld1q_f32(cur_input);
            _p = vmaxq_f32(_p, _zero);
            vst1q_f32(cur_output, _p);
            cur_input += 4;
            cur_output += 4;
        }
        for (int i = step & ~3; i < step; i++)
        {
            *cur_output++ = MAX(*cur_input++, 0.f);
        }
    }
    else
    {
        float32x4_t _slope = vdupq_n_f32(slope);
        for (int l = 0; l < (step & -4); l += 4)
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
        for (int i = step & ~3; i < step; i++)
        {
            *cur_output++ = MAX(cur_input[0], 0.f) + slope * MIN(cur_input[0], 0.f);
            cur_input++;
        }
    }
    return 0;
}

int relu_arm_run(struct tensor* output_tensor, struct tensor* input_tensor, struct relu_param* relu_param,
                 int num_thread)
{
    float* data = (float*)input_tensor->data;
    float* out_data = (float*)output_tensor->data;
    float negativeslope = relu_param->negative_slope;

    int chan_num = input_tensor->dims[0] * input_tensor->dims[1];
    int chan_size = input_tensor->dims[2] * input_tensor->dims[3];

    //    #pragma omp parallel for num_threads(num_thread)
    for (int i = 0; i < chan_num; i++)
    {
        int offset = i * chan_size;
        relu_kernel(0, 0, &chan_size, data + offset, out_data + offset, negativeslope);
    }

    return 0;
}
