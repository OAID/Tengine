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

#include "tanh_kernel_arm.h"

#include <math.h>

#include <arm_neon.h>

#define T_MAX(a, b) ((a) > (b) ? (a) : (b))
#define T_MIN(a, b) ((a) < (b) ? (a) : (b))

static inline float fast_exp(float x)
{
    union
    {
        uint32_t i;
        float f;
    } v;
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

static float exp10_f32(float x)
{
    x = 1.0 + x * 0.0009765625f;
    x *= x;
    x *= x;
    x *= x;
    x *= x;
    x *= x;
    x *= x;
    x *= x;
    x *= x;
    x *= x;
    x *= x;
    return x;
}
/*
exp(x) = lim(1+x/n)^n       // n=10
*/
static inline float32x4_t vexpq10_f32(float32x4_t x)
{
    x = vmlaq_n_f32(vdupq_n_f32(1.0f), x, 0.0009765625f); // n = 10
    x = vmulq_f32(x, x);
    x = vmulq_f32(x, x);
    x = vmulq_f32(x, x);
    x = vmulq_f32(x, x);
    x = vmulq_f32(x, x);
    x = vmulq_f32(x, x);
    x = vmulq_f32(x, x);
    x = vmulq_f32(x, x);
    x = vmulq_f32(x, x);
    x = vmulq_f32(x, x);
    return x;
}

static void tanh_kernel(int i, int id, void* data, const float* input, float* output)
{
    int step = ((int*)data)[0];
    float32x4_t min = vdupq_n_f32(-30.0f);
    float32x4_t max = vdupq_n_f32(30.0f);
    const float* cur_input = input + id * step;
    float* cur_output = output + id * step;
    for (int i = 0; i < (step & -4); i += 4)
    {
        float32x4_t _input = vld1q_f32(cur_input);
        _input = vmaxq_f32(_input, min);
        _input = vminq_f32(_input, max);
        /// float32x4_t positive_exp = vexpq10_f32(_input);
        /// float32x4_t negative_exp = vexpq10_f32(vmulq_n_f32(_input, -1.0f));
        float32x4_t denominator = vaddq_f32(vexpq10_f32(_input), vexpq10_f32(vmulq_n_f32(_input, -1.0f)));
        float32x4_t numerator = vsubq_f32(vexpq10_f32(_input), vexpq10_f32(vmulq_n_f32(_input, -1.0f)));

        float32x4_t tmp_recip = vrecpeq_f32(denominator);
        tmp_recip = vmulq_f32(vrecpsq_f32(denominator, tmp_recip), tmp_recip);
        tmp_recip = vmulq_f32(vrecpsq_f32(denominator, tmp_recip), tmp_recip);
        float32x4_t out = vmulq_f32(numerator, tmp_recip);
        vst1q_f32(cur_output, out);
        cur_input += 4;
        cur_output += 4;
    }
    for (int i = step & ~3; i < step; i++)
    {
        float tmp = *input++;
        tmp = T_MIN(tmp, 30.0f);
        tmp = T_MAX(tmp, -30.0f);
        *cur_output++ = (exp10_f32(tmp) - exp10_f32(-tmp)) / (exp10_f32(tmp) + exp10_f32(-tmp));
    }
}

int tanh_run(struct tensor* output_tensor, struct tensor* input_tensor, int num_thread)
{
    float* data = (float*)input_tensor->data;
    float* out_data = (float*)output_tensor->data;

    int chan_num = (input_tensor->dims[0]) * (input_tensor->dims[1]);
    int chan_size = (input_tensor->dims[2]) * (input_tensor->dims[3]);

#pragma omp parallel for num_threads(num_thread)
    for (int i = 0; i < chan_num; i++)
    {
        int offset = i * chan_size;
        tanh_kernel(0, 0, &chan_size, data + offset, out_data + offset);
    }

    return 0;
}
