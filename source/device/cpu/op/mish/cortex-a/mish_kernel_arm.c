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
 * Author: 942002795@qq.com
 */

#include "mish_kernel_arm.h"

#include "mish_math_func.h"

#include <math.h>

#include <arm_neon.h>

static void mish_kernel(int i, int id, void* data, const float* input, float* output)
{
    int step = ((int*)data)[0];
    const float* cur_input = input + id * step;
    float* cur_output = output + id * step;
    for (int i = 0; i < (step & -4); i += 4)
    {
        float32x4_t _input = vld1q_f32(cur_input);
        float32x4_t out = vmulq_f32(_input, tanh_ps(log_ps(vaddq_f32(exp_ps(_input), vdupq_n_f32(1.f)))));
        vst1q_f32(cur_output, out);
        cur_input += 4;
        cur_output += 4;
    }
    for (int i = step & ~3; i < step; i++)
    {
        float tmp = *input++;
        *cur_output++ = tanh(log(exp(tmp) + 1.f));
    }
}

int mish_run(struct tensor* output_tensor, struct tensor* input_tensor, int num_thread)
{
    float* data = (float*)input_tensor->data;
    float* out_data = (float*)output_tensor->data;

    int chan_num = (input_tensor->dims[0]) * (input_tensor->dims[1]);
    int chan_size = (input_tensor->dims[2]) * (input_tensor->dims[3]);

#pragma omp parallel for num_threads(num_thread)
    for (int i = 0; i < chan_num; i++)
    {
        int offset = i * chan_size;
        mish_kernel(0, 0, &chan_size, data + offset, out_data + offset);
    }
    return 0;
}
