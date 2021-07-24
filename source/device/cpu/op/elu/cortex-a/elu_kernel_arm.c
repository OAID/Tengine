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

#include "elu_kernel_arm.h"

#include "neon_mathfun.h"

#include <math.h>

#include <arm_neon.h>

static void elu_kernel(int i, int id, void* data, const float* input, float* output, float alpha)
{
    int elem_num = ((int*)data)[0];
    float32x4_t _one = vdupq_n_f32(1.f);
    float32x4_t _zero = vdupq_n_f32(0.f);
    float32x4_t _alpha = vdupq_n_f32(alpha);
    const float* cur_input = input + id * elem_num;
    float* cur_output = output + id * elem_num;
    for (int i = 0; i < (elem_num & -4); i += 4)
    {
        float32x4_t _p = vld1q_f32(cur_input);
        uint32x4_t _lemask = vcleq_f32(_p, _zero);

        float32x4_t _nps = exp_ps(_p);
        _nps = vsubq_f32(_nps, _one);
        _nps = vmulq_f32(_nps, _alpha);

        _p = vbslq_f32(_lemask, _nps, _p);
        vst1q_f32(cur_output, _p);
        cur_input += 4;
        cur_output += 4;
    }
    for (int i = elem_num & ~3; i < elem_num; i++)
    {
        if (*cur_input < 0.f)
            *cur_output = (exp(*cur_input) - 1.f) * alpha;
        else
            *cur_output = *cur_input;
        cur_input++;
        cur_output++;
    }
}

int elu_run(struct tensor* output_tensor, struct tensor* input_tensor, struct elu_param* elu_param,
            int num_thread)
{
    float* data = (float*)input_tensor->data;
    float* out_data = (float*)output_tensor->data;
    float alpha = elu_param->alpha;

    int chan_num = (input_tensor->dims[0]) * (input_tensor->dims[1]);
    int chan_size = (input_tensor->dims[2]) * (input_tensor->dims[3]);

#pragma omp parallel for num_threads(num_thread)
    for (int i = 0; i < chan_num; i++)
    {
        int offset = i * chan_size;
        elu_kernel(0, 0, &chan_size, data + offset, out_data + offset, alpha);
    }

    return 0;
}
