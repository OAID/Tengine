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

#include "sigmoid_kernel_arm.h"

#include <math.h>

#include <arm_neon.h>

#define SIGMOID_MAX(a, b) ((a) > (b) ? (a) : (b))
#define SIGMOID_MIN(a, b) ((a) < (b) ? (a) : (b))

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

static float fast_exp1(float x)
{
    volatile union
    {
        float f;
        unsigned int i;
    } cvt;

    /* exp(x) = 2^i * 2^f; i = floor (log2(e) * x), 0 <= f <= 1 */
    float t = x * 1.442695041f;
    float fi = floorf(t);
    float f = t - fi;
    int i = (int)fi;
    cvt.f = (0.3371894346f * f + 0.657636276f) * f + 1.00172476f; /* compute 2^f */
    cvt.i += (i << 23);                                           /* scale by 2^i */
    return cvt.f;
}

static float acl_exp(float x)
{
    volatile union
    {
        float f;
        unsigned int i;
    } cvt;

    /* exp(x) = = 2^k * exp(x-k ln2); k = round（x/ln2）*/
    float t = x * 1.4426950408f;
    float f = x - ((int)t) * 0.6931471805f;
    int i = (int)t;
    /// cvt.f = (0.3371894346f * f + 0.657636276f) * f + 1.00172476f;       /* compute 2^f */
    cvt.f = 1 + f * 1.00000011921f + (0.0416598916054f + f * 0.00833693705499f) * f * f + ((0.500000596046f + f * 0.166665703058f) + (0.0014122662833f + f * 0.000195780929062f) * f * f) * f * f * f * f;
    cvt.i += (i << 23); /* scale by 2^i */
    return cvt.f;
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

static struct tab exp_tab;

static void init_tab(void)
{
    exp_tab.a0 = vdupq_n_f32(1.f);
    exp_tab.a1 = vdupq_n_f32(0.0416598916054f);
    exp_tab.a2 = vdupq_n_f32(0.500000596046f);
    exp_tab.a3 = vdupq_n_f32(0.0014122662833f);
    exp_tab.a4 = vdupq_n_f32(1.00000011921f);
    exp_tab.a5 = vdupq_n_f32(0.00833693705499f);
    exp_tab.a6 = vdupq_n_f32(0.166665703058f);
    exp_tab.a7 = vdupq_n_f32(0.000195780929062f);
}

static inline float32x4_t vtaylor_polyq_f32(float32x4_t x, struct tab* coeffs)
{
    float32x4_t A = vmlaq_f32(coeffs->a0, coeffs->a4, x);
    float32x4_t B = vmlaq_f32(coeffs->a2, coeffs->a6, x);
    float32x4_t C = vmlaq_f32(coeffs->a1, coeffs->a5, x);
    float32x4_t D = vmlaq_f32(coeffs->a3, coeffs->a7, x);
    float32x4_t x2 = vmulq_f32(x, x);
    float32x4_t x4 = vmulq_f32(x2, x2);
    float32x4_t res = vmlaq_f32(vmlaq_f32(A, B, x2), vmlaq_f32(C, D, x2), x4);
    return res;
}
/* ACL exp function impelement */
static inline float32x4_t vexpq_f32(float32x4_t x)
{
    const float32x4_t CONST_LN2 = vdupq_n_f32(0.6931471805f);     // ln(2)
    const float32x4_t CONST_INV_LN2 = vdupq_n_f32(1.4426950408f); // 1/ln(2)
    const float32x4_t CONST_0 = vdupq_n_f32(0.f);
    const int32x4_t CONST_NEGATIVE_126 = vdupq_n_s32(-126);

    // Perform range reduction [-log(2),log(2)]
    int32x4_t m = vcvtq_s32_f32(vmulq_f32(x, CONST_INV_LN2));
    float32x4_t val = vmlsq_f32(x, vcvtq_f32_s32(m), CONST_LN2);

    // Polynomial Approximation
    float32x4_t poly = vtaylor_polyq_f32(val, &exp_tab);

    // Reconstruct
    poly = vreinterpretq_f32_s32(vqaddq_s32(vreinterpretq_s32_f32(poly), vqshlq_n_s32(m, 23)));
    poly = vbslq_f32(vcltq_s32(m, CONST_NEGATIVE_126), CONST_0, poly);

    return poly;
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

int sigmoid_run(struct tensor* output_tensor, struct tensor* input_tensor, int num_thread)
{
    init_tab();
    float* input = (float*)input_tensor->data;
    float* output = (float*)output_tensor->data;

    float32x4_t min = vdupq_n_f32(-30.0f);
    float32x4_t max = vdupq_n_f32(30.0f);
    float32x4_t tmp_vec = vdupq_n_f32(1);

    int chan_num = input_tensor->dims[0] * input_tensor->dims[1];
    int chan_size = input_tensor->dims[2] * input_tensor->dims[3];

#pragma omp parallel for num_threads(num_thread)
    for (int j = 0; j < chan_num; j++)
    {
        float* pinput = input + j * chan_size;
        float* poutput = output + j * chan_size;

        for (int i = 0; i < (chan_size & -4); i += 4)
        {
            float32x4_t _input = vld1q_f32(pinput + i);
            _input = vmaxq_f32(_input, min);
            _input = vminq_f32(_input, max);
            float32x4_t tmp_exp = vaddq_f32(tmp_vec, vexpq10_f32(vmulq_n_f32(_input, -1.0f)));
            float32x4_t out = vrecpeq_f32(tmp_exp);
            out = vmulq_f32(vrecpsq_f32(tmp_exp, out), out);
            out = vmulq_f32(vrecpsq_f32(tmp_exp, out), out);
            vst1q_f32(poutput, out);
            poutput += 4;
        }
        for (int i = chan_size & ~3; i < chan_size; i++)
        {
            pinput[i] = SIGMOID_MIN(pinput[i], 30.0f);
            pinput[i] = SIGMOID_MAX(pinput[i], -30.0f);
            float tmp_exp = exp10_f32(-pinput[i]);
            *poutput++ = 1 / (1 + tmp_exp);
        }
    }

    return 0;
}
