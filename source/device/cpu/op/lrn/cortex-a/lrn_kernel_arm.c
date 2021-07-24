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

#include "lrn_kernel_arm.h"

#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "utility/sys_port.h"
#include "utility/float.h"
#include "utility/log.h"
#include "device/cpu/cpu_node.h"
#include "device/cpu/cpu_graph.h"
#include "device/cpu/cpu_module.h"

#include <math.h>

#include <arm_neon.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

static struct tab exp_tab;
static struct tab log_tab;

static void init_tab(void)
{
    /* Exponent polynomial coefficients */
    exp_tab.a0 = vdupq_n_f32(1.f);
    exp_tab.a1 = vdupq_n_f32(0.0416598916054f);
    exp_tab.a2 = vdupq_n_f32(0.500000596046f);
    exp_tab.a3 = vdupq_n_f32(0.0014122662833f);
    exp_tab.a4 = vdupq_n_f32(1.00000011921f);
    exp_tab.a5 = vdupq_n_f32(0.00833693705499f);
    exp_tab.a6 = vdupq_n_f32(0.166665703058f);
    exp_tab.a7 = vdupq_n_f32(0.000195780929062f);

    /* Logarithm polynomial coefficients */
    log_tab.a0 = vdupq_n_f32(-2.29561495781f);
    log_tab.a1 = vdupq_n_f32(-2.47071170807f);
    log_tab.a2 = vdupq_n_f32(-5.68692588806f);
    log_tab.a3 = vdupq_n_f32(-0.165253549814f);
    log_tab.a4 = vdupq_n_f32(5.17591238022f);
    log_tab.a5 = vdupq_n_f32(0.844007015228f);
    log_tab.a6 = vdupq_n_f32(4.58445882797f);
    log_tab.a7 = vdupq_n_f32(0.0141278216615f);
}

static inline float32x4_t vfloorq_f32(float32x4_t val)
{
    const float32x4_t CONST_1 = vdupq_n_f32(1.f);

    const int32x4_t z = vcvtq_s32_f32(val);
    const float32x4_t r = vcvtq_f32_s32(z);

    return vbslq_f32(vcgtq_f32(r, val), vsubq_f32(r, CONST_1), r);
}

static inline float32x2_t vinvsqrt_f32(float32x2_t x)
{
    float32x2_t sqrt_reciprocal = vrsqrte_f32(x);
    sqrt_reciprocal = vmul_f32(vrsqrts_f32(vmul_f32(x, sqrt_reciprocal), sqrt_reciprocal), sqrt_reciprocal);
    sqrt_reciprocal = vmul_f32(vrsqrts_f32(vmul_f32(x, sqrt_reciprocal), sqrt_reciprocal), sqrt_reciprocal);

    return sqrt_reciprocal;
}

static inline float32x4_t vinvsqrtq_f32(float32x4_t x)
{
    float32x4_t sqrt_reciprocal = vrsqrteq_f32(x);
    sqrt_reciprocal = vmulq_f32(vrsqrtsq_f32(vmulq_f32(x, sqrt_reciprocal), sqrt_reciprocal), sqrt_reciprocal);
    sqrt_reciprocal = vmulq_f32(vrsqrtsq_f32(vmulq_f32(x, sqrt_reciprocal), sqrt_reciprocal), sqrt_reciprocal);

    return sqrt_reciprocal;
}

static inline float32x2_t vinv_f32(float32x2_t x)
{
    float32x2_t recip = vrecpe_f32(x);
    recip = vmul_f32(vrecps_f32(x, recip), recip);
    recip = vmul_f32(vrecps_f32(x, recip), recip);
    return recip;
}

static inline float32x4_t vinvq_f32(float32x4_t x)
{
    float32x4_t recip = vrecpeq_f32(x);
    recip = vmulq_f32(vrecpsq_f32(x, recip), recip);
    recip = vmulq_f32(vrecpsq_f32(x, recip), recip);
    return recip;
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

static inline float32x4_t vlogq_f32(float32x4_t x)
{
    const int32x4_t CONST_127 = vdupq_n_s32(127);             // 127
    const float32x4_t CONST_LN2 = vdupq_n_f32(0.6931471805f); // ln(2)

    // Extract exponent
    int32x4_t m = vsubq_s32(vreinterpretq_s32_u32(vshrq_n_u32(vreinterpretq_u32_f32(x), 23)), CONST_127);
    float32x4_t val = vreinterpretq_f32_s32(vsubq_s32(vreinterpretq_s32_f32(x), vshlq_n_s32(m, 23)));

    // Polynomial Approximation
    float32x4_t poly = vtaylor_polyq_f32(val, &log_tab);

    // Reconstruct
    poly = vmlaq_f32(poly, vcvtq_f32_s32(m), CONST_LN2);

    return poly;
}

static inline float32x4_t vtanhq_f32(float32x4_t val)
{
    const float32x4_t CONST_1 = vdupq_n_f32(1.f);
    const float32x4_t CONST_2 = vdupq_n_f32(2.f);
    const float32x4_t CONST_MIN_TANH = vdupq_n_f32(-10.f);
    const float32x4_t CONST_MAX_TANH = vdupq_n_f32(10.f);

    float32x4_t x = vminq_f32(vmaxq_f32(val, CONST_MIN_TANH), CONST_MAX_TANH);
    float32x4_t exp2x = vexpq_f32(vmulq_f32(CONST_2, x));
    float32x4_t num = vsubq_f32(exp2x, CONST_1);
    float32x4_t den = vaddq_f32(exp2x, CONST_1);
    float32x4_t tanh = vmulq_f32(num, vinvq_f32(den));
    return tanh;
}

static inline float32x4_t vpowq_f32(float32x4_t val, float32x4_t n)
{
    return vexpq_f32(vmulq_f32(n, vlogq_f32(val)));
}

static void lrn_kernel(int i, int id, void* data, const float* input, float* output, float* square, float alpha,
                       float beta, float bias, int local_size, int channel_size, int channel_num, int num_thread)
{
    int step = ((int*)data)[0];
    const float32x4_t alpha_vec = vdupq_n_f32(alpha / local_size);
    const float32x4_t beta_vec = vdupq_n_f32(beta);
    const float32x4_t bias_vec = vdupq_n_f32(bias);
    int mod = channel_size / 4;
    int start_c = step * id;
    int end_c = step * id + step;

    //    #pragma omp parallel for num_threads(num_thread)
    for (int j = start_c; j < end_c; j++)
    {
        int c_start = j - local_size / 2;
        int c_end = j + local_size / 2;

        c_start = MAX(0, c_start);
        c_end = MIN(c_end, channel_num - 1);

        const float* cur_input = input + j * channel_size;
        float* cur_output = output + j * channel_size;
        for (int m = 0; m < mod; m++)
        {
            float32x4_t accu = vdupq_n_f32(0.f);

            for (int l = c_start; l <= c_end; l++)
            {
                accu = vaddq_f32(accu, vld1q_f32(square + l * channel_size + m * 4));
            }
            const float32x4_t normalized = vpowq_f32(vmlaq_f32(bias_vec, alpha_vec, accu), beta_vec);
            const float32x4_t normalized_pixel = vmulq_f32(vld1q_f32(cur_input), vinvq_f32(normalized));
            vst1q_f32(cur_output, normalized_pixel);
            cur_input += 4;
            cur_output += 4;
        }

        float alpha_over_size = alpha / local_size;

        for (int m = 4 * mod; m < channel_size; m++)
        {
            float sum = 0;
            for (int l = c_start; l <= c_end; l++)
            {
                sum = sum + square[l * channel_size + m];
            }
            *cur_output++ = *cur_input++ * pow(bias + alpha_over_size * sum, -beta);
        }
    }
}

int lrn_run(struct tensor* output_tensor, struct tensor* input_tensor, struct lrn_param* lrn_param,
            int num_thread)
{
    init_tab();
    const float* input = (float*)input_tensor->data;
    float* output = (float*)output_tensor->data;
    float* square = (float*)(malloc(input_tensor->elem_num * sizeof(float)));

    int n = input_tensor->dims[0];
    int c = input_tensor->dims[1];
    int h = input_tensor->dims[2];
    int w = input_tensor->dims[3];

    int img_size = c * h * w;
    int channel_size = h * w;
    float alpha = lrn_param->alpha;
    float beta = lrn_param->beta;
    float bias = lrn_param->k;
    int local_size = lrn_param->local_size;

    for (int i = 0; i < n; i++)
    {
        /* get square value */
        const float* img_base = input + i * img_size;
        float* out_base = output + i * img_size;

        int j = 0;
        for (; j < (img_size & -4); j += 4)
        {
            float32x4_t in = vld1q_f32(img_base + j);
            in = vmulq_f32(in, in);
            vst1q_f32(square + j, in);
        }
        for (; j < img_size; j++)
            square[j] = img_base[j] * img_base[j];

        if (lrn_param->norm_region != 0)
        {
            sys_free(square);
            TLOG_ERR("LRN: Only support across channels\n");
            return -1;
        }

        lrn_kernel(0, 0, &c, img_base, out_base, square, alpha, beta, bias, local_size, channel_size, c, num_thread);
    }

    free(square);

    return 0;
}
