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

/* 
refer to ncnn
https://github.com/Tencent/ncnn/blob/master/src/layer/arm/neon_mathfun_tanh.h
https://github.com/Tencent/ncnn/blob/master/src/layer/arm/neon_mathfun.h
*/

#include <arm_neon.h>


static inline float32x4_t div_ps(float32x4_t a, float32x4_t b)
{
#if __aarch64__
    return vdivq_f32(a, b);
#else
    float32x4_t reciprocal = vrecpeq_f32(b);
    reciprocal = vmulq_f32(vrecpsq_f32(b, reciprocal), reciprocal);
    // reciprocal = vmulq_f32(vrecpsq_f32(b, reciprocal), reciprocal);
    return vmulq_f32(a, reciprocal);
#endif
}

#define c_exp_hi 88.3762626647949f
#define c_exp_lo -88.3762626647949f

#define c_cephes_LOG2EF 1.44269504088896341
#define c_cephes_exp_C1 0.693359375
#define c_cephes_exp_C2 -2.12194440e-4

#define c_cephes_exp_p0 1.9875691500E-4
#define c_cephes_exp_p1 1.3981999507E-3
#define c_cephes_exp_p2 8.3334519073E-3
#define c_cephes_exp_p3 4.1665795894E-2
#define c_cephes_exp_p4 1.6666665459E-1
#define c_cephes_exp_p5 5.0000001201E-1

/* exp() computed for 4 float at once */
static inline float32x4_t exp_ps(float32x4_t x)
{
    float32x4_t tmp, fx;

    float32x4_t one = vdupq_n_f32(1);
    x = vminq_f32(x, vdupq_n_f32(c_exp_hi));
    x = vmaxq_f32(x, vdupq_n_f32(c_exp_lo));

    /* express exp(x) as exp(g + n*log(2)) */
    fx = vmlaq_f32(vdupq_n_f32(0.5f), x, vdupq_n_f32(c_cephes_LOG2EF));

    /* perform a floorf */
    tmp = vcvtq_f32_s32(vcvtq_s32_f32(fx));

    /* if greater, substract 1 */
    uint32x4_t mask = vcgtq_f32(tmp, fx);
    mask = vandq_u32(mask, vreinterpretq_u32_f32(one));

    fx = vsubq_f32(tmp, vreinterpretq_f32_u32(mask));

    tmp = vmulq_f32(fx, vdupq_n_f32(c_cephes_exp_C1));
    float32x4_t z = vmulq_f32(fx, vdupq_n_f32(c_cephes_exp_C2));
    x = vsubq_f32(x, tmp);
    x = vsubq_f32(x, z);

    static const float cephes_exp_p[6] = {c_cephes_exp_p0, c_cephes_exp_p1, c_cephes_exp_p2, c_cephes_exp_p3, c_cephes_exp_p4, c_cephes_exp_p5};
    float32x4_t y = vld1q_dup_f32(cephes_exp_p + 0);
    float32x4_t c1 = vld1q_dup_f32(cephes_exp_p + 1);
    float32x4_t c2 = vld1q_dup_f32(cephes_exp_p + 2);
    float32x4_t c3 = vld1q_dup_f32(cephes_exp_p + 3);
    float32x4_t c4 = vld1q_dup_f32(cephes_exp_p + 4);
    float32x4_t c5 = vld1q_dup_f32(cephes_exp_p + 5);

    y = vmulq_f32(y, x);
    z = vmulq_f32(x, x);

    y = vaddq_f32(y, c1);
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, c2);
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, c3);
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, c4);
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, c5);

    y = vmulq_f32(y, z);
    y = vaddq_f32(y, x);
    y = vaddq_f32(y, one);

    /* build 2^n */
    int32x4_t mm;
    mm = vcvtq_s32_f32(fx);
    mm = vaddq_s32(mm, vdupq_n_s32(0x7f));
    mm = vshlq_n_s32(mm, 23);
    float32x4_t pow2n = vreinterpretq_f32_s32(mm);

    y = vmulq_f32(y, pow2n);
    return y;
}

// tanh neon vector version
// refer the scalar version from Cephes Math Library

#define c_cephes_HALFMAXLOGF 44.014845935754205f
#define c_cephes_tanh_C1     0.625f

#define c_cephes_tanh_p0 -5.70498872745E-3
#define c_cephes_tanh_p1 +2.06390887954E-2
#define c_cephes_tanh_p2 -5.37397155531E-2
#define c_cephes_tanh_p3 +1.33314422036E-1
#define c_cephes_tanh_p4 -3.33332819422E-1

/* Single precision hyperbolic tangent computed for 4 simultaneous float */
static inline float32x4_t tanh_ps(float32x4_t x)
{
    float32x4_t x2 = vabsq_f32(x);

    uint32x4_t mask_l = vcgeq_f32(x2, vdupq_n_f32(c_cephes_tanh_C1));
    uint32x4_t mask_l2 = vcgtq_f32(x2, vdupq_n_f32(c_cephes_HALFMAXLOGF));

    // abs(x) >= 0.625
    // tanh(x) = 1 − 2 / (exp(2x) + 1)
    float32x4_t _one = vdupq_n_f32(1.f);
    float32x4_t _two = vdupq_n_f32(2.f);
    float32x4_t exp_x_x = exp_ps(vaddq_f32(x, x));
#if __aarch64__
    float32x4_t y0 = vsubq_f32(_one, vdivq_f32(_two, vaddq_f32(exp_x_x, _one)));
#else
    float32x4_t y0 = vsubq_f32(_one, div_ps(_two, vaddq_f32(exp_x_x, _one)));
#endif

    // abs(x) < 0.625
    /*
        z = x2 * x2;
        z =
        (((( -5.70498872745E-3 * z
        + 2.06390887954E-2) * z
        - 5.37397155531E-2) * z
        + 1.33314422036E-1) * z
        - 3.33332819422E-1) * z * x
        + x;
    */
    static const float cephes_tanh_p[5] = {c_cephes_tanh_p0, c_cephes_tanh_p1, c_cephes_tanh_p2, c_cephes_tanh_p3, c_cephes_tanh_p4};
    float32x4_t y = vld1q_dup_f32(cephes_tanh_p + 0);
    float32x4_t c1 = vld1q_dup_f32(cephes_tanh_p + 1);
    float32x4_t c2 = vld1q_dup_f32(cephes_tanh_p + 2);
    float32x4_t c3 = vld1q_dup_f32(cephes_tanh_p + 3);
    float32x4_t c4 = vld1q_dup_f32(cephes_tanh_p + 4);

    float32x4_t z = vmulq_f32(x, x);

    y = vmulq_f32(y, z);
    y = vaddq_f32(y, c1);
    y = vmulq_f32(y, z);
    y = vaddq_f32(y, c2);
    y = vmulq_f32(y, z);
    y = vaddq_f32(y, c3);
    y = vmulq_f32(y, z);
    y = vaddq_f32(y, c4);

    y = vmulq_f32(y, z);
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, x);

    // abs(x) > HALFMAXLOGF
    // return 1.0 or -1.0
    uint32x4_t mask_pos = vcgtq_f32(x, vdupq_n_f32(0.f));
    float32x4_t y1 = vreinterpretq_f32_u32(vbslq_u32(mask_pos, vreinterpretq_u32_f32(vdupq_n_f32(1.f)), vreinterpretq_u32_f32(vdupq_n_f32(-1.f))));

    y = vreinterpretq_f32_u32(vbslq_u32(mask_l, vreinterpretq_u32_f32(y0), vreinterpretq_u32_f32(y)));
    y = vreinterpretq_f32_u32(vbslq_u32(mask_l2, vreinterpretq_u32_f32(y1), vreinterpretq_u32_f32(y)));
    return y;
}


#define c_inv_mant_mask ~0x7f800000u
#define c_cephes_SQRTHF 0.707106781186547524
#define c_cephes_log_p0 7.0376836292E-2
#define c_cephes_log_p1 -1.1514610310E-1
#define c_cephes_log_p2 1.1676998740E-1
#define c_cephes_log_p3 -1.2420140846E-1
#define c_cephes_log_p4 +1.4249322787E-1
#define c_cephes_log_p5 -1.6668057665E-1
#define c_cephes_log_p6 +2.0000714765E-1
#define c_cephes_log_p7 -2.4999993993E-1
#define c_cephes_log_p8 +3.3333331174E-1
#define c_cephes_log_q1 -2.12194440e-4
#define c_cephes_log_q2 0.693359375

/* natural logarithm computed for 4 simultaneous float
 *   return NaN for x <= 0
 */
static inline float32x4_t log_ps(float32x4_t x)
{
    float32x4_t one = vdupq_n_f32(1);

    x = vmaxq_f32(x, vdupq_n_f32(0)); /* force flush to zero on denormal values */
    uint32x4_t invalid_mask = vcleq_f32(x, vdupq_n_f32(0));

    int32x4_t ux = vreinterpretq_s32_f32(x);

    int32x4_t emm0 = vshrq_n_s32(ux, 23);

    /* keep only the fractional part */
    ux = vandq_s32(ux, vdupq_n_s32(c_inv_mant_mask));
    ux = vorrq_s32(ux, vreinterpretq_s32_f32(vdupq_n_f32(0.5f)));
    x = vreinterpretq_f32_s32(ux);

    emm0 = vsubq_s32(emm0, vdupq_n_s32(0x7f));
    float32x4_t e = vcvtq_f32_s32(emm0);

    e = vaddq_f32(e, one);

    /* part2:
     *     if( x < SQRTHF ) {
     *       e -= 1;
     *       x = x + x - 1.0;
     *     } else { x = x - 1.0; }
     */
    uint32x4_t mask = vcltq_f32(x, vdupq_n_f32(c_cephes_SQRTHF));
    float32x4_t tmp = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(x), mask));
    x = vsubq_f32(x, one);
    e = vsubq_f32(e, vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(one), mask)));
    x = vaddq_f32(x, tmp);

    float32x4_t z = vmulq_f32(x, x);

    float32x4_t y = vdupq_n_f32(c_cephes_log_p0);
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, vdupq_n_f32(c_cephes_log_p1));
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, vdupq_n_f32(c_cephes_log_p2));
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, vdupq_n_f32(c_cephes_log_p3));
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, vdupq_n_f32(c_cephes_log_p4));
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, vdupq_n_f32(c_cephes_log_p5));
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, vdupq_n_f32(c_cephes_log_p6));
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, vdupq_n_f32(c_cephes_log_p7));
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, vdupq_n_f32(c_cephes_log_p8));
    y = vmulq_f32(y, x);

    y = vmulq_f32(y, z);

    tmp = vmulq_f32(e, vdupq_n_f32(c_cephes_log_q1));
    y = vaddq_f32(y, tmp);

    tmp = vmulq_f32(z, vdupq_n_f32(0.5f));
    y = vsubq_f32(y, tmp);

    tmp = vmulq_f32(e, vdupq_n_f32(c_cephes_log_q2));
    x = vaddq_f32(x, y);
    x = vaddq_f32(x, tmp);
    x = vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(x), invalid_mask)); // negative arg will be NAN
    return x;
}
