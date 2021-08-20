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

#include "interp_kernel_arm.h"

#include "utility/sys_port.h"

#include <math.h>

#include <arm_neon.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))

static void linear_coeffs(int w, int outw, int* xofs, float* alpha, int align_corner)
{
    double scale = (double)w / outw;
    if (align_corner)
    {
        scale = (double)(w - 1) / (outw - 1);
    }
    for (int dx = 0; dx < outw; dx++)
    {
        float fx = (float)((dx + 0.5) * scale - 0.5);
        if (align_corner)
        {
            fx = (float)(dx * scale);
        }

        int sx = floor(fx);
        fx -= sx;

        if (sx < 0)
        {
            sx = 0;
            fx = 0.f;
        }
        if (sx >= w - 1)
        {
            sx = w - 2;
            fx = 1.f;
        }

        xofs[dx] = sx;

        alpha[dx * 2] = 1.f - fx;
        alpha[dx * 2 + 1] = fx;
    }
}

static void resize_bilinear_image(float* src, float* dst, float* alpha, int* xofs, float* beta, int* yofs, int out_h,
                                  int out_w, int in_h, int in_w)
{
    int w = out_w; // dst.w;
    int h = out_h; // dst.h;

    // loop body
    float* rowsbuf0 = (float*)sys_malloc(w * sizeof(float));
    float* rowsbuf1 = (float*)sys_malloc(w * sizeof(float));
    float* rows0 = rowsbuf0;
    float* rows1 = rowsbuf1;

    int prev_sy1 = -2;

    for (int dy = 0; dy < h; dy++)
    {
        int sy = yofs[dy];

        if (sy == prev_sy1)
        {
            // reuse all rows
        }
        else if (sy == prev_sy1 + 1)
        {
            // hresize one row
            float* rows0_old = rows0;
            rows0 = rows1;
            rows1 = rows0_old;
            const float* S1 = src + (sy + 1) * in_w; // src.row(sy+1);

            const float* alphap = alpha;
            float* rows1p = rows1;
            // neon
            for (int dx = 0; dx + 1 < w; dx += 2)
            {
                int sx = xofs[dx];
                int sxn = xofs[dx + 1];
                const float* S1p = S1 + sx;
                const float* S1np = S1 + sxn;

                float32x4_t _a = vld1q_f32(alphap);
                float32x2_t _S1 = vld1_f32(S1p);
                float32x2_t _S1n = vld1_f32(S1np);

                float32x4_t _S1S1n = vcombine_f32(_S1, _S1n);
                float32x4_t _ms1 = vmulq_f32(_S1S1n, _a);
                float32x2_t _rows1 = vpadd_f32(vget_low_f32(_ms1), vget_high_f32(_ms1));

                vst1_f32(rows1p + dx, _rows1);

                alphap += 4;
            }
        }
        else
        {
            // hresize two rows
            const float* S0 = src + sy * in_w;       // src.row(sy);
            const float* S1 = src + (sy + 1) * in_w; // src.row(sy+1);

            const float* alphap = alpha;
            float* rows0p = rows0;
            float* rows1p = rows1;
            for (int dx = 0; dx + 1 < w; dx += 2)
            {
                int sx = xofs[dx];
                int sxn = xofs[dx + 1];
                const float* S0p = S0 + sx;
                const float* S1p = S1 + sx;
                const float* S0np = S0 + sxn;
                const float* S1np = S1 + sxn;

                float32x4_t _a = vld1q_f32(alphap);
                float32x2_t _S0 = vld1_f32(S0p);
                float32x2_t _S1 = vld1_f32(S1p);
                float32x2_t _S0n = vld1_f32(S0np);
                float32x2_t _S1n = vld1_f32(S1np);

                float32x4_t _S0S0n = vcombine_f32(_S0, _S0n);
                float32x4_t _S1S1n = vcombine_f32(_S1, _S1n);
                float32x4_t _ms0 = vmulq_f32(_S0S0n, _a);
                float32x4_t _ms1 = vmulq_f32(_S1S1n, _a);
                float32x2_t _rows0 = vpadd_f32(vget_low_f32(_ms0), vget_high_f32(_ms0));
                float32x2_t _rows1 = vpadd_f32(vget_low_f32(_ms1), vget_high_f32(_ms1));

                vst1_f32(rows0p + dx, _rows0);
                vst1_f32(rows1p + dx, _rows1);

                alphap += 4;
            }
        }

        prev_sy1 = sy;

        // vresize
        float b0 = beta[0];
        float b1 = beta[1];

        float* rows0p = rows0;
        float* rows1p = rows1;
        float* Dp = dst + dy * out_w; // dst.row(dy);

        int nn = w >> 3;
        int remain = w - (nn << 3);
        float32x4_t _b0 = vdupq_n_f32(b0);
        float32x4_t _b1 = vdupq_n_f32(b1);
        for (; nn > 0; nn--)
        {
            float32x4_t _rows0 = vld1q_f32(rows0p);
            float32x4_t _rows1 = vld1q_f32(rows1p);

            float32x4_t _D = vmulq_f32(_rows0, _b0);
            _D = vmlaq_f32(_D, _rows1, _b1);

            vst1q_f32(Dp, _D);

            float32x4_t _rows0n = vld1q_f32(rows0p + 4);
            float32x4_t _rows1n = vld1q_f32(rows1p + 4);

            float32x4_t _Dn = vmulq_f32(_rows0n, _b0);
            _Dn = vmlaq_f32(_Dn, _rows1n, _b1);

            vst1q_f32(Dp + 4, _Dn);

            Dp += 8;
            rows0p += 8;
            rows1p += 8;
        }
        for (; remain; --remain)
        {
            *Dp++ = *rows0p++ * b0 + *rows1p++ * b1;
        }

        beta += 2;
    }

    sys_free(rowsbuf0);
    sys_free(rowsbuf1);
}

static inline void interpolate_cubic(float fx, float* coeffs)
{
    const float A = -0.75f;

    float fx0 = fx + 1;
    float fx1 = fx;
    float fx2 = 1 - fx;

    coeffs[0] = A * fx0 * fx0 * fx0 - 5 * A * fx0 * fx0 + 8 * A * fx0 - 4 * A;
    coeffs[1] = (A + 2) * fx1 * fx1 * fx1 - (A + 3) * fx1 * fx1 + 1;
    coeffs[2] = (A + 2) * fx2 * fx2 * fx2 - (A + 3) * fx2 * fx2 + 1;
    coeffs[3] = 1.f - coeffs[0] - coeffs[1] - coeffs[2];
}
static void cubic_coeffs(int w, int outw, int* xofs, float* alpha)
{
    double scale = (double)w / outw;

    for (int dx = 0; dx < outw; dx++)
    {
        float fx = (float)((dx + 0.5) * scale - 0.5);
        int sx = floor(fx);
        fx -= sx;

        interpolate_cubic(fx, alpha + dx * 4);

        if (sx <= -1)
        {
            sx = 1;
            alpha[dx * 4 + 0] = 1.f - alpha[dx * 4 + 3];
            alpha[dx * 4 + 1] = alpha[dx * 4 + 3];
            alpha[dx * 4 + 2] = 0.f;
            alpha[dx * 4 + 3] = 0.f;
        }
        if (sx == 0)
        {
            sx = 1;
            alpha[dx * 4 + 0] = alpha[dx * 4 + 0] + alpha[dx * 4 + 1];
            alpha[dx * 4 + 1] = alpha[dx * 4 + 2];
            alpha[dx * 4 + 2] = alpha[dx * 4 + 3];
            alpha[dx * 4 + 3] = 0.f;
        }
        if (sx == w - 2)
        {
            sx = w - 3;
            alpha[dx * 4 + 3] = alpha[dx * 4 + 2] + alpha[dx * 4 + 3];
            alpha[dx * 4 + 2] = alpha[dx * 4 + 1];
            alpha[dx * 4 + 1] = alpha[dx * 4 + 0];
            alpha[dx * 4 + 0] = 0.f;
        }
        if (sx >= w - 1)
        {
            sx = w - 3;
            alpha[dx * 4 + 3] = 1.f - alpha[dx * 4 + 0];
            alpha[dx * 4 + 2] = alpha[dx * 4 + 0];
            alpha[dx * 4 + 1] = 0.f;
            alpha[dx * 4 + 0] = 0.f;
        }

        xofs[dx] = sx;
    }
}

static void resize_bicubic_image(float* src, float* dst, float* alpha, int* xofs, float* beta, int* yofs, int out_h,
                                 int out_w, int in_h, int in_w)
{
    int w = out_w; // dst.w;
    int h = out_h; // dst.h;

    // loop body
    float* rowsbuf0 = (float*)sys_malloc(w * sizeof(float));
    float* rowsbuf1 = (float*)sys_malloc(w * sizeof(float));
    float* rowsbuf2 = (float*)sys_malloc(w * sizeof(float));
    float* rowsbuf3 = (float*)sys_malloc(w * sizeof(float));
    float* rows0 = rowsbuf0;
    float* rows1 = rowsbuf1;
    float* rows2 = rowsbuf2;
    float* rows3 = rowsbuf3;

    int prev_sy1 = -3;

    for (int dy = 0; dy < h; dy++)
    {
        int sy = yofs[dy];

        if (sy == prev_sy1)
        {
            // reuse all rows
        }
        else if (sy == prev_sy1 + 1)
        {
            // hresize one row
            float* rows0_old = rows0;
            rows0 = rows1;
            rows1 = rows2;
            rows2 = rows3;
            rows3 = rows0_old;
            const float* S3 = src + (sy + 2) * in_w; // src.row(sy+2);

            const float* alphap = alpha;
            float* rows3p = rows3;
            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx];
                const float* S3p = S3 + sx;

                float a0 = alphap[0];
                float a1 = alphap[1];
                float a2 = alphap[2];
                float a3 = alphap[3];
                rows3p[dx] = S3p[-1] * a0 + S3p[0] * a1 + S3p[1] * a2 + S3p[2] * a3;

                alphap += 4;
            }
        }
        else if (sy == prev_sy1 + 2)
        {
            // hresize two rows
            float* rows0_old = rows0;
            float* rows1_old = rows1;
            rows0 = rows2;
            rows1 = rows3;
            rows2 = rows0_old;
            rows3 = rows1_old;
            const float* S2 = src + (sy + 1) * in_w; // src.row(sy+1);
            const float* S3 = src + (sy + 2) * in_w; // src.row(sy+2);

            const float* alphap = alpha;
            float* rows2p = rows2;
            float* rows3p = rows3;
            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx];
                const float* S2p = S2 + sx;
                const float* S3p = S3 + sx;

                float a0 = alphap[0];
                float a1 = alphap[1];
                float a2 = alphap[2];
                float a3 = alphap[3];
                rows2p[dx] = S2p[-1] * a0 + S2p[0] * a1 + S2p[1] * a2 + S2p[2] * a3;
                rows3p[dx] = S3p[-1] * a0 + S3p[0] * a1 + S3p[1] * a2 + S3p[2] * a3;

                alphap += 4;
            }
        }
        else if (sy == prev_sy1 + 3)
        {
            // hresize three rows
            float* rows0_old = rows0;
            float* rows1_old = rows1;
            float* rows2_old = rows2;
            rows0 = rows3;
            rows1 = rows0_old;
            rows2 = rows1_old;
            rows3 = rows2_old;
            const float* S1 = src + sy * in_w;       // src.row(sy);
            const float* S2 = src + (sy + 1) * in_w; // src.row(sy+1);
            const float* S3 = src + (sy + 2) * in_w; // src.row(sy+2);

            const float* alphap = alpha;
            float* rows1p = rows1;
            float* rows2p = rows2;
            float* rows3p = rows3;
            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx];
                const float* S1p = S1 + sx;
                const float* S2p = S2 + sx;
                const float* S3p = S3 + sx;

                float a0 = alphap[0];
                float a1 = alphap[1];
                float a2 = alphap[2];
                float a3 = alphap[3];
                rows1p[dx] = S1p[-1] * a0 + S1p[0] * a1 + S1p[1] * a2 + S1p[2] * a3;
                rows2p[dx] = S2p[-1] * a0 + S2p[0] * a1 + S2p[1] * a2 + S2p[2] * a3;
                rows3p[dx] = S3p[-1] * a0 + S3p[0] * a1 + S3p[1] * a2 + S3p[2] * a3;

                alphap += 4;
            }
        }
        else
        {
            // hresize four rows
            const float* S0 = src + (sy - 1) * in_w; // src.row(sy-1);
            const float* S1 = src + sy * in_w;       // src.row(sy);
            const float* S2 = src + (sy + 1) * in_w; // src.row(sy+1);
            const float* S3 = src + (sy + 2) * in_w; // src.row(sy+2);

            const float* alphap = alpha;
            float* rows0p = rows0;
            float* rows1p = rows1;
            float* rows2p = rows2;
            float* rows3p = rows3;
            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx];
                const float* S0p = S0 + sx;
                const float* S1p = S1 + sx;
                const float* S2p = S2 + sx;
                const float* S3p = S3 + sx;

                float a0 = alphap[0];
                float a1 = alphap[1];
                float a2 = alphap[2];
                float a3 = alphap[3];
                rows0p[dx] = S0p[-1] * a0 + S0p[0] * a1 + S0p[1] * a2 + S0p[2] * a3;
                rows1p[dx] = S1p[-1] * a0 + S1p[0] * a1 + S1p[1] * a2 + S1p[2] * a3;
                rows2p[dx] = S2p[-1] * a0 + S2p[0] * a1 + S2p[1] * a2 + S2p[2] * a3;
                rows3p[dx] = S3p[-1] * a0 + S3p[0] * a1 + S3p[1] * a2 + S3p[2] * a3;

                alphap += 4;
            }
        }

        prev_sy1 = sy;

        // vresize
        float b0 = beta[0];
        float b1 = beta[1];
        float b2 = beta[2];
        float b3 = beta[3];

        float* rows0p = rows0;
        float* rows1p = rows1;
        float* rows2p = rows2;
        float* rows3p = rows3;
        float* Dp = dst + dy * out_w; // dst.row(dy);
        for (int dx = 0; dx < w; dx++)
        {
            *Dp++ = *rows0p++ * b0 + *rows1p++ * b1 + *rows2p++ * b2 + *rows3p++ * b3;
        }

        beta += 4;
    }

    sys_free(rowsbuf0);
    sys_free(rowsbuf1);
    sys_free(rowsbuf2);
    sys_free(rowsbuf3);
}

int interp_run(struct tensor* output_tensor, struct tensor* input_tensor, struct interp_param* interp_param,
               int num_thread)
{
    int resize_type = interp_param->resize_type;
    int out_w = interp_param->output_width;
    int out_h = interp_param->output_height;
    float width_scale = interp_param->width_scale;
    float height_scale = interp_param->height_scale;

    int in_c = input_tensor->dims[1];
    int in_h = input_tensor->dims[2];
    int in_w = input_tensor->dims[3];

    float* data = (float*)input_tensor->data;
    float* out_data = (float*)output_tensor->data;

    if (out_h == 0 || out_w == 0)
    {
        out_h = in_h * height_scale;
        out_w = in_w * width_scale;
    }
    if (out_h == in_h && out_w == in_w)
    {
        out_data = data;
        return 0;
    }
    int out_channel_size = out_h * out_w;
    int in_channel_size = in_h * in_w;
    if (input_tensor->dim_num == 1)
    {
#pragma omp parallel for num_threads(num_thread)
        for (int q = 0; q < input_tensor->dims[0]; ++q)
        {
            for (int i = 0; i < out_h * out_w; i++)
            {
                out_data[q * out_h * out_w + i] = data[q];
            }
        }
        return 0;
    }

    if (resize_type == 1) // nearest
    {
#pragma omp parallel for num_threads(num_thread)
        for (int q = 0; q < in_c; q++)
        {
            for (int y = 0; y < out_h; ++y)
            {
                const int in_y = MIN((int)(y / height_scale), (in_h - 1));
                for (int x = 0; x < out_w; ++x)
                {
                    const int in_x = MIN((int)(x / width_scale), (in_w - 1));
                    out_data[out_w * y + x + out_w * out_h * q] = data[in_y * in_w + in_x + q * in_w * in_h];
                }
            }
        }
    }
    else if (resize_type == 2 || resize_type == 4) // bilinear
    {
        int* buf = (int*)sys_malloc((out_w + out_h + out_w * 2 + out_h * 2) * sizeof(int));

        int* xofs = buf;         // new int[ow];
        int* yofs = buf + out_w; // new int[oh];

        float* alpha = (float*)(buf + out_w + out_h);            // new float[ow * 2];
        float* beta = (float*)(buf + out_w + out_h + out_w * 2); // new float[oh * 2];

        int align_corner = interp_param->resize_type == 2 ? 0 : 1;
        linear_coeffs(in_w, out_w, xofs, alpha, align_corner);
        linear_coeffs(in_h, out_h, yofs, beta, align_corner);

#pragma omp parallel for num_threads(num_thread)
        for (int q = 0; q < in_c; ++q)
        {
            resize_bilinear_image(data + in_channel_size * q, out_data + out_channel_size * q, alpha, xofs, beta, yofs,
                                  out_h, out_w, in_h, in_w);
        }

        sys_free(buf);
    }
    else if (resize_type == 3) // bicubic
    {
        int* buf = (int*)sys_malloc((out_w + out_h + out_w * 4 + out_h * 4) * sizeof(int));

        int* xofs = buf;         // new int[ow];
        int* yofs = buf + out_w; // new int[oh];

        float* alpha = (float*)(buf + out_w + out_h);            // new float[ow * 4];
        float* beta = (float*)(buf + out_w + out_h + out_w * 4); // new float[oh * 4];

        cubic_coeffs(in_w, out_w, xofs, alpha);
        cubic_coeffs(in_h, out_h, yofs, beta);

#pragma omp parallel for num_threads(num_thread)
        for (int q = 0; q < in_c; q++)
        {
            resize_bicubic_image(data + in_channel_size * q, out_data + out_channel_size * q, alpha, xofs, beta, yofs,
                                 out_h, out_w, in_h, in_w);
        }

        sys_free(buf);

        return 0;
    }

    return 0;
}
