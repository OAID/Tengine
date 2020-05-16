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
 * Parts of the following code in this file refs to
 * https://github.com/Tencent/ncnn/blob/master/src/layer/x86/convolution_3x3.h
 * BUG1989 is pleased to support the open source community by supporting ncnn available.
 *
 * Copyright (C) 2019 BUG1989. All rights reserved.
 *
 * Licensed under the BSD 3-Clause License (the "License"); you may not use this
 * file except in compliance with the License. You may obtain a copy of the License at
 *
 * https://opensource.org/licenses/BSD-3-Clause
 */

/*
 * Copyright (c) 2020, OPEN AI LAB
 * Author: qwang02@openailab.com
 */

#ifndef __CONVOLUTION_WINO_X86_H__
#define __CONVOLUTION_WINO_X86_H__

#include <stdlib.h>

#if __SSE2__
#include <emmintrin.h>
#endif
#if __AVX__
#include <immintrin.h>
#endif

static void pad_0_align_2D(float* dst, float* src, int m, int n, int m_align, int n_align, int pad_h, int pad_w)
{
    int i;
    if(n >= n_align && m >= m_align)
        return;
    for(i = 0; i < m; ++i)
    {
        memcpy(dst + (i + pad_h) * n_align + pad_w, src + i * n, n * sizeof(float));
    }
}

// pad 0 in right and down side on 3D
void pad_0_align_3D(float* dst, float* src, int m, int n, int m_align, int n_align, int c, int pad_h, int pad_w)
{
    int i;
    if(n >= n_align && m >= m_align)
        return;
    for(i = 0; i < c; ++i)
    {
        pad_0_align_2D(dst + i * m_align * n_align, src + i * m * n, m, n, m_align, n_align, pad_h, pad_w);
    }
}

static void delete_0_2D(float* dst, float* src, int m_align, int n_align, int m, int n, int pad_h, int pad_w)
{
    int i;
    if(n >= n_align && m >= m_align)
        return;
    for(i = 0; i < m; ++i)
    {
        memcpy(dst + i * n, src + (i + pad_h) * n_align + pad_w, n * sizeof(float));
    }
}

// pad 0 in right and down side on 3D
void delete_0_3D(float* dst, float* src, int m_align, int n_align, int m, int n, int c, int pad_h, int pad_w)
{
    int i;
    if(n >= n_align && m >= m_align)
        return;
    for(i = 0; i < c; ++i)
    {
        delete_0_2D(dst + i * m * n, src + i * m_align * n_align, m_align, n_align, m, n, pad_h, pad_w);
    }
}

void conv3x3s1_winograd43_sse(float* bottom_blob, float* top_blob, float* kernel_tm_test, float* dot_block,
                              float* transform_input, float* output_bordered, float* _bias, int w, int h, int inch,
                              int outw, int outh, int outch)
{
    size_t elemsize = sizeof(float);
    const float* bias = _bias;

    // pad to 4n+2, winograd F(4,3)
    float* bottom_blob_bordered = bottom_blob;
    int outw_align = (outw + 3) / 4 * 4;
    int outh_align = (outh + 3) / 4 * 4;

    w = outw_align + 2;
    h = outh_align + 2;

    // BEGIN transform input
    float* bottom_blob_tm = nullptr;
    {
        int w_tm = outw_align / 4 * 6;
        int h_tm = outh_align / 4 * 6;

        int nColBlocks = h_tm / 6;    // may be the block num in Feathercnn
        int nRowBlocks = w_tm / 6;

        const int tiles = nColBlocks * nRowBlocks;
        const int tiles_n = 4 * inch * tiles;

        bottom_blob_tm = transform_input;

        // BT
        // const float itm[4][4] = {
        //     {4.0f, 0.0f, -5.0f, 0.0f, 1.0f, 0.0f},
        //     {0.0f,-4.0f, -4.0f, 1.0f, 1.0f, 0.0f},
        //     {0.0f, 4.0f, -4.0f,-1.0f, 1.0f, 0.0f},
        //     {0.0f,-2.0f, -1.0f, 2.0f, 1.0f, 0.0f},
        //     {0.0f, 2.0f, -1.0f,-2.0f, 1.0f, 0.0f},
        //     {0.0f, 4.0f,  0.0f,-5.0f, 0.0f, 1.0f}
        // };

        // 0 =	4 * r00  - 5 * r02	+ r04
        // 1 = -4 * (r01 + r02)  + r03 + r04
        // 2 =	4 * (r01 - r02)  - r03 + r04
        // 3 = -2 * r01 - r02 + 2 * r03 + r04
        // 4 =	2 * r01 - r02 - 2 * r03 + r04
        // 5 =	4 * r01 - 5 * r03 + r05

        // 0 =	4 * r00  - 5 * r02	+ r04
        // 1 = -4 * (r01 + r02)  + r03 + r04
        // 2 =	4 * (r01 - r02)  - r03 + r04
        // 3 = -2 * r01 - r02 + 2 * r03 + r04
        // 4 =	2 * r01 - r02 - 2 * r03 + r04
        // 5 =	4 * r01 - 5 * r03 + r05

#if __AVX__
        __m256 _1_n = _mm256_set1_ps(-1);
        __m256 _2_p = _mm256_set1_ps(2);
        __m256 _2_n = _mm256_set1_ps(-2);
        __m256 _4_p = _mm256_set1_ps(4);
        __m256 _4_n = _mm256_set1_ps(-4);
        __m256 _5_n = _mm256_set1_ps(-5);
#endif

#pragma omp parallel for num_threads(opt.num_threads)
        for(int q = 0; q < inch; q++)
        {
            const float* img = bottom_blob_bordered + q * w * h;

            for(int j = 0; j < nColBlocks; j++)
            {
                const float* r0 = img + w * j * 4;
                const float* r1 = r0 + w;
                const float* r2 = r1 + w;
                const float* r3 = r2 + w;
                const float* r4 = r3 + w;
                const float* r5 = r4 + w;

                for(int i = 0; i < nRowBlocks; i++)
                {
                    float* out_tm0 = bottom_blob_tm + 4 * inch * (j * nRowBlocks + i) + 4 * q;
                    float* out_tm1 = out_tm0 + tiles_n;
                    float* out_tm2 = out_tm0 + 2 * tiles_n;
                    float* out_tm3 = out_tm0 + 3 * tiles_n;
                    float* out_tm4 = out_tm0 + 4 * tiles_n;
                    float* out_tm5 = out_tm0 + 5 * tiles_n;
                    float* out_tm6 = out_tm0 + 6 * tiles_n;
                    float* out_tm7 = out_tm0 + 7 * tiles_n;
                    float* out_tm8 = out_tm0 + 8 * tiles_n;

#if __AVX__
                    __m256 _d0, _d1, _d2, _d3, _d4, _d5;
                    __m256 _w0, _w1, _w2, _w3, _w4, _w5;
                    __m256 _t0, _t1, _t2, _t3, _t4, _t5;
                    __m256 _n0, _n1, _n2, _n3, _n4, _n5;
                    // load
                    _d0 = _mm256_loadu_ps(r0);
                    _d1 = _mm256_loadu_ps(r1);
                    _d2 = _mm256_loadu_ps(r2);
                    _d3 = _mm256_loadu_ps(r3);
                    _d4 = _mm256_loadu_ps(r4);
                    _d5 = _mm256_loadu_ps(r5);

                    // w = B_t * d
                    _w0 = _mm256_mul_ps(_d0, _4_p);
                    _w0 = _mm256_fmadd_ps(_d2, _5_n, _w0);
                    _w0 = _mm256_add_ps(_w0, _d4);

                    _w1 = _mm256_mul_ps(_d1, _4_n);
                    _w1 = _mm256_fmadd_ps(_d2, _4_n, _w1);
                    _w1 = _mm256_add_ps(_w1, _d3);
                    _w1 = _mm256_add_ps(_w1, _d4);

                    _w2 = _mm256_mul_ps(_d1, _4_p);
                    _w2 = _mm256_fmadd_ps(_d2, _4_n, _w2);
                    _w2 = _mm256_fmadd_ps(_d3, _1_n, _w2);
                    _w2 = _mm256_add_ps(_w2, _d4);

                    _w3 = _mm256_mul_ps(_d1, _2_n);
                    _w3 = _mm256_fmadd_ps(_d2, _1_n, _w3);
                    _w3 = _mm256_fmadd_ps(_d3, _2_p, _w3);
                    _w3 = _mm256_add_ps(_w3, _d4);

                    _w4 = _mm256_mul_ps(_d1, _2_p);
                    _w4 = _mm256_fmadd_ps(_d2, _1_n, _w4);
                    _w4 = _mm256_fmadd_ps(_d3, _2_n, _w4);
                    _w4 = _mm256_add_ps(_w4, _d4);

                    _w5 = _mm256_mul_ps(_d1, _4_p);
                    _w5 = _mm256_fmadd_ps(_d3, _5_n, _w5);
                    _w5 = _mm256_add_ps(_w5, _d5);
                    // transpose d to d_t
#ifdef _WIN32
                    {
                        _t0.m256_f32[0] = _w0.m256_f32[0];
                        _t1.m256_f32[0] = _w0.m256_f32[1];
                        _t2.m256_f32[0] = _w0.m256_f32[2];
                        _t3.m256_f32[0] = _w0.m256_f32[3];
                        _t4.m256_f32[0] = _w0.m256_f32[4];
                        _t5.m256_f32[0] = _w0.m256_f32[5];
                        _t0.m256_f32[1] = _w1.m256_f32[0];
                        _t1.m256_f32[1] = _w1.m256_f32[1];
                        _t2.m256_f32[1] = _w1.m256_f32[2];
                        _t3.m256_f32[1] = _w1.m256_f32[3];
                        _t4.m256_f32[1] = _w1.m256_f32[4];
                        _t5.m256_f32[1] = _w1.m256_f32[5];
                        _t0.m256_f32[2] = _w2.m256_f32[0];
                        _t1.m256_f32[2] = _w2.m256_f32[1];
                        _t2.m256_f32[2] = _w2.m256_f32[2];
                        _t3.m256_f32[2] = _w2.m256_f32[3];
                        _t4.m256_f32[2] = _w2.m256_f32[4];
                        _t5.m256_f32[2] = _w2.m256_f32[5];
                        _t0.m256_f32[3] = _w3.m256_f32[0];
                        _t1.m256_f32[3] = _w3.m256_f32[1];
                        _t2.m256_f32[3] = _w3.m256_f32[2];
                        _t3.m256_f32[3] = _w3.m256_f32[3];
                        _t4.m256_f32[3] = _w3.m256_f32[4];
                        _t5.m256_f32[3] = _w3.m256_f32[5];
                        _t0.m256_f32[4] = _w4.m256_f32[0];
                        _t1.m256_f32[4] = _w4.m256_f32[1];
                        _t2.m256_f32[4] = _w4.m256_f32[2];
                        _t3.m256_f32[4] = _w4.m256_f32[3];
                        _t4.m256_f32[4] = _w4.m256_f32[4];
                        _t5.m256_f32[4] = _w4.m256_f32[5];
                        _t0.m256_f32[5] = _w5.m256_f32[0];
                        _t1.m256_f32[5] = _w5.m256_f32[1];
                        _t2.m256_f32[5] = _w5.m256_f32[2];
                        _t3.m256_f32[5] = _w5.m256_f32[3];
                        _t4.m256_f32[5] = _w5.m256_f32[4];
                        _t5.m256_f32[5] = _w5.m256_f32[5];
                    }
#else
                    {
                        _t0[0] = _w0[0];
                        _t1[0] = _w0[1];
                        _t2[0] = _w0[2];
                        _t3[0] = _w0[3];
                        _t4[0] = _w0[4];
                        _t5[0] = _w0[5];
                        _t0[1] = _w1[0];
                        _t1[1] = _w1[1];
                        _t2[1] = _w1[2];
                        _t3[1] = _w1[3];
                        _t4[1] = _w1[4];
                        _t5[1] = _w1[5];
                        _t0[2] = _w2[0];
                        _t1[2] = _w2[1];
                        _t2[2] = _w2[2];
                        _t3[2] = _w2[3];
                        _t4[2] = _w2[4];
                        _t5[2] = _w2[5];
                        _t0[3] = _w3[0];
                        _t1[3] = _w3[1];
                        _t2[3] = _w3[2];
                        _t3[3] = _w3[3];
                        _t4[3] = _w3[4];
                        _t5[3] = _w3[5];
                        _t0[4] = _w4[0];
                        _t1[4] = _w4[1];
                        _t2[4] = _w4[2];
                        _t3[4] = _w4[3];
                        _t4[4] = _w4[4];
                        _t5[4] = _w4[5];
                        _t0[5] = _w5[0];
                        _t1[5] = _w5[1];
                        _t2[5] = _w5[2];
                        _t3[5] = _w5[3];
                        _t4[5] = _w5[4];
                        _t5[5] = _w5[5];
                    }
#endif
                    // d = B_t * d_t
                    _n0 = _mm256_mul_ps(_t0, _4_p);
                    _n0 = _mm256_fmadd_ps(_t2, _5_n, _n0);
                    _n0 = _mm256_add_ps(_n0, _t4);

                    _n1 = _mm256_mul_ps(_t1, _4_n);
                    _n1 = _mm256_fmadd_ps(_t2, _4_n, _n1);
                    _n1 = _mm256_add_ps(_n1, _t3);
                    _n1 = _mm256_add_ps(_n1, _t4);

                    _n2 = _mm256_mul_ps(_t1, _4_p);
                    _n2 = _mm256_fmadd_ps(_t2, _4_n, _n2);
                    _n2 = _mm256_fmadd_ps(_t3, _1_n, _n2);
                    _n2 = _mm256_add_ps(_n2, _t4);

                    _n3 = _mm256_mul_ps(_t1, _2_n);
                    _n3 = _mm256_fmadd_ps(_t2, _1_n, _n3);
                    _n3 = _mm256_fmadd_ps(_t3, _2_p, _n3);
                    _n3 = _mm256_add_ps(_n3, _t4);

                    _n4 = _mm256_mul_ps(_t1, _2_p);
                    _n4 = _mm256_fmadd_ps(_t2, _1_n, _n4);
                    _n4 = _mm256_fmadd_ps(_t3, _2_n, _n4);
                    _n4 = _mm256_add_ps(_n4, _t4);

                    _n5 = _mm256_mul_ps(_t1, _4_p);
                    _n5 = _mm256_fmadd_ps(_t3, _5_n, _n5);
                    _n5 = _mm256_add_ps(_n5, _t5);
                    // save to out_tm
                    float output_n0[8] = {0.f};
                    _mm256_storeu_ps(output_n0, _n0);
                    float output_n1[8] = {0.f};
                    _mm256_storeu_ps(output_n1, _n1);
                    float output_n2[8] = {0.f};
                    _mm256_storeu_ps(output_n2, _n2);
                    float output_n3[8] = {0.f};
                    _mm256_storeu_ps(output_n3, _n3);
                    float output_n4[8] = {0.f};
                    _mm256_storeu_ps(output_n4, _n4);
                    float output_n5[8] = {0.f};
                    _mm256_storeu_ps(output_n5, _n5);

                    out_tm0[0] = output_n0[0];
                    out_tm0[1] = output_n0[1];
                    out_tm0[2] = output_n0[2];
                    out_tm0[3] = output_n0[3];
                    out_tm1[0] = output_n0[4];
                    out_tm1[1] = output_n0[5];
                    out_tm1[2] = output_n1[0];
                    out_tm1[3] = output_n1[1];
                    out_tm2[0] = output_n1[2];
                    out_tm2[1] = output_n1[3];
                    out_tm2[2] = output_n1[4];
                    out_tm2[3] = output_n1[5];

                    out_tm3[0] = output_n2[0];
                    out_tm3[1] = output_n2[1];
                    out_tm3[2] = output_n2[2];
                    out_tm3[3] = output_n2[3];
                    out_tm4[0] = output_n2[4];
                    out_tm4[1] = output_n2[5];
                    out_tm4[2] = output_n3[0];
                    out_tm4[3] = output_n3[1];
                    out_tm5[0] = output_n3[2];
                    out_tm5[1] = output_n3[3];
                    out_tm5[2] = output_n3[4];
                    out_tm5[3] = output_n3[5];

                    out_tm6[0] = output_n4[0];
                    out_tm6[1] = output_n4[1];
                    out_tm6[2] = output_n4[2];
                    out_tm6[3] = output_n4[3];
                    out_tm7[0] = output_n4[4];
                    out_tm7[1] = output_n4[5];
                    out_tm7[2] = output_n5[0];
                    out_tm7[3] = output_n5[1];
                    out_tm8[0] = output_n5[2];
                    out_tm8[1] = output_n5[3];
                    out_tm8[2] = output_n5[4];
                    out_tm8[3] = output_n5[5];
#else
                    float d0[6], d1[6], d2[6], d3[6], d4[6], d5[6];
                    float w0[6], w1[6], w2[6], w3[6], w4[6], w5[6];
                    float t0[6], t1[6], t2[6], t3[6], t4[6], t5[6];

                    // load
                    for(int n = 0; n < 6; n++)
                    {
                        d0[n] = r0[n];
                        d1[n] = r1[n];
                        d2[n] = r2[n];
                        d3[n] = r3[n];
                        d4[n] = r4[n];
                        d5[n] = r5[n];
                    }
                    // w = B_t * d
                    for(int n = 0; n < 6; n++)
                    {
                        w0[n] = 4 * d0[n] - 5 * d2[n] + d4[n];
                        w1[n] = -4 * d1[n] - 4 * d2[n] + d3[n] + d4[n];
                        w2[n] = 4 * d1[n] - 4 * d2[n] - d3[n] + d4[n];
                        w3[n] = -2 * d1[n] - d2[n] + 2 * d3[n] + d4[n];
                        w4[n] = 2 * d1[n] - d2[n] - 2 * d3[n] + d4[n];
                        w5[n] = 4 * d1[n] - 5 * d3[n] + d5[n];
                    }
                    // transpose d to d_t
                    {
                        t0[0] = w0[0];
                        t1[0] = w0[1];
                        t2[0] = w0[2];
                        t3[0] = w0[3];
                        t4[0] = w0[4];
                        t5[0] = w0[5];
                        t0[1] = w1[0];
                        t1[1] = w1[1];
                        t2[1] = w1[2];
                        t3[1] = w1[3];
                        t4[1] = w1[4];
                        t5[1] = w1[5];
                        t0[2] = w2[0];
                        t1[2] = w2[1];
                        t2[2] = w2[2];
                        t3[2] = w2[3];
                        t4[2] = w2[4];
                        t5[2] = w2[5];
                        t0[3] = w3[0];
                        t1[3] = w3[1];
                        t2[3] = w3[2];
                        t3[3] = w3[3];
                        t4[3] = w3[4];
                        t5[3] = w3[5];
                        t0[4] = w4[0];
                        t1[4] = w4[1];
                        t2[4] = w4[2];
                        t3[4] = w4[3];
                        t4[4] = w4[4];
                        t5[4] = w4[5];
                        t0[5] = w5[0];
                        t1[5] = w5[1];
                        t2[5] = w5[2];
                        t3[5] = w5[3];
                        t4[5] = w5[4];
                        t5[5] = w5[5];
                    }
                    // d = B_t * d_t
                    for(int n = 0; n < 6; n++)
                    {
                        d0[n] = 4 * t0[n] - 5 * t2[n] + t4[n];
                        d1[n] = -4 * t1[n] - 4 * t2[n] + t3[n] + t4[n];
                        d2[n] = 4 * t1[n] - 4 * t2[n] - t3[n] + t4[n];
                        d3[n] = -2 * t1[n] - t2[n] + 2 * t3[n] + t4[n];
                        d4[n] = 2 * t1[n] - t2[n] - 2 * t3[n] + t4[n];
                        d5[n] = 4 * t1[n] - 5 * t3[n] + t5[n];
                    }
                    // save to out_tm
                    {
                        out_tm0[0] = d0[0];
                        out_tm0[1] = d0[1];
                        out_tm0[2] = d0[2];
                        out_tm0[3] = d0[3];
                        out_tm1[0] = d0[4];
                        out_tm1[1] = d0[5];
                        out_tm1[2] = d1[0];
                        out_tm1[3] = d1[1];
                        out_tm2[0] = d1[2];
                        out_tm2[1] = d1[3];
                        out_tm2[2] = d1[4];
                        out_tm2[3] = d1[5];

                        out_tm3[0] = d2[0];
                        out_tm3[1] = d2[1];
                        out_tm3[2] = d2[2];
                        out_tm3[3] = d2[3];
                        out_tm4[0] = d2[4];
                        out_tm4[1] = d2[5];
                        out_tm4[2] = d3[0];
                        out_tm4[3] = d3[1];
                        out_tm5[0] = d3[2];
                        out_tm5[1] = d3[3];
                        out_tm5[2] = d3[4];
                        out_tm5[3] = d3[5];

                        out_tm6[0] = d4[0];
                        out_tm6[1] = d4[1];
                        out_tm6[2] = d4[2];
                        out_tm6[3] = d4[3];
                        out_tm7[0] = d4[4];
                        out_tm7[1] = d4[5];
                        out_tm7[2] = d5[0];
                        out_tm7[3] = d5[1];
                        out_tm8[0] = d5[2];
                        out_tm8[1] = d5[3];
                        out_tm8[2] = d5[4];
                        out_tm8[3] = d5[5];
                    }
#endif    // __AVX__
                    r0 += 4;
                    r1 += 4;
                    r2 += 4;
                    r3 += 4;
                    r4 += 4;
                    r5 += 4;
                }
            }
        }
    }

    // BEGIN dot
    float* top_blob_tm = nullptr;
    {
        int w_tm = outw_align / 4 * 6;
        int h_tm = outh_align / 4 * 6;

        int nColBlocks = h_tm / 6;    // may be the block num in Feathercnn
        int nRowBlocks = w_tm / 6;

        const int tiles = nColBlocks * nRowBlocks;
        const int tiles_n = 36 * tiles;

        top_blob_tm = dot_block;

#pragma omp parallel for num_threads(opt.num_threads)
        for(int r = 0; r < 9; r++)
        {
            int nn_outch = 0;
            int remain_outch_start = 0;

            nn_outch = outch >> 3;
            remain_outch_start = nn_outch << 3;

            for(int pp = 0; pp < nn_outch; pp++)
            {
                int p = pp << 3;

                float* output0_tm = top_blob_tm + tiles_n * p;
                float* output1_tm = top_blob_tm + tiles_n * (p + 1);
                float* output2_tm = top_blob_tm + tiles_n * (p + 2);
                float* output3_tm = top_blob_tm + tiles_n * (p + 3);
                float* output4_tm = top_blob_tm + tiles_n * (p + 4);
                float* output5_tm = top_blob_tm + tiles_n * (p + 5);
                float* output6_tm = top_blob_tm + tiles_n * (p + 6);
                float* output7_tm = top_blob_tm + tiles_n * (p + 7);

                output0_tm = output0_tm + r * 4;
                output1_tm = output1_tm + r * 4;
                output2_tm = output2_tm + r * 4;
                output3_tm = output3_tm + r * 4;
                output4_tm = output4_tm + r * 4;
                output5_tm = output5_tm + r * 4;
                output6_tm = output6_tm + r * 4;
                output7_tm = output7_tm + r * 4;

                for(int i = 0; i < tiles; i++)
                {
                    const float* kptr = kernel_tm_test + 4 * r * inch * outch + p / 8 * inch * 32;
                    const float* r0 = bottom_blob_tm + 4 * inch * (tiles * r + i);
#if __AVX__ || __SSE__
#if __AVX__
                    float zero_val = 0.f;
                    __m128 _sum0 = _mm_broadcast_ss(&zero_val);
                    __m128 _sum1 = _mm_broadcast_ss(&zero_val);
                    __m128 _sum2 = _mm_broadcast_ss(&zero_val);
                    __m128 _sum3 = _mm_broadcast_ss(&zero_val);
                    __m128 _sum4 = _mm_broadcast_ss(&zero_val);
                    __m128 _sum5 = _mm_broadcast_ss(&zero_val);
                    __m128 _sum6 = _mm_broadcast_ss(&zero_val);
                    __m128 _sum7 = _mm_broadcast_ss(&zero_val);
#else
                    __m128 _sum0 = _mm_set1_ps(0.f);
                    __m128 _sum1 = _mm_set1_ps(0.f);
                    __m128 _sum2 = _mm_set1_ps(0.f);
                    __m128 _sum3 = _mm_set1_ps(0.f);
                    __m128 _sum4 = _mm_set1_ps(0.f);
                    __m128 _sum5 = _mm_set1_ps(0.f);
                    __m128 _sum6 = _mm_set1_ps(0.f);
                    __m128 _sum7 = _mm_set1_ps(0.f);
#endif
                    int q = 0;
                    for(; q + 3 < inch; q = q + 4)
                    {
                        __m128 _r0 = _mm_loadu_ps(r0);
                        __m128 _r1 = _mm_loadu_ps(r0 + 4);
                        __m128 _r2 = _mm_loadu_ps(r0 + 8);
                        __m128 _r3 = _mm_loadu_ps(r0 + 12);

                        __m128 _k0 = _mm_loadu_ps(kptr);
                        __m128 _k1 = _mm_loadu_ps(kptr + 4);
                        __m128 _k2 = _mm_loadu_ps(kptr + 8);
                        __m128 _k3 = _mm_loadu_ps(kptr + 12);
                        __m128 _k4 = _mm_loadu_ps(kptr + 16);
                        __m128 _k5 = _mm_loadu_ps(kptr + 20);
                        __m128 _k6 = _mm_loadu_ps(kptr + 24);
                        __m128 _k7 = _mm_loadu_ps(kptr + 28);
#if __AVX__
                        _sum0 = _mm_fmadd_ps(_r0, _k0, _sum0);
                        _sum1 = _mm_fmadd_ps(_r0, _k1, _sum1);
                        _sum2 = _mm_fmadd_ps(_r0, _k2, _sum2);
                        _sum3 = _mm_fmadd_ps(_r0, _k3, _sum3);
                        _sum4 = _mm_fmadd_ps(_r0, _k4, _sum4);
                        _sum5 = _mm_fmadd_ps(_r0, _k5, _sum5);
                        _sum6 = _mm_fmadd_ps(_r0, _k6, _sum6);
                        _sum7 = _mm_fmadd_ps(_r0, _k7, _sum7);
#else
                        _sum0 = _mm_add_ps(_sum0, _mm_mul_ps(_r0, _k0));
                        _sum1 = _mm_add_ps(_sum1, _mm_mul_ps(_r0, _k1));
                        _sum2 = _mm_add_ps(_sum2, _mm_mul_ps(_r0, _k2));
                        _sum3 = _mm_add_ps(_sum3, _mm_mul_ps(_r0, _k3));
                        _sum4 = _mm_add_ps(_sum4, _mm_mul_ps(_r0, _k4));
                        _sum5 = _mm_add_ps(_sum5, _mm_mul_ps(_r0, _k5));
                        _sum6 = _mm_add_ps(_sum6, _mm_mul_ps(_r0, _k6));
                        _sum7 = _mm_add_ps(_sum7, _mm_mul_ps(_r0, _k7));
#endif
                        kptr += 32;
                        _k0 = _mm_loadu_ps(kptr);
                        _k1 = _mm_loadu_ps(kptr + 4);
                        _k2 = _mm_loadu_ps(kptr + 8);
                        _k3 = _mm_loadu_ps(kptr + 12);
                        _k4 = _mm_loadu_ps(kptr + 16);
                        _k5 = _mm_loadu_ps(kptr + 20);
                        _k6 = _mm_loadu_ps(kptr + 24);
                        _k7 = _mm_loadu_ps(kptr + 28);
#if __AVX__
                        _sum0 = _mm_fmadd_ps(_r1, _k0, _sum0);
                        _sum1 = _mm_fmadd_ps(_r1, _k1, _sum1);
                        _sum2 = _mm_fmadd_ps(_r1, _k2, _sum2);
                        _sum3 = _mm_fmadd_ps(_r1, _k3, _sum3);
                        _sum4 = _mm_fmadd_ps(_r1, _k4, _sum4);
                        _sum5 = _mm_fmadd_ps(_r1, _k5, _sum5);
                        _sum6 = _mm_fmadd_ps(_r1, _k6, _sum6);
                        _sum7 = _mm_fmadd_ps(_r1, _k7, _sum7);
#else
                        _sum0 = _mm_add_ps(_sum0, _mm_mul_ps(_r1, _k0));
                        _sum1 = _mm_add_ps(_sum1, _mm_mul_ps(_r1, _k1));
                        _sum2 = _mm_add_ps(_sum2, _mm_mul_ps(_r1, _k2));
                        _sum3 = _mm_add_ps(_sum3, _mm_mul_ps(_r1, _k3));
                        _sum4 = _mm_add_ps(_sum4, _mm_mul_ps(_r1, _k4));
                        _sum5 = _mm_add_ps(_sum5, _mm_mul_ps(_r1, _k5));
                        _sum6 = _mm_add_ps(_sum6, _mm_mul_ps(_r1, _k6));
                        _sum7 = _mm_add_ps(_sum7, _mm_mul_ps(_r1, _k7));
#endif

                        kptr += 32;
                        _k0 = _mm_loadu_ps(kptr);
                        _k1 = _mm_loadu_ps(kptr + 4);
                        _k2 = _mm_loadu_ps(kptr + 8);
                        _k3 = _mm_loadu_ps(kptr + 12);
                        _k4 = _mm_loadu_ps(kptr + 16);
                        _k5 = _mm_loadu_ps(kptr + 20);
                        _k6 = _mm_loadu_ps(kptr + 24);
                        _k7 = _mm_loadu_ps(kptr + 28);
#if __AVX__
                        _sum0 = _mm_fmadd_ps(_r2, _k0, _sum0);
                        _sum1 = _mm_fmadd_ps(_r2, _k1, _sum1);
                        _sum2 = _mm_fmadd_ps(_r2, _k2, _sum2);
                        _sum3 = _mm_fmadd_ps(_r2, _k3, _sum3);
                        _sum4 = _mm_fmadd_ps(_r2, _k4, _sum4);
                        _sum5 = _mm_fmadd_ps(_r2, _k5, _sum5);
                        _sum6 = _mm_fmadd_ps(_r2, _k6, _sum6);
                        _sum7 = _mm_fmadd_ps(_r2, _k7, _sum7);
#else
                        _sum0 = _mm_add_ps(_sum0, _mm_mul_ps(_r2, _k0));
                        _sum1 = _mm_add_ps(_sum1, _mm_mul_ps(_r2, _k1));
                        _sum2 = _mm_add_ps(_sum2, _mm_mul_ps(_r2, _k2));
                        _sum3 = _mm_add_ps(_sum3, _mm_mul_ps(_r2, _k3));
                        _sum4 = _mm_add_ps(_sum4, _mm_mul_ps(_r2, _k4));
                        _sum5 = _mm_add_ps(_sum5, _mm_mul_ps(_r2, _k5));
                        _sum6 = _mm_add_ps(_sum6, _mm_mul_ps(_r2, _k6));
                        _sum7 = _mm_add_ps(_sum7, _mm_mul_ps(_r2, _k7));
#endif
                        kptr += 32;
                        _k0 = _mm_loadu_ps(kptr);
                        _k1 = _mm_loadu_ps(kptr + 4);
                        _k2 = _mm_loadu_ps(kptr + 8);
                        _k3 = _mm_loadu_ps(kptr + 12);
                        _k4 = _mm_loadu_ps(kptr + 16);
                        _k5 = _mm_loadu_ps(kptr + 20);
                        _k6 = _mm_loadu_ps(kptr + 24);
                        _k7 = _mm_loadu_ps(kptr + 28);
#if __AVX__
                        _sum0 = _mm_fmadd_ps(_r3, _k0, _sum0);
                        _sum1 = _mm_fmadd_ps(_r3, _k1, _sum1);
                        _sum2 = _mm_fmadd_ps(_r3, _k2, _sum2);
                        _sum3 = _mm_fmadd_ps(_r3, _k3, _sum3);
                        _sum4 = _mm_fmadd_ps(_r3, _k4, _sum4);
                        _sum5 = _mm_fmadd_ps(_r3, _k5, _sum5);
                        _sum6 = _mm_fmadd_ps(_r3, _k6, _sum6);
                        _sum7 = _mm_fmadd_ps(_r3, _k7, _sum7);
#else
                        _sum0 = _mm_add_ps(_sum0, _mm_mul_ps(_r3, _k0));
                        _sum1 = _mm_add_ps(_sum1, _mm_mul_ps(_r3, _k1));
                        _sum2 = _mm_add_ps(_sum2, _mm_mul_ps(_r3, _k2));
                        _sum3 = _mm_add_ps(_sum3, _mm_mul_ps(_r3, _k3));
                        _sum4 = _mm_add_ps(_sum4, _mm_mul_ps(_r3, _k4));
                        _sum5 = _mm_add_ps(_sum5, _mm_mul_ps(_r3, _k5));
                        _sum6 = _mm_add_ps(_sum6, _mm_mul_ps(_r3, _k6));
                        _sum7 = _mm_add_ps(_sum7, _mm_mul_ps(_r3, _k7));
#endif
                        kptr += 32;
                        r0 += 16;
                    }

                    for(; q < inch; q++)
                    {
                        __m128 _r0 = _mm_loadu_ps(r0);
                        __m128 _k0 = _mm_loadu_ps(kptr);
                        __m128 _k1 = _mm_loadu_ps(kptr + 4);
                        __m128 _k2 = _mm_loadu_ps(kptr + 8);
                        __m128 _k3 = _mm_loadu_ps(kptr + 12);
                        __m128 _k4 = _mm_loadu_ps(kptr + 16);
                        __m128 _k5 = _mm_loadu_ps(kptr + 20);
                        __m128 _k6 = _mm_loadu_ps(kptr + 24);
                        __m128 _k7 = _mm_loadu_ps(kptr + 28);

#if __AVX__
                        _sum0 = _mm_fmadd_ps(_r0, _k0, _sum0);
                        _sum1 = _mm_fmadd_ps(_r0, _k1, _sum1);
                        _sum2 = _mm_fmadd_ps(_r0, _k2, _sum2);
                        _sum3 = _mm_fmadd_ps(_r0, _k3, _sum3);
                        _sum4 = _mm_fmadd_ps(_r0, _k4, _sum4);
                        _sum5 = _mm_fmadd_ps(_r0, _k5, _sum5);
                        _sum6 = _mm_fmadd_ps(_r0, _k6, _sum6);
                        _sum7 = _mm_fmadd_ps(_r0, _k7, _sum7);
#else
                        _sum0 = _mm_add_ps(_sum0, _mm_mul_ps(_r0, _k0));
                        _sum1 = _mm_add_ps(_sum1, _mm_mul_ps(_r0, _k1));
                        _sum2 = _mm_add_ps(_sum2, _mm_mul_ps(_r0, _k2));
                        _sum3 = _mm_add_ps(_sum3, _mm_mul_ps(_r0, _k3));
                        _sum4 = _mm_add_ps(_sum4, _mm_mul_ps(_r0, _k4));
                        _sum5 = _mm_add_ps(_sum5, _mm_mul_ps(_r0, _k5));
                        _sum6 = _mm_add_ps(_sum6, _mm_mul_ps(_r0, _k6));
                        _sum7 = _mm_add_ps(_sum7, _mm_mul_ps(_r0, _k7));
#endif

                        kptr += 32;
                        r0 += 4;
                    }

                    _mm_storeu_ps(output0_tm, _sum0);
                    _mm_storeu_ps(output1_tm, _sum1);
                    _mm_storeu_ps(output2_tm, _sum2);
                    _mm_storeu_ps(output3_tm, _sum3);
                    _mm_storeu_ps(output4_tm, _sum4);
                    _mm_storeu_ps(output5_tm, _sum5);
                    _mm_storeu_ps(output6_tm, _sum6);
                    _mm_storeu_ps(output7_tm, _sum7);
#else
                    float sum0[4] = {0};
                    float sum1[4] = {0};
                    float sum2[4] = {0};
                    float sum3[4] = {0};
                    float sum4[4] = {0};
                    float sum5[4] = {0};
                    float sum6[4] = {0};
                    float sum7[4] = {0};

                    for(int q = 0; q < inch; q++)
                    {
                        for(int n = 0; n < 4; n++)
                        {
                            sum0[n] += r0[n] * kptr[n];
                            sum1[n] += r0[n] * kptr[n + 4];
                            sum2[n] += r0[n] * kptr[n + 8];
                            sum3[n] += r0[n] * kptr[n + 12];
                            sum4[n] += r0[n] * kptr[n + 16];
                            sum5[n] += r0[n] * kptr[n + 20];
                            sum6[n] += r0[n] * kptr[n + 24];
                            sum7[n] += r0[n] * kptr[n + 28];
                        }
                        kptr += 32;
                        r0 += 4;
                    }

                    for(int n = 0; n < 4; n++)
                    {
                        output0_tm[n] = sum0[n];
                        output1_tm[n] = sum1[n];
                        output2_tm[n] = sum2[n];
                        output3_tm[n] = sum3[n];
                        output4_tm[n] = sum4[n];
                        output5_tm[n] = sum5[n];
                        output6_tm[n] = sum6[n];
                        output7_tm[n] = sum7[n];
                    }
#endif    // __AVX__
                    output0_tm += 36;
                    output1_tm += 36;
                    output2_tm += 36;
                    output3_tm += 36;
                    output4_tm += 36;
                    output5_tm += 36;
                    output6_tm += 36;
                    output7_tm += 36;
                }
            }

            nn_outch = (outch - remain_outch_start) >> 2;
            for(int pp = 0; pp < nn_outch; pp++)
            {
                int p = remain_outch_start + pp * 4;

                float* output0_tm = top_blob_tm + tiles_n * p;
                float* output1_tm = top_blob_tm + tiles_n * (p + 1);
                float* output2_tm = top_blob_tm + tiles_n * (p + 2);
                float* output3_tm = top_blob_tm + tiles_n * (p + 3);

                output0_tm = output0_tm + r * 4;
                output1_tm = output1_tm + r * 4;
                output2_tm = output2_tm + r * 4;
                output3_tm = output3_tm + r * 4;

                for(int i = 0; i < tiles; i++)
                {
                    const float* kptr = kernel_tm_test + 4 * r * inch * outch + (p / 8 + (p % 8) / 4) * inch * 32;
                    const float* r0 = bottom_blob_tm + 4 * inch * (tiles * r + i);
#if __AVX__ || __SSE__
#if __AVX__
                    float zero_val = 0.f;
                    __m128 _sum0 = _mm_broadcast_ss(&zero_val);
                    __m128 _sum1 = _mm_broadcast_ss(&zero_val);
                    __m128 _sum2 = _mm_broadcast_ss(&zero_val);
                    __m128 _sum3 = _mm_broadcast_ss(&zero_val);
#else
                    __m128 _sum0 = _mm_set1_ps(0.f);
                    __m128 _sum1 = _mm_set1_ps(0.f);
                    __m128 _sum2 = _mm_set1_ps(0.f);
                    __m128 _sum3 = _mm_set1_ps(0.f);
#endif
                    for(int q = 0; q < inch; q++)
                    {
                        __m128 _r0 = _mm_loadu_ps(r0);
                        __m128 _k0 = _mm_loadu_ps(kptr);
                        __m128 _k1 = _mm_loadu_ps(kptr + 4);
                        __m128 _k2 = _mm_loadu_ps(kptr + 8);
                        __m128 _k3 = _mm_loadu_ps(kptr + 12);
#if __AVX__
                        _sum0 = _mm_fmadd_ps(_r0, _k0, _sum0);
                        _sum1 = _mm_fmadd_ps(_r0, _k1, _sum1);
                        _sum2 = _mm_fmadd_ps(_r0, _k2, _sum2);
                        _sum3 = _mm_fmadd_ps(_r0, _k3, _sum3);
#else
                        _sum0 = _mm_add_ps(_sum0, _mm_mul_ps(_r0, _k0));
                        _sum1 = _mm_add_ps(_sum1, _mm_mul_ps(_r0, _k1));
                        _sum2 = _mm_add_ps(_sum2, _mm_mul_ps(_r0, _k2));
                        _sum3 = _mm_add_ps(_sum3, _mm_mul_ps(_r0, _k3));
#endif
                        kptr += 16;
                        r0 += 4;
                    }

                    _mm_storeu_ps(output0_tm, _sum0);
                    _mm_storeu_ps(output1_tm, _sum1);
                    _mm_storeu_ps(output2_tm, _sum2);
                    _mm_storeu_ps(output3_tm, _sum3);
#else
                    float sum0[4] = {0};
                    float sum1[4] = {0};
                    float sum2[4] = {0};
                    float sum3[4] = {0};

                    for(int q = 0; q < inch; q++)
                    {
                        for(int n = 0; n < 4; n++)
                        {
                            sum0[n] += r0[n] * kptr[n];
                            sum1[n] += r0[n] * kptr[n + 4];
                            sum2[n] += r0[n] * kptr[n + 8];
                            sum3[n] += r0[n] * kptr[n + 12];
                        }
                        kptr += 16;
                        r0 += 4;
                    }

                    for(int n = 0; n < 4; n++)
                    {
                        output0_tm[n] = sum0[n];
                        output1_tm[n] = sum1[n];
                        output2_tm[n] = sum2[n];
                        output3_tm[n] = sum3[n];
                    }
#endif    // __AVX__
                    output0_tm += 36;
                    output1_tm += 36;
                    output2_tm += 36;
                    output3_tm += 36;
                }
            }

            remain_outch_start += nn_outch << 2;

            for(int p = remain_outch_start; p < outch; p++)
            {
                float* output0_tm = top_blob_tm + 36 * tiles * p;

                output0_tm = output0_tm + r * 4;

                for(int i = 0; i < tiles; i++)
                {
                    const float* kptr =
                        kernel_tm_test + 4 * r * inch * outch + (p / 8 + (p % 8) / 4 + p % 4) * inch * 32;
                    const float* r0 = bottom_blob_tm + 4 * inch * (tiles * r + i);
#if __AVX__ || __SSE__
#if __AVX__
                    float zero_val = 0.f;
                    __m128 _sum0 = _mm_broadcast_ss(&zero_val);
#else
                    __m128 _sum0 = _mm_set1_ps(0.f);
#endif

                    for(int q = 0; q < inch; q++)
                    {
                        __m128 _r0 = _mm_loadu_ps(r0);
                        __m128 _k0 = _mm_loadu_ps(kptr);
#if __AVX__
                        _sum0 = _mm_fmadd_ps(_r0, _k0, _sum0);
#else
                        _sum0 = _mm_add_ps(_sum0, _mm_mul_ps(_r0, _k0));
#endif
                        kptr += 16;
                        r0 += 4;
                    }
                    _mm_storeu_ps(output0_tm, _sum0);
#else
                    float sum0[4] = {0};

                    for(int q = 0; q < inch; q++)
                    {
                        for(int n = 0; n < 4; n++)
                        {
                            sum0[n] += ( int )r0[n] * kptr[n];
                        }
                        kptr += 4;
                        r0 += 4;
                    }

                    for(int n = 0; n < 4; n++)
                    {
                        output0_tm[n] = sum0[n];
                    }
#endif    // __AVX__ || __SSE__
                    output0_tm += 36;
                }
            }
        }
    }
    // END dot

    // BEGIN transform output
    float* top_blob_bordered = nullptr;
    if(outw_align == outw && outh_align == outh)
    {
        top_blob_bordered = top_blob;
    }
    else
    {
        top_blob_bordered = output_bordered;
    }
    {
        // AT
        // const float itm[4][6] = {
        //     {1.0f, 1.0f,  1.0f, 1.0f,  1.0f, 0.0f},
        //     {0.0f, 1.0f, -1.0f, 2.0f, -2.0f, 0.0f},
        //     {0.0f, 1.0f,  1.0f, 4.0f,  4.0f, 0.0f},
        //     {0.0f, 1.0f, -1.0f, 8.0f, -8.0f, 1.0f}
        // };

        // 0 =	r00 + r01 + r02 + r03 +	r04
        // 1 =		  r01 - r02 + 2 * (r03 - r04)
        // 2 =		  r01 + r02 + 4 * (r03 + r04)
        // 3 =		  r01 - r02 + 8 * (r03 - r04)  + r05

        int w_tm = outw_align / 4 * 6;
        int h_tm = outh_align / 4 * 6;

        int nColBlocks = h_tm / 6;    // may be the block num in Feathercnn
        int nRowBlocks = w_tm / 6;

        const int tiles = nColBlocks * nRowBlocks;

#pragma omp parallel for num_threads(opt.num_threads)
        for(int p = 0; p < outch; p++)
        {
            float* out_tile = top_blob_tm + 36 * tiles * p;
            float* outRow0 = top_blob_bordered + outw_align * outh_align * p;
            float* outRow1 = outRow0 + outw_align;
            float* outRow2 = outRow0 + outw_align * 2;
            float* outRow3 = outRow0 + outw_align * 3;

            const float bias0 = bias ? bias[p] : 0.f;

            for(int j = 0; j < nColBlocks; j++)
            {
                for(int i = 0; i < nRowBlocks; i++)
                {
                    // TODO AVX2
                    float s0[6], s1[6], s2[6], s3[6], s4[6], s5[6];
                    float w0[6], w1[6], w2[6], w3[6];
                    float d0[4], d1[4], d2[4], d3[4], d4[4], d5[4];
                    float o0[4], o1[4], o2[4], o3[4];

                    // load
                    for(int n = 0; n < 6; n++)
                    {
                        s0[n] = out_tile[n];
                        s1[n] = out_tile[n + 6];
                        s2[n] = out_tile[n + 12];
                        s3[n] = out_tile[n + 18];
                        s4[n] = out_tile[n + 24];
                        s5[n] = out_tile[n + 30];
                    }
                    // w = A_T * W
                    for(int n = 0; n < 6; n++)
                    {
                        w0[n] = s0[n] + s1[n] + s2[n] + s3[n] + s4[n];
                        w1[n] = s1[n] - s2[n] + 2 * s3[n] - 2 * s4[n];
                        w2[n] = s1[n] + s2[n] + 4 * s3[n] + 4 * s4[n];
                        w3[n] = s1[n] - s2[n] + 8 * s3[n] - 8 * s4[n] + s5[n];
                    }
                    // transpose w to w_t
                    {
                        d0[0] = w0[0];
                        d0[1] = w1[0];
                        d0[2] = w2[0];
                        d0[3] = w3[0];
                        d1[0] = w0[1];
                        d1[1] = w1[1];
                        d1[2] = w2[1];
                        d1[3] = w3[1];
                        d2[0] = w0[2];
                        d2[1] = w1[2];
                        d2[2] = w2[2];
                        d2[3] = w3[2];
                        d3[0] = w0[3];
                        d3[1] = w1[3];
                        d3[2] = w2[3];
                        d3[3] = w3[3];
                        d4[0] = w0[4];
                        d4[1] = w1[4];
                        d4[2] = w2[4];
                        d4[3] = w3[4];
                        d5[0] = w0[5];
                        d5[1] = w1[5];
                        d5[2] = w2[5];
                        d5[3] = w3[5];
                    }
                    // Y = A_T * w_t
                    for(int n = 0; n < 4; n++)
                    {
                        o0[n] = d0[n] + d1[n] + d2[n] + d3[n] + d4[n];
                        o1[n] = d1[n] - d2[n] + 2 * d3[n] - 2 * d4[n];
                        o2[n] = d1[n] + d2[n] + 4 * d3[n] + 4 * d4[n];
                        o3[n] = d1[n] - d2[n] + 8 * d3[n] - 8 * d4[n] + d5[n];
                    }
                    // save to top blob tm
                    for(int n = 0; n < 4; n++)
                    {
                        outRow0[n] = o0[n] + bias0;
                        outRow1[n] = o1[n] + bias0;
                        outRow2[n] = o2[n] + bias0;
                        outRow3[n] = o3[n] + bias0;
                    }

                    out_tile += 36;

                    outRow0 += 4;
                    outRow1 += 4;
                    outRow2 += 4;
                    outRow3 += 4;
                }

                outRow0 += outw_align * 3;
                outRow1 += outw_align * 3;
                outRow2 += outw_align * 3;
                outRow3 += outw_align * 3;
            }
        }
    }

    // END transform output
    if(outw_align != outw || outh_align != outw)
    {
        delete_0_3D(top_blob, top_blob_bordered, outh_align, outw_align, outh, outw, outch, 0, 0);
    }
}

#endif
