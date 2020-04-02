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
 * https://github.com/Tencent/ncnn/blob/master/src/layer/x86/convolution_sgemm.h
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
 * Author: qtang@openailab.com
 */

#ifndef __CONVOLUTION_X86_H__
#define __CONVOLUTION_X86_H__

#include <stdlib.h>

#if __SSE2__
#include <emmintrin.h>
#endif
#if __AVX__
#include <immintrin.h>
#endif

#if __AVX__
// int M = outch;       // outch
// int N = outw * outh; // outsize or out stride
// int K = kernel_w * kernel_h * inch; // ksize * inch
// float* pA = kernel
// float* pB = input_data
// float* pC = output_data
static void sgemm(int M, int N, int K, float* pA, float* pB, float* pC) // unloop output M, unloop N, packet 8x8, using intrinsic
{
    // printf("sgemm avx2 start\n");
    // kernel pack 8
    float* pA_t = (float* )malloc((8*K * (M/8 + (M%8)/4 + M%4)) * sizeof(float));
    {
        int nn_outch = M >> 3;
        int remain_outch_start = nn_outch << 3;

        for (int pp=0; pp<nn_outch; pp++)
        {
            int p = pp * 8;

            const float* k0 = pA + (p+0)*K;
            const float* k1 = pA + (p+1)*K;
            const float* k2 = pA + (p+2)*K;
            const float* k3 = pA + (p+3)*K;
            const float* k4 = pA + (p+4)*K;
            const float* k5 = pA + (p+5)*K;
            const float* k6 = pA + (p+6)*K;
            const float* k7 = pA + (p+7)*K;

            float* ktmp = pA_t + (p/8) * 8*K;

            for (int q=0; q<K; q++)
            {
                ktmp[0] = k0[0];
                ktmp[1] = k1[0];
                ktmp[2] = k2[0];
                ktmp[3] = k3[0];
                ktmp[4] = k4[0];
                ktmp[5] = k5[0];
                ktmp[6] = k6[0];
                ktmp[7] = k7[0];
                ktmp += 8;

                k0 += 1;
                k1 += 1;
                k2 += 1;
                k3 += 1;
                k4 += 1;
                k5 += 1;
                k6 += 1;
                k7 += 1;
            }
        }

        nn_outch = (M - remain_outch_start) >> 2;

        for (int pp=0; pp<nn_outch; pp++)
        {
            int p = remain_outch_start + pp * 4;

            const float* k0 = pA + (p+0)*K;
            const float* k1 = pA + (p+1)*K;
            const float* k2 = pA + (p+2)*K;
            const float* k3 = pA + (p+3)*K;

            float* ktmp = pA_t + (p/8 + (p%8)/4) * 8*K;

            for (int q=0; q<K; q++)
            {
                ktmp[0] = k0[0];
                ktmp[1] = k1[0];
                ktmp[2] = k2[0];
                ktmp[3] = k3[0];
                ktmp += 4;

                k0 += 1;
                k1 += 1;
                k2 += 1;
                k3 += 1;
            }
        }

        remain_outch_start += nn_outch << 2;
        
        for (int p=remain_outch_start; p<M; p++)
        {
            const float* k0 = pA + (p+0)*K;

            float* ktmp = pA_t + (p/8 + (p%8)/4 + p%4) * 8*K;

            for (int q=0; q<K; q++)
            {
                ktmp[0] = k0[0];
                ktmp++;
                k0++;
            }
        }
    }

    // printf("kernel interleave\n");

    // data, col2row, pack 8x8
    float* pB_t = (float* )malloc((8*K * (N/8 + N%8)) * sizeof(float));
    {
        int nn_size = N >> 3;
        int remian_size_start = nn_size << 3;

        // [ch00, ch10, ch20, ch30, ch40, ch50, ch60, ch70, ch10, ch11, ch12, ch13, ch14, ch15, ch16, ch17 ....]
        for (int ii=0; ii<nn_size; ii++)
        {
            int i = ii * 8;
            const float* img = pB + i;
            float* tmp = pB_t + (i/8) * 8*K;

            for (int j=0; j<K; j++)
            {
#if __AVX__
                _mm256_storeu_ps(tmp, _mm256_loadu_ps(img));
#else
                tmp[0] = img[0];
                tmp[1] = img[1];
                tmp[2] = img[2];
                tmp[3] = img[3];
                tmp[4] = img[4];
                tmp[5] = img[5];
                tmp[6] = img[6];
                tmp[7] = img[7];
#endif // __AVX__
                tmp += 8;
                img += N;
            }
        }

        // [ch00, ch01, ch02, ch03 ....]
        for (int i=remian_size_start; i<N; i++)
        {
            const float* img = pB + i;
            float* tmp = pB_t + (i/8 + i%8) * 8*K;

            for (int j=0; j<K; j++)
            {
                tmp[0] = img[0];

                tmp += 1;
                img += N;
            }
        }
    }

    // printf("data interleave\n");

    int nn_outch = 0;
    int remain_outch_start = 0;

    nn_outch = M >> 3;
    remain_outch_start = nn_outch << 3;

    // output ch0 - ch7
    for (int pp=0; pp<nn_outch; pp++)
    {
        int i = pp * 8;

        float* output0 = pC + (i  )*N;
        float* output1 = pC + (i+1)*N;
        float* output2 = pC + (i+2)*N;
        float* output3 = pC + (i+3)*N;
        float* output4 = pC + (i+4)*N;
        float* output5 = pC + (i+5)*N;
        float* output6 = pC + (i+6)*N;
        float* output7 = pC + (i+7)*N;

        int j=0;
        for (; j+7<N; j+=8)
        {
            float* va = pA_t + (i/8) * 8*K;
            float* vb = pB_t + (j/8) * 8*K;
#if __AVX__
            __m256 _sum0 = _mm256_set1_ps(0.0);
            __m256 _sum1 = _mm256_set1_ps(0.0);
            __m256 _sum2 = _mm256_set1_ps(0.0);
            __m256 _sum3 = _mm256_set1_ps(0.0);
            __m256 _sum4 = _mm256_set1_ps(0.0);
            __m256 _sum5 = _mm256_set1_ps(0.0);
            __m256 _sum6 = _mm256_set1_ps(0.0);
            __m256 _sum7 = _mm256_set1_ps(0.0);

            int k=0;
            for (; k+3<K; k=k+4)
            {
                // k0
                __m256 _va0 = _mm256_broadcast_ss(va);
                __m256 _va1 = _mm256_broadcast_ss(va+1);
                __m256 _va2 = _mm256_broadcast_ss(va+2);
                __m256 _va3 = _mm256_broadcast_ss(va+3);
                __m256 _vb0 = _mm256_loadu_ps(vb);
                __m256 _vb1 = _mm256_loadu_ps(vb+8);
                __m256 _vb2 = _mm256_loadu_ps(vb+16);
                __m256 _vb3 = _mm256_loadu_ps(vb+24);
                _sum0 = _mm256_fmadd_ps(_vb0, _va0, _sum0);    // sum0 = (a00-a07) * k00
                _sum1 = _mm256_fmadd_ps(_vb0, _va1, _sum1);    // sum1 = (a00-a07) * k10
                _sum2 = _mm256_fmadd_ps(_vb0, _va2, _sum2);    // sum2 = (a00-a07) * k20
                _sum3 = _mm256_fmadd_ps(_vb0, _va3, _sum3);    // sum3 = (a00-a07) * k30
                _va0 = _mm256_broadcast_ss(va+4);
                _va1 = _mm256_broadcast_ss(va+5);
                _va2 = _mm256_broadcast_ss(va+6);
                _va3 = _mm256_broadcast_ss(va+7); 
                _sum4 = _mm256_fmadd_ps(_vb0, _va0, _sum4);    // sum4 = (a00-a07) * k40
                _sum5 = _mm256_fmadd_ps(_vb0, _va1, _sum5);    // sum5 = (a00-a07) * k50
                _sum6 = _mm256_fmadd_ps(_vb0, _va2, _sum6);    // sum6 = (a00-a07) * k60
                _sum7 = _mm256_fmadd_ps(_vb0, _va3, _sum7);    // sum7 = (a00-a07) * k70

                va += 8;

                // k1
                _va0 = _mm256_broadcast_ss(va);
                _va1 = _mm256_broadcast_ss(va+1);
                _va2 = _mm256_broadcast_ss(va+2);
                _va3 = _mm256_broadcast_ss(va+3);                  
                _sum0 = _mm256_fmadd_ps(_vb1, _va0, _sum0);    // sum0 += (a10-a17) * k01
                _sum1 = _mm256_fmadd_ps(_vb1, _va1, _sum1);    // sum1 += (a10-a17) * k11
                _sum2 = _mm256_fmadd_ps(_vb1, _va2, _sum2);    // sum2 += (a10-a17) * k21
                _sum3 = _mm256_fmadd_ps(_vb1, _va3, _sum3);    // sum3 += (a10-a17) * k31
                _va0 = _mm256_broadcast_ss(va+4);
                _va1 = _mm256_broadcast_ss(va+5);
                _va2 = _mm256_broadcast_ss(va+6);
                _va3 = _mm256_broadcast_ss(va+7);                     
                _sum4 = _mm256_fmadd_ps(_vb1, _va0, _sum4);    // sum4 += (a10-a17) * k41
                _sum5 = _mm256_fmadd_ps(_vb1, _va1, _sum5);    // sum5 += (a10-a17) * k51
                _sum6 = _mm256_fmadd_ps(_vb1, _va2, _sum6);    // sum6 += (a10-a17) * k61
                _sum7 = _mm256_fmadd_ps(_vb1, _va3, _sum7);    // sum7 += (a10-a17) * k71

                va += 8;

                // k2
                _va0 = _mm256_broadcast_ss(va);
                _va1 = _mm256_broadcast_ss(va+1);
                _va2 = _mm256_broadcast_ss(va+2);
                _va3 = _mm256_broadcast_ss(va+3);
                _sum0 = _mm256_fmadd_ps(_vb2, _va0, _sum0);    // sum0 += (a20-a27) * k02
                _sum1 = _mm256_fmadd_ps(_vb2, _va1, _sum1);    // sum1 += (a20-a27) * k12
                _sum2 = _mm256_fmadd_ps(_vb2, _va2, _sum2);    // sum2 += (a20-a27) * k22
                _sum3 = _mm256_fmadd_ps(_vb2, _va3, _sum3);    // sum3 += (a20-a27) * k32
                _va0 = _mm256_broadcast_ss(va+4);
                _va1 = _mm256_broadcast_ss(va+5);
                _va2 = _mm256_broadcast_ss(va+6);
                _va3 = _mm256_broadcast_ss(va+7);                     
                _sum4 = _mm256_fmadd_ps(_vb2, _va0, _sum4);    // sum4 += (a20-a27) * k42
                _sum5 = _mm256_fmadd_ps(_vb2, _va1, _sum5);    // sum5 += (a20-a27) * k52
                _sum6 = _mm256_fmadd_ps(_vb2, _va2, _sum6);    // sum6 += (a20-a27) * k62
                _sum7 = _mm256_fmadd_ps(_vb2, _va3, _sum7);    // sum7 += (a20-a27) * k72  

                va += 8;                  

                // k3
                _va0 = _mm256_broadcast_ss(va);
                _va1 = _mm256_broadcast_ss(va+1);
                _va2 = _mm256_broadcast_ss(va+2);
                _va3 = _mm256_broadcast_ss(va+3);
                _sum0 = _mm256_fmadd_ps(_vb3, _va0, _sum0);    // sum0 += (a30-a37) * k03
                _sum1 = _mm256_fmadd_ps(_vb3, _va1, _sum1);    // sum1 += (a30-a37) * k13
                _sum2 = _mm256_fmadd_ps(_vb3, _va2, _sum2);    // sum2 += (a30-a37) * k23
                _sum3 = _mm256_fmadd_ps(_vb3, _va3, _sum3);    // sum3 += (a30-a37) * k33
                _va0 = _mm256_broadcast_ss(va+4);
                _va1 = _mm256_broadcast_ss(va+5);
                _va2 = _mm256_broadcast_ss(va+6);
                _va3 = _mm256_broadcast_ss(va+7);                     
                _sum4 = _mm256_fmadd_ps(_vb3, _va0, _sum4);    // sum4 += (a30-a37) * k43
                _sum5 = _mm256_fmadd_ps(_vb3, _va1, _sum5);    // sum5 += (a30-a37) * k53
                _sum6 = _mm256_fmadd_ps(_vb3, _va2, _sum6);    // sum6 += (a30-a37) * k63
                _sum7 = _mm256_fmadd_ps(_vb3, _va3, _sum7);    // sum7 += (a30-a37) * k73

                va += 8;
                vb += 32;
            }

            for (; k<K; k++)
            {
                // k0
                __m256 _va0 = _mm256_broadcast_ss(va);
                __m256 _va1 = _mm256_broadcast_ss(va+1);
                __m256 _va2 = _mm256_broadcast_ss(va+2);
                __m256 _va3 = _mm256_broadcast_ss(va+3);
                __m256 _va4 = _mm256_broadcast_ss(va+4);
                __m256 _va5 = _mm256_broadcast_ss(va+5);
                __m256 _va6 = _mm256_broadcast_ss(va+6);
                __m256 _va7 = _mm256_broadcast_ss(va+7); 
                __m256 _vb0 = _mm256_loadu_ps(vb);
                _sum0 = _mm256_fmadd_ps(_vb0, _va0, _sum0);    // sum0 = (a00-a07) * k00
                _sum1 = _mm256_fmadd_ps(_vb0, _va1, _sum1);    // sum1 = (a00-a07) * k10
                _sum2 = _mm256_fmadd_ps(_vb0, _va2, _sum2);    // sum2 = (a00-a07) * k20
                _sum3 = _mm256_fmadd_ps(_vb0, _va3, _sum3);    // sum3 = (a00-a07) * k30
                _sum4 = _mm256_fmadd_ps(_vb0, _va4, _sum4);    // sum4 = (a00-a07) * k40
                _sum5 = _mm256_fmadd_ps(_vb0, _va5, _sum5);    // sum5 = (a00-a07) * k50
                _sum6 = _mm256_fmadd_ps(_vb0, _va6, _sum6);    // sum6 = (a00-a07) * k60
                _sum7 = _mm256_fmadd_ps(_vb0, _va7, _sum7);    // sum7 = (a00-a07) * k70

                va += 8;
                vb += 8;
            }

            _mm256_storeu_ps(output0, _sum0);
            _mm256_storeu_ps(output1, _sum1); 
            _mm256_storeu_ps(output2, _sum2);
            _mm256_storeu_ps(output3, _sum3); 
            _mm256_storeu_ps(output4, _sum4);
            _mm256_storeu_ps(output5, _sum5); 
            _mm256_storeu_ps(output6, _sum6);
            _mm256_storeu_ps(output7, _sum7);
#else
            float sum0[8] = {0};
            float sum1[8] = {0};
            float sum2[8] = {0};
            float sum3[8] = {0};
            float sum4[8] = {0};
            float sum5[8] = {0};
            float sum6[8] = {0};
            float sum7[8] = {0};

            for (int k=0; k<K; k++)
            {
                for (int n=0; n<8; n++)
                {
                    sum0[n] += va[0] * vb[n];
                    sum1[n] += va[1] * vb[n];
                    sum2[n] += va[2] * vb[n];
                    sum3[n] += va[3] * vb[n];
                    sum4[n] += va[4] * vb[n];
                    sum5[n] += va[5] * vb[n];
                    sum6[n] += va[6] * vb[n];
                    sum7[n] += va[7] * vb[n];
                }
                
                va += 8;
                vb += 8;
            }

            for (int n=0; n<8; n++)
            {
                output0[n] = sum0[n];
                output1[n] = sum1[n];
                output2[n] = sum2[n];
                output3[n] = sum3[n];
                output4[n] = sum4[n];
                output5[n] = sum5[n];
                output6[n] = sum6[n];
                output7[n] = sum7[n];
            }
#endif // __AVX__
            output0 += 8;
            output1 += 8;
            output2 += 8;
            output3 += 8;
            output4 += 8;
            output5 += 8;
            output6 += 8;
            output7 += 8;
        }

        for (; j<N; j++)
        {
            float* va = pA_t + (i/8) * 8*K;
            float* vb = pB_t + (j/8 + j%8) * 8*K;

#if __AVX__
            __m256 _sum0_7 = _mm256_set1_ps(0.0);
            __m256 _sum0 = _mm256_set1_ps(0.0);
            __m256 _sum1 = _mm256_set1_ps(0.0);
            __m256 _sum2 = _mm256_set1_ps(0.0);
            __m256 _sum3 = _mm256_set1_ps(0.0);

            int k=0;
            for (; k+3<K; k=k+4)
            {
                __m256 _vb0 = _mm256_broadcast_ss(vb);
                __m256 _vb1 = _mm256_broadcast_ss(vb+1);
                __m256 _vb2 = _mm256_broadcast_ss(vb+2);
                __m256 _vb3 = _mm256_broadcast_ss(vb+3);
                __m256 _va0 = _mm256_loadu_ps(va);
                __m256 _va1 = _mm256_loadu_ps(va+8);
                __m256 _va2 = _mm256_loadu_ps(va+16);
                __m256 _va3 = _mm256_loadu_ps(va+24);

                _sum0 = _mm256_fmadd_ps(_va0, _vb0, _sum0);// sum0 += (k00-k70) * a00
                _sum1 = _mm256_fmadd_ps(_va1, _vb1, _sum1);// sum1 += (k01-k71) * a10
                _sum2 = _mm256_fmadd_ps(_va2, _vb2, _sum2);// sum2 += (k02-k72) * a20
                _sum3 = _mm256_fmadd_ps(_va3, _vb3, _sum3);// sum3 += (k03-k73) * a30

                va += 32;
                vb += 4;
            }

            _sum0 = _mm256_add_ps(_sum0, _sum1);
            _sum2 = _mm256_add_ps(_sum2, _sum3);
            _sum0_7 = _mm256_add_ps(_sum0_7, _sum0);
            _sum0_7 = _mm256_add_ps(_sum0_7, _sum2);

            for (; k<K; k++)
            {
                __m256 _vb0 = _mm256_broadcast_ss(vb);
                __m256 _va = _mm256_loadu_ps(va); 

                _sum0_7 = _mm256_fmadd_ps(_va, _vb0, _sum0_7);// sum0 += (k00-k70) * a00

                va += 8;
                vb += 1;
            }

            float output_sum0_7[8] = {0.f};
            _mm256_storeu_ps(output_sum0_7, _sum0_7); 

            output0[0] = output_sum0_7[0];
            output1[0] = output_sum0_7[1];
            output2[0] = output_sum0_7[2];
            output3[0] = output_sum0_7[3];
            output4[0] = output_sum0_7[4];
            output5[0] = output_sum0_7[5];
            output6[0] = output_sum0_7[6];
            output7[0] = output_sum0_7[7];
#else
            float sum0 = 0;
            float sum1 = 0;
            float sum2 = 0;
            float sum3 = 0;
            float sum4 = 0;
            float sum5 = 0;
            float sum6 = 0;
            float sum7 = 0;
            
            for (int k=0; k<K; k++)
            {
                sum0 += va[0] * vb[0];
                sum1 += va[1] * vb[0];
                sum2 += va[2] * vb[0];
                sum3 += va[3] * vb[0];
                sum4 += va[4] * vb[0];
                sum5 += va[5] * vb[0];
                sum6 += va[6] * vb[0];
                sum7 += va[7] * vb[0];
                
                va += 8;
                vb += 1;
            }
            output0[0] = sum0;
            output1[0] = sum1;
            output2[0] = sum2;
            output3[0] = sum3;
            output4[0] = sum4;
            output5[0] = sum5;
            output6[0] = sum6;
            output7[0] = sum7;
#endif // __AVX__
            output0++;
            output1++;
            output2++;
            output3++;
            output4++;
            output5++;
            output6++;
            output7++;
        }
    }

    nn_outch = (M - remain_outch_start) >> 2;

    for (int pp=0; pp<nn_outch; pp++)
    {
        int i = remain_outch_start + pp * 4;

        float* output0 = pC + (i  )*N;
        float* output1 = pC + (i+1)*N;
        float* output2 = pC + (i+2)*N;
        float* output3 = pC + (i+3)*N;

        int j=0;
        for (; j+7<N; j+=8)
        {
            float* va = pA_t + (i/8 + (i%8)/4) * 8*K;
            float* vb = pB_t + (j/8) * 8*K;
#if __AVX__
            __m256 _sum0 = _mm256_set1_ps(0.0);
            __m256 _sum1 = _mm256_set1_ps(0.0);
            __m256 _sum2 = _mm256_set1_ps(0.0);
            __m256 _sum3 = _mm256_set1_ps(0.0);

            int k=0;
            for (; k+3<K; k=k+4)
            {
                // k0
                __m256 _va0 = _mm256_broadcast_ss(va);
                __m256 _va1 = _mm256_broadcast_ss(va+1);
                __m256 _va2 = _mm256_broadcast_ss(va+2);
                __m256 _va3 = _mm256_broadcast_ss(va+3);
                __m256 _vb0 = _mm256_loadu_ps(vb);
                __m256 _vb1 = _mm256_loadu_ps(vb+8);
                __m256 _vb2 = _mm256_loadu_ps(vb+16);
                __m256 _vb3 = _mm256_loadu_ps(vb+24);
                _sum0 = _mm256_fmadd_ps(_vb0, _va0, _sum0);    // sum0 = (a00-a07) * k00
                _sum1 = _mm256_fmadd_ps(_vb0, _va1, _sum1);    // sum1 = (a00-a07) * k10
                _sum2 = _mm256_fmadd_ps(_vb0, _va2, _sum2);    // sum2 = (a00-a07) * k20
                _sum3 = _mm256_fmadd_ps(_vb0, _va3, _sum3);    // sum3 = (a00-a07) * k30

                va += 4;

                // k1
                _va0 = _mm256_broadcast_ss(va);
                _va1 = _mm256_broadcast_ss(va+1);
                _va2 = _mm256_broadcast_ss(va+2);
                _va3 = _mm256_broadcast_ss(va+3);                  
                _sum0 = _mm256_fmadd_ps(_vb1, _va0, _sum0);    // sum0 += (a10-a17) * k01
                _sum1 = _mm256_fmadd_ps(_vb1, _va1, _sum1);    // sum1 += (a10-a17) * k11
                _sum2 = _mm256_fmadd_ps(_vb1, _va2, _sum2);    // sum2 += (a10-a17) * k21
                _sum3 = _mm256_fmadd_ps(_vb1, _va3, _sum3);    // sum3 += (a10-a17) * k31

                va += 4;

                // k2
                _va0 = _mm256_broadcast_ss(va);
                _va1 = _mm256_broadcast_ss(va+1);
                _va2 = _mm256_broadcast_ss(va+2);
                _va3 = _mm256_broadcast_ss(va+3);
                _sum0 = _mm256_fmadd_ps(_vb2, _va0, _sum0);    // sum0 += (a20-a27) * k02
                _sum1 = _mm256_fmadd_ps(_vb2, _va1, _sum1);    // sum1 += (a20-a27) * k12
                _sum2 = _mm256_fmadd_ps(_vb2, _va2, _sum2);    // sum2 += (a20-a27) * k22
                _sum3 = _mm256_fmadd_ps(_vb2, _va3, _sum3);    // sum3 += (a20-a27) * k32

                va += 4;                  

                // k3
                _va0 = _mm256_broadcast_ss(va);
                _va1 = _mm256_broadcast_ss(va+1);
                _va2 = _mm256_broadcast_ss(va+2);
                _va3 = _mm256_broadcast_ss(va+3);
                _sum0 = _mm256_fmadd_ps(_vb3, _va0, _sum0);    // sum0 += (a30-a37) * k03
                _sum1 = _mm256_fmadd_ps(_vb3, _va1, _sum1);    // sum1 += (a30-a37) * k13
                _sum2 = _mm256_fmadd_ps(_vb3, _va2, _sum2);    // sum2 += (a30-a37) * k23
                _sum3 = _mm256_fmadd_ps(_vb3, _va3, _sum3);    // sum3 += (a30-a37) * k33

                va += 4;
                vb += 32;
            }

            for (; k<K; k++)
            {
                // k0
                __m256 _va0 = _mm256_broadcast_ss(va);
                __m256 _va1 = _mm256_broadcast_ss(va+1);
                __m256 _va2 = _mm256_broadcast_ss(va+2);
                __m256 _va3 = _mm256_broadcast_ss(va+3);
                __m256 _vb0 = _mm256_loadu_ps(vb);
                _sum0 = _mm256_fmadd_ps(_vb0, _va0, _sum0);    // sum0 = (a00-a07) * k00
                _sum1 = _mm256_fmadd_ps(_vb0, _va1, _sum1);    // sum1 = (a00-a07) * k10
                _sum2 = _mm256_fmadd_ps(_vb0, _va2, _sum2);    // sum2 = (a00-a07) * k20
                _sum3 = _mm256_fmadd_ps(_vb0, _va3, _sum3);    // sum3 = (a00-a07) * k30

                va += 4;
                vb += 8;
            }

            _mm256_storeu_ps(output0, _sum0);
            _mm256_storeu_ps(output1, _sum1); 
            _mm256_storeu_ps(output2, _sum2);
            _mm256_storeu_ps(output3, _sum3);   
#else
            float sum0[8] = {0};
            float sum1[8] = {0};
            float sum2[8] = {0};
            float sum3[8] = {0};

            for (int k=0; k<K; k++)
            {
                for (int n=0; n<8; n++)
                {
                    sum0[n] += va[0] * vb[n];
                    sum1[n] += va[1] * vb[n];
                    sum2[n] += va[2] * vb[n];
                    sum3[n] += va[3] * vb[n];
                }
                
                va += 4;
                vb += 8;
            }

            for (int n=0; n<8; n++)
            {
                output0[n] = sum0[n];
                output1[n] = sum1[n];
                output2[n] = sum2[n];
                output3[n] = sum3[n];
            }
#endif // __AVX__
            output0 += 8;
            output1 += 8;
            output2 += 8;
            output3 += 8;
        }

        for (; j<N; j++)
        {
            float* va = pA_t + (i/8 + (i%8)/4) * 8*K;
            float* vb = pB_t + (j/8 + j%8) * 8*K;
#if __AVX__
            __m128 _sum0_3 = _mm_set1_ps(0.0);
            __m128 _sum0 = _mm_set1_ps(0.0);
            __m128 _sum1 = _mm_set1_ps(0.0);
            __m128 _sum2 = _mm_set1_ps(0.0);
            __m128 _sum3 = _mm_set1_ps(0.0);

            int k=0;
            for (; k+3<K; k=k+4)
            {
                __m128 _vb0 = _mm_set1_ps(vb[0]);
                __m128 _vb1 = _mm_set1_ps(vb[1]);
                __m128 _vb2 = _mm_set1_ps(vb[2]);
                __m128 _vb3 = _mm_set1_ps(vb[3]);
                __m128 _va0 = _mm_loadu_ps(va);
                __m128 _va1 = _mm_loadu_ps(va+4);
                __m128 _va2 = _mm_loadu_ps(va+8);
                __m128 _va3 = _mm_loadu_ps(va+12);

                _sum0 = _mm_fmadd_ps(_va0, _vb0, _sum0);// sum0 += (k00-k30) * a00
                _sum1 = _mm_fmadd_ps(_va1, _vb1, _sum1);// sum1 += (k01-k31) * a10
                _sum2 = _mm_fmadd_ps(_va2, _vb2, _sum2);// sum2 += (k02-k32) * a20
                _sum3 = _mm_fmadd_ps(_va3, _vb3, _sum3);// sum3 += (k03-k33) * a30

                va += 16;
                vb += 4;
            }

            _sum0 = _mm_add_ps(_sum0, _sum1);
            _sum2 = _mm_add_ps(_sum2, _sum3);
            _sum0_3 = _mm_add_ps(_sum0_3, _sum0);
            _sum0_3 = _mm_add_ps(_sum0_3, _sum2);

            for (; k<K; k++)
            {
                __m128 _vb0 = _mm_set1_ps(vb[0]);
                __m128 _va = _mm_loadu_ps(va); 

                _sum0_3 = _mm_fmadd_ps(_va, _vb0, _sum0_3);// sum0 += (k00-k30) * a00

                va += 4;
                vb += 1;
            }

            float output_sum0_3[4] = {0.f};
            _mm_storeu_ps(output_sum0_3, _sum0_3); 
            output0[0] = output_sum0_3[0];
            output1[0] = output_sum0_3[1];
            output2[0] = output_sum0_3[2];
            output3[0] = output_sum0_3[3];
#else
            float sum0 = 0;
            float sum1 = 0;
            float sum2 = 0;
            float sum3 = 0;
            
            for (int k=0; k<K; k++)
            {
                sum0 += va[0] * vb[0];
                sum1 += va[1] * vb[0];
                sum2 += va[2] * vb[0];
                sum3 += va[3] * vb[0];
                
                va += 4;
                vb += 1;
            }
            output0[0] = sum0;
            output1[0] = sum1;
            output2[0] = sum2;
            output3[0] = sum3;
#endif // __AVX__
            output0++;
            output1++;
            output2++;
            output3++;
        }         
    }

    remain_outch_start += nn_outch << 2;

    // output ch0
    for (int i=remain_outch_start; i<M; i++)
    {
        float* output = pC + i*N;

        int j=0;
        for (; j+7<N; j+=8)
        {
            float* va = pA_t + (i/8 + (i%8)/4 + i%4) * 8*K;
            float* vb = pB_t + (j/8) * 8*K;
#if __AVX__
            __m256 _sum0 = _mm256_set1_ps(0.0);

            int k=0;
            for (; k+3<K; k=k+4)
            {
                // k0
                __m256 _va0 = _mm256_broadcast_ss(va);
                __m256 _va1 = _mm256_broadcast_ss(va+1);
                __m256 _va2 = _mm256_broadcast_ss(va+2);
                __m256 _va3 = _mm256_broadcast_ss(va+3);
                __m256 _vb0 = _mm256_loadu_ps(vb);
                __m256 _vb1 = _mm256_loadu_ps(vb+8);
                __m256 _vb2 = _mm256_loadu_ps(vb+16);
                __m256 _vb3 = _mm256_loadu_ps(vb+24);

                _sum0 = _mm256_fmadd_ps(_vb0, _va0, _sum0);    // sum0 = (a00-a07) * k00
                _sum0 = _mm256_fmadd_ps(_vb1, _va1, _sum0);    // sum0 += (a10-a17) * k01
                _sum0 = _mm256_fmadd_ps(_vb2, _va2, _sum0);    // sum0 += (a20-a27) * k02
                _sum0 = _mm256_fmadd_ps(_vb3, _va3, _sum0);    // sum0 += (a30-a37) * k03
            
                va += 4;
                vb += 32;
            }

            for (; k<K; k++)
            {
                // k0
                __m256 _va0 = _mm256_broadcast_ss(va);
                __m256 _vb0 = _mm256_loadu_ps(vb);

                _sum0 = _mm256_fmadd_ps(_vb0, _va0, _sum0);    // sum0 = (a00-a07) * k00

                va += 1;
                vb += 8;
            }

            _mm256_storeu_ps(output, _sum0); 
#else                
            float sum[8] = {0};
            
            for (int k=0; k<K; k++)
            {
                for (int n=0; n<8; n++)
                {
                    sum[n] += va[0] * vb[n];
                }

                va += 1;
                vb += 8;
            }

            for (int n=0; n<8; n++)
            {
                output[n] = sum[n];
            }
#endif // __AVX__
            output += 8;
        }

        for (; j<N; j++)
        {
            float* va = pA_t + (i/8 + (i%8)/4 + i%4) * 8*K;
            float* vb = pB_t + (j/8 + j%8) * 8*K;

            int k=0;
#if __AVX__
            __m128 _sum0 = _mm_set1_ps(0.f);

            for (; k+3<K; k+=4)
            {
                __m128 _p0 = _mm_loadu_ps(vb);
                __m128 _k0 = _mm_loadu_ps(va);
                _sum0 = _mm_add_ps(_sum0, _mm_mul_ps(_p0, _k0));

                va += 4;
                vb += 4;
            }
            float sum0 = _sum0[0] + _sum0[1] + _sum0[2] + _sum0[3];
#else
            float sum0 = 0.f;
#endif // __AVX__
            for (; k<K; k++)
            {
                sum0 += va[0] * vb[0];

                va += 1;
                vb += 1;
            }
            output[0] = sum0;

            output++;
        }
    }

    // printf("sgemm avx2 done\n");

    free(pA_t);
    free(pB_t);
}
#else // SSE2
// int M = outch;       // outch
// int N = outw * outh; // outsize or out stride
// int K = kernel_w * kernel_h * inch; // ksize * inch
// float* pA = kernel
// float* pB = input_data
// float* pC = output_data
void sgemm(int M, int N, int K, float* pA, float* pB, float* pC) // unloop output M, unloop N, packet 4x4, using intrinsic
{
    // kernel pack 4
    float* pA_t = (float* )malloc((4*K * (M/4 + M%4)) * sizeof(float));
    {
        int nn_outch = M >> 2;
        int remain_outch_start = nn_outch << 2;

        for (int pp=0; pp<nn_outch; pp++)
        {
            int p = pp * 4;

            const float* k0 = pA + (p+0)*K;
            const float* k1 = pA + (p+1)*K;
            const float* k2 = pA + (p+2)*K;
            const float* k3 = pA + (p+3)*K;

            float* ktmp = pA_t + (p/4) * 4*K;

            for (int q=0; q<K; q++)
            {
                ktmp[0] = k0[0];
                ktmp[1] = k1[0];
                ktmp[2] = k2[0];
                ktmp[3] = k3[0];
                ktmp += 4;

                k0 += 1;
                k1 += 1;
                k2 += 1;
                k3 += 1;
            }
        }
        
        for (int p=remain_outch_start; p<M; p++)
        {
            const float* k0 = pA + (p+0)*K;

            float* ktmp = pA_t + (p/4 + p%4) * 4*K;

            for (int q=0; q<K; q++)
            {
                ktmp[0] = k0[0];
                ktmp++;
                k0++;
            }
        }
    }

    // data, col2row, pack
    float* pB_t = (float* )malloc((4*K * (N/4 + N%4)) * sizeof(float));
    {
        int nn_size = N >> 2;
        int remian_size_start = nn_size << 2;

        // [ch00, ch10, ch20, ch30, ch01, ch11, ch21, ch31, ch02, ch12, ch22, ch32, ch03, ch13, ch23, ch33 ....]
        for (int ii=0; ii<nn_size; ii++)
        {
            int i = ii * 4;
            const float* img = pB + i;
            float* tmp = pB_t + (i/4) * 4*K;

            for (int j=0; j<K; j++)
            {
#if __SSE__
                _mm_storeu_ps(tmp, _mm_loadu_ps(img));
#else                                  
                tmp[0] = img[0];
                tmp[1] = img[1];
                tmp[2] = img[2];
                tmp[3] = img[3];
#endif // __SSE__
                tmp += 4;
                img += N;
            }
        }

        // [ch00, ch01, ch02, ch03 ....]   
        for (int i=remian_size_start; i<N; i++)
        {
            const float* img = pB + i;
            float* tmp = pB_t + (i/4 + i%4) * 4*K;

            for (int j=0; j<K; j++)
            {
                tmp[0] = img[0];

                tmp += 1;
                img += N;
            }                
        }
    }

    // output ch0 - ch3
    int i=0;
    for (; i+3<M; i+=4)
    {
        float* output0 = pC + (i  )*N;
        float* output1 = pC + (i+1)*N;
        float* output2 = pC + (i+2)*N;
        float* output3 = pC + (i+3)*N;

        int j=0;
        for (; j+3<N; j+=4)
        {
            float* va = pA_t + (i/4) * 4*K;
            float* vb = pB_t + (j/4) * 4*K;
#if __SSE__
            __m128 _sum0 = _mm_set1_ps(0.f);
            __m128 _sum1 = _mm_set1_ps(0.f);
            __m128 _sum2 = _mm_set1_ps(0.f);
            __m128 _sum3 = _mm_set1_ps(0.f);

            int k=0;
            for (; k+3<K; k=k+4)
            {
                // k0
                __m128 _vb = _mm_loadu_ps(vb);
                __m128 _va0 = _mm_set1_ps(va[0]);
                __m128 _va1 = _mm_set1_ps(va[1]);
                __m128 _va2 = _mm_set1_ps(va[2]);
                __m128 _va3 = _mm_set1_ps(va[3]);
                _sum0 = _mm_add_ps(_sum0, _mm_mul_ps(_vb, _va0));// sum0 = (a00-a03) * k00
                _sum1 = _mm_add_ps(_sum1, _mm_mul_ps(_vb, _va1));// sum1 = (a00-a03) * k10
                _sum2 = _mm_add_ps(_sum2, _mm_mul_ps(_vb, _va2));// sum2 = (a00-a03) * k20
                _sum3 = _mm_add_ps(_sum3, _mm_mul_ps(_vb, _va3));// sum3 = (a00-a03) * k30

                // k1
                _vb = _mm_loadu_ps(vb+4);
                _va0 = _mm_set1_ps(va[4]);
                _va1 = _mm_set1_ps(va[5]);
                _va2 = _mm_set1_ps(va[6]);
                _va3 = _mm_set1_ps(va[7]);
                _sum0 = _mm_add_ps(_sum0, _mm_mul_ps(_vb, _va0));// sum0 = (a10-a13) * k01
                _sum1 = _mm_add_ps(_sum1, _mm_mul_ps(_vb, _va1));// sum1 = (a10-a13) * k11
                _sum2 = _mm_add_ps(_sum2, _mm_mul_ps(_vb, _va2));// sum2 = (a10-a13) * k21
                _sum3 = _mm_add_ps(_sum3, _mm_mul_ps(_vb, _va3));// sum3 = (a10-a13) * k31

                // k2
                _vb = _mm_loadu_ps(vb+8);
                _va0 = _mm_set1_ps(va[8]);
                _va1 = _mm_set1_ps(va[9]);
                _va2 = _mm_set1_ps(va[10]);
                _va3 = _mm_set1_ps(va[11]);
                _sum0 = _mm_add_ps(_sum0, _mm_mul_ps(_vb, _va0));// sum0 = (a20-a23) * k02
                _sum1 = _mm_add_ps(_sum1, _mm_mul_ps(_vb, _va1));// sum1 = (a20-a23) * k12
                _sum2 = _mm_add_ps(_sum2, _mm_mul_ps(_vb, _va2));// sum2 = (a20-a23) * k22
                _sum3 = _mm_add_ps(_sum3, _mm_mul_ps(_vb, _va3));// sum3 = (a20-a23) * k32

                // k3
                _vb = _mm_loadu_ps(vb+12);
                _va0 = _mm_set1_ps(va[12]);
                _va1 = _mm_set1_ps(va[13]);
                _va2 = _mm_set1_ps(va[14]);
                _va3 = _mm_set1_ps(va[15]);
                _sum0 = _mm_add_ps(_sum0, _mm_mul_ps(_vb, _va0));// sum0 = (a30-a33) * k03
                _sum1 = _mm_add_ps(_sum1, _mm_mul_ps(_vb, _va1));// sum1 = (a30-a33) * k13
                _sum2 = _mm_add_ps(_sum2, _mm_mul_ps(_vb, _va2));// sum2 = (a30-a33) * k23
                _sum3 = _mm_add_ps(_sum3, _mm_mul_ps(_vb, _va3));// sum3 = (a30-a33) * k33

                va += 16;
                vb += 16;
            }

            for (; k<K; k++)
            {
                // k0
                __m128 _vb = _mm_loadu_ps(vb);
                __m128 _va0 = _mm_set1_ps(va[0]);
                __m128 _va1 = _mm_set1_ps(va[1]);
                __m128 _va2 = _mm_set1_ps(va[2]);
                __m128 _va3 = _mm_set1_ps(va[3]);
                _sum0 = _mm_add_ps(_sum0, _mm_mul_ps(_vb, _va0));// sum0 = (a00-a03) * k00
                _sum1 = _mm_add_ps(_sum1, _mm_mul_ps(_vb, _va1));// sum1 = (a00-a03) * k10
                _sum2 = _mm_add_ps(_sum2, _mm_mul_ps(_vb, _va2));// sum2 = (a00-a03) * k20
                _sum3 = _mm_add_ps(_sum3, _mm_mul_ps(_vb, _va3));// sum3 = (a00-a03) * k30
                
                va += 4;
                vb += 4;
            }
            _mm_storeu_ps(output0, _sum0);
            _mm_storeu_ps(output1, _sum1);
            _mm_storeu_ps(output2, _sum2);
            _mm_storeu_ps(output3, _sum3);
#else
            float sum0[4] = {0};
            float sum1[4] = {0};
            float sum2[4] = {0};
            float sum3[4] = {0};

            for (int k=0; k<K; k++)
            {
                for (int n=0; n<4; n++)
                {
                    sum0[n] += va[0] * vb[n];
                    sum1[n] += va[1] * vb[n];
                    sum2[n] += va[2] * vb[n];
                    sum3[n] += va[3] * vb[n];
                }                    
                
                va += 4;
                vb += 4;
            }

            for (int n=0; n<4; n++)
            {
                output0[n] = sum0[n];
                output1[n] = sum1[n];
                output2[n] = sum2[n];
                output3[n] = sum3[n];
            }
#endif // __SSE__
            output0 += 4;
            output1 += 4;
            output2 += 4;
            output3 += 4;
        }

        for (; j<N; j++)
        {
            float* va = pA_t + (i/4) * 4*K;
            float* vb = pB_t + (j/4 + j%4) * 4*K;

#if __SSE__
            __m128 _sum0_3 = _mm_set1_ps(0.f);
            __m128 _sum0 = _mm_set1_ps(0.f);
            __m128 _sum1 = _mm_set1_ps(0.f);
            __m128 _sum2 = _mm_set1_ps(0.f);
            __m128 _sum3 = _mm_set1_ps(0.f);

            int k=0;
            for (; k+3<K; k=k+4)
            {
                __m128 _vb0 = _mm_set1_ps(vb[0]);
                __m128 _vb1 = _mm_set1_ps(vb[1]);
                __m128 _vb2 = _mm_set1_ps(vb[2]);
                __m128 _vb3 = _mm_set1_ps(vb[3]);
                __m128 _va0 = _mm_loadu_ps(va);
                __m128 _va1 = _mm_loadu_ps(va+4);
                __m128 _va2 = _mm_loadu_ps(va+8);
                __m128 _va3 = _mm_loadu_ps(va+12);

                _sum0 = _mm_add_ps(_sum0, _mm_mul_ps(_va0, _vb0));// sum0 += (k00-k30) * a00
                _sum1 = _mm_add_ps(_sum1, _mm_mul_ps(_va1, _vb1));// sum1 += (k01-k31) * a10
                _sum2 = _mm_add_ps(_sum2, _mm_mul_ps(_va2, _vb2));// sum2 += (k02-k32) * a20
                _sum3 = _mm_add_ps(_sum3, _mm_mul_ps(_va3, _vb3));// sum3 += (k03-k33) * a30

                va += 16;
                vb += 4;
            }

            _sum0 = _mm_add_ps(_sum0, _sum1);
            _sum2 = _mm_add_ps(_sum2, _sum3);
            _sum0_3 = _mm_add_ps(_sum0_3, _sum0);
            _sum0_3 = _mm_add_ps(_sum0_3, _sum2);

            for (; k<K; k++)
            {
                __m128 _vb0 = _mm_set1_ps(vb[0]);
                __m128 _va = _mm_loadu_ps(va); 

                _sum0_3 = _mm_add_ps(_sum0_3, _mm_mul_ps(_va, _vb0));// sum0 += (k00-k30) * a00

                va += 4;
                vb += 1;
            }         
            output0[0] = _sum0_3[0];
            output1[0] = _sum0_3[1];
            output2[0] = _sum0_3[2];
            output3[0] = _sum0_3[3];
#else
            float sum0 = 0;
            float sum1 = 0;
            float sum2 = 0;
            float sum3 = 0;                
            
            for (int k=0; k<K; k++)
            {
                sum0 += va[0] * vb[0];
                sum1 += va[1] * vb[0];
                sum2 += va[2] * vb[0];
                sum3 += va[3] * vb[0];
                
                va += 4;
                vb += 1;
            }
            output0[0] = sum0;
            output1[0] = sum1;
            output2[0] = sum2;
            output3[0] = sum3;
#endif // __SSE__
            output0++;
            output1++;
            output2++;
            output3++;
        }
    }

    // output ch0
    for (; i<M; i++)
    {
        float* output = pC + i*N;

        int j=0;
        for (; j+3<N; j+=4)
        {
            float* va = pA_t + (i/4 + i%4) * 4*K;
            float* vb = pB_t + (j/4) * 4*K;
#if __SSE__
            __m128 _sum0 = _mm_set1_ps(0.f);

            int k=0;
            for (; k+3<K; k=k+4)
            {
                // k0
                __m128 _va0 = _mm_set1_ps(va[0]);
                __m128 _va1 = _mm_set1_ps(va[1]);
                __m128 _va2 = _mm_set1_ps(va[2]);
                __m128 _va3 = _mm_set1_ps(va[3]);
                __m128 _vb0 = _mm_loadu_ps(vb);
                __m128 _vb1 = _mm_loadu_ps(vb+4);
                __m128 _vb2 = _mm_loadu_ps(vb+8);
                __m128 _vb3 = _mm_loadu_ps(vb+12);

                _sum0 = _mm_add_ps(_sum0, _mm_mul_ps(_vb0, _va0));// sum0 = (a00-a03) * k00                
                _sum0 = _mm_add_ps(_sum0, _mm_mul_ps(_vb1, _va1));// sum0 += (a10-a13) * k01
                _sum0 = _mm_add_ps(_sum0, _mm_mul_ps(_vb2, _va2));// sum0 += (a20-a23) * k02
                _sum0 = _mm_add_ps(_sum0, _mm_mul_ps(_vb3, _va3));// sum0 += (a30-a33) * k03
            
                va += 4;
                vb += 16;
            }

            for (; k<K; k++)
            {
                // k0
                __m128 _va0 = _mm_set1_ps(va[0]);
                __m128 _vb0 = _mm_loadu_ps(vb);

                _sum0 = _mm_add_ps(_sum0, _mm_mul_ps(_vb0, _va0));    // sum0 = (a00-a03) * k00

                va += 1;
                vb += 4;
            }
            _mm_storeu_ps(output, _sum0);
#else                
            float sum[4] = {0};
            
            for (int k=0; k<K; k++)
            {
                for (int n=0; n<4; n++)
                {
                    sum[n] += va[0] * vb[n];
                }

                va += 1;
                vb += 4;
            }

            for (int n=0; n<4; n++)
            {
                output[n] = sum[n];
            }
#endif // __SSE__
            output += 4;
        }

        for (; j<N; j++)
        {
            float* va = pA_t + (i/4 + i%4) * 4*K;
            float* vb = pB_t + (j/4 + j%4) * 4*K;

            int k=0;
#if __SSE__
            __m128 _sum0 = _mm_set1_ps(0.f);

            for (; k+3<K; k+=4)
            {
                __m128 _p0 = _mm_loadu_ps(vb);
                __m128 _k0 = _mm_loadu_ps(va);
                _sum0 = _mm_add_ps(_sum0, _mm_mul_ps(_p0, _k0));

                va += 4;
                vb += 4;
            }
            float sum0 = _sum0[0] + _sum0[1] + _sum0[2] + _sum0[3];
#else
            float sum0 = 0.f;
#endif // __SSE__
            for (; k<K; k++)
            {
                sum0 += va[0] * vb[0];

                va += 1;
                vb += 1;
            }
            output[0] = sum0;

            output++;
        }
    }

    free(pA_t);
    free(pB_t);
}
#endif // __AVX2__

#endif
