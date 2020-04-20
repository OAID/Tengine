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
 * Copyright (c) 2020, OPEN AI LAB
 * Author: qtang@openailab.com
 */

#ifndef __FULLY_CONNECTED_X86_H__
#define __FULLY_CONNECTED_X86_H__

#include <stdlib.h>

#if __SSE2__
#include <emmintrin.h>
#endif
#if __AVX__
#include <immintrin.h>
#endif

#if 0 //__AVX__
// TODO
#else
// int M = batch size;          // outch
// int N = outpuh channel num;  // outsize or out stride
// int K = kernel_w * kernel_h * inch; // ksize * inch
// float* pA = kernel
// float* pB = input_data
// float* pC = output_data
static void sgemm(int M, int N, int K, float* pA, float* pB, float* pC) // kernel pack 4, using intrinsic
{
    // kernel pack 4
    float* pA_t = (float* )malloc((4*K * (N/4 + N%4)) * sizeof(float));
    {
        int nn_outch = N >> 2;
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
        
        for (int p=remain_outch_start; p<N; p++)
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

    for (int i=0; i<M; i++)         // batch num
    {
        float* output = pC + i*N;   // output image step

        int j=0;
        for (; j+3<N; j+=4)         // output ch0 - ch3
        {
            float* va = pA_t + (j/4) * 4*K;
            float* vb = pB + i*K;
#if __SSE2__
            __m128 _sum0 = _mm_set1_ps(0.f);

            int k=0;
            for (; k+3<K; k=k+4)
            {
                __m128 _va0 = _mm_loadu_ps(va);
                __m128 _va1 = _mm_loadu_ps(va+4);
                __m128 _va2 = _mm_loadu_ps(va+8);
                __m128 _va3 = _mm_loadu_ps(va+12);
                __m128 _vb0 = _mm_set1_ps(vb[0]);
                __m128 _vb1 = _mm_set1_ps(vb[1]);
                __m128 _vb2 = _mm_set1_ps(vb[2]);
                __m128 _vb3 = _mm_set1_ps(vb[3]);

                _sum0 = _mm_add_ps(_sum0, _mm_mul_ps(_va0, _vb0));  // sum0 =  (a00-a03) * b00
                _sum0 = _mm_add_ps(_sum0, _mm_mul_ps(_va1, _vb1));  // sum0 += (a10-a13) * b01
                _sum0 = _mm_add_ps(_sum0, _mm_mul_ps(_va2, _vb2));  // sum0 += (a20-a23) * b02
                _sum0 = _mm_add_ps(_sum0, _mm_mul_ps(_va3, _vb3));  // sum0 += (a30-a33) * b03
            
                va += 16;
                vb += 4;
            }

            for (; k<K; k++)
            {
                __m128 _vb0 = _mm_set1_ps(vb[0]);
                __m128 _va = _mm_loadu_ps(va); 

                _sum0 = _mm_add_ps(_sum0, _mm_mul_ps(_va, _vb0));   // sum0 += (a00-a30) * b00

                va += 4;
                vb += 1;
            }
            _mm_storeu_ps(output, _sum0);
#else
            float sum[4] = {0};

            for (int k=0; k<K; k++)
            {
                for (int n=0; n<4; n++)
                {
                    sum[n] += va[n] * vb[0];
                }

                va += 4;
                vb += 1;
            }

            for (int n=0; n<4; n++)
            {
                output[n] = sum[n];
            }
#endif
            output += 4;
        }

        for (; j<N; j++)            // output ch0
        {
            float sum = 0;

            float* va = pA_t + (j/4 + j%4) * 4*K;
            float* vb = pB + i*K;

            for (int k=0; k<K; k++)
            {
                sum += va[0] * vb[0];

                va += 1;
                vb += 1;
            }

            output[0] = sum;
            output++;
        }            
    }

    free(pA_t);
}
#endif

#endif
