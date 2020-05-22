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
 * https://github.com/Tencent/ncnn/blob/master/src/layer/innerproduct.h
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

#ifndef __FULLY_CONNECTED_X86_H__
#define __FULLY_CONNECTED_X86_H__

#include <stdlib.h>

#if __SSE2__
#include <emmintrin.h>
#endif
#if __AVX__
#include <immintrin.h>
#endif

int innerproduct(int inn, int inc, int inh, int inw, int outc, float* weight, float* input, float* output, float* _bias)
{
    size_t elemsize = sizeof(float);
    int size = inw * inh;

// outc
#pragma omp parallel for num_threads(opt.num_threads)
    for(int n = 0; n < inn; n++)
    {
        for(int p = 0; p < outc; p++)
        {
            int q = 0;
            float sum = _bias ? _bias[p] : 0.f;
            const float* weight1 = weight + p * inc * size;
            const float* input1 = input + n * inc * size;
#if __AVX__ || __SSE__
#if __SSE__
            float _sum[4] = {0.f};
            __m128 _sum0 = _mm_set1_ps(0.f);
            for(; q + 3 < inc * size; q = q + 4)
            {
                __m128 _input = _mm_loadu_ps(input1 + q);
                __m128 _weight = _mm_loadu_ps(weight1 + q);
                __m128 _sum1 = _mm_mul_ps(_input, _weight);
                _sum0 = _mm_add_ps(_sum0, _sum1);
            }
            _mm_storeu_ps(_sum, _sum0);
            float tmp = _sum[0] + _sum[1] + _sum[2] + _sum[3];
            sum = sum + tmp;
#else    //__AVX__
         // TODO
#endif
#endif
            for(; q < inc * size; q++)
            {
                float tmp = input1[q] * weight1[q];
                sum = sum + tmp;
            }

            output[n * outc + p] = sum;
        }
    }

    return 0;
}
#endif