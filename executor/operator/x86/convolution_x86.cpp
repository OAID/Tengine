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
#include <iostream>
#include <functional>
#include <cstring>
#include <algorithm>

#include "logger.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "graph.hpp"
#include "operator/convolution.hpp"
#include <math.h>

#if __SSE2__
#include <emmintrin.h>
#endif

namespace TEngine {

namespace ConvolutionImpl {
const char* conv_name = "CONV_X86";
const int default_prio = 500;

struct ConvolutionOps : public NodeOps
{
    void im2col(float* data_img, float* data_col, int inh, int inw, int inc, int outh, int outw, int outc, int ksize_h,
                int ksize_w, int sh, int sw, int ph, int pw, int dh, int dw)
    {
        const int channels_col = ksize_h * ksize_w * inc;

        for(int c = 0; c < channels_col; ++c)
        {
            const int kw = c % ksize_w;
            int c_ = c / ksize_w;
            const int kh = c_ % ksize_h;
            c_ = c_ / ksize_h;
            const int im_col = kw * dw - pw;
            const int w_low = std::max(0, -im_col / sw + (-im_col % sw > 0));
            const int w_high = std::min(outw, (inw - im_col) / sw + ((inw - im_col) % sw > 0));

            for(int h = 0; h < outh; ++h)
            {
                const int im_row = kh * dh + h * sh - ph;
                float* out = data_col + (c * outh + h) * outw;
                const float* end = out + w_high;

                if(im_row >= 0 && im_row < inh)
                {
                    float* in = data_img + inw * (im_row + inh * c_) + im_col + (w_low - 1) * sw;

                    memset(out, 0, w_low * sizeof(float));
                    out += w_low;
                    while(out < end)
                    {
                        in += sw;
                        *(out++) = *in;
                    }
                    memset(out, 0, (outw - w_high) * sizeof(float));
                }
                else
                {
                    memset(out, 0, outw * sizeof(float));
                }
            }
        }
    }
    void relu(float* data, int size, int activation)
    {
        for(int i = 0; i < size; i++)
        {
            data[i] = std::max(data[i], ( float )0);

            if(activation > 0)
            {
                data[i] = std::min(data[i], ( float )activation);
            }
        }
    }
    void add_bias(float* output, float* bias, int c_out, int hw)
    {
        for(int c = 0; c < c_out; ++c)
        {
            for(int i = 0; i < hw; ++i)
            {
                output[c * hw + i] += bias[c];
            }
        }
    }
    bool Prerun(Node* node)
    {
        Convolution* conv_op = dynamic_cast<Convolution*>(node->GetOp());
        ConvParam* param = conv_op->GetParam();

        const Tensor* input_tensor = node->GetInputTensor(0);
        const TShape& shape = input_tensor->GetShape();
        const std::vector<int> in_dims = shape.GetDim();

        Tensor* output_tensor = node->GetOutputTensor(0);
        TShape& shape1 = output_tensor->GetShape();
        std::vector<int> out_dims = shape1.GetDim();

        int size = param->kernel_h * param->kernel_w * in_dims[1] / param->group * out_dims[2] * out_dims[3];
        float* buffer = ( float* )std::malloc(sizeof(float) * size);
        (*node)["buffer"] = buffer;
        return true;
    }

    bool Reshape(Node* node)
    {
        if(node->ExistAttr("buffer"))
        {
            float* buffer = any_cast<float*>(node->GetAttr("buffer"));
            free(buffer);
            node->RemoveAttr("buffer");
        }

        return Prerun(node);
    }

    // int M = outch;       // outch
    // int N = outw * outh; // outsize or out stride
    // int K = kernel_w * kernel_h * inch; // ksize * inch
    // float* pA = kernel
    // float* pB = input_data
    // float* pC = output_data
	/* 
    void sgemm(int M, int N, int K, float* pA, float* pB, float* pC) // round 1, 'tu fa' sgemm
    {
        for (int i=0; i<M; i++)
        {
            float* output = pC + i*N;

            for (int j=0; j<N; j++)
            {
                float sum = 0;

                for (int k=0; k<K; k++)
                {
                    sum += pA[i*K + k] * pB[k*N + j];
                }

                output[0] = sum;
                output++;
            }
        }
    }
    */
    /*
    void sgemm(int M, int N, int K, float* pA, float* pB, float* pC) // round 2, reorder data matrix
    {
        float* pB_t = (float* )malloc(N * K * sizeof(float));

        // data, col2row
        for (int i=0; i<K; i++)
        {
            for (int j=0; j<N; j++)
            {
                pB_t[j*K + i] = pB[i*N + j];
            }
        }

        for (int i=0; i<M; i++)
        {
            float* output = pC + i*N;

            for (int j=0; j<N; j++)
            {
                float sum = 0;

                float* va = pA + i*K;
                float* vb = pB_t + j*K;

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

        free(pB_t);
    }
    */
    /* 
    void sgemm(int M, int N, int K, float* pA, float* pB, float* pC) // round 3, unloop output ch
    {
        float* pB_t = (float* )malloc(N * K * sizeof(float));
            
        // data, col2row
        for (int i=0; i<K; i++)
        {
            for (int j=0; j<N; j++)
            {
                pB_t[j*K + i] = pB[i*N + j];
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

            for (int j=0; j<N; j++)
            {
                float sum0 = 0;
                float sum1 = 0;
                float sum2 = 0;
                float sum3 = 0;
                
                float* va0 = pA + (i  )*K;
                float* va1 = pA + (i+1)*K;
                float* va2 = pA + (i+2)*K;
                float* va3 = pA + (i+3)*K;
                
                float* vb = pB_t + j*K;
                
                for (int k=0; k<K; k++)
                {
                    sum0 += va0[0] * vb[0];
                    sum1 += va1[0] * vb[0];
                    sum2 += va2[0] * vb[0];
                    sum3 += va3[0] * vb[0];
                    
                    va0 += 1;
                    va1 += 1;
                    va2 += 1;
                    va3 += 1;
                    
                    vb += 1;
                }
                output0[0] = sum0;
                output1[0] = sum1;
                output2[0] = sum2;
                output3[0] = sum3;

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

            for (int j=0; j<N; j++)
            {
                float sum = 0;
                
                float* va = pA   + i*K;
                float* vb = pB_t + j*K;
                
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

        free(pB_t);
    }
    */
    void sgemm(int M, int N, int K, float* pA, float* pB, float* pC) // round 4, unloop output M, unloop N, packet 4x4, using intrinsic
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
                float* va = pA_t + (i/4 + j%4) * 4*K;
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
                float* va = pA_t + (i/4 + j%4) * 4*K;
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

    bool Run(Node* node)
    {
        bool debug_conv = false;
        const char* debug_env = std::getenv("DEBUG_CONV");
        if((debug_env) && (debug_env[0] == '1'))
        {
            debug_conv = true;
        }

        Tensor* input_tensor = node->GetInputTensor(0);
        Tensor* output_tensor = node->GetOutputTensor(0);
        Tensor* weight_tensor = node->GetInputTensor(1);
        float* buffer = any_cast<float*>(node->GetAttr("buffer"));

        Convolution* conv_op = dynamic_cast<Convolution*>(node->GetOp());
        ConvParam* param = conv_op->GetParam();
        int activation = param->activation;

        float* input = ( float* )get_tensor_mem(input_tensor);
        float* output = ( float* )get_tensor_mem(output_tensor);
        float* kernel = ( float* )get_tensor_mem(weight_tensor);

        const TShape& shape = input_tensor->GetShape();
        const std::vector<int> in_dims = shape.GetDim();
        const TShape& shape1 = output_tensor->GetShape();
        const std::vector<int> out_dims = shape1.GetDim();

        int batch_number = in_dims[0];
        int inc = in_dims[1];
        int inh = in_dims[2];
        int inw = in_dims[3];
        int in_chw = inc * inh * inw;

        int outc = out_dims[1];
        int outh = out_dims[2];
        int outw = out_dims[3];
        int out_hw = outh * outw;
        int out_chw = out_hw * outc;

        int ksize_h = param->kernel_h;
        int ksize_w = param->kernel_w;
        int pad_w = param->pad_w0;
        int pad_h = param->pad_h0;

        int stride_w = param->stride_w;
        int stride_h = param->stride_h;
        int dilation_w = param->dilation_w;
        int dilation_h = param->dilation_h;
        int group = param->group;

        int inc_g = inc / group;
        int outc_g = outc / group;

        int m = outc_g;
        int k = ksize_h * ksize_w * inc_g;
        int n = outh * outw;

        int in_chw_g = inh * inw * inc_g;
        int out_chw_g = outc_g * out_hw;
        int kernel_size_g = outc_g * ksize_h * ksize_w * inc_g;

        bool have_biases = (node->GetInputNum() > 2);
        float* biases = nullptr;

        if(debug_conv)
        {
            std::cout << inc << " " << inh << " " << inw << "\tksp dg: " << ksize_h << " " << stride_h << " " << pad_h
                      << " " << dilation_w << " " << group << "\t" << outc << " " << outh << " " << outw << "\t";
        }

        for(int i = 0; i < batch_number; i++)
        {
            for(int g = 0; g < group; g++)
            {
                im2col(input + i * in_chw + g * in_chw_g, buffer, inh, inw, inc_g, outh, outw, outc_g, ksize_h, ksize_w,
                       stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w);
                sgemm(m, n, k, kernel + g * kernel_size_g, buffer, output + i * out_chw + g * out_chw_g);
            }
        }
        if(have_biases)
        {
            biases = ( float* )get_tensor_mem(node->GetInputTensor(2));
            for(int i = 0; i < batch_number; i++)
            {
                add_bias(output + i * out_chw, biases, outc, out_hw);
            }
        }
        if(activation >= 0)
        {
            relu(output, batch_number * out_chw, activation);
        }
        if(debug_conv)
        {
            std::cout << output[0] << " " << output[10] << "\n";
        }

        return true;
    }

    bool Postrun(Node* node)
    {
        float* addr;
        if(node->ExistAttr("buffer"))
        {
            addr = any_cast<float*>(node->GetAttr("buffer"));
            std::free(addr);
        }
        return true;
    }
};

NodeOps* SelectFunc(const CPUInfo* cpu_info, Node* node)
{
    Tensor* input = node->GetInputTensor(0);
    const int data_type = input->GetDataType();
    const ExecAttr* exec_attr = any_cast<const ExecAttr*>(node->GetAttr(ATTR_EXEC_ATTR));
    if(data_type != TENGINE_DT_FP32 || exec_attr->graph_layout != TENGINE_LAYOUT_NCHW)
        return nullptr;

    ConvolutionOps* ops = new ConvolutionOps();

    return ops;
}

}    // namespace ConvolutionImpl

void RegisterConvNodeExec_x86(void)
{
    if(!NodeOpsRegistryManager::RegisterOPImplementor("x86", "Convolution", ConvolutionImpl::SelectFunc, ConvolutionImpl::default_prio))
    {
        LOG_ERROR() << __FUNCTION__ << " :Regist OP failed for prio [" << ConvolutionImpl::default_prio << "]\n";
        printf("%s :Regist OP failed for prio %d\n", __FUNCTION__, ConvolutionImpl::default_prio);
    }
    else
    {
        printf("%s :Regist OP succeed for prio %d\n", __FUNCTION__, ConvolutionImpl::default_prio);
    }
}

}    // namespace TEngine
