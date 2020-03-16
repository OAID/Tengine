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

#include <iostream>
#include <functional>
#include <cstring>
#include <algorithm>

#include "logger.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "graph.hpp"
#include "operator/fully_connected.hpp"
#include <math.h>

#if __SSE2__
#include <emmintrin.h>
#endif

namespace TEngine {

namespace FCImpl {

struct FcBlasOps : public NodeOps
{
    // int M = batch size;          // outch
    // int N = outpuh channel num;  // outsize or out stride
    // int K = kernel_w * kernel_h * inch; // ksize * inch
    // float* pA = kernel
    // float* pB = input_data
    // float* pC = output_data
    void sgemm(int M, int N, int K, float* pA, float* pB, float* pC) // round 3, kernel pack 4, using intrinsic
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
                float* vb = pB;
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
                float* vb = pB;

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

    bool Run(Node* node)
    {
        Tensor* input_tensor = node->GetInputTensor(0);
        Tensor* output_tensor = node->GetOutputTensor(0);
        Tensor* weight_tensor = node->GetInputTensor(1);
        bool has_bias = node->GetInputNum() > 2 ? true : false;

        float* input = ( float* )get_tensor_mem(input_tensor);
        float* output = ( float* )get_tensor_mem(output_tensor);
        float* weight = ( float* )get_tensor_mem(weight_tensor);

        Tensor* bias_tensor;
        float* bias = nullptr;

        if(has_bias)
        {
            bias_tensor = node->GetInputTensor(2);
            bias = ( float* )get_tensor_mem(bias_tensor);
        }

        const TShape& shape = input_tensor->GetShape();
        const std::vector<int> in_dims = shape.GetDim();
        const TShape& shape1 = output_tensor->GetShape();
        const std::vector<int> out_dims = shape1.GetDim();

        int batch_number = in_dims[0];
        int inc = in_dims[1];
        int inh = 1;
        int inw = 1;

        if(in_dims.size() > 2)
            inh = in_dims[2];

        if(in_dims.size() > 3)
            inw = in_dims[3];

        int in_size = inc * inh * inw;

        /* specially handling on tensorflow models */
        float* converted = nullptr;

        if(exec_attr->model_format == MODEL_FORMAT_TENSORFLOW && (inh * inw > 1))
        {
            converted = ( float* )malloc(batch_number * inc * inh * inw * sizeof(float));

            for(int n = 0; n < batch_number; n++)
            {
                int img_size = inc * inh * inw;

                float* img = converted + n * img_size;
                float* src_img = input + n * img_size;

                for(int c = 0; c < inc; c++)
                    for(int h = 0; h < inh; h++)
                        for(int w = 0; w < inw; w++)
                        {
                            img[h * (inw * inc) + w * inc + c] = src_img[c * inh * inw + h * inw + w];
                        }
            }

            input = converted;
        }

        int outc = out_dims[1];

        int m = batch_number;
        int k = in_size;
        int n = outc;
        
        sgemm(m, n, k, weight, input, output);

        if(has_bias)
        {
            for(int b = 0; b < batch_number; b++)
            {
                float* out_ptr = output + b * outc;
                for(int i = 0; i < outc; ++i)
                {
                    out_ptr[i] += bias[i];
                }
            }
        }

        if(converted)
            free(converted);

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

    FcBlasOps* ops = new FcBlasOps();

    return ops;
}

}    // namespace FCImpl

void RegisterFcNodeExec_x86(void)
{
    if (!NodeOpsRegistryManager::RegisterOPImplementor("x86", "FullyConnected", FCImpl::SelectFunc, 500))
        LOG_ERROR() << __FUNCTION__ << " :Regist OP failed for prio \n";
}

}    // namespace Tengine
