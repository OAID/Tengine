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
 * Copyright (c) 2017, Open AI Lab
 * Author: haitao@openailab.com
 *         chunyinglv@openailab.com
 */
#include <iostream>
#include <functional>
#include <cstring>
#include <algorithm>
#include <complex>
#include <array>

#include "logger.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "graph.hpp"
#include "operator/softmax.hpp"
#include "data_type.hpp"
#include "arm_neon.h"

namespace TEngine {

namespace SoftmaxFP32Impl64 {

const int default_prio = 300;

const std::array<float32x2_t, 8> exp_tab = {{
    vdup_n_f32(1.f),
    vdup_n_f32(0.0416598916054f),
    vdup_n_f32(0.500000596046f),
    vdup_n_f32(0.0014122662833f),
    vdup_n_f32(1.00000011921f),
    vdup_n_f32(0.00833693705499f),
    vdup_n_f32(0.166665703058f),
    vdup_n_f32(0.000195780929062f),
}};

    inline float32x2_t vtaylor_poly_f32(float32x2_t x, const std::array<float32x2_t, 8>& coeffs)
    {
        float32x2_t A = vmla_f32(coeffs[0], coeffs[4], x);
        float32x2_t B = vmla_f32(coeffs[2], coeffs[6], x);
        float32x2_t C = vmla_f32(coeffs[1], coeffs[5], x);
        float32x2_t D = vmla_f32(coeffs[3], coeffs[7], x);
        float32x2_t x2 = vmul_f32(x, x);
        float32x2_t x4 = vmul_f32(x2, x2);
        float32x2_t res = vmla_f32(vmla_f32(A, B, x2), vmla_f32(C, D, x2), x4);
        return res;
    }

    inline float32x2_t vexp_f32(float32x2_t x)
    {
        static const float32x2_t CONST_LN2 = vdup_n_f32(0.6931471805f);    // ln(2)
        static const float32x2_t CONST_INV_LN2 = vdup_n_f32(1.4426950408f);    // 1/ln(2)
        static const float32x2_t CONST_0 = vdup_n_f32(0.f);
        static const int32x2_t CONST_NEGATIVE_126 = vdup_n_s32(-126);

        // Perform range reduction [-log(2),log(2)]
        int32x2_t m = vcvt_s32_f32(vmul_f32(x, CONST_INV_LN2));
        float32x2_t val = vmls_f32(x, vcvt_f32_s32(m), CONST_LN2);

        // Polynomial Approximation
        float32x2_t poly = vtaylor_poly_f32(val, exp_tab);

        // Reconstruct
        poly = vreinterpret_f32_s32(vqadd_s32(vreinterpret_s32_f32(poly), vqshl_n_s32(m, 23)));
        poly = vbsl_f32(vclt_s32(m, CONST_NEGATIVE_126), CONST_0, poly);

        return poly;
    }
/*
exp(x) = lim(1+x/n)^n       // n=10
*/
float exp10_f32(float x)
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
inline float32x4_t vexpq10_f32(float32x4_t x)
{
    x = vmlaq_n_f32(vdupq_n_f32(1.0f), x, 0.0009765625f);    // n = 10
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
inline float32x2_t vexp10_f32(float32x2_t x)
{
    x = vmla_n_f32(vdup_n_f32(1.0f), x, 0.0009765625f);    // n = 10
    x = vmul_f32(x, x);
    x = vmul_f32(x, x);
    x = vmul_f32(x, x);
    x = vmul_f32(x, x);
    x = vmul_f32(x, x);
    x = vmul_f32(x, x);
    x = vmul_f32(x, x);
    x = vmul_f32(x, x);
    x = vmul_f32(x, x);
    x = vmul_f32(x, x);
    return x;
}

struct SoftmaxOps : public NodeOps
{
    SoftmaxOps()
    {
        name_ = "arm_softmax_fp32";
    }

    static void GetMaxArray(float* input, float* array, int in_size, int on_size)
    {
        float* input_ptr = ( float* )input;
        float* array_ptr = ( float* )array;
        memset(array, 0, in_size * sizeof(float));

        for(int j = 0; j < on_size; j++)
        {
            for(int i = 0; i < (in_size & -4); i += 4)
            {
                float32x4_t _p = vld1q_f32(array_ptr + i);
                float32x4_t _in = vld1q_f32(input_ptr + j* in_size +i);
                _p = vpmaxq_f32(_p, _in);

                vst1q_f32(array_ptr+i, _p);
            }
            for(int i = in_size & ~3; i < in_size; i++)
            {
                if(array_ptr[i] < input_ptr[j * in_size + i])
                    array_ptr[i] = input_ptr[j * in_size + i];
            }
            /*
            for(int l = 0; l < in_size; l++)
            {
                if(array_ptr[l] < input_ptr[j * in_size + l])
                    array_ptr[l] = input_ptr[j * in_size + l];
            }
            */
        }   
    }

    static void GetOutResult(float* input, float* output, float* maxarray, float* sum_array, int in_size, int on_size)
    {
        float* input_ptr = ( float* )input;
        float* output_ptr = ( float* )output;
        float* maxarray_ptr = ( float* )maxarray;
        float* sum_array_ptr = ( float* )sum_array;

        memset(sum_array, 0x0, in_size * sizeof(float));

        /* get the exp and the summary */

        for(int j = 0; j < on_size; j++)
        {
            for(int i = 0; i < (in_size & -4); i += 4)
            {
                int index = j * in_size + i;
                float32x4_t out = vexpq10_f32(vsubq_f32(vld1q_f32(input_ptr + index), vld1q_f32(maxarray_ptr + i)));
                float32x4_t sum = vaddq_f32(vld1q_f32(sum_array_ptr+i), out);
                vst1q_f32(output_ptr+index, out);
                vst1q_f32(sum_array_ptr+i, sum);

            }
            for(int i = in_size & ~3; i < in_size; i++)
            {
                int index = j * in_size + i;
                output_ptr[index] = exp(input_ptr[index] - maxarray_ptr[i]);
                sum_array_ptr[i] += output_ptr[index];
            }
        }
        /*
            for(int l = 0; l < in_size; l++)
            {
                int index = j * in_size + l;
                output_ptr[index] = exp(input_ptr[index] - array_ptr[l]);
                sum_array_ptr[l] += output_ptr[index];
            }
        */
        /* the final result */
        for(int j = 0; j < on_size; j++)
            for(int l = 0; l < in_size; l++)
            {
                int index = j * in_size + l;
                output_ptr[index] /= sum_array_ptr[l];
            }
    }

    bool Run(Node* node)
    {
        Tensor* input_tensor = node->GetInputTensor(0);
        Tensor* output_tensor = node->GetOutputTensor(0);
        const std::vector<int>& dims = input_tensor->GetShape().GetDim();
        Softmax* softmax_op = dynamic_cast<Softmax*>(node->GetOp());
        SoftmaxParam* param_ = softmax_op->GetParam();
        float* input = ( float* )get_tensor_mem(input_tensor);
        float* output = ( float* )get_tensor_mem(output_tensor);


        int dim_size = dims.size();
        int axis = param_->axis;

        if (dim_size == 1) // axis == 0
        {
            int w = dims[0];
            float* ptr = input;
            float* output_ptr = output;

            // get max
            float max = -__FLT_MAX__;
            float32x4_t _max4 = vdupq_n_f32(max);
            for(int i = 0; i < (w & -4); i += 4)
            {
                float32x4_t _in = vld1q_f32(ptr + i);
                _max4 = vmaxq_f32(_max4, _in);
            }
            max = vmaxvq_f32(_max4);
            float32x2_t _max2 = vdup_n_f32(max);
            for (int j = (w & ~3);j<(w & -2);j+=2)
            {
                float32x2_t _in = vld1_f32(ptr + j);
                _max2 = vmax_f32(_max2, _in);
            }
            max = vpmaxs_f32(_max2);
            for (int j=(w & ~1); j<w; j++)
            {
                max = std::max(max, ptr[j]);
            }

            // get sum
            float sum = 0.f;
            float32x4_t _sum4 = vdupq_n_f32(0.0f);
            _max4 = vdupq_n_f32(max);
            for(int i = 0; i < (w & -4); i += 4)
            {
                float32x4_t _in = vld1q_f32(ptr + i);
                float32x4_t _out = vexpq10_f32(vsubq_f32(_in, _max4));
                vst1q_f32(output_ptr+i, _out);
                _sum4 = vaddq_f32(_sum4, _out);
            }
            float32x2_t _sum2 = vdup_n_f32(0.0f);
            _max2 = vdup_n_f32(max);
            for(int i = (w & ~3); i < (w & -2); i += 2)
            {
                float32x2_t _in = vld1_f32(ptr + i);
                float32x2_t _out = vexp10_f32(vsub_f32(_in, _max2));
                vst1_f32(output_ptr+i, _out);
                _sum2 = vadd_f32(_sum2, _out);
            }
            sum = vaddvq_f32(_sum4);
            sum += vpadds_f32(_sum2);
            for (int i=(w & ~1); i<w; i++)
            {
                output_ptr[i] = exp(ptr[i] - max);
                sum += output_ptr[i];
            }
            // get result
            for (int i=0; i<w; i++)
            {
                output[i] =output_ptr[i] / sum;
            }
            return true;
        }

        if (dim_size == 2 && axis == 0)
        {
            int w = dims[1];
            int h = dims[0];
            // get max
            float* max = ( float* )std::malloc(w * sizeof(float));
            std::memset(max, 0.0f, w * sizeof(float));
            for (int i=0; i<h; i++)
            {
                const float* ptr = input + i * w;
                for(int j=0; j<(w&-4); j+=4)
                {
                    float32x4_t _in = vld1q_f32(ptr + j);
                    float32x4_t _max = vld1q_f32(max + j);
                    _max = vmaxq_f32(_max, _in);
                    vst1q_f32(max+j, _max);
                }
                for(int j=(w&~3); j<(w&-2); j+=2)
                {
                    float32x2_t _in = vld1_f32(ptr + j);
                    float32x2_t _max = vld1_f32(max + j);
                    _max = vmax_f32(_max, _in);
                    vst1_f32(max+j, _max);
                }
                for(int j=(w&~1); j<w; j++)
                {
                    max[j] = std::max(max[j], ptr[j]);
                }
            }
            // get sum
            float* sum = ( float* )std::malloc(w * sizeof(float));
            std::memset(sum, 0.0f, w * sizeof(float));

            for (int i = 0; i<h; i++)
            {
                float* ptr = input + i * w;
                float* output_ptr = output + i * w;
                for(int j=0; j<(w&-4); j+=4)
                {
                    float32x4_t _in = vld1q_f32(ptr + j);
                    float32x4_t _max = vld1q_f32(max + j);
                    float32x4_t _out = vexpq10_f32(vsubq_f32(_in, _max));
                    vst1q_f32(output_ptr+j, _out);
                    float32x4_t _sum = vld1q_f32(sum + j);
                    _sum = vaddq_f32(_sum, _out);
                    vst1q_f32(sum + j, _sum);
                }
                for(int j=(w&~3); j<(w&-2); j+=2)
                {
                    float32x2_t _in = vld1_f32(ptr + j);
                    float32x2_t _max = vld1_f32(max + j);
                    float32x2_t _out = vexp10_f32(vsub_f32(_in, _max));
                    vst1_f32(output_ptr+j, _out);
                    float32x2_t _sum = vld1_f32(sum + j);
                    _sum = vadd_f32(_sum, _out);
                    vst1_f32(sum + j, _sum);
                }
                for(int j=(w&~1); j<w; j++)
                {
                    output_ptr[j] = exp(ptr[j] - max[j]);
                    sum[j] += output_ptr[j];
                }
            }
            // get result
            for (int i=0; i<h; i++)
            {
                float* output_ptr = output + i * w;
                for (int j=0; j<w; j++)
                {
                    output_ptr[j] = output_ptr[j] / sum[j];
                }
            }
            return true;
        }

        if (dim_size == 2 && axis == 1)
        {
            int w = dims[1];
            int h = dims[0];

            for (int i=0; i<h; i++)
            {
                // get max
                float* ptr = input + i * w;
                float* output_ptr = output + i * w;
                float max = -__FLT_MAX__;
                float32x4_t _max4 = vdupq_n_f32(max);
                for (int j=0; j<(w & -4); j+=4)
                {
                    float32x4_t _in = vld1q_f32(ptr + j);
                    _max4 = vmaxq_f32(_max4, _in);
                }
                max = vmaxvq_f32(_max4);
                float32x2_t _max2 = vdup_n_f32(max);
                for (int j = (w & ~3);j<(w & -2);j+=2)
                {
                    float32x2_t _in = vld1_f32(ptr + j);
                    _max2 = vmax_f32(_max2, _in);
                }
                max = vpmaxs_f32(_max2);
                for (int j=(w & ~1); j<w; j++)
                {
                    max = std::max(max, ptr[j]);
                }

                // get sum
                
                float sum = 0.0f;
                float32x4_t _sum4 = vdupq_n_f32(0.0f);
                _max4 = vdupq_n_f32(max);
                for(int i = 0; i < (w & -4); i += 4)
                {
                    float32x4_t _in = vld1q_f32(ptr + i);
                    float32x4_t _out = vexpq10_f32(vsubq_f32(_in, _max4));
                    vst1q_f32(output_ptr+i, _out);
                    _sum4 = vaddq_f32(_sum4, _out);
                }
                float32x2_t _sum2 = vdup_n_f32(0.0f);
                _max2 = vdup_n_f32(max);
                for (int j=(w&~3);j<(w & -2);j+=2)
                {
                    float32x2_t _in = vld1_f32(ptr + j);
                    float32x2_t _out = vexp10_f32(vsub_f32(_in, _max2));
                    vst1_f32(output_ptr+j, _out);
                    _sum2 = vadd_f32(_sum2, _out);
                }
                sum = vaddvq_f32(_sum4);
                sum += vpadds_f32(_sum2);
                for (int j=(w & ~1); j<w; j++)
                {
                    output_ptr[j] = std::exp(ptr[j] - max);
                    sum += output_ptr[j];
                }
                // get result
                for (int j=0; j<w; j++)
                {
                    output_ptr[j]  = output_ptr[j] / sum;
                }
            }
            return true;
        }

        if (dim_size == 3 && axis == 0)
        {
            int w = dims[2];
            int h = dims[1];
            int channels = dims[0];
            int size = w * h;

            // get max
            float* max = ( float* )std::malloc(w * h * sizeof(float));
            std::memset(max, 0.0f, w * h * sizeof(float));
            for (int q=0; q<channels; q++)
            {
                const float* ptr = input + q * size;
                for(int j=0; j<(size&-4); j+=4)
                {
                    float32x4_t _in = vld1q_f32(ptr + j);
                    float32x4_t _max = vld1q_f32(max + j);
                    _max = vmaxq_f32(_max, _in);
                    vst1q_f32(max+j, _max);
                }
                for(int j=(size&~3); j<(size&-2); j+=2)
                {
                    float32x2_t _in = vld1_f32(ptr + j);
                    float32x2_t _max = vld1_f32(max + j);
                    _max = vmax_f32(_max, _in);
                    vst1_f32(max+j, _max);
                }
                for(int j=(size&~1); j<w; j++)
                {
                    max[j] = std::max(max[j], ptr[j]);
                }
            }

            // get sum
            float* sum = ( float* )std::malloc(w * h * sizeof(float));
            std::memset(sum, 0.0f, w * h * sizeof(float));

            for (int q=0; q<channels; q++)
            {
                float* ptr = input + q * size;
                float* output_ptr = output + q * size;
                for(int j=0; j<(size&-4); j+=4)
                {
                    float32x4_t _in = vld1q_f32(ptr + j);
                    float32x4_t _max = vld1q_f32(max + j);
                    float32x4_t _out = vexpq10_f32(vsubq_f32(_in, _max));
                    vst1q_f32(output_ptr+j, _out);
                    float32x4_t _sum = vld1q_f32(sum + j);
                    _sum = vaddq_f32(_sum, _out);
                    vst1q_f32(sum + j, _sum);
                }
                for(int j=(size&~3); j<(size&-2); j+=2)
                {
                    float32x2_t _in = vld1_f32(ptr + j);
                    float32x2_t _max = vld1_f32(max + j);
                    float32x2_t _out = vexp10_f32(vsub_f32(_in, _max));
                    vst1_f32(output_ptr+j, _out);
                    float32x2_t _sum = vld1_f32(sum + j);
                    _sum = vadd_f32(_sum, _out);
                    vst1_f32(sum + j, _sum);
                }
                for(int j=(size&~1); j<size; j++)
                {
                    output_ptr[j] = exp(ptr[j] - max[j]);
                    sum[j] += output_ptr[j];
                }
            }
            // get result
            for (int q=0; q<channels; q++)
            {
                float* output_ptr = output + q * size;
                for (int i=0; i<size; i++)
                {
                    output_ptr[i] = output_ptr[i] / sum[i];
                }
            }

            return true;
        }

        if (dim_size == 3 && axis == 1)
        {
            int w = dims[2];
            int h = dims[1];
            int channels = dims[0];

            // get max
            float* max = ( float* )std::malloc(w * channels * sizeof(float));
            std::memset(max, 0.0f, w * channels * sizeof(float));

            for (int q=0; q<channels; q++)
            {
                const float* ptr = input+ q * h * w;
                float* maxptr = max + q * w;
                for (int i=0; i<h; i++)
                {
                    for(int j=0; j<(w&-4); j+=4)
                    {
                        float32x4_t _in = vld1q_f32(ptr + j);
                        float32x4_t _max = vld1q_f32(max + j);
                        _max = vmaxq_f32(_max, _in);
                        vst1q_f32(max+j, _max);
                    }
                    for(int j=(w&~3); j<(w&-2); j+=2)
                    {
                        float32x2_t _in = vld1_f32(ptr + j);
                        float32x2_t _max = vld1_f32(max + j);
                        _max = vmax_f32(_max, _in);
                        vst1_f32(maxptr+j, _max);
                    }
                    for(int j=(w&~1); j<w; j++)
                    {
                        max[j] = std::max(max[j], ptr[j]);
                    }
                    ptr += w;
                }
            }
            // get sum
            float* sum = ( float* )std::malloc(w * channels * sizeof(float));
            std::memset(sum, 0.0f, w * channels * sizeof(float));

            for (int q=0; q<channels; q++)
            {
                float* ptr = input+ q * h * w;
                float* output_ptr = output + q * h * w;
                float* maxptr = max + q * w;
                float* sumptr = sum + q * w;

                for (int i=0; i<h; i++)
                {
                    for(int j=0; j<(w&-4); j+=4)
                    {
                        float32x4_t _in = vld1q_f32(ptr + j);
                        float32x4_t _max = vld1q_f32(maxptr + j);
                        float32x4_t _out = vexpq10_f32(vsubq_f32(_in, _max));
                        vst1q_f32(output_ptr+j, _out);
                        float32x4_t _sum = vld1q_f32(sumptr + j);
                        _sum = vaddq_f32(_sum, _out);
                        vst1q_f32(sumptr + j, _sum);
                    }
                    for(int j=(w&~3); j<(w&-2); j+=2)
                    {
                        float32x2_t _in = vld1_f32(ptr + j);
                        float32x2_t _max = vld1_f32(maxptr + j);
                        float32x2_t _out = vexp10_f32(vsub_f32(_in, _max));
                        vst1_f32(output_ptr+j, _out);
                        float32x2_t _sum = vld1_f32(sumptr + j);
                        _sum = vadd_f32(_sum, _out);
                        vst1_f32(sumptr + j, _sum);
                    }
                    for(int j=(w&~1); j<w; j++)
                    {
                        output_ptr[j] = exp(ptr[j] - maxptr[j]);
                        sumptr[j] += output_ptr[j];
                    }

                    ptr += w;
                    output_ptr += w;
                }
            }

            for (int q=0; q<channels; q++)
            {
                float* output_ptr = output + q * h * w;
                float* sumptr = sum + q * w;

                for (int i=0; i<h; i++)
                {
                    for (int j=0; j<w; j++)
                    {
                        output_ptr[j] = output_ptr[j] / sumptr[j];
                    }
                    output_ptr += w;
                }
            }

            return true;
        }

        if (dim_size == 3 && axis == 2)
        {
            int w = dims[2];
            int h = dims[1];
            int channels = dims[0];
            int channel_size = w*h;
            for (int q=0; q<channels; q++)
            {
                float* ptr = input+q*channel_size;
                float* output_ptr = output+q*channel_size;
                for (int i=0; i<h; i++)
                {
                    // get max
                    float max = -__FLT_MAX__;
                    float32x4_t _max4 = vdupq_n_f32(max);
                    for(int i = 0; i < (w & -4); i += 4)
                    {
                        float32x4_t _in = vld1q_f32(ptr + i);
                        _max4 = vmaxq_f32(_max4, _in);
                    }
                    max = vmaxvq_f32(_max4);
                    float32x2_t _max2 = vdup_n_f32(max);
                    for (int j = (w & ~3);j<(w & -2);j+=2)
                    {
                        float32x2_t _in = vld1_f32(ptr + j);
                        _max2 = vmax_f32(_max2, _in);
                    }
                    max = vpmaxs_f32(_max2);
                    for (int j=(w & ~1); j<w; j++)
                    {
                        max = std::max(max, ptr[j]);
                    }
                    // get sum
                    float sum = 0.f;
                    float32x4_t _sum4 = vdupq_n_f32(0.0f);
                    _max4 = vdupq_n_f32(max);
                    for(int i = 0; i < (w & -4); i += 4)
                    {
                        float32x4_t _in = vld1q_f32(ptr + i);
                        float32x4_t _out = vexpq10_f32(vsubq_f32(_in, _max4));
                        vst1q_f32(output_ptr+i, _out);
                        _sum4 = vaddq_f32(_sum4, _out);
                    }
                    float32x2_t _sum2 = vdup_n_f32(0.0f);
                    _max2 = vdup_n_f32(max);
                    for(int i = (w & ~3); i < (w & -2); i += 2)
                    {
                        float32x2_t _in = vld1_f32(ptr + i);
                        float32x2_t _out = vexp10_f32(vsub_f32(_in, _max2));
                        vst1_f32(output_ptr+i, _out);
                        _sum2 = vadd_f32(_sum2, _out);
                    }
                    sum = vaddvq_f32(_sum4);
                    sum += vpadds_f32(_sum2);
                    for (int i=(w & ~1); i<w; i++)
                    {
                        output_ptr[i] = exp(ptr[i] - max);
                        sum += output_ptr[i];
                    }
                    // get result
                    for (int j=0; j<w; j++)
                    {
                        output_ptr[j] = output_ptr[j] / sum;
                    }

                    ptr += w;
                    output_ptr +=w;
                }
            }
            return true;
        }

        if(axis > dim_size)
            axis = dim_size - 1;
        int out_size, in_size, on_size;
        out_size = 1;
        for(int i = 0; i < axis; i++)
        {
            out_size *= dims[i];
        }
        in_size = 1;
        for(size_t i = axis + 1; i < dims.size(); i++)
        {
            in_size *= dims[i];
        }
        on_size = dims[axis];

        float* max_array = ( float* )std::malloc(in_size * sizeof(float));
        float* sum_array = ( float* )std::malloc(in_size * sizeof(float));

        int on_in_size = on_size * in_size;

        for(int i = 0; i < out_size; i++)
        {
            /* get max */
            int img_base = i * on_in_size; 
            
            GetMaxArray(input + img_base, max_array, in_size, on_size);
            GetOutResult(input + img_base, output + img_base, max_array, sum_array, in_size, on_size);
        }

        std::free(max_array);
        std::free(sum_array);
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

    SoftmaxOps* ops = new SoftmaxOps();

    return ops;
}

}    // namespace SoftmaxFP32Impl64

using namespace SoftmaxFP32Impl64;

void RegisterSoftmaxFP32(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor("arm64", "Softmax", SoftmaxFP32Impl64::SelectFunc, SoftmaxFP32Impl64::default_prio);
}

}    // namespace TEngine

