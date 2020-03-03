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
 * Copyright (c) 2019, Open AI Lab
 * Author: ddzhao@openailab.com
 */

#include <vector>

#include "data_type.hpp"
#include "kernel_registry.hpp"
#include "tengine_errno.hpp"
#include "logger.hpp"
#include "graph.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "math.h"
#include <sys/time.h>
#include <arm_neon.h>

namespace TEngine {

namespace SigmoidFP32Impl64 {

const int default_prio = 300;

inline float fast_exp(float x)
{
    union
    {
        uint32_t i;
        float f;
    } v;
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

float fast_exp1(float x)
{
    volatile union
    {
        float f;
        unsigned int i;
    } cvt;

    /* exp(x) = 2^i * 2^f; i = floor (log2(e) * x), 0 <= f <= 1 */
    float t = x * 1.442695041f;
    float fi = floorf(t);
    float f = t - fi;
    int i = ( int )fi;
    cvt.f = (0.3371894346f * f + 0.657636276f) * f + 1.00172476f; /* compute 2^f */
    cvt.i += (i << 23); /* scale by 2^i */
    return cvt.f;
}

float acl_exp(float x)
{
    volatile union
    {
        float f;
        unsigned int i;
    } cvt;

    /* exp(x) = = 2^k * exp(x-k ln2); k = round（x/ln2）*/
    float t = x * 1.4426950408f;
    float f = x - (( int )t) * 0.6931471805f;
    int i = ( int )t;
    /// cvt.f = (0.3371894346f * f + 0.657636276f) * f + 1.00172476f;       /* compute 2^f */
    cvt.f =
        1 + f * 1.00000011921f + (0.0416598916054f + f * 0.00833693705499f) * f * f +
        ((0.500000596046f + f * 0.166665703058f) + (0.0014122662833f + f * 0.000195780929062f) * f * f) * f * f * f * f;
    cvt.i += (i << 23); /* scale by 2^i */
    return cvt.f;
}

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

const std::vector<float32x4_t> exp_tab = {{
    vdupq_n_f32(1.f),
    vdupq_n_f32(0.0416598916054f),
    vdupq_n_f32(0.500000596046f),
    vdupq_n_f32(0.0014122662833f),
    vdupq_n_f32(1.00000011921f),
    vdupq_n_f32(0.00833693705499f),
    vdupq_n_f32(0.166665703058f),
    vdupq_n_f32(0.000195780929062f),
}};

inline float32x4_t vtaylor_polyq_f32(float32x4_t x, const std::vector<float32x4_t>& coeffs)
{
    float32x4_t A = vmlaq_f32(coeffs[0], coeffs[4], x);
    float32x4_t B = vmlaq_f32(coeffs[2], coeffs[6], x);
    float32x4_t C = vmlaq_f32(coeffs[1], coeffs[5], x);
    float32x4_t D = vmlaq_f32(coeffs[3], coeffs[7], x);
    float32x4_t x2 = vmulq_f32(x, x);
    float32x4_t x4 = vmulq_f32(x2, x2);
    float32x4_t res = vmlaq_f32(vmlaq_f32(A, B, x2), vmlaq_f32(C, D, x2), x4);
    return res;
}
/* ACL exp function impelement */
inline float32x4_t vexpq_f32(float32x4_t x)
{
    static const float32x4_t CONST_LN2 = vdupq_n_f32(0.6931471805f);    // ln(2)
    static const float32x4_t CONST_INV_LN2 = vdupq_n_f32(1.4426950408f);    // 1/ln(2)
    static const float32x4_t CONST_0 = vdupq_n_f32(0.f);
    static const int32x4_t CONST_NEGATIVE_126 = vdupq_n_s32(-126);

    // Perform range reduction [-log(2),log(2)]
    int32x4_t m = vcvtq_s32_f32(vmulq_f32(x, CONST_INV_LN2));
    float32x4_t val = vmlsq_f32(x, vcvtq_f32_s32(m), CONST_LN2);

    // Polynomial Approximation
    float32x4_t poly = vtaylor_polyq_f32(val, exp_tab);

    // Reconstruct
    poly = vreinterpretq_f32_s32(vqaddq_s32(vreinterpretq_s32_f32(poly), vqshlq_n_s32(m, 23)));
    poly = vbslq_f32(vcltq_s32(m, CONST_NEGATIVE_126), CONST_0, poly);

    return poly;
}
/*
exp(x) = lim(1+x/n)^n       // n=10
*/
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

struct SigmoidOps : public MTNodeOps
{
    bool OnBind(Node* node)
    {
        inplace_t io_map;

        io_map[0] = 0;

        node->SetAttr(ATTR_INPLACE, io_map);
        return true;
    }

    bool Prerun(Node* node)
    {
        return true;
    }

    bool Run(Node* node)
    {
        Tensor* input_tensor = node->GetInputTensor(0);
        Tensor* output_tensor = node->GetOutputTensor(0);
        const TShape& shape = input_tensor->GetShape();
        int elem_num = shape.GetSize();
        float* input = ( float* )get_tensor_mem(input_tensor);
        float* output = ( float* )get_tensor_mem(output_tensor);
        /// void* data = get_tensor_mem(input_tensor);
        /// int ret = kernel_run(data, elem_num, &op_param);
#define SIGMOID_MAX(a, b) ((a) > (b) ? (a) : (b))
#define SIGMOID_MIN(a, b) ((a) < (b) ? (a) : (b))
        /*
        for(int i = 0; i < elem_num; i++)
        {
            input[i] = SIGMOID_MIN(input[i], 30.0f);
            input[i] = SIGMOID_MAX(input[i], -30.0f);
            std::cout<< input[i] << " ";
            float tmp_exp = exp8mul(-input[i]);
            output[i] = 1 / (1 + tmp_exp);
            std::cout<< tmp_exp << " " <<output[i] << "\n";
        }
        */

        float32x4_t min = vdupq_n_f32(-30.0f);
        float32x4_t max = vdupq_n_f32(30.0f);
        float32x4_t tmp_vec = vdupq_n_f32(1);
        for(int i = 0; i < (elem_num & -4); i += 4)
        {
            float32x4_t _input = vld1q_f32(input + i);
            _input = vmaxq_f32(_input, min);
            _input = vminq_f32(_input, max);
            float32x4_t tmp_exp = vaddq_f32(tmp_vec, vexpq10_f32(vmulq_n_f32(_input, -1.0f)));
            /// float32x4_t tmp_exp = vaddq_f32(tmp_vec, vexpq_f32(vmulq_n_f32(_input, -1.0f)));
            float32x4_t out = vrecpeq_f32(tmp_exp);
            out = vmulq_f32(vrecpsq_f32(tmp_exp, out), out);
            out = vmulq_f32(vrecpsq_f32(tmp_exp, out), out);
            vst1q_f32(output, out);
            output += 4;
        }
        for(int i = elem_num & ~3; i < elem_num; i++)
        {
            input[i] = SIGMOID_MIN(input[i], 30.0f);
            input[i] = SIGMOID_MAX(input[i], -30.0f);
            float tmp_exp = exp10_f32(-input[i]);
            *output++ = 1 / (1 + tmp_exp);
        }

        return true;
    }
};

NodeOps* SelectFunc(const CPUInfo* info, Node* node)
{
    SigmoidOps* ops = new SigmoidOps();
    return ops;
}

}    // namespace SigmoidFP32Impl64
using namespace SigmoidFP32Impl64;

void RegisterSigmoidFP32(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor("arm64", "Sigmoid", SigmoidFP32Impl64::SelectFunc,
                                                  SigmoidFP32Impl64::default_prio);
}

}    // namespace TEngine
