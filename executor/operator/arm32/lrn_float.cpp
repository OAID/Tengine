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
 */
#include <iostream>
#include <functional>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <arm_neon.h>
#include <array>

#include "logger.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "graph.hpp"
#include "operator/lrn.hpp"

namespace TEngine {

namespace LRNImplArm {

const std::array<float32x4_t, 8> exp_tab = {{
    vdupq_n_f32(1.f),
    vdupq_n_f32(0.0416598916054f),
    vdupq_n_f32(0.500000596046f),
    vdupq_n_f32(0.0014122662833f),
    vdupq_n_f32(1.00000011921f),
    vdupq_n_f32(0.00833693705499f),
    vdupq_n_f32(0.166665703058f),
    vdupq_n_f32(0.000195780929062f),
}};

/* Logarithm polynomial coefficients */
const std::array<float32x4_t, 8> log_tab = {{
    vdupq_n_f32(-2.29561495781f),
    vdupq_n_f32(-2.47071170807f),
    vdupq_n_f32(-5.68692588806f),
    vdupq_n_f32(-0.165253549814f),
    vdupq_n_f32(5.17591238022f),
    vdupq_n_f32(0.844007015228f),
    vdupq_n_f32(4.58445882797f),
    vdupq_n_f32(0.0141278216615f),
}};

struct LRNOps : public NodeOps
{
    LRNOps()
    {
        name_ = "arm_lrn_fp32";
    }

    inline float32x4_t vfloorq_f32(float32x4_t val)
    {
        static const float32x4_t CONST_1 = vdupq_n_f32(1.f);

        const int32x4_t z = vcvtq_s32_f32(val);
        const float32x4_t r = vcvtq_f32_s32(z);

        return vbslq_f32(vcgtq_f32(r, val), vsubq_f32(r, CONST_1), r);
    }

    inline float32x2_t vinvsqrt_f32(float32x2_t x)
    {
        float32x2_t sqrt_reciprocal = vrsqrte_f32(x);
        sqrt_reciprocal = vmul_f32(vrsqrts_f32(vmul_f32(x, sqrt_reciprocal), sqrt_reciprocal), sqrt_reciprocal);
        sqrt_reciprocal = vmul_f32(vrsqrts_f32(vmul_f32(x, sqrt_reciprocal), sqrt_reciprocal), sqrt_reciprocal);

        return sqrt_reciprocal;
    }

    inline float32x4_t vinvsqrtq_f32(float32x4_t x)
    {
        float32x4_t sqrt_reciprocal = vrsqrteq_f32(x);
        sqrt_reciprocal = vmulq_f32(vrsqrtsq_f32(vmulq_f32(x, sqrt_reciprocal), sqrt_reciprocal), sqrt_reciprocal);
        sqrt_reciprocal = vmulq_f32(vrsqrtsq_f32(vmulq_f32(x, sqrt_reciprocal), sqrt_reciprocal), sqrt_reciprocal);

        return sqrt_reciprocal;
    }

    inline float32x2_t vinv_f32(float32x2_t x)
    {
        float32x2_t recip = vrecpe_f32(x);
        recip = vmul_f32(vrecps_f32(x, recip), recip);
        recip = vmul_f32(vrecps_f32(x, recip), recip);
        return recip;
    }

    inline float32x4_t vinvq_f32(float32x4_t x)
    {
        float32x4_t recip = vrecpeq_f32(x);
        recip = vmulq_f32(vrecpsq_f32(x, recip), recip);
        recip = vmulq_f32(vrecpsq_f32(x, recip), recip);
        return recip;
    }

    inline float32x4_t vtaylor_polyq_f32(float32x4_t x, const std::array<float32x4_t, 8>& coeffs)
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

    inline float32x4_t vlogq_f32(float32x4_t x)
    {
        static const int32x4_t CONST_127 = vdupq_n_s32(127);    // 127
        static const float32x4_t CONST_LN2 = vdupq_n_f32(0.6931471805f);    // ln(2)

        // Extract exponent
        int32x4_t m = vsubq_s32(vreinterpretq_s32_u32(vshrq_n_u32(vreinterpretq_u32_f32(x), 23)), CONST_127);
        float32x4_t val = vreinterpretq_f32_s32(vsubq_s32(vreinterpretq_s32_f32(x), vshlq_n_s32(m, 23)));

        // Polynomial Approximation
        float32x4_t poly = vtaylor_polyq_f32(val, log_tab);

        // Reconstruct
        poly = vmlaq_f32(poly, vcvtq_f32_s32(m), CONST_LN2);

        return poly;
    }

    inline float32x4_t vtanhq_f32(float32x4_t val)
    {
        static const float32x4_t CONST_1 = vdupq_n_f32(1.f);
        static const float32x4_t CONST_2 = vdupq_n_f32(2.f);
        static const float32x4_t CONST_MIN_TANH = vdupq_n_f32(-10.f);
        static const float32x4_t CONST_MAX_TANH = vdupq_n_f32(10.f);

        float32x4_t x = vminq_f32(vmaxq_f32(val, CONST_MIN_TANH), CONST_MAX_TANH);
        float32x4_t exp2x = vexpq_f32(vmulq_f32(CONST_2, x));
        float32x4_t num = vsubq_f32(exp2x, CONST_1);
        float32x4_t den = vaddq_f32(exp2x, CONST_1);
        float32x4_t tanh = vmulq_f32(num, vinvq_f32(den));
        return tanh;
    }

    inline float32x4_t vpowq_f32(float32x4_t val, float32x4_t n)
    {
        return vexpq_f32(vmulq_f32(n, vlogq_f32(val)));
    }

    void lrn_kernel(int i, int id, void* data, const float* input, float* output,float* square,
                float alpha, float beta, float bias, int local_size, int channel_size, int channel_num)
    {
        int step = ((int*)data)[0];
        const float32x4_t alpha_vec = vdupq_n_f32(alpha / local_size);
        const float32x4_t beta_vec = vdupq_n_f32(beta);
        const float32x4_t bias_vec = vdupq_n_f32(bias);
        int mod = channel_size / 4;
        int start_c = step * id;
        int end_c = step * id + step;
        
        for(int j = start_c; j < end_c; j++)
        {
            int c_start = j - local_size / 2;
            int c_end = j + local_size / 2;

            c_start = std::max(0, c_start);
            c_end = std::min(c_end, channel_num - 1);

            const float* cur_input = input + j * channel_size;
            float* cur_output = output + j * channel_size;
            for(int m = 0; m < mod; m++)
            {
                float32x4_t accu = vdupq_n_f32(0.f);

                for(int l = c_start; l <= c_end; l++)
                {
                    accu = vaddq_f32(accu, vld1q_f32(square + l * channel_size + m * 4));
                }
                const float32x4_t normalized = vpowq_f32(vmlaq_f32(bias_vec, alpha_vec, accu), beta_vec);
                const float32x4_t normalized_pixel =
                    vmulq_f32(vld1q_f32(cur_input), vinvq_f32(normalized));
                vst1q_f32(cur_output, normalized_pixel);
                cur_input += 4;
                cur_output += 4;
            }

            float alpha_over_size = alpha / local_size;

            for(int m = 4 * mod; m < channel_size; m++)
            {
                float sum = 0;
                for(int l = c_start; l <= c_end; l++)
                {
                    sum = sum + square[l * channel_size + m];
                }
                *cur_output++ = *cur_input++ * std::pow(bias + alpha_over_size * sum, -beta);
            }
        }
    }

    bool Run(Node* node)
    {
        Tensor* input_tensor = node->GetInputTensor(0);
        Tensor* output_tensor = node->GetOutputTensor(0);

        LRN* lrn_op = dynamic_cast<LRN*>(node->GetOp());
        LRNParam* param = lrn_op->GetParam();

        const float* input = ( float* )get_tensor_mem(input_tensor);
        float* output = ( float* )get_tensor_mem(output_tensor);

        float* square = ( float* )(std::malloc(input_tensor->GetTotalSize()));

        const TShape& shape = input_tensor->GetShape();
        const std::vector<int>& dims = shape.GetDim();

        int n = dims[0];
        int c = dims[1];
        int h = dims[2];
        int w = dims[3];

        int img_size = c * h * w;
        int channel_size = h * w;
        float alpha = param->alpha;
        float beta = param->beta;
        float bias = param->k;
        int local_size = param->local_size;

        int cpu_number = cpu_info->GetCPUNumber();
        int num_task = c < cpu_number ? c : cpu_number;
        int step = c / num_task;
        for(int i = 0; i < n; i++)
        {
            /* get square value */
            const float* img_base = input + i * img_size;
            float* out_base = output + i * img_size;

            int j = 0;
            for(j = 0; j < (img_size & -4); j+=4)
            {
                float32x4_t in = vld1q_f32(img_base + j);
                in = vmulq_f32(in, in);
                vst1q_f32(square + j, in);
            }
            for(; j < img_size; j++)
                square[j] = img_base[j] * img_base[j];

            if(param->norm_region != LRN_ACROSS_CHANNELS)
            {
                std::free(square);
                LOG_ERROR() << "LRN: Only support Across_channels\n";
                return false;
            }
            if(num_task == 1)
            {
                lrn_kernel(0, 0, &step, img_base, out_base, square, alpha, beta, bias,
                    local_size, channel_size, c);
            }
            else
            {
                MULTI_THREAD_START(num_task, step, id, param)
                    lrn_kernel(0, id, param, img_base, out_base, square, alpha, beta, bias,
                        local_size, channel_size, c);
                MULTI_THREAD_END();
            }
            if(num_task * step != c)
            {
                int offset = num_task * step;
                int remain_num = c - offset;
                img_base += offset * channel_size;
                out_base += offset * channel_size;
                lrn_kernel(0, 0, &remain_num, img_base, out_base, square, alpha, beta, bias,
                    local_size, channel_size, c);
            }
        }

        std::free(square);

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

    LRNOps* ops = new LRNOps();

    return ops;
}

}    // namespace LRNImplArm

using namespace LRNImplArm;

void RegisterLRNNodeExec(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor("arm32", "LRN", LRNImplArm::SelectFunc, 1000);
}

}    // namespace TEngine
