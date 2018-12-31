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
 * Author: chunyinglv@openailab.com
 */
#include <iostream>
#include <functional>
#include <cstring>
#include <algorithm>

#include "logger.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "graph.hpp"
#include "operator/batch_norm.hpp"
#include <arm_neon.h>
#include <cmath>
namespace TEngine {

namespace BatchNormImpl {

struct BNOps : public NodeOps
{
    bool OnBind(Node* node)
    {
        // set the inplace feature
        inplace_t io_map;

        io_map[0] = 0;

        node->SetAttr(ATTR_INPLACE, io_map);

        return true;
    }

    bool Prerun(Node* node)
    {
        const Tensor* mean_tensor = node->GetInputTensor(3);
        const TShape& shape = mean_tensor->GetShape();

        const std::vector<int> dims = shape.GetDim();

        int channel_num = dims[0];

        float* scale_mean = ( float* )mem_alloc(channel_num * sizeof(float));
        float* scale_var_inv = ( float* )mem_alloc(channel_num * sizeof(float));

        const Tensor* var_tensor = node->GetInputTensor(4);
        const float* mean = ( const float* )get_tensor_mem(mean_tensor);
        const float* var = ( const float* )get_tensor_mem(var_tensor);

        BatchNorm* bn_op = dynamic_cast<BatchNorm*>(node->GetOp());
        BatchNormParam* param = bn_op->GetParam();

        float rescale_factor;
        float eps = param->eps;

        rescale_factor = param->rescale_factor ? 1 / param->rescale_factor : 0;
        for(int c = 0; c < channel_num; c++)
        {
            scale_var_inv[c] = 1.f / std::sqrt(var[c] * rescale_factor + eps);
            scale_mean[c] = -mean[c] * rescale_factor * scale_var_inv[c];
        }

        node->SetAttr("scale_mean", scale_mean);
        node->SetAttr("scale_var_inv", scale_var_inv);

        return true;
    }

    bool Run(Node* node)
    {
        const Tensor* input_tensor = node->GetInputTensor(0);
        Tensor* output_tensor = node->GetOutputTensor(0);
        const TShape& shape = input_tensor->GetShape();
        const std::vector<int> dims = shape.GetDim();

        int batch_number = dims[0];
        int channel_num = dims[1];
        int channel_size = dims[2] * dims[3];
        int img_size = channel_num * channel_size;

        BatchNorm* bn_op = dynamic_cast<BatchNorm*>(node->GetOp());
        BatchNormParam* param = bn_op->GetParam();

        const float* input = ( const float* )get_tensor_mem(input_tensor);
        float* output = ( float* )get_tensor_mem(output_tensor);

        if(param->caffe_flavor)
        {
            float* scale_mean = any_cast<float*>(node->GetAttr("scale_mean"));
            float* scale_var_inv = any_cast<float*>(node->GetAttr("scale_var_inv"));

            /* only use mean and var */
            for(int i = 0; i < batch_number; i++)
            {
                for(int c = 0; c < channel_num; c++)
                {
                    float s_mean = scale_mean[c];
                    float s_var = scale_var_inv[c];
                    float32x4_t _mean = vdupq_n_f32(s_mean);
                    float32x4_t _var = vdupq_n_f32(s_var);
                    int offset = i * img_size + c * channel_size;
                    const float* input_ptr = input + offset;
                    float* output_ptr = output + offset;

                    for(int l = 0; l < channel_size - 3; l += 4)
                    {
                        float32x4_t _input = vld1q_f32(input_ptr);
                        vst1q_f32(output_ptr, vmlaq_f32(_mean, _input, _var));
                        input_ptr += 4;
                        output_ptr += 4;
                        // output[offset]= input[offset]*scale_var_inv[c] - scale_mean[c];
                    }
                    for(int l = channel_size & ~3; l < channel_size; l++)
                    {
                        *output_ptr = (*input_ptr) * s_var + s_mean;
                        input_ptr++;
                        output_ptr++;
                    }
                }
            }
        }
        else
        {
            float* scale_mean = any_cast<float*>(node->GetAttr("scale_mean"));
            float* scale_var_inv = any_cast<float*>(node->GetAttr("scale_var_inv"));

            const Tensor* gamma_tensor = node->GetInputTensor(1);
            const Tensor* beta_tensor = node->GetInputTensor(2);
            const float* gamma = ( const float* )get_tensor_mem(gamma_tensor);
            const float* beta = ( const float* )get_tensor_mem(beta_tensor);

            for(int i = 0; i < batch_number; i++)
            {
                for(int c = 0; c < channel_num; c++)
                {
                    float s_mean = scale_mean[c];
                    float s_var = scale_var_inv[c];
                    float s_gamma = gamma[c];
                    float s_beta = beta[c];

                    float s_val1 = s_beta + s_gamma * s_mean;
                    float s_val2 = s_gamma * s_var;

                    float32x4_t _mean = vdupq_n_f32(s_mean);
                    float32x4_t _var = vdupq_n_f32(s_var);
                    float32x4_t _gamma = vdupq_n_f32(s_gamma);
                    float32x4_t _beta = vdupq_n_f32(s_beta);

                    float32x4_t val1 = vmlaq_f32(_beta, _gamma, _mean);
                    float32x4_t val2 = vmulq_f32(_gamma, _var);

                    int offset = i * img_size + c * channel_size;
                    const float* input_ptr = input + offset;
                    float* output_ptr = output + offset;

                    for(int l = 0; l < channel_size - 3; l += 4)
                    {
                        float32x4_t _input = vld1q_f32(input_ptr);
                        // output = val1 + _input * val2
                        vst1q_f32(output_ptr, vmlaq_f32(val1, _input, val2));
                        input_ptr += 4;
                        output_ptr += 4;
                    }
                    for(int l = channel_size & ~3; l < channel_size; l++)
                    {
                        *output_ptr = (*input_ptr) * s_val2 + s_val1;
                        input_ptr++;
                        output_ptr++;
                    }
                }
            }
        }

        return true;
    }

    bool Postrun(Node* node)
    {
        float* scale_mean = any_cast<float*>(node->GetAttr("scale_mean"));
        float* scale_var = any_cast<float*>(node->GetAttr("scale_var_inv"));

        mem_free(scale_mean);
        mem_free(scale_var);

        return true;
    }
};

}    // namespace BatchNormImpl

using namespace BatchNormImpl;

void RegisterBatchNormNodeExec(void)
{
    BNOps* ops = new BNOps();

    NodeOpsRegistryManager::RegisterOPImplementor("arm64", BatchNormName, ops);
}

}    // namespace TEngine
