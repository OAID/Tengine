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

namespace BatchNormImpl64 {

void batchnorm_kernel(int i, int id, void* data, const float* input, float* output, float* scale_mean, float* scale_var, int channel_size)
{
    int step = ((int*)data)[0];
    for(int c = 0; c < step; c++)
    {
        int cur_c = id * step + c;
        float s_mean = scale_mean[cur_c];
        float s_var = scale_var[cur_c];
        float32x4_t _mean = vdupq_n_f32(s_mean);
        float32x4_t _var = vdupq_n_f32(s_var);
        int offset = cur_c * channel_size;
        const float* input_ptr = input + offset;
        float* output_ptr = output + offset;

        // output[offset]= input[offset]*scale_var_inv[c] - scale_mean[c];
        for(int l = 0; l < (channel_size & -4); l += 4)
        {
            float32x4_t _input = vld1q_f32(input_ptr);
            vst1q_f32(output_ptr, vmlaq_f32(_mean, _input, _var));
            input_ptr += 4;
            output_ptr += 4;
        }
        for(int l = channel_size & ~3; l < channel_size; l++)
        {
            *output_ptr = (*input_ptr) * s_var + s_mean;
            input_ptr++;
            output_ptr++;
        }
    }
}

struct BNOps : public NodeOps
{
    BNOps()
    {
        name_ = "arm_batchnorm_fp32";
    }

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
            float tmp = std::sqrt(var[c] * rescale_factor + eps);
            scale_var_inv[c] = (float)(1.f / tmp);
            tmp = rescale_factor * scale_var_inv[c];
            scale_mean[c] = (float)(-mean[c] * tmp);
        }
        if(!param->caffe_flavor)
        {
            const Tensor* gamma_tensor = node->GetInputTensor(1);
            const Tensor* beta_tensor = node->GetInputTensor(2);
            const float* gamma = ( const float* )get_tensor_mem(gamma_tensor);
            const float* beta = ( const float* )get_tensor_mem(beta_tensor);
            for(int c = 0; c < channel_num; c++)
            {
                scale_var_inv[c] *= gamma[c];
                scale_mean[c] *= gamma[c];
                scale_mean[c] += beta[c];
            }
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

        const float* input = ( const float* )get_tensor_mem(input_tensor);
        float* output = ( float* )get_tensor_mem(output_tensor);

        float* scale_mean = any_cast<float*>(node->GetAttr("scale_mean"));
        float* scale_var_inv = any_cast<float*>(node->GetAttr("scale_var_inv"));


        int cpu_number = cpu_info->GetCPUNumber();
        int block = channel_num;
        block = block > 0 ? block : 1;
        int num_task = cpu_number < block ? cpu_number : block;
        int step = channel_num / num_task;
        /* only use mean and var */
        for(int i = 0; i < batch_number; i++)
        {
            const float* cur_input = input + i * img_size;
            float* cur_output = output + i * img_size;

            if(num_task == 1)
                batchnorm_kernel( 0, 0, &step, cur_input, cur_output, scale_mean, scale_var_inv, channel_size);
            else
            {
                MULTI_THREAD_START(num_task, step, p_id, p_param)
                    batchnorm_kernel( 0, p_id, p_param, cur_input, cur_output, scale_mean, scale_var_inv, channel_size);
                MULTI_THREAD_END();
            }
            if(num_task * step != channel_num)
            {
                int offset = num_task * step;
                int remain_num = channel_num - offset;
                cur_input += offset * channel_size;
                cur_output += offset * channel_size;
                batchnorm_kernel( 0, 0, &remain_num, cur_input, cur_output, scale_mean + offset, scale_var_inv + offset, channel_size);
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

NodeOps* SelectFunc(const CPUInfo* cpu_info, Node* node)
{
    Tensor* input = node->GetInputTensor(0);
    const ExecAttr* exec_attr = any_cast<const ExecAttr*>(node->GetAttr(ATTR_EXEC_ATTR));
    if(input->GetDataType() != TENGINE_DT_FP32 || exec_attr->graph_layout != TENGINE_LAYOUT_NCHW)
        return nullptr;
    if(input->GetShape().GetDim().size() != 4 && input->GetShape().GetDim().size() != 3)
        return nullptr;

    BNOps* ops = new BNOps();

    return ops;
}

}    // namespace BatchNormImpl64

using namespace BatchNormImpl64;

void RegisterBatchNormNodeExec(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor("arm64", BatchNormName, BatchNormImpl64::SelectFunc, 1000);
}

}    // namespace TEngine
