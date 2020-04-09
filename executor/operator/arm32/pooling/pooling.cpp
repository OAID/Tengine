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
#include <stdlib.h>

#include "node_ops.hpp"
#include "graph.hpp"
#include "operator/pooling.hpp"
#include "tensor_mem.hpp"
#include "pooling_kernel.h"

#ifdef CONFIG_AUTH_DEVICE
#include "auth_nodeops.hpp"
#endif

namespace TEngine {

namespace PoolingImpl {

const int default_prio = 100;

typedef void (*pool_kernel_t)(const float* input, float* output, int inc, int in_h, int in_w, int out_h, int out_w,
                              int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h0, int pad_w0,
                              int pad_h1, int pad_w1, int is_caffe);

struct PoolingOps : public MTNodeOps
{
    PoolingOps()
    {
        name_ = "arm_pool_fp32";
    }

    PoolingSize pooling_size = POOL_GENERIC;
    pool_kernel_t kernel_run = nullptr;

    void pool_kernel(int i, int id, void* data, const float* input, float* output, int in_h, int in_w,
                            int out_h, int out_w, int kernel_h, int kernel_w, int stride_h, int stride_w,
                            int pad_h0, int pad_w0, int pad_h1, int pad_w1, int is_caffe)
    {
        int step = ((int*)data)[0];
        int in_channel_size = in_h * in_w;
        int out_channel_size = out_h * out_w;
        const float* cur_input = input + id * step * in_channel_size;
        float* cur_output = output + id * step * out_channel_size;
        kernel_run(cur_input, cur_output, step, in_h, in_w, out_h, out_w, kernel_h, kernel_w,
                stride_h, stride_w, pad_h0, pad_w0, pad_h1, pad_w1, is_caffe);
    }

    bool Prerun(Node* node)
    {
        // operator, param
        Pooling* pooling_op = dynamic_cast<Pooling*>(node->GetOp());
        PoolParam* param_ = pooling_op->GetParam();

        if(param_->stride_h == 2 && param_->stride_w == 2)
        {
            if(param_->kernel_h == 2 && param_->kernel_w == 2)
                pooling_size = POOL_K2S2;
            else if(param_->kernel_h == 3 && param_->kernel_w == 3)
                pooling_size = POOL_K3S2;
        }
        else if(param_->stride_h == 1 && param_->stride_w == 1 && param_->kernel_h == 3 && param_->kernel_w == 3)
        {
            pooling_size = POOL_K3S1;
        }

        if(param_->alg == kPoolMax)
        {
            if(param_->global == 1)
            {
                kernel_run = Global_MaxPool;
                return true;
            }

            kernel_run = Generic_MaxPool;
            if(param_->pad_h0 == 0 && param_->pad_w0 == 0)
            {
                if(pooling_size == POOL_K2S2)
                    kernel_run = MaxPool_2x2s2;
                else if(pooling_size == POOL_K3S2)
                    kernel_run = MaxPool_3x3s2;
            }
            else if(param_->pad_h0 == 1 && param_->pad_w0 == 1)
            {
                if(pooling_size == POOL_K2S2)
                    kernel_run = MaxPool_2x2s2_pad1;
                else if(pooling_size == POOL_K3S2)
                    kernel_run = MaxPool_3x3s2_pad1;
                else if(pooling_size == POOL_K3S1)
                    kernel_run = MaxPool_3x3s1_pad1;
            }

            return true;
        }

        if(param_->alg == kPoolAvg)
        {
            if(param_->global == 1)
            {
                kernel_run = Global_AvgPool;
                return true;
            }

            kernel_run = Generic_AvgPool;
            if(param_->pad_h0 == 0 && param_->pad_w0 == 0)
            {
                if(pooling_size == POOL_K2S2)
                    kernel_run = AvgPool_2x2s2;
                else if(pooling_size == POOL_K3S2)
                    kernel_run = AvgPool_3x3s2;
            }

            if(param_->pad_h0 == 1 && param_->pad_w0 == 1)
            {
                if(pooling_size == POOL_K2S2)
                    kernel_run = AvgPool_2x2s2_pad1;
                else if(pooling_size == POOL_K3S2)
                    kernel_run = AvgPool_3x3s2_pad1;
            }

            return true;
        }

        return false;
    }

    bool Run(Node* node)
    {
        // operator, param
        Pooling* pooling_op = dynamic_cast<Pooling*>(node->GetOp());
        PoolParam* param_ = pooling_op->GetParam();

        // input, output, shape
        Tensor* itensor = node->GetInputTensor(0);
        const TShape& ishape = itensor->GetShape();
        Tensor* otensor = node->GetOutputTensor(0);
        TShape& oshape = otensor->GetShape();
        // dim=[n,c,h,w]
        const std::vector<int>& in_dim = ishape.GetDim();
        const std::vector<int>& out_dim = oshape.GetDim();
        int in_hw = in_dim[3] * in_dim[2];
        int in_chw = in_dim[1] * in_hw;

        int out_hw = out_dim[2] * out_dim[3];
        int out_chw = out_dim[1] * out_hw;
        // data
        float* input_data = ( float* )get_tensor_mem(itensor);
        float* output_data = ( float* )get_tensor_mem(otensor);

#if 0
    printf("input: %d,%d,%d   --> output: %d,%d \n",
                in_dim[1], in_dim[2], in_dim[3], out_dim[2], out_dim[3]);
    printf("kernel: %d, stride: %d, arg: %d, pad: %d,%d,%d,%d\n",
                param_->kernel_h, param_->stride_h, param_->alg,
                param_->pad_h0,param_->pad_w0,param_->pad_h1,param_->pad_w1);
#endif
        int is_caffe = param_->caffe_flavor;

        int cpu_number = cpu_info->GetCPUNumber();
        int block = in_dim[1];
        block = block > 0 ? block : 1;
        int num_task = cpu_number < block ? cpu_number : block;
        int step = in_dim[1] / num_task;

        for(int n = 0; n < in_dim[0]; n++)
        {
            float* in_ptr = input_data + n * in_chw;
            float* out_ptr = output_data + n * out_chw;
            if(num_task == 1)
                pool_kernel(0, 0, &step, in_ptr, out_ptr, in_dim[2], in_dim[3], out_dim[2], out_dim[3],
                        param_->kernel_h, param_->kernel_w, param_->stride_h, param_->stride_w,
                        param_->pad_h0, param_->pad_w0, param_->pad_h1, param_->pad_w1, is_caffe);
            else
            {
                MULTI_THREAD_START(num_task, step, p_id, p_param)
                    pool_kernel(0, p_id, p_param, in_ptr, out_ptr, in_dim[2], in_dim[3], out_dim[2], out_dim[3],
                        param_->kernel_h, param_->kernel_w, param_->stride_h, param_->stride_w,
                        param_->pad_h0, param_->pad_w0, param_->pad_h1, param_->pad_w1, is_caffe);
                MULTI_THREAD_END();
            }
            if(num_task * step != in_dim[1])
            {
                int offset = num_task * step;
                int remain_num = in_dim[1] - offset;
                in_ptr += offset * in_hw;
                out_ptr += offset * out_hw;
                pool_kernel(0, 0, &remain_num, in_ptr, out_ptr, in_dim[2], in_dim[3], out_dim[2], out_dim[3],
                        param_->kernel_h, param_->kernel_w, param_->stride_h, param_->stride_w,
                        param_->pad_h0, param_->pad_w0, param_->pad_h1, param_->pad_w1, is_caffe);
            }
        }

        return true;
    }
};

NodeOps* SelectFunc(const CPUInfo* cpu_info, Node* node)
{
#ifdef CONFIG_AUTH_DEVICE
    if(!get_auth_float_enabled())
        return nullptr;
#endif

    Tensor* input = node->GetInputTensor(0);
    const int data_type = input->GetDataType();
    const ExecAttr* exec_attr = any_cast<const ExecAttr*>(node->GetAttr(ATTR_EXEC_ATTR));
    if(data_type != TENGINE_DT_FP32 || exec_attr->graph_layout != TENGINE_LAYOUT_NCHW)
        return nullptr;

    PoolingOps* ops = new PoolingOps();

    return ops;
}

}    // namespace PoolingImpl

void RegisterPoolingNodeExec(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor("arm32", "Pooling", PoolingImpl::SelectFunc,
                                                  PoolingImpl::default_prio);
}

}    // namespace TEngine
