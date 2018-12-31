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

namespace TEngine {

namespace PoolingImpl {

typedef void (*pool_kernel_t)(const float* input, float* output, int inc, int in_h, int inw, int out_h, int out_w,
                              int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h0, int pad_w0,
                              int pad_h1, int pad_w1, int is_caffe);

struct PoolingOps : public NodeOps
{
    PoolingSize pooling_size = POOL_GENERIC;
    pool_kernel_t kernel_run = nullptr;

    struct pooling_param
    {
        pool_kernel_t fun;
        float* input;
        float* output;
        int in_h;
        int in_w;
        int out_h;
        int out_w;
        int kernel_h;
        int kernel_w;
        int stride_h;
        int stride_w;
        int pad_h0;
        int pad_w0;
        int pad_h1;
        int pad_w1;
        int is_caffe;
    };

    bool pooling_aider(int cpu, int seq, void* data)
    {
        pooling_param* param = ( pooling_param* )(data);
        param->fun(param->input, param->output, 1, param->in_h, param->in_w, param->out_h, param->out_w,
                   param->kernel_h, param->kernel_w, param->stride_h, param->stride_w, param->pad_h0, param->pad_w0,
                   param->pad_h1, param->pad_w1, param->is_caffe);

        return true;
    }

    bool Prerun(Node* node)
    {
        // operator, param
        Pooling* pooling_op = dynamic_cast<Pooling*>(node->GetOp());
        PoolParam* param_ = pooling_op->GetParam();

        if(param_->strides[0] == 2 && param_->strides[1] == 2)
        {
            if(param_->kernel_shape[0] == 2 && param_->kernel_shape[1] == 2)
                pooling_size = POOL_K2S2;
            else if(param_->kernel_shape[0] == 3 && param_->kernel_shape[1] == 3)
                pooling_size = POOL_K3S2;
        }
        else if(param_->strides[0] == 1 && param_->strides[1] == 1 && param_->kernel_shape[0] == 3 &&
                param_->kernel_shape[1] == 3)
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
            if(param_->pads[0] == 0 && param_->pads[1] == 0)
            {
                if(pooling_size == POOL_K2S2)
                    kernel_run = MaxPool_2x2s2;
                else if(pooling_size == POOL_K3S2)
                    kernel_run = MaxPool_3x3s2;
            }
            else if(param_->pads[0] == 1 && param_->pads[1] == 1)
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
            if(param_->pads[0] == 0 && param_->pads[1] == 0)
            {
                if(pooling_size == POOL_K2S2)
                    kernel_run = AvgPool_2x2s2;
                else if(pooling_size == POOL_K3S2)
                    kernel_run = AvgPool_3x3s2;
            }

            if(param_->pads[0] == 1 && param_->pads[1] == 1)
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
                param_->kernel_shape[0], param_->strides[0], param_->alg,
                param_->pads[0],param_->pads[1],param_->pads[2],param_->pads[3]);
#endif
        int is_caffe = param_->caffe_flavor;
        for(int n = 0; n < in_dim[0]; n++)
        {
            float* in_ptr = input_data + n * in_chw;
            float* out_ptr = output_data + n * out_chw;
            if(!exec_attr->pooling_mt)
            {
                kernel_run(in_ptr, out_ptr, in_dim[1], in_dim[2], in_dim[3], out_dim[2], out_dim[3],
                           param_->kernel_shape[0], param_->kernel_shape[1], param_->strides[0], param_->strides[1],
                           param_->pads[0], param_->pads[1], param_->pads[2], param_->pads[3], is_caffe);
            }
            else
            {
                std::vector<sub_op_task> task_list;
                std::vector<pooling_param> param_list;

                task_list.resize(in_dim[1]);
                param_list.resize(in_dim[1]);

                auto f = std::bind(&PoolingOps::pooling_aider, this, std::placeholders::_1, std::placeholders::_2,
                                   std::placeholders::_3);

                for(int i = 0; i < in_dim[1]; i++)
                {
                    pooling_param* param = &param_list[i];
                    sub_op_task* task = &task_list[i];

                    task->exec_func = f;
                    task->seq = i;
                    task->data = param;

                    param->fun = kernel_run;
                    param->input = in_ptr + in_hw * i;
                    param->output = out_ptr + out_hw * i;
                    param->in_h = in_dim[2];
                    param->in_w = in_dim[3];
                    param->out_h = out_dim[2];
                    param->out_w = out_dim[3];
                    param->kernel_h = param_->kernel_shape[0];
                    param->kernel_w = param_->kernel_shape[1];
                    param->stride_h = param_->strides[0];
                    param->stride_w = param_->strides[1];
                    param->pad_h0 = param_->pads[0];
                    param->pad_w0 = param_->pads[1];
                    param->pad_h1 = param_->pads[2];
                    param->pad_w1 = param_->pads[3];
                    param->is_caffe = is_caffe;
                }

                task_dispatch(task_list, -1);
                wait_done();
            }
        }

        return true;
    }
};

const int default_prio = 100;
NodeOps* SelectFunc(const CPUInfo* cpu_info, Node* node)
{
    const ExecAttr* exec_attr = any_cast<const ExecAttr*>(node->GetAttr(ATTR_EXEC_ATTR));
    if(exec_attr->layout == TENGINE_LAYOUT_NHWC)
        return nullptr;

    PoolingOps* ops = new PoolingOps();

    ops->need_free = true;

    return ops;
}

}    // namespace PoolingImpl

void RegisterPoolingNodeExec(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor("arm64", "Pooling", PoolingImpl::SelectFunc,
                                                  PoolingImpl::default_prio);
}

}    // namespace TEngine
