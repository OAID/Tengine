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
 * Author: haoluo@openailab.com
 */
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <arm_neon.h>
#include <math.h>

#include "logger.hpp"
#include "tensor_mem.hpp"

#include "graph.hpp"
#include "node_ops.hpp"
#include "operator/convolution.hpp"
#include "depthwise_conv.hpp"

#ifdef CONFIG_AUTH_DEVICE
#include "auth_nodeops.hpp"
#endif

namespace TEngine {

namespace conv_2d_dw_k7 {


const char* conv_name = "CONV_DW_K7";
const int default_prio = 30;

struct dw_param
{
    float* input_buf;
    float* weight_buf;
    float* bias;
    float* output_buf;
    int input_h;
    int input_w;
    int channel_num;
    int output_h;
    int output_w;
    int stride;
};

struct Conv2dDepthK7 : public MTNodeOps
{
    Conv2dDepthK7()
    {
        name_ = "arm_dw7x7_conv_fp32";
    }

    bool Run(Node* node);

    int activation;

    bool Aider(int cpu, int seq, void* data);
};

bool Conv2dDepthK7::Aider(int cpu, int seq, void* data)
{
    dw_param* param = ( dw_param* )data;
    if(param->stride == 1)
        depthwise_conv_k7s1(param->input_buf, param->weight_buf, param->bias, param->output_buf, param->input_h, param->input_w,
               param->channel_num, param->output_h, param->output_w, activation);
    else if(param->stride == 2)
        depthwise_conv_k7s2(param->input_buf, param->weight_buf, param->bias, param->output_buf, param->input_h, param->input_w,
               param->channel_num, param->output_h, param->output_w, activation);
    else
        return false;

    return true;
}

bool Conv2dDepthK7::Run(Node* node)
{
    Tensor* input_tensor = node->GetInputTensor(0);
    Convolution* conv_op = dynamic_cast<Convolution*>(node->GetOp());
    ConvParam* param = conv_op->GetParam();
    const TShape& input_shape = input_tensor->GetShape();

    int input_c = input_shape.GetC();
    int input_h = input_shape.GetH();
    int input_w = input_shape.GetW();

    /* output */
    Tensor* output_tensor = node->GetOutputTensor(0);
    TShape& output_shape = output_tensor->GetShape();

    int output_n = output_shape.GetN();
    int output_h = output_shape.GetH();
    int output_w = output_shape.GetW();

    Tensor* weight_tensor = node->GetInputTensor(1);
    float* weight_buf = ( float* )get_tensor_mem(weight_tensor);
    float* input_buf = ( float* )get_tensor_mem(input_tensor);
    float* output_buf = ( float* )get_tensor_mem(output_tensor);

    int cpu_number = cpu_info->GetCPUNumber();
    int stride_h = param->stride_h;

    // get bias
    float* bias = NULL;
    if(node->GetInputNum() > 2)
    {
        Tensor* bias_tensor = node->GetInputTensor(2);
        bias = ( float* )get_tensor_mem(bias_tensor);
    }

    for(int i = 0; i < output_n; i++)
    {
        if(cpu_number == 1)
        {
            if(stride_h == 1)
                depthwise_conv_k7s1(input_buf, weight_buf, bias, output_buf, input_h, input_w, input_c, output_h, output_w, activation);
            else if(stride_h == 2)
                depthwise_conv_k7s2(input_buf, weight_buf, bias, output_buf, input_h, input_w, input_c, output_h, output_w, activation);
            else
                return false;
        }
        else
        {
            // partition into 4 tasks
            std::vector<sub_op_task> task_list;
            std::vector<dw_param> param_list;

            auto f = std::bind(&Conv2dDepthK7::Aider, this, std::placeholders::_1, std::placeholders::_2,
                               std::placeholders::_3);

            int step = input_c / cpu_number;
            if(step < 1)
                step = 1;
            int max_task_num = input_c / step;

            task_list.resize(max_task_num);
            param_list.resize(max_task_num);

            int input_hw = input_h * input_w;
            int output_hw = output_h * output_w;

            for(int i = 0; i < max_task_num; i++)
            {
                dw_param* param = &param_list[i];
                sub_op_task* task = &task_list[i];

                task->exec_func = f;
                task->seq = i;
                task->data = param;

                param->input_buf = input_buf + i * step * input_hw;
                param->weight_buf = weight_buf + i * step * 49;
                param->bias = bias ? bias + i * step : nullptr;
                param->output_buf = output_buf + i * step * output_hw;
                param->input_h = input_h;
                param->input_w = input_w;
                param->channel_num = step;
                param->output_h = output_h;
                param->output_w = output_w;
                param->stride = stride_h;
            }

            // the last left ones
            param_list[max_task_num - 1].channel_num += input_c - max_task_num * step;

            task_dispatch(task_list, -1);

            wait_done();
        }
    }

    return true;
}

static bool isDepthwiseSupported(const ConvParam* param, const TShape& input_shape)
{
    int input_c = input_shape.GetC();
    int input_h = input_shape.GetH();
    int input_w = input_shape.GetW();
    int group = param->group;
    int kernel_h = param->kernel_h;
    int kernel_w = param->kernel_w;
    int stride_h = param->stride_h;
    int stride_w = param->stride_w;
    int dilation_h = param->dilation_h;
    int dilation_w = param->dilation_w;
    int pad_h0 = param->pad_h0;
    int pad_w0 = param->pad_w0;
    int pad_h1 = param->pad_h1;
    int pad_w1 = param->pad_w1;

    if(input_h < 8 || input_w < 8)
        return false;
    if(group == 1 || input_c != group || kernel_h != 7 || kernel_w != 7 || dilation_h != dilation_w ||dilation_h != 1||
       pad_h1 != pad_h0 || pad_w1 != pad_w0 || pad_h0 != 3 || pad_w0 != 3 || stride_w != stride_h )
    {
        return false;
    }
    return true;
}

NodeOps* SelectFunc(const CPUInfo* cpu_info, Node* node)
{
    Tensor* input = node->GetInputTensor(0);
    const int data_type = input->GetDataType();
    const ExecAttr* exec_attr = any_cast<const ExecAttr*>(node->GetAttr(ATTR_EXEC_ATTR));

#ifdef CONFIG_AUTH_DEVICE
    bool float_enabled = get_auth_float_enabled();

    if(!float_enabled)
        return nullptr;
#endif

    if(data_type != TENGINE_DT_FP32 || exec_attr->graph_layout != TENGINE_LAYOUT_NCHW)

        return nullptr;

    Operator* op = node->GetOp();

    Convolution* conv_op = dynamic_cast<Convolution*>(op);

    ConvParam* param = conv_op->GetParam();

    const TShape& input_shape = node->GetInputTensor(0)->GetShape();

    if(!isDepthwiseSupported(param, input_shape))
        return nullptr;

    Conv2dDepthK7* ops = new Conv2dDepthK7();

    ops->activation = param->activation;

    return ops;
}

}    // namespace conv_2d_dw_k7

void RegisterConv2dDepthK7(void)
{
    if(!NodeOpsRegistryManager::RegisterOPImplementor("arm64", "Convolution", conv_2d_dw_k7::SelectFunc,
                                                      conv_2d_dw_k7::default_prio))
        LOG_ERROR() << __FUNCTION__ << " :Regist OP failed for prio[" << conv_2d_dw_k7::default_prio << "]\n";
}

}    // namespace TEngine
