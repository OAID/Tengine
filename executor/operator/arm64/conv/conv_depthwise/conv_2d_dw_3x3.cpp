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
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "logger.hpp"
#include "tensor_mem.hpp"

#include "graph.hpp"
#include "node_ops.hpp"
#include "operator/convolution.hpp"
#include "op_utils.hpp"

#ifdef CONFIG_AUTH_DEVICE
#include "auth_nodeops.hpp"
#endif

#include <math.h>
namespace TEngine {

namespace conv_2d_dw_3x3 {

#define TYPE_A53 0
#define TYPE_A72 1

const char* conv_name = "CONV_DW_3x3";
const int default_prio = 10;

#include "conv_2d_dw_3x3_kernel/A72.inl"

extern "C" void dw_k3s2p0(float* data, int h, int w, float* kernel, float* output, float* bias, int out_w, int act);
extern "C" void dw_k3s2p0p1(float* data, int h, int w, float* kernel, float* output, float* bias, int out_w, int act);

struct dw_param
{
    float* input_buf;
    int input_h;
    int input_w;
    float* output_buf;
    int output_h;
    int output_w;
    float* weight_buf;
    int channel_num;
    int stride;
    float* bias;
    int pads[4];
};

struct Conv2dDepth : public MTNodeOps
{
    Conv2dDepth()
    {
        name_ = "arm_dw3x3_conv_fp32";
    }

    bool Run(Node* node);

    int activation;

    void DirectConv(float* input_buf, int input_h, int input_w, float* output_buf, int output_h, int output_w,
                    float* weight_buf, int channel_num, int stride, float* bias, int* pads, int cpu_type);

    bool Aider(int cpu, int seq, void* data);
};

bool Conv2dDepth::Aider(int cpu, int seq, void* data)
{
    dw_param* param = ( dw_param* )data;

    int cpu_type = -1;
    if(cpu_info->GetCPUModel(cpu) == CPU_A72)
        cpu_type = TYPE_A72;
    else
        cpu_type = TYPE_A53;

    DirectConv(param->input_buf, param->input_h, param->input_w, param->output_buf, param->output_h, param->output_w,
               param->weight_buf, param->channel_num, param->stride, param->bias, param->pads, cpu_type);

    return true;
}

void Conv2dDepth::DirectConv(float* input_buf, int input_h, int input_w, float* output_buf, int output_h, int output_w,
                             float* weight_buf, int channel_num, int stride, float* bias, int* pads, int cpu_type)
{
    int channel_size = input_h * input_w;
    float* bias_tmp = bias;

    int pad_h0 = pads[0];
    int pad_h1 = pads[2];

    {
        for(int i = 0; i < channel_num; i++)
        {
            if(bias)
                bias_tmp = bias + i;

            if(stride == 1)
            {
                dw_k3s1p1_a72(input_buf, input_h, input_w, weight_buf, output_buf, bias_tmp, activation);
            }
            else if(stride == 2)
            {
                if(pad_h0 == 0)
                {
                    if(pad_h1 == 0)
                        dw_k3s2p0(input_buf, input_h, input_w, weight_buf, output_buf, bias_tmp, output_w, activation);
                    else
                        dw_k3s2p0p1(input_buf, input_h, input_w, weight_buf, output_buf, bias_tmp, output_w,
                                    activation);
                }
                else
                    dw_k3s2p1_a72(input_buf, input_h, input_w, weight_buf, output_buf, bias_tmp, activation);
            }

            input_buf += channel_size;
            output_buf += output_h * output_w;
            weight_buf += 9;
        }
    }
}

bool Conv2dDepth::Run(Node* node)
{
    Tensor* input_tensor = node->GetInputTensor(0);
    Convolution* conv_op = dynamic_cast<Convolution*>(node->GetOp());
    ConvParam* param = conv_op->GetParam();
    const TShape& input_shape = input_tensor->GetShape();
    int cpu_type = -1;

    int input_c = input_shape.GetC();
    int input_h = input_shape.GetH();
    int input_w = input_shape.GetW();

    /* output */
    Tensor* output_tensor = node->GetOutputTensor(0);
    TShape& output_shape = output_tensor->GetShape();

    int output_h = output_shape.GetH();
    int output_w = output_shape.GetW();
    int output_n = output_shape.GetN();

    Tensor* weight_tensor = node->GetInputTensor(1);
    float* weight_buf = ( float* )get_tensor_mem(weight_tensor);
    float* input_buf = ( float* )get_tensor_mem(input_tensor);
    float* output_buf = ( float* )get_tensor_mem(output_tensor);

    int stride_h = param->stride_h;
    int cpu_number = cpu_info->GetCPUNumber();
    int pad_h0 = param->pad_h0;
    int pad_w0 = param->pad_w0;
    int pad_h1 = param->pad_h1;
    int pad_w1 = param->pad_w1;
    float* bias = NULL;

    if(cpu_info->GetCPUModel(cpu_info->GetMasterCPU()) == CPU_A72)
        cpu_type = TYPE_A72;
    else
        cpu_type = TYPE_A53;

    // get bias
    if(node->GetInputNum() > 2)
    {
        Tensor* bias_tensor = node->GetInputTensor(2);
        bias = ( float* )get_tensor_mem(bias_tensor);
    }

    for(int i = 0; i < output_n; i++)
    {
        if(cpu_number == 1)
            DirectConv(input_buf, input_h, input_w, output_buf, output_h, output_w, weight_buf, input_c, stride_h, bias,
                       ( int* )&param->pad_h0, cpu_type);
        else
        {
            // partition into 4 tasks
            std::vector<sub_op_task> task_list;
            std::vector<dw_param> param_list;

            auto f = std::bind(&Conv2dDepth::Aider, this, std::placeholders::_1, std::placeholders::_2,
                               std::placeholders::_3);

            int step = input_c / cpu_number;
            if(step < 1)
                step = 1;
            int max_task_num = input_c / step;

            task_list.resize(max_task_num);
            param_list.resize(max_task_num);

            int channel_size = input_h * input_w;

            for(int i = 0; i < max_task_num; i++)
            {
                dw_param* param = &param_list[i];
                sub_op_task* task = &task_list[i];

                task->exec_func = f;
                task->seq = i;
                task->data = param;

                param->input_buf = input_buf;
                param->input_h = input_h;
                param->input_w = input_w;
                param->output_buf = output_buf;
                param->output_h = output_h;
                param->output_w = output_w;
                param->weight_buf = weight_buf;
                param->channel_num = step;
                param->stride = stride_h;
                param->pads[0] = pad_h0;
                param->pads[1] = pad_w0;
                param->pads[2] = pad_h1;
                param->pads[3] = pad_w1;
                if(NULL != bias)
                    param->bias = bias + i * step;
                else
                    param->bias = NULL;

                input_buf += channel_size * step;
                if(stride_h == 1)
                    output_buf += channel_size * step;
                else
                    output_buf += output_h * output_w * step;
                weight_buf += 9 * step;
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
#if 0
    if(group == 1 || input_c != group || kernel_h != 3 || kernel_w != 3 ||
       pad_h0 != 1 || pad_w0 !=1 || pad_h0 != pad_h1 || pad_w0 != pad_w1 ||
       dilation_h != 1 || dilation_w != 1 || stride_w != stride_h)
    {
        return false;
    }
#endif
    if(input_h < 4 || input_w < 4)
        return false;
    if(group == 1 || input_c != group || kernel_h != 3 || kernel_w != 3 || (pad_h0 != 0 && pad_h0 != 1) ||
       (pad_w0 != 0 && pad_w0 != 1) || pad_h0 != pad_w0 || pad_h1 != pad_w1 || dilation_h != 1 || dilation_w != 1 ||
       stride_w != stride_h)
    {
        return false;
    }

    if (kernel_h == 3 && pad_h0 == 0 && dilation_h == 1 && stride_w == 1)
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

    Conv2dDepth* ops = new Conv2dDepth();

    ops->activation = param->activation;

    return ops;
}

}    // namespace conv_2d_dw_3x3

void RegisterConv2dDepth3x3(void)
{
    if(!NodeOpsRegistryManager::RegisterOPImplementor("arm64", "Convolution", conv_2d_dw_3x3::SelectFunc,
                                                      conv_2d_dw_3x3::default_prio))
        LOG_ERROR() << __FUNCTION__ << " :Regist OP failed for prio[" << conv_2d_dw_3x3::default_prio << "]\n";
}

}    // namespace TEngine
