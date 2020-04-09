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
 * Author: chunyinglv@openailab.com
 */

#include <iostream>
#include <cstring>
#include <cstdlib>

#include "logger.hpp"
#include "tensor_mem.hpp"
#include "node_ops.hpp"

#include "graph.hpp"
#include "operator/convolution.hpp"

#ifdef CONFIG_AUTH_DEVICE
#include "auth_nodeops.hpp"
#endif

#define CONV_DW_MAX(a, b) ((a) > (b) ? (a) : (b))
#define CONV_DW_MIN(a, b) ((a) < (b) ? (a) : (b))


namespace TEngine {

namespace conv_2d_dw {

inline float do_activation(float input, int activation)
{
    if(activation == 0)
    {
        input = CONV_DW_MAX(input, 0);
        if(activation == 6)
            input =CONV_DW_MIN(input, 6);
    }
    return input;
}

const char* conv_name = "CONV_DW";
const int default_prio = 50;

struct dw_param
{
    float* input_buf;
    float* weight_buf;
    float* output_buf;

    int group_start;
    int group_end;
    int activation;

    int input_c;
    int input_h;
    int input_w;

    int output_c;
    int output_h;
    int output_w;

    int ker_h;
    int ker_w;
    int pad_h;
    int pad_w;
    int stride_h;
    int stride_w;
    int dilation_h;
    int dilation_w;
};

void initial_output(float* output, float* bias, int output_ch, int output_wh)
{
    int i, j;
    // no bias
    if(bias == nullptr)
    {
        memset(output, 0.f, output_ch * output_wh* sizeof(float));
    }
    else
    {
        float* out_ptr= output;
        for(i = 0; i < output_ch; i++)
            for(j = 0; j < output_wh; j++)
                *out_ptr++ = bias[i];
    }
}
struct Conv2dDepth : public MTNodeOps
{
    Conv2dDepth()
    {
        name_ = "arm_dw_conv_fp32";
    }
    bool Run(Node* node);

    int activation;

    bool Aider(int cpu, int seq, void* data);
};
void conv_dw_genreal_kernel(const float *input, const float *kernel,float *output, 
                            int group_start,int group_end, int activation,
                            int input_c, int input_h, int input_w,
                            int output_c, int output_h, int output_w,
                            int kernel_h, int kernel_w,
                            int pad_h, int pad_w, int stride_h, int stride_w,
                            int dilation_h, int dilation_w)
{
    int c, h, w, kc, k_h, k_w;
    int input_offset = 0;
    int kernel_offset = 0;
    int output_offset = 0;
    int kernel_size = input_c * kernel_h * kernel_w;
    int out_hw = output_w * output_h;

    for(int g= group_start; g < group_end; g++)
    {
        for (c = 0; c < output_c; c++)
        {
            for (h = 0; h < output_h; h++)
            {
                for (w = 0; w < output_w; w++)
                {
                    const int h_start = (h * stride_h) - pad_h;
                    const int w_start = (w * stride_w) - pad_w;
                    float total = 0.f;
                    output_offset = (g * output_c  + c) * out_hw + h * output_w + w;
                    for (kc = 0; kc < input_c; kc++)
                    {
                        for (k_h = 0; k_h < kernel_h; k_h++)
                        {
                            for (k_w = 0; k_w < kernel_w; k_w++)
                            {
                                const int cur_y = h_start + dilation_h * k_h;
                                const int cur_x = w_start + dilation_w * k_w;
                                if((cur_x >= 0) && (cur_x < input_w) && (cur_y >= 0) && (cur_y < input_h))
                                {
                                    input_offset = (g * input_c + kc)* input_h * input_w + cur_y * input_w + cur_x;
                                    kernel_offset = (g * output_c + c) * kernel_size + kc * kernel_h* kernel_w +
                                                    k_h * kernel_w + k_w;
                                    total += (input[input_offset] * kernel[kernel_offset]);
                                }
                            }
                        }
                    }
                    output[output_offset] = do_activation(output[output_offset]+total, activation);
                }
            }
        }
    }
}

bool Conv2dDepth::Aider(int cpu, int seq, void* data)
{
    dw_param* param = ( dw_param* )data;

    conv_dw_genreal_kernel(param->input_buf, param->weight_buf,param->output_buf,
                param->group_start,param->group_end,param->activation,
                param->input_c,param->input_h, param->input_w,  
                param->output_c, param->output_h, param->output_w,
                param->ker_h,param->ker_w,param->pad_h,param->pad_w,param->stride_h,param->stride_w,
                param->dilation_h,param->dilation_w);

    return true;
}


bool Conv2dDepth::Run(Node* node)
{
    Tensor* input_tensor = node->GetInputTensor(0);
    Convolution* conv_op = dynamic_cast<Convolution*>(node->GetOp());
    ConvParam* param_ = conv_op->GetParam();

    int group=param_->group;
    int kernel_h = param_->kernel_h;
    int kernel_w = param_->kernel_w;
    int stride_h = param_->stride_h;
    int stride_w = param_->stride_w;
    int dilation_h = param_->dilation_h;
    int dilation_w = param_->dilation_w;
    int pad_h = param_->pad_h0;
    int pad_w = param_->pad_w0;

    const TShape& input_shape = input_tensor->GetShape();

    int input_c = input_shape.GetC();
    int input_h = input_shape.GetH();
    int input_w = input_shape.GetW();

    /* output */
    Tensor* output_tensor = node->GetOutputTensor(0);
    TShape& output_shape = output_tensor->GetShape();

    int output_h = output_shape.GetH();
    int output_w = output_shape.GetW();
    int output_n = output_shape.GetN();
    int output_c = output_shape.GetC();
    int output_hw = output_h * output_w;

    int input_c0 = input_c/group;
    int output_c0 = output_c/group;

    Tensor* weight_tensor = node->GetInputTensor(1);
    float* weight_buf = ( float* )get_tensor_mem(weight_tensor);
    float* input_buf = ( float* )get_tensor_mem(input_tensor);
    float* output_buf = ( float* )get_tensor_mem(output_tensor);

    int input_size = input_c * input_h * input_w;
    int output_size = output_c * output_h * output_w;

    int cpu_number = cpu_info->GetCPUNumber();


    float* bias = nullptr;

    if(node->GetInputNum() > 2)
    {
        Tensor* bias_tensor = node->GetInputTensor(2);
        bias = ( float* )get_tensor_mem(bias_tensor);
    }

    for(int b = 0; b < output_n; b++)
    {
        float* cur_input = input_buf + b * input_size;
        float* cur_output = output_buf + b * output_size;

        initial_output(cur_output, bias, output_c, output_hw);
        if(cpu_number == 1)
        {
            conv_dw_genreal_kernel(cur_input,
                            weight_buf,
                            cur_output,
                            0,group, 
                            activation,
                            input_c0,input_h, input_w,
                            output_c0,output_h,output_w,
                            kernel_h,kernel_w,
                            pad_h,pad_w,stride_h,stride_w,
                            dilation_h,dilation_w);
        }
        else
        {
            std::vector<sub_op_task> task_list;
            std::vector<dw_param> param_list;
            int step = group/cpu_number;
            int task_number = cpu_number;
            if(group <=cpu_number)
            {
                task_number=group;
                step=1;
            }
            task_list.resize(task_number);
            param_list.resize(task_number);

            auto f = std::bind(&Conv2dDepth::Aider, this, std::placeholders::_1, std::placeholders::_2,
                                std::placeholders::_3);
            for(int i = 0; i < task_number; i++)
            {
                
                dw_param* param = &param_list[i];
                sub_op_task* task = &task_list[i];
                task->exec_func = f;
                task->seq = i;
                task->data = param;

                param->input_buf = cur_input;
                param->weight_buf = weight_buf;
                param->output_buf = cur_output;
                param->group_start = i*step;
                param->group_end = param->group_start + step;
                param->activation = activation;
                param->input_c=input_c0;
                param->input_h = input_h;
                param->input_w = input_w;
                param->output_c = output_c0;
                param->output_h = output_h;
                param->output_w = output_w;

                param->ker_h = kernel_h;
                param->ker_w = kernel_w;
                param->pad_h = pad_h;
                param->pad_w = pad_w;
                param->stride_h = stride_h;
                param->stride_w = stride_w;
                param->dilation_h = dilation_h;
                param->dilation_w = dilation_w;

            }
            param_list[task_number - 1].group_end = group;
            task_dispatch(task_list, -1);
            wait_done();
        }

 
    }
    return true;
}

NodeOps* SelectFunc(const CPUInfo* cpu_info, Node* node)
{
#ifdef CONFIG_AUTH_DEVICE
    if(!get_auth_float_enabled())
        return nullptr;
#endif

    Tensor* input = node->GetInputTensor(0);
    const int data_type = input->GetDataType();
    int input_c = input->GetShape().GetC();

    Operator* op = node->GetOp();

    Convolution* conv_op = dynamic_cast<Convolution*>(op);

    ConvParam* param = conv_op->GetParam();

    int group = param->group;
    int out_c = param->output_channel;
    if(group == 1 || group != out_c || input_c != group)
    {
        return nullptr;
    }

    const ExecAttr* exec_attr = any_cast<const ExecAttr*>(node->GetAttr(ATTR_EXEC_ATTR));

    if(data_type != TENGINE_DT_FP32 || exec_attr->graph_layout != TENGINE_LAYOUT_NCHW)
        return nullptr;

    Conv2dDepth* ops = new Conv2dDepth();

    ops->activation = param->activation;

    return ops;
}

}    // namespace conv_2d_dw

void RegisterConv2dDepth(void)
{
    if(!NodeOpsRegistryManager::RegisterOPImplementor("arm32", "Convolution", conv_2d_dw::SelectFunc,
                                                      conv_2d_dw::default_prio))
        LOG_ERROR() << __FUNCTION__ << " :Regist OP failed for prio[" << conv_2d_dw::default_prio << "]\n";
}

}    // namespace TEngine
