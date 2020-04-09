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

#include "logger.hpp"
#include "tensor_mem.hpp"

#include "graph.hpp"
#include "node_ops.hpp"
#include "operator/convolution.hpp"

#ifdef CONFIG_AUTH_DEVICE
#include "auth_nodeops.hpp"
#endif

#include <math.h>
namespace TEngine {

namespace conv_2d_dw_dilation {

#define TYPE_A72 1

const char* conv_name = "CONV_DW_DILATION";
const int default_prio = 40;

static float elem_activation(float tmp, int type)
{
    if(type == 0)
    {
        if(tmp < 0.0f)
            tmp = 0;
        if(type > 0)
            tmp = tmp < type ? tmp : type;
    }

    return tmp;
}

static float32x4_t vector_activation(float32x4_t tmp, int type)
{
    if(type == 0)
    {
        float32x4_t zero = vdupq_n_f32(0.0);
        tmp = vmaxq_f32(tmp, zero);
        if(type > 0)
        {
            float32x4_t max = vdupq_n_f32((float)type);
            tmp = vminq_f32(tmp, max);
        }
    }

    return tmp;
}

struct dw_param
{
    float* input_buf;
    float* weight_buf;
    float* bias;
    float* output_buf;
    int input_h;
    int input_w;
    int channel_num;
    int pad;
};

struct Conv2dDepthDilation : public MTNodeOps
{
    Conv2dDepthDilation()
    {
        name_ = "arm_dw_dilat_conv_fp32";
    }
    bool Run(Node* node);

    int activation;

    void DirectConv(float* input_buf, float* weight_buf, float* bias, float* output_buf, int input_h, int input_w,
                    int channel_num, int pad);

    bool Aider(int cpu, int seq, void* data);
};

bool Conv2dDepthDilation::Aider(int cpu, int seq, void* data)
{
    dw_param* param = ( dw_param* )data;

    DirectConv(param->input_buf, param->weight_buf, param->bias, param->output_buf, param->input_h, param->input_w,
               param->channel_num, param->pad);

    return true;
}

void Conv2dDepthDilation::DirectConv(float* input_buf, float* weight_buf, float* bias, float* output_buf, int input_h,
                                     int input_w, int channel, int pad)
{
    int channel_size = input_h * input_w;
    int mid_w = input_w - pad * 2;
    int mid_block_end = (mid_w & -4) + pad;
    int mid_end = mid_w + pad;
    int w = 0;
    for(int c = 0; c < channel; c++)
    {
        float* input_buf_c = input_buf + c * channel_size;
        float* output_buf_c = output_buf + c * channel_size;
        float* weight_buf_c = weight_buf + c * 9;
        float bias_c = bias ? bias[c] : 0;
        for(int h = 0; h < pad; h++)
        {
            for(w = 0; w < pad; w++)
            {
                float tmp = bias_c;
                tmp += weight_buf_c[4] * input_buf_c[h * input_w + w];
                tmp += weight_buf_c[5] * input_buf_c[h * input_w + w + pad];
                tmp += weight_buf_c[7] * input_buf_c[(h + pad) * input_w + w];
                tmp += weight_buf_c[8] * input_buf_c[(h + pad) * input_w + w + pad];
                output_buf_c[h * input_w + w] = elem_activation(tmp, activation);
            }
            for(; w < mid_block_end; w += 4)
            {
                float32x4_t tmp_4 = vdupq_n_f32(bias_c);

                tmp_4 = vmlaq_f32(tmp_4, vdupq_n_f32(weight_buf_c[3]), vld1q_f32(input_buf_c + h * input_w + w - pad));
                tmp_4 = vmlaq_f32(tmp_4, vdupq_n_f32(weight_buf_c[4]), vld1q_f32(input_buf_c + h * input_w + w));
                tmp_4 = vmlaq_f32(tmp_4, vdupq_n_f32(weight_buf_c[5]), vld1q_f32(input_buf_c + h * input_w + w + pad));
                tmp_4 = vmlaq_f32(tmp_4, vdupq_n_f32(weight_buf_c[6]),
                                  vld1q_f32(input_buf_c + (h + pad) * input_w + w - pad));
                tmp_4 =
                    vmlaq_f32(tmp_4, vdupq_n_f32(weight_buf_c[7]), vld1q_f32(input_buf_c + (h + pad) * input_w + w));
                tmp_4 = vmlaq_f32(tmp_4, vdupq_n_f32(weight_buf_c[8]),
                                  vld1q_f32(input_buf_c + (h + pad) * input_w + w + pad));
                tmp_4 = vector_activation(tmp_4, activation);
                vst1q_f32(output_buf_c + h * input_w + w, tmp_4);
            }
            for(; w < mid_end; w++)
            {
                float tmp = bias_c;
                tmp += weight_buf_c[3] * input_buf_c[h * input_w + w - pad];
                tmp += weight_buf_c[4] * input_buf_c[h * input_w + w];
                tmp += weight_buf_c[5] * input_buf_c[h * input_w + w + pad];
                tmp += weight_buf_c[6] * input_buf_c[(h + pad) * input_w + w - pad];
                tmp += weight_buf_c[7] * input_buf_c[(h + pad) * input_w + w];
                tmp += weight_buf_c[8] * input_buf_c[(h + pad) * input_w + w + pad];
                output_buf_c[h * input_w + w] = elem_activation(tmp, activation);;
            }
            for(; w < input_w; w++)
            {
                float tmp = bias_c;
                tmp += weight_buf_c[3] * input_buf_c[h * input_w + w - pad];
                tmp += weight_buf_c[4] * input_buf_c[h * input_w + w];
                tmp += weight_buf_c[6] * input_buf_c[(h + pad) * input_w + w - pad];
                tmp += weight_buf_c[7] * input_buf_c[(h + pad) * input_w + w];
                output_buf_c[h * input_w + w] = elem_activation(tmp, activation);;
            }
        }
        for(int h = pad; h < input_h - pad; h++)
        {
            for(w = 0; w < pad; w++)
            {
                float tmp = bias_c;
                tmp += weight_buf_c[1] * input_buf_c[(h - pad) * input_w + w];
                tmp += weight_buf_c[2] * input_buf_c[(h - pad) * input_w + w + pad];
                tmp += weight_buf_c[4] * input_buf_c[h * input_w + w];
                tmp += weight_buf_c[5] * input_buf_c[h * input_w + w + pad];
                tmp += weight_buf_c[7] * input_buf_c[(h + pad) * input_w + w];
                tmp += weight_buf_c[8] * input_buf_c[(h + pad) * input_w + w + pad];
                output_buf_c[h * input_w + w] = elem_activation(tmp, activation);;
            }
            for(; w < mid_block_end; w += 4)
            {
                float32x4_t tmp_4 = vdupq_n_f32(bias_c);

                tmp_4 = vmlaq_f32(tmp_4, vdupq_n_f32(weight_buf_c[0]),
                                  vld1q_f32(input_buf_c + (h - pad) * input_w + w - pad));
                tmp_4 =
                    vmlaq_f32(tmp_4, vdupq_n_f32(weight_buf_c[1]), vld1q_f32(input_buf_c + (h - pad) * input_w + w));
                tmp_4 = vmlaq_f32(tmp_4, vdupq_n_f32(weight_buf_c[2]),
                                  vld1q_f32(input_buf_c + (h - pad) * input_w + w + pad));
                tmp_4 = vmlaq_f32(tmp_4, vdupq_n_f32(weight_buf_c[3]), vld1q_f32(input_buf_c + h * input_w + w - pad));
                tmp_4 = vmlaq_f32(tmp_4, vdupq_n_f32(weight_buf_c[4]), vld1q_f32(input_buf_c + h * input_w + w));
                tmp_4 = vmlaq_f32(tmp_4, vdupq_n_f32(weight_buf_c[5]), vld1q_f32(input_buf_c + h * input_w + w + pad));
                tmp_4 = vmlaq_f32(tmp_4, vdupq_n_f32(weight_buf_c[6]),
                                  vld1q_f32(input_buf_c + (h + pad) * input_w + w - pad));
                tmp_4 =
                    vmlaq_f32(tmp_4, vdupq_n_f32(weight_buf_c[7]), vld1q_f32(input_buf_c + (h + pad) * input_w + w));
                tmp_4 = vmlaq_f32(tmp_4, vdupq_n_f32(weight_buf_c[8]),
                                  vld1q_f32(input_buf_c + (h + pad) * input_w + w + pad));
                tmp_4 = vector_activation(tmp_4, activation);
                vst1q_f32(output_buf_c + h * input_w + w, tmp_4);
            }
            for(; w < mid_end; w++)
            {
                float tmp = bias_c;
                tmp += weight_buf_c[0] * input_buf_c[(h - pad) * input_w + w - pad];
                tmp += weight_buf_c[1] * input_buf_c[(h - pad) * input_w + w];
                tmp += weight_buf_c[2] * input_buf_c[(h - pad) * input_w + w + pad];
                tmp += weight_buf_c[3] * input_buf_c[h * input_w + w - pad];
                tmp += weight_buf_c[4] * input_buf_c[h * input_w + w];
                tmp += weight_buf_c[5] * input_buf_c[h * input_w + w + pad];
                tmp += weight_buf_c[6] * input_buf_c[(h + pad) * input_w + w - pad];
                tmp += weight_buf_c[7] * input_buf_c[(h + pad) * input_w + w];
                tmp += weight_buf_c[8] * input_buf_c[(h + pad) * input_w + w + pad];
                output_buf_c[h * input_w + w] = elem_activation(tmp, activation);;
            }
            for(; w < input_w; w++)
            {
                float tmp = bias_c;
                tmp += weight_buf_c[0] * input_buf_c[(h - pad) * input_w + w - pad];
                tmp += weight_buf_c[1] * input_buf_c[(h - pad) * input_w + w];
                tmp += weight_buf_c[3] * input_buf_c[h * input_w + w - pad];
                tmp += weight_buf_c[4] * input_buf_c[h * input_w + w];
                tmp += weight_buf_c[6] * input_buf_c[(h + pad) * input_w + w - pad];
                tmp += weight_buf_c[7] * input_buf_c[(h + pad) * input_w + w];
                output_buf_c[h * input_w + w] = elem_activation(tmp, activation);;
            }
        }
        for(int h = input_h - pad; h < input_h; h++)
        {
            for(w = 0; w < pad; w++)
            {
                float tmp = bias_c;
                tmp += weight_buf_c[1] * input_buf_c[(h - pad) * input_w + w];
                tmp += weight_buf_c[2] * input_buf_c[(h - pad) * input_w + w + pad];
                tmp += weight_buf_c[4] * input_buf_c[h * input_w + w];
                tmp += weight_buf_c[5] * input_buf_c[h * input_w + w + pad];
                output_buf_c[h * input_w + w] = elem_activation(tmp, activation);;
            }
            for(; w < mid_block_end; w += 4)
            {
                float32x4_t tmp_4 = vdupq_n_f32(bias_c);

                tmp_4 = vmlaq_f32(tmp_4, vdupq_n_f32(weight_buf_c[0]),
                                  vld1q_f32(input_buf_c + (h - pad) * input_w + w - pad));
                tmp_4 =
                    vmlaq_f32(tmp_4, vdupq_n_f32(weight_buf_c[1]), vld1q_f32(input_buf_c + (h - pad) * input_w + w));
                tmp_4 = vmlaq_f32(tmp_4, vdupq_n_f32(weight_buf_c[2]),
                                  vld1q_f32(input_buf_c + (h - pad) * input_w + w + pad));
                tmp_4 = vmlaq_f32(tmp_4, vdupq_n_f32(weight_buf_c[3]), vld1q_f32(input_buf_c + h * input_w + w - pad));
                tmp_4 = vmlaq_f32(tmp_4, vdupq_n_f32(weight_buf_c[4]), vld1q_f32(input_buf_c + h * input_w + w));
                tmp_4 = vmlaq_f32(tmp_4, vdupq_n_f32(weight_buf_c[5]), vld1q_f32(input_buf_c + h * input_w + w + pad));
                tmp_4 = vector_activation(tmp_4, activation);
                vst1q_f32(output_buf_c + h * input_w + w, tmp_4);
            }
            for(; w < mid_end; w++)
            {
                float tmp = bias_c;
                tmp += weight_buf_c[0] * input_buf_c[(h - pad) * input_w + w - pad];
                tmp += weight_buf_c[1] * input_buf_c[(h - pad) * input_w + w];
                tmp += weight_buf_c[2] * input_buf_c[(h - pad) * input_w + w + pad];
                tmp += weight_buf_c[3] * input_buf_c[h * input_w + w - pad];
                tmp += weight_buf_c[4] * input_buf_c[h * input_w + w];
                tmp += weight_buf_c[5] * input_buf_c[h * input_w + w + pad];
                output_buf_c[h * input_w + w] = elem_activation(tmp, activation);;
            }
            for(; w < input_w; w++)
            {
                float tmp = bias_c;
                tmp += weight_buf_c[0] * input_buf_c[(h - pad) * input_w + w - pad];
                tmp += weight_buf_c[1] * input_buf_c[(h - pad) * input_w + w];
                tmp += weight_buf_c[3] * input_buf_c[h * input_w + w - pad];
                tmp += weight_buf_c[4] * input_buf_c[h * input_w + w];
                output_buf_c[h * input_w + w] = elem_activation(tmp, activation);;
            }
        }
    }
}

bool Conv2dDepthDilation::Run(Node* node)
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

    Tensor* weight_tensor = node->GetInputTensor(1);
    float* weight_buf = ( float* )get_tensor_mem(weight_tensor);
    float* input_buf = ( float* )get_tensor_mem(input_tensor);
    float* output_buf = ( float* )get_tensor_mem(output_tensor);

    int cpu_number = cpu_info->GetCPUNumber();
    int pad_h0 = param->pad_h0;
    // int pad_w0 = param->pad_w0;
    // int pad_h1 = param->pad_h1;
    // int pad_w1 = param->pad_w1;

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
            DirectConv(input_buf, weight_buf, bias, output_buf, input_h, input_w, input_c, pad_h0);
        else
        {
            // partition into 4 tasks
            std::vector<sub_op_task> task_list;
            std::vector<dw_param> param_list;

            auto f = std::bind(&Conv2dDepthDilation::Aider, this, std::placeholders::_1, std::placeholders::_2,
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

                param->input_buf = input_buf + i * step * channel_size;
                param->weight_buf = weight_buf + i * step * 9;
                param->bias = bias ? bias + i * step : nullptr;
                param->output_buf = output_buf + i * step * channel_size;
                param->input_h = input_h;
                param->input_w = input_w;
                param->channel_num = step;
                param->pad = pad_h0;
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

    if(input_h < 4 || input_w < 4)
        return false;
    if(group == 1 || input_c != group || kernel_h != 3 || kernel_w != 3 || dilation_h != dilation_w ||
       dilation_h != pad_h0 || dilation_w != pad_w0 || pad_h0 != pad_w0 || pad_h1 != pad_w1 || stride_w != stride_h ||
       stride_w != 1)
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

    Conv2dDepthDilation* ops = new Conv2dDepthDilation();

    ops->activation = param->activation;

    return ops;
}

}    // namespace conv_2d_dw_dilation

void RegisterConv2dDepthDilation(void)
{
    if(!NodeOpsRegistryManager::RegisterOPImplementor("arm32", "Convolution", conv_2d_dw_dilation::SelectFunc,
                                                      conv_2d_dw_dilation::default_prio))
        LOG_ERROR() << __FUNCTION__ << " :Regist OP failed for prio[" << conv_2d_dw_dilation::default_prio << "]\n";
}

}    // namespace TEngine
