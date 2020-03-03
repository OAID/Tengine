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

namespace conv_2d_direct_3x3_dilation {

const char* conv_name = "CONV_DIRECT_3X3_DILATION";
const int default_prio = 150;


static void vector_set_value(float* buf, int size, float value)
{
    float32x4_t value_4 = vdupq_n_f32(value);
    int i = 0;
    for(i = 0; i + 3 < size; i +=4)
    {
        vst1q_f32(buf + i, value_4);
    }
    for(;i < size;i++)
        buf[i] = value;
}

static void vector_activation(float* buf, int size, int type)
{
    if(type == 0)
    {
        float32x4_t zero = vdupq_n_f32(0.0);
        float32x4_t max = vdupq_n_f32((float)type);
        int i = 0;
        for(i = 0; i + 3 < size; i +=4)
        {
            float32x4_t value_4 = vld1q_f32(buf + i);
            value_4 = vmaxq_f32(value_4, zero);
            if(type > 0)
            {
                value_4 = vminq_f32(value_4, max);
                
            }
            vst1q_f32(buf + i, value_4);
        }
        for(;i < size;i++)
        {
            float value = buf[i];
            value = value > 0? value:0.f;
            if(type > 0)
                value = value > type? (float)type : value;
            buf[i] = value;
        }

    }
}

struct direct_3x3_param
{
    float* input_buf;
    float* weight_buf;
    float* bias;
    float* output_buf;
    int input_h;
    int input_w;
    int input_c;
    int output_c;
    int pad;
};

struct Conv2dDirect3x3Dilation : public MTNodeOps
{
    Conv2dDirect3x3Dilation()
    {
        name_ = "arm_direct3x3_dialtion_fp32";
    }

    bool Run(Node* node);

    int activation;

    void DirectConv(float* input_buf, float* weight_buf, float* bias, float* output_buf, int input_h, int input_w,
                    int input_c, int output_c, int pad);

    bool Aider(int cpu, int seq, void* data);
};

bool Conv2dDirect3x3Dilation::Aider(int cpu, int seq, void* data)
{
    direct_3x3_param* param = ( direct_3x3_param* )data;

    DirectConv(param->input_buf, param->weight_buf, param->bias, param->output_buf, param->input_h, param->input_w,
               param->input_c, param->output_c, param->pad);

    return true;
}

void Conv2dDirect3x3Dilation::DirectConv(float* input_buf, float* weight_buf, float* bias, float* output_buf, int input_h,
                                     int input_w, int input_c, int output_c, int pad)
{
    int channel_size = input_h * input_w;
    int mid_w = input_h - pad * 2;
    int mid_block_end = (mid_w & -4) + pad;
    int mid_end = mid_w + pad;
    int h = 0, w = 0;
    for(int c = 0; c < output_c; c++)
    {
        float* output_buf_c = output_buf + c * channel_size;
        float bias_c = bias ? bias[c] : 0;
        vector_set_value(output_buf_c, channel_size, bias_c);
        
        for(int inc = 0; inc < input_c; inc++)
        {
            float* input_buf_c = input_buf + inc * channel_size;
            float* weight_buf_c = weight_buf + (c * input_c + inc) * 9;
            float32x4_t kernel_0 = vdupq_n_f32(weight_buf_c[0]);
            float32x4_t kernel_1 = vdupq_n_f32(weight_buf_c[1]);
            float32x4_t kernel_2 = vdupq_n_f32(weight_buf_c[2]);
            float32x4_t kernel_3 = vdupq_n_f32(weight_buf_c[3]);
            float32x4_t kernel_4 = vdupq_n_f32(weight_buf_c[4]);
            float32x4_t kernel_5 = vdupq_n_f32(weight_buf_c[5]);
            float32x4_t kernel_6 = vdupq_n_f32(weight_buf_c[6]);
            float32x4_t kernel_7 = vdupq_n_f32(weight_buf_c[7]);
            float32x4_t kernel_8 = vdupq_n_f32(weight_buf_c[8]);
            for(h = 0; h < pad; h++)
            {
                for(w = 0; w < pad; w++)
                {
                    float tmp = weight_buf_c[4] * input_buf_c[h * input_w + w];
                    tmp += weight_buf_c[5] * input_buf_c[h * input_w + w + pad];
                    tmp += weight_buf_c[7] * input_buf_c[(h + pad) * input_w + w];
                    tmp += weight_buf_c[8] * input_buf_c[(h + pad) * input_w + w + pad];
                    output_buf_c[h * input_w + w] += tmp;
                    // if(h==0 && w==0)
                        // printf("output[%d]= %f\n",c, output_buf_c[0]);
                }
                for(; w < mid_block_end; w += 4)
                {
                    float32x4_t out_4 = vld1q_f32(output_buf_c + h * input_w + w);
                    out_4 = vmlaq_f32(out_4, kernel_3, vld1q_f32(input_buf_c + h * input_w + w - pad));
                    out_4 = vmlaq_f32(out_4, kernel_4, vld1q_f32(input_buf_c + h * input_w + w));
                    out_4 = vmlaq_f32(out_4, kernel_5, vld1q_f32(input_buf_c + h * input_w + w + pad));
                    out_4 = vmlaq_f32(out_4, kernel_6, vld1q_f32(input_buf_c + (h + pad) * input_w + w - pad));
                    out_4 = vmlaq_f32(out_4, kernel_7, vld1q_f32(input_buf_c + (h + pad) * input_w + w));
                    out_4 = vmlaq_f32(out_4, kernel_8, vld1q_f32(input_buf_c + (h + pad) * input_w + w + pad));
                    vst1q_f32(output_buf_c + h * input_w + w, out_4);
                }
                for(; w < mid_end; w++)
                {
                    float tmp = weight_buf_c[3] * input_buf_c[h * input_w + w - pad];
                    tmp += weight_buf_c[4] * input_buf_c[h * input_w + w];
                    tmp += weight_buf_c[5] * input_buf_c[h * input_w + w + pad];
                    tmp += weight_buf_c[6] * input_buf_c[(h + pad) * input_w + w - pad];
                    tmp += weight_buf_c[7] * input_buf_c[(h + pad) * input_w + w];
                    tmp += weight_buf_c[8] * input_buf_c[(h + pad) * input_w + w + pad];
                    output_buf_c[h * input_w + w] += tmp;
                }
                for(; w < input_w; w++)
                {
                    float tmp = weight_buf_c[3] * input_buf_c[h * input_w + w - pad];
                    tmp += weight_buf_c[4] * input_buf_c[h * input_w + w];
                    tmp += weight_buf_c[6] * input_buf_c[(h + pad) * input_w + w - pad];
                    tmp += weight_buf_c[7] * input_buf_c[(h + pad) * input_w + w];
                    output_buf_c[h * input_w + w] += tmp;
                }
            }
            for(; h < input_h - pad; h++)
            {
                for(w = 0; w < pad; w++)
                {
                    float tmp = weight_buf_c[1] * input_buf_c[(h - pad) * input_w + w];
                    tmp += weight_buf_c[2] * input_buf_c[(h - pad) * input_w + w + pad];
                    tmp += weight_buf_c[4] * input_buf_c[h * input_w + w];
                    tmp += weight_buf_c[5] * input_buf_c[h * input_w + w + pad];
                    tmp += weight_buf_c[7] * input_buf_c[(h + pad) * input_w + w];
                    tmp += weight_buf_c[8] * input_buf_c[(h + pad) * input_w + w + pad];
                    output_buf_c[h * input_w + w] += tmp;
                }
                for(; w < mid_block_end; w += 4)
                {
                    float32x4_t out_4 = vld1q_f32(output_buf_c + h * input_w + w);
                    out_4 = vmlaq_f32(out_4, kernel_0, vld1q_f32(input_buf_c + (h - pad) * input_w + w - pad));
                    out_4 = vmlaq_f32(out_4, kernel_1, vld1q_f32(input_buf_c + (h - pad) * input_w + w));
                    out_4 = vmlaq_f32(out_4, kernel_2, vld1q_f32(input_buf_c + (h - pad) * input_w + w + pad));
                    out_4 = vmlaq_f32(out_4, kernel_3, vld1q_f32(input_buf_c + h * input_w + w - pad));
                    out_4 = vmlaq_f32(out_4, kernel_4, vld1q_f32(input_buf_c + h * input_w + w));
                    out_4 = vmlaq_f32(out_4, kernel_5, vld1q_f32(input_buf_c + h * input_w + w + pad));
                    out_4 = vmlaq_f32(out_4, kernel_6, vld1q_f32(input_buf_c + (h + pad) * input_w + w - pad));
                    out_4 = vmlaq_f32(out_4, kernel_7, vld1q_f32(input_buf_c + (h + pad) * input_w + w));
                    out_4 = vmlaq_f32(out_4, kernel_8, vld1q_f32(input_buf_c + (h + pad) * input_w + w + pad));
                    vst1q_f32(output_buf_c + h * input_w + w, out_4);
                }
                for(; w < mid_end; w++)
                {
                    float tmp = weight_buf_c[0] * input_buf_c[(h - pad) * input_w + w - pad];
                    tmp += weight_buf_c[1] * input_buf_c[(h - pad) * input_w + w];
                    tmp += weight_buf_c[2] * input_buf_c[(h - pad) * input_w + w + pad];
                    tmp += weight_buf_c[3] * input_buf_c[h * input_w + w - pad];
                    tmp += weight_buf_c[4] * input_buf_c[h * input_w + w];
                    tmp += weight_buf_c[5] * input_buf_c[h * input_w + w + pad];
                    tmp += weight_buf_c[6] * input_buf_c[(h + pad) * input_w + w - pad];
                    tmp += weight_buf_c[7] * input_buf_c[(h + pad) * input_w + w];
                    tmp += weight_buf_c[8] * input_buf_c[(h + pad) * input_w + w + pad];
                    output_buf_c[h * input_w + w] += tmp;
                }
                for(; w < input_w; w++)
                {
                    float tmp = weight_buf_c[0] * input_buf_c[(h - pad) * input_w + w - pad];
                    tmp += weight_buf_c[1] * input_buf_c[(h - pad) * input_w + w];
                    tmp += weight_buf_c[3] * input_buf_c[h * input_w + w - pad];
                    tmp += weight_buf_c[4] * input_buf_c[h * input_w + w];
                    tmp += weight_buf_c[6] * input_buf_c[(h + pad) * input_w + w - pad];
                    tmp += weight_buf_c[7] * input_buf_c[(h + pad) * input_w + w];
                    output_buf_c[h * input_w + w] += tmp;
                }
            }
            for(; h < input_h; h++)
            {
                for(w = 0; w < pad; w++)
                {
                    float tmp = weight_buf_c[1] * input_buf_c[(h - pad) * input_w + w];
                    tmp += weight_buf_c[2] * input_buf_c[(h - pad) * input_w + w + pad];
                    tmp += weight_buf_c[4] * input_buf_c[h * input_w + w];
                    tmp += weight_buf_c[5] * input_buf_c[h * input_w + w + pad];
                    output_buf_c[h * input_w + w] += tmp;
                }
                for(; w < mid_block_end; w += 4)
                {
                    float32x4_t out_4 = vld1q_f32(output_buf_c + h * input_w + w);
                    out_4 = vmlaq_f32(out_4, kernel_0, vld1q_f32(input_buf_c + (h - pad) * input_w + w - pad));
                    out_4 = vmlaq_f32(out_4, kernel_1, vld1q_f32(input_buf_c + (h - pad) * input_w + w));
                    out_4 = vmlaq_f32(out_4, kernel_2, vld1q_f32(input_buf_c + (h - pad) * input_w + w + pad));
                    out_4 = vmlaq_f32(out_4, kernel_3, vld1q_f32(input_buf_c + h * input_w + w - pad));
                    out_4 = vmlaq_f32(out_4, kernel_4, vld1q_f32(input_buf_c + h * input_w + w));
                    out_4 = vmlaq_f32(out_4, kernel_5, vld1q_f32(input_buf_c + h * input_w + w + pad));
                    vst1q_f32(output_buf_c + h * input_w + w, out_4);
                }
                for(; w < mid_end; w++)
                {
                    float tmp = weight_buf_c[0] * input_buf_c[(h - pad) * input_w + w - pad];
                    tmp += weight_buf_c[1] * input_buf_c[(h - pad) * input_w + w];
                    tmp += weight_buf_c[2] * input_buf_c[(h - pad) * input_w + w + pad];
                    tmp += weight_buf_c[3] * input_buf_c[h * input_w + w - pad];
                    tmp += weight_buf_c[4] * input_buf_c[h * input_w + w];
                    tmp += weight_buf_c[5] * input_buf_c[h * input_w + w + pad];
                    output_buf_c[h * input_w + w] += tmp;
                }
                for(; w < input_w; w++)
                {
                    float tmp = weight_buf_c[0] * input_buf_c[(h - pad) * input_w + w - pad];
                    tmp += weight_buf_c[1] * input_buf_c[(h - pad) * input_w + w];
                    tmp += weight_buf_c[3] * input_buf_c[h * input_w + w - pad];
                    tmp += weight_buf_c[4] * input_buf_c[h * input_w + w];
                    output_buf_c[h * input_w + w] += tmp;
                }
            }
        }
        vector_activation(output_buf_c, channel_size, activation);
    }
}

bool Conv2dDirect3x3Dilation::Run(Node* node)
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
    int output_c = output_shape.GetC();

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
            DirectConv(input_buf, weight_buf, bias, output_buf, input_h, input_w, input_c, output_c, pad_h0);
        else
        {
            // partition into 4 tasks
            std::vector<sub_op_task> task_list;
            std::vector<direct_3x3_param> param_list;

            auto f = std::bind(&Conv2dDirect3x3Dilation::Aider, this, std::placeholders::_1, std::placeholders::_2,
                               std::placeholders::_3);

            int max_task_num = output_c > cpu_number ? cpu_number : output_c;
            int step = output_c / max_task_num;

            task_list.resize(max_task_num);
            param_list.resize(max_task_num);

            int channel_size = input_h * input_w;

            for(int i = 0; i < max_task_num; i++)
            {
                direct_3x3_param* param = &param_list[i];
                sub_op_task* task = &task_list[i];

                task->exec_func = f;
                task->seq = i;
                task->data = param;

                param->input_buf = input_buf;
                param->weight_buf = weight_buf + i * step * 9 * input_c;
                param->bias = bias ? bias + i * step : nullptr;
                param->output_buf = output_buf + i * step * channel_size;
                param->input_h = input_h;
                param->input_w = input_w;
                param->input_c = input_c;
                param->output_c = step;
                param->pad = pad_h0;
            }

            // the last left ones
            param_list[max_task_num - 1].output_c += output_c - max_task_num * step;

            task_dispatch(task_list, -1);

            wait_done();
        }
    }

    return true;
}

static bool isDirect3x3Supported(const ConvParam* param, const TShape& output_shape)
{
    
    int output_c = output_shape.GetC();
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
    if(dilation_h != dilation_w || dilation_h == 1 || dilation_h != pad_h0)
        return false;

    if(group > 1 || kernel_h != 3 || kernel_w != 3 || stride_w != stride_h || stride_w != 1 ||
       pad_h0 != pad_w0 || pad_h1 != pad_w1 || output_c > 8)
    {
        return false;
    }
    return true;
}

NodeOps* SelectFunc(const CPUInfo* cpu_info, Node* node)
{
    const char* env = std::getenv("NO_3X3");
    if(env)
        return nullptr;
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

    const TShape& output_shape = node->GetOutputTensor(0)->GetShape();

    if(!isDirect3x3Supported(param, output_shape))
        return nullptr;

    Conv2dDirect3x3Dilation* ops = new Conv2dDirect3x3Dilation();

    ops->activation = param->activation;

    return ops;
}

}    // namespace conv_2d_direct_3x3_dilation

void RegisterConv2dDirect3x3Dilation(void)
{
    if(!NodeOpsRegistryManager::RegisterOPImplementor("arm64", "Convolution", conv_2d_direct_3x3_dilation::SelectFunc,
                                                      conv_2d_direct_3x3_dilation::default_prio))
        LOG_ERROR() << __FUNCTION__ << " :Regist OP failed for prio[" << conv_2d_direct_3x3_dilation::default_prio << "]\n";
}

}    // namespace TEngine
