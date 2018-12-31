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
 * Author: haoluo@openailab.com
 */
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <math.h>
#include <cmath>

#include "logger.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "data_type.hpp"

#include "graph.hpp"
#include "operator/convolution.hpp"

namespace TEngine {

namespace conv_ref {

struct op_data
{
    float i_scale;
    int i_zero;
    float k_scale;
    int k_zero;
    float o_scale;
    int o_zero;
    int activation_min;
    int activation_max;
};

const char* conv_name = "CONV_REF";
const int default_prio = 1500;
/*
template <typename data_type>
void interleave_kernel(void* kernel_org , void* kernel_interleaved,int output_chan ,
            int kernel_h, int kernel_w,int kernel_c)
{
    data_type* kernel = (data_type*) kernel_org;
    data_type* kernel_inter = (data_type*) kernel_interleaved;

    int kernel_size = kernel_h * kernel_w * kernel_c;
    for(int i =0;i<output_chan; i++)
    {
        data_type* kernel_interleaved_cur = kernel_inter + i*kernel_size;
        data_type* kernel_cur = kernel + i* kernel_size;
        for(int h=0;h<kernel_h;h++)
            for(int w=0;w<kernel_w;w++)
                for(int c=0;c<kernel_c;c++)
                {
                    kernel_interleaved_cur[c*kernel_h*kernel_w + h*kernel_w + w] = *kernel_cur++;
                }
    }

}
*/

bool GetQuantizedActivationMinMax(op_data& op_param, int activation_type)
{
    const float scale = op_param.o_scale;
    const int zero = op_param.o_zero;
    auto quantize = [scale, zero](float f) { return zero + static_cast<int>(std::round(f / scale)); };

    if(activation_type == 0)
    {
        op_param.activation_max = 255;
        op_param.activation_min = std::max(0, quantize(0));
    }
    else if(activation_type == 6)
    {
        op_param.activation_max = std::min(255, quantize(6));
        op_param.activation_min = std::max(0, quantize(0));
    }
    else if(activation_type == 1)
    {
        op_param.activation_max = std::min(255, quantize(1));
        op_param.activation_min = std::max(0, quantize(-1));
    }
    else
    {
        op_param.activation_max = 255;
        op_param.activation_min = 0;
    }

    return true;
}
/*
bool GetQuantizedMultiplerShift(op_data& op_param)
{
    const double input_product_scale = op_param.i_scale*op_param.k_scale;
    double double_multiplier = input_product_scale/op_param.o_scale;
    int shift = 0;
    if(double_multiplier<1)
    {
        while(double_multiplier < 0.5)
        {
            double_multiplier*=2;
            shift ++;
        }
    }
    else if(double_multiplier>=1)
    {
        while(double_multiplier>1)
        {
            double_multiplier/=2;
            shift --;
        }
    }
    op_param.multiplier = std::round(double_multiplier * 256);
    op_param.shift = -shift;
    //printf("%f, %f, %f, %f,%d, %d\n",op_param.i_scale,op_param.k_scale,op_param.o_scale,
    //            dd, op_param.multiplier, shift);
    //printf("%d, %d\n",op_param.i_zero,op_param.k_zero);

    return true;
}
*/
template <typename data_type>
void im2col(void* input_org, void* im2col, int input_chan, int input_x, int input_y, int kernel_x, int kernel_y,
            int stride_x, int stride_y, int pad_x0, int pad_y0, int pad_x1, int pad_y1, int output_x, int output_y,
            int group, int i_zero)
{
    data_type* input = ( data_type* )input_org;
    data_type* col = ( data_type* )im2col;

    int input_c = input_chan * group;
    int kernel_size = input_chan * kernel_x * kernel_y;
    for(int h = 0; h < output_y; h++)
    {
        data_type* col_h = col + output_x * kernel_size * h;
        for(int w = 0; w < output_x; w++)
        {
            data_type* col_w = col_h + kernel_size * w;
            int w_start = w * stride_x - pad_x0;
            int w_end = w_start + kernel_x;
            int h_start = h * stride_y - pad_y0;
            int h_end = h_start + kernel_y;

            for(int kh = h_start; kh < h_end; kh++)
                for(int kw = w_start; kw < w_end; kw++)
                    for(int kc = 0; kc < input_chan; kc++)
                    {
                        if(kh < 0 || kh >= input_y || kw < 0 || kw >= input_x)
                        {
                            *col_w++ = ( data_type )i_zero;
                        }
                        else
                            *col_w++ = input[kh * input_c * input_x + kw * input_c + kc];
                    }
        }
    }
}

template <typename data_type>
static void run_kernel(void* input, void* output, void* kernel, void* bias, int activation, int kernel_h, int kernel_w,
                       int input_c, int output_chan, int output_x, int output_y, int group, op_data param)
{
    data_type* output0 = ( data_type* )output;
    data_type* kernel0 = ( data_type* )kernel;

    int in_chan_rel = input_c * group;
    int out_chan_real = output_chan * group;
    int kernel_size = input_c * kernel_h * kernel_w;

    for(int c = 0; c < output_chan; c++)
    {
        data_type* kernel_cur = kernel0 + c * in_chan_rel * kernel_h * kernel_w;
        if(sizeof(data_type) == 4)
        {
            float* bias0 = ( float* )bias;
            float bias_cur = bias0 ? bias0[c] : 0;
            for(int h = 0; h < output_y; h++)
                for(int w = 0; w < output_x; w++)
                {
                    int index = h * output_x * out_chan_real + w * out_chan_real + c;
                    float tmp = bias_cur;
                    float* input_cur = ( float* )input + kernel_size * h * output_x + w * kernel_size;
                    for(int i = 0; i < kernel_h; i++)
                        for(int j = 0; j < kernel_w; j++)
                            for(int k = 0; k < input_c; k++)
                            {
                                int pos = i * kernel_w * in_chan_rel + j * in_chan_rel + k;
                                tmp += *input_cur * kernel_cur[pos];
                                input_cur++;
                            }

                    if(activation == 0)
                    {
                        if(tmp < 0)
                            tmp = 0;
                    }
                    if(activation == 6)
                    {
                        if(tmp < 0)
                            tmp = 0;
                        if(tmp > 6)
                            tmp = 6;
                    }
                    output0[index] = tmp;
                }
        }
        else
        {
            int* bias0 = ( int* )bias;
            int bias_cur = bias0 ? bias0[c] : 0;
            for(int h = 0; h < output_y; h++)
                for(int w = 0; w < output_x; w++)
                {
                    int index = h * output_x * out_chan_real + w * out_chan_real + c;
                    int tmp = bias_cur;
                    uint8_t* input_cur = ( uint8_t* )input + kernel_size * h * output_x + w * kernel_size;
                    for(int i = 0; i < kernel_h; i++)
                        for(int j = 0; j < kernel_w; j++)
                            for(int k = 0; k < input_c; k++)
                            {
                                int pos = i * kernel_w * in_chan_rel + j * in_chan_rel + k;
                                tmp += (*input_cur - param.i_zero) * (kernel_cur[pos] - param.k_zero);
                                input_cur++;
                            }
                    tmp = std::round(tmp * param.i_scale * param.k_scale / param.o_scale);

                    tmp += param.o_zero;
                    tmp = std::max(param.activation_min, tmp);
                    tmp = std::min(param.activation_max, tmp);
                    output0[index] = tmp;
                }
        }
    }
}

struct ConvRef : public MTNodeOps
{
    bool Prerun(Node* node) override;
    bool Run(Node* node) override;
    bool Reshape(Node* node) override;
    bool Postrun(Node* node) override;
    bool GetSharedMemorySize(Node*, unsigned int& mem_size) override;
    bool SetSharedMemoryAddr(Node*, void* mem_addr, int mem_size) override;

    bool RunNHWC(Node* node);
    bool RunNCHW(Node* node);

    op_data op_param;
    int element_size;
    bool dynamic_shape;
};

bool ConvRef::Reshape(Node* node)
{
    unsigned int new_col_size;

    GetSharedMemorySize(node, new_col_size);

    if(node->ExistAttr("col_buf_allocated"))
    {
        unsigned int col_size = any_cast<unsigned int>(node->GetAttr("col_buf_allocated"));

        if(new_col_size == col_size)
            return true;

        float* addr = any_cast<float*>(node->GetAttr("col_buf"));
        mem_free(addr);
    }

    float* col_buf = ( float* )mem_alloc(new_col_size);
    node->SetAttr("col_buf", col_buf);
    node->SetAttr("col_buf_allocated", new_col_size);
    return true;
}

bool ConvRef::SetSharedMemoryAddr(Node* node, void* mem_addr, int mem_size)
{
    (*node)["shared_col_buf"] = mem_addr;
    return true;
}

bool ConvRef::GetSharedMemorySize(Node* node, unsigned int& mem_size)
{
    Convolution* conv_op = dynamic_cast<Convolution*>(node->GetOp());
    ConvParam* param = conv_op->GetParam();
    int group = param->group;

    Tensor* output_tensor = node->GetOutputTensor(0);
    TShape& output_shape = output_tensor->GetShape();
    int output_y = output_shape.GetH();
    int output_x = output_shape.GetW();

    Tensor* input_tensor = node->GetInputTensor(0);
    TShape& input_shape = input_tensor->GetShape();
    element_size = DataType::GetTypeSize(input_tensor->GetDataType());

    int input_chan = input_shape.GetC();
    int kernel_size = input_chan / group * param->kernel_h * param->kernel_w;
    int output_xy = output_x * output_y;

    mem_size = (element_size * kernel_size * output_xy);

    return true;
}

bool ConvRef::Prerun(Node* node)
{
    if(!dynamic_shape)
    {
        if(node->ExistAttr("shared_col_buf"))
        {
            void* addr = any_cast<void*>(node->GetAttr("shared_col_buf"));

            (*node)["col_buf"] = addr;
        }
        else
        {
            unsigned int col_size;

            GetSharedMemorySize(node, col_size);

            void* col_buf = mem_alloc(col_size);
            (*node)["col_buf"] = col_buf;
            node->SetAttr("col_buf_allocated", col_size);
        }
    }
    if(element_size == 1)
    {
        Convolution* conv_op = dynamic_cast<Convolution*>(node->GetOp());
        ConvParam* param = conv_op->GetParam();
        Tensor* input_tensor = node->GetInputTensor(0);
        Tensor* kernel_tensor = node->GetInputTensor(1);
        Tensor* output_tensor = node->GetOutputTensor(0);

        auto* in_quant = input_tensor->GetQuantParam();
        op_param.i_scale = (*in_quant)[0].scale;
        op_param.i_zero = (*in_quant)[0].zero_point;
        auto* k_quant = kernel_tensor->GetQuantParam();
        op_param.k_scale = (*k_quant)[0].scale;
        op_param.k_zero = (*k_quant)[0].zero_point;
        auto* o_quant = output_tensor->GetQuantParam();
        op_param.o_scale = (*o_quant)[0].scale;
        op_param.o_zero = (*o_quant)[0].zero_point;
        // GetQuantizedMultiplerShift(op_param);
        GetQuantizedActivationMinMax(op_param, param->activation);
    }

    return true;
}

bool ConvRef::Run(Node* node)
{
    if(exec_attr->layout == TENGINE_LAYOUT_NHWC)
    {
        return RunNHWC(node);
    }
    else
    {
        // TODO: support NCHW
        return false;
    }
}

bool ConvRef::RunNHWC(Node* node)
{
    Convolution* conv_op = dynamic_cast<Convolution*>(node->GetOp());
    ConvParam* param = conv_op->GetParam();

    int kernel_h = param->kernel_h;
    int kernel_w = param->kernel_w;
    int stride_h = param->stride_h;
    int stride_w = param->stride_w;
    // int  pad_h = param->pad_h;
    // int  pad_w = param->pad_w;
    int dilation_h = param->dilation_h;
    int dilation_w = param->dilation_w;
    int pad_x0 = param->pads[1];    // left padding columns
    int pad_x1 = param->pads[3];    // right padding columns
    int pad_y0 = param->pads[0];    // top padding rows
    int pad_y1 = param->pads[2];    // bottom padding rows
    int group = param->group;
    int activation = param->activation;
    if(dilation_h != 1 || dilation_w != 1)
        return false;

    Tensor* input_tensor = node->GetInputTensor(0);
    uint8_t* input_org = ( uint8_t* )get_tensor_mem(input_tensor);
    TShape& input_shape = input_tensor->GetShape();
    int input_w = input_shape.GetW();
    int input_h = input_shape.GetH();
    int input_c = input_shape.GetC() / group;
    int input_n = input_shape.GetN();
    int input_size = input_w * input_h * input_c;

    Tensor* kernel_tensor = node->GetInputTensor(1);
    uint8_t* kernel = ( uint8_t* )get_tensor_mem(kernel_tensor);

    uint8_t* bias_data = nullptr;
    if(node->GetInputNum() > 2)
    {
        Tensor* bias_tensor = node->GetInputTensor(2);
        bias_data = ( uint8_t* )get_tensor_mem(bias_tensor);
    }

    Tensor* output_tensor = node->GetOutputTensor(0);
    uint8_t* output_org = ( uint8_t* )get_tensor_mem(output_tensor);

    TShape& output_shape = output_tensor->GetShape();
    int output_w = output_shape.GetW();
    int output_h = output_shape.GetH();
    int output_c = output_shape.GetC() / group;
    int output_xy = output_h * output_w;

    void* col_buf = any_cast<void*>(node->GetAttr("col_buf"));
    uint8_t* col = ( uint8_t* )col_buf;

    for(int n = 0; n < input_n; n++)
    {
        uint8_t* input = input_org + n * input_size * group * element_size;
        uint8_t* output = output_org + n * output_xy * output_c * group * element_size;

        for(int g = 0; g < group; g++)
        {
            uint8_t* input_g = input + input_c * g * element_size;
            uint8_t* output_g = output + output_c * g * element_size;
            uint8_t* kernel_g = kernel + input_c * g * element_size;
            uint8_t* bias_g = bias_data ? bias_data + output_c * g * 4 : nullptr;
            if(element_size == 4)
            {
                im2col<float>(input_g, col, input_c, input_w, input_h, kernel_w, kernel_h, stride_w, stride_h, pad_x0,
                              pad_y0, pad_x1, pad_y1, output_w, output_h, group, 0);
                run_kernel<float>(col, output_g, kernel_g, bias_g, activation, kernel_h, kernel_w, input_c, output_c,
                                  output_w, output_h, group, op_param);
            }

            if(element_size == 1)
            {
                im2col<uint8_t>(input_g, col, input_c, input_w, input_h, kernel_w, kernel_h, stride_w, stride_h, pad_x0,
                                pad_y0, pad_x1, pad_y1, output_w, output_h, group, op_param.i_zero);
                run_kernel<uint8_t>(col, output_g, kernel_g, bias_g, activation, kernel_h, kernel_w, input_c, output_c,
                                    output_w, output_h, group, op_param);
            }
        }
    }

    return true;
}

bool ConvRef::Postrun(Node* node)
{
    if(node->ExistAttr("col_buf_allocated"))
    {
        void* addr = any_cast<void*>(node->GetAttr("col_buf"));
        mem_free(addr);
        node->RemoveAttr("col_buf_allocated");
    }
    return true;
}

NodeOps* SelectFunc(const CPUInfo* cpu_info, Node* node)
{
    ConvRef* ops = new ConvRef();

    ops->need_free = true;
    if(node->IsDynamicShape())
        ops->dynamic_shape = true;
    else
        ops->dynamic_shape = false;

    return ops;
}

}    // namespace conv_ref

void RegisterConv2dRef(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor("common", "Convolution", conv_ref::SelectFunc,
                                                  conv_ref::default_prio);
}

}    // namespace TEngine
