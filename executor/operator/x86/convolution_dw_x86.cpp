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
 * Copyright (c) 2020, OPEN AI LAB
 * Author: qtang@openailab.com
 */

#include <iostream>
#include <functional>
#include <cstring>
#include <algorithm>

#include "logger.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "graph.hpp"
#include "operator/convolution.hpp"
#include "dwconvolution_x86.h"
#include <math.h>

namespace TEngine {

namespace ConvolutionDwImpl {

struct ConvolutionDwOps : public NodeOps
{
    bool Prerun(Node* node) override;
    bool Run(Node* node) override;
    bool Postrun(Node* node) override;
    bool Reshape(Node* node) override;
    void pad(float* input, float* output, int in_h, int in_w, int out_h, int out_w, int top, int left, float v);
    void relu(float* data, int size, int activation);
};

void ConvolutionDwOps::relu(float* data, int size, int activation)
{
    for(int i = 0; i < size; i++)
    {
        data[i] = std::max(data[i], ( float )0);

        if(activation > 0)
        {
            data[i] = std::min(data[i], ( float )activation);
        }
    }
}

bool ConvolutionDwOps::Prerun(Node* node)
{
    Convolution* conv_op = dynamic_cast<Convolution*>(node->GetOp());
    ConvParam* param = conv_op->GetParam();

    const Tensor* input_tensor = node->GetInputTensor(0);
    const TShape& shape = input_tensor->GetShape();
    const std::vector<int> in_dims = shape.GetDim();

    Tensor* output_tensor = node->GetOutputTensor(0);
    TShape& shape1 = output_tensor->GetShape();
    std::vector<int> out_dims = shape1.GetDim();

    int size = param->kernel_h * param->kernel_w * in_dims[1] / param->group * out_dims[2] * out_dims[3];
    float* buffer = ( float* )std::malloc(sizeof(float) * size);
    (*node)["buffer"] = buffer;
    return true;
}

bool ConvolutionDwOps::Reshape(Node* node)
{
    if(node->ExistAttr("buffer"))
    {
        float* buffer = any_cast<float*>(node->GetAttr("buffer"));
        free(buffer);
        node->RemoveAttr("buffer");
    }

    return Prerun(node);
}

void ConvolutionDwOps::pad(float* input, float* output, int in_h, int in_w, int out_h, int out_w, int top, int left, float v)
{
    float* ptr = input;
    float* outptr = output;

    int y = 0;
    // fill top
    for (; y < top; y++)
    {
        int x = 0;
        for (; x < out_w; x++)
        {
            outptr[x] = v;
        }
        outptr += out_w;
    }
    // fill center
    for (; y < (top + in_h); y++)
    {
        int x = 0;
        for (; x < left; x++)
        {
            outptr[x] = v;
        }
        if (in_w < 12)
        {
            for (; x < (left + in_w); x++)
            {
                outptr[x] = ptr[x - left];
            }
        }
        else
        {
            memcpy(outptr + left, ptr, in_w * sizeof(float));
            x += in_w;
        }
        for (; x < out_w; x++)
        {
            outptr[x] = v;
        }
        ptr += in_w;
        outptr += out_w;
    }
    // fill bottom
    for (; y < out_h; y++)
    {
        int x = 0;
        for (; x < out_w; x++)
        {
            outptr[x] = v;
        }
        outptr += out_w;
    }
}

bool ConvolutionDwOps::Run(Node* node)
{
    bool debug_conv = false;
    const char* debug_env = std::getenv("DEBUG_CONV");
    if((debug_env) && (debug_env[0] == '1'))
    {
        debug_conv = true;
    }

    Tensor* input_tensor = node->GetInputTensor(0);
    Tensor* output_tensor = node->GetOutputTensor(0);
    Tensor* weight_tensor = node->GetInputTensor(1);
    float* buffer = any_cast<float*>(node->GetAttr("buffer"));

    Convolution* conv_op = dynamic_cast<Convolution*>(node->GetOp());
    ConvParam* param = conv_op->GetParam();
    int activation = param->activation;

    float* input = ( float* )get_tensor_mem(input_tensor);
    float* output = ( float* )get_tensor_mem(output_tensor);
    float* kernel = ( float* )get_tensor_mem(weight_tensor);

    const TShape& shape = input_tensor->GetShape();
    const std::vector<int> in_dims = shape.GetDim();
    const TShape& shape1 = output_tensor->GetShape();
    const std::vector<int> out_dims = shape1.GetDim();

    int batch_number = in_dims[0];
    int inc = in_dims[1];
    int inh = in_dims[2];
    int inw = in_dims[3];
    int in_chw = inc * inh * inw;

    int outc = out_dims[1];
    int outh = out_dims[2];
    int outw = out_dims[3];
    int out_hw = outh * outw;
    int out_chw = out_hw * outc;

    int ksize_h = param->kernel_h;
    int ksize_w = param->kernel_w;
    int pad_w = param->pad_w0;
    int pad_h = param->pad_h0;

    int stride_w = param->stride_w;
    int stride_h = param->stride_h;
    int dilation_w = param->dilation_w;
    int dilation_h = param->dilation_h;
    int group = param->group;

    if(debug_conv)
    {
        std::cout << inc << " " << inh << " " << inw << "\tksp dg: " << ksize_h << " " << stride_h << " " << pad_h
                  << " " << dilation_w << " " << group << "\t" << outc << " " << outh << " " << outw << "\t";
    }

    float* biases = nullptr;
    bool have_biases = node->GetInputNum() > 2;
    if (have_biases)
        biases = (float*)get_tensor_mem(node->GetInputTensor(2));

    /* pading */
    int inh_tmp = inh + pad_h + pad_h;
    int inw_tmp = inw + pad_w + pad_w;
    float* input_tmp = NULL;
    if (inh_tmp == inh && inw_tmp == inw)
        input_tmp = input;
    else
    {
        input_tmp = (float*)malloc(inh_tmp * inw_tmp * group * sizeof(float));
        for (int g=0; g<group; g++)
        {
            float* pad_in  = input + g * inh * inw;
            float* pad_out = input_tmp + g * inh_tmp * inw_tmp;
            pad(pad_in, pad_out, inh, inw, inh_tmp, inw_tmp, pad_h, pad_w, 0.f);
        }
    }

    /* process */
    for(int i = 0; i < batch_number; i++)
    {
        if (stride_h == 1)
            dwconv3x3s1d1(inc, inw_tmp, inh_tmp, outw, outh, kernel, input_tmp, biases, output, have_biases);
        else
            dwconv3x3s2d1(inc, inw_tmp, inh_tmp, outw, outh, kernel, input_tmp, biases, output, have_biases);
    }

    /* relu */
    if (activation >= 0)
        relu(output, batch_number * out_chw, activation);

    if (!(inh_tmp == inh && inw_tmp == inw))
        free(input_tmp);

    return true;
}

bool ConvolutionDwOps::Postrun(Node* node)
{
    float* addr;
    if(node->ExistAttr("buffer"))
    {
        addr = any_cast<float*>(node->GetAttr("buffer"));
        std::free(addr);
    }
    return true;
}

NodeOps* SelectFunc(const CPUInfo* cpu_info, Node* node)
{
    Tensor* input = node->GetInputTensor(0);
    const int data_type = input->GetDataType();
    int input_c = input->GetShape().GetC();
    const ExecAttr* exec_attr = any_cast<const ExecAttr*>(node->GetAttr(ATTR_EXEC_ATTR));

    if(data_type != TENGINE_DT_FP32 || exec_attr->graph_layout != TENGINE_LAYOUT_NCHW)
        return nullptr;

    Operator* op = node->GetOp();
    Convolution* conv_op = dynamic_cast<Convolution*>(op);
    ConvParam* param = conv_op->GetParam();

    int group = param->group;
    int out_c = param->output_channel;
    int kernel_h = param->kernel_h;
    int kernel_w = param->kernel_w;
    int stride_h = param->stride_h;
    int stride_w = param->stride_w;

    /* filter perf case, just for convdw3x3s1, convdw3x3s2, auto padding */
    if (group == 1 || group != out_c || input_c != group)
        return nullptr;
    if (kernel_h != 3 || kernel_w != 3 || !(stride_h == stride_w) || (stride_h != 2 && stride_h != 1))
        return nullptr;

    ConvolutionDwOps* ops = new ConvolutionDwOps();

    return ops;
}

}    // namespace ConvolutionDwImpl

void RegisterConvDwNodeExec_x86(void)
{
    if(!NodeOpsRegistryManager::RegisterOPImplementor("x86", "Convolution", ConvolutionDwImpl::SelectFunc, 200))
        LOG_ERROR() << __FUNCTION__ << " :Regist OP failed for prio \n";
}

}    // namespace TEngine
