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
 * Parts of the following code in this file refs to
 * https://github.com/Tencent/ncnn/blob/master/src/layer/x86/convolution_3x3.h
 * BUG1989 is pleased to support the open source community by supporting ncnn available.
 *
 * Copyright (C) 2019 BUG1989. All rights reserved.
 *
 * Licensed under the BSD 3-Clause License (the "License"); you may not use this
 * file except in compliance with the License. You may obtain a copy of the License at
 *
 * https://opensource.org/licenses/BSD-3-Clause
 */

/*
 * Copyright (c) 2020, OPEN AI LAB
 * Author: qwang02@openailab.com
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
#include "convolution_wino_x86.h"
#include <math.h>

namespace TEngine {

namespace ConvolutionWinoImpl {

struct ConvolutionWionOps : public NodeOps
{
    int cpu_type;
    int cpu_number;
    int activation;
    bool Reshape(Node* node);
    bool Prerun(Node* node) override;
    bool Run(Node* node) override;
    bool Postrun(Node* node) override;
    void relu(float* data, int size, int activation);
    void add_bias(float* output, float* bias, int c_out, int hw);
    void conv3x3s1_winograd43_transform_kernel_sse(const float* kernel, float* kernel_wino, int inch, int outch);
};

// prerun
bool ConvolutionWionOps::Prerun(Node* node)
{
    Convolution* conv_op = dynamic_cast<Convolution*>(node->GetOp());
    ConvParam* param = conv_op->GetParam();
    int pad_h = param->pad_h0;
    int pad_w = param->pad_w0;

    const Tensor* input_tensor = node->GetInputTensor(0);
    const TShape& input_shape = input_tensor->GetShape();
    const std::vector<int> in_dims = input_shape.GetDim();
    int input_c = input_shape.GetC();

    const Tensor* kernel_tensor = node->GetInputTensor(1);
    const TShape& kernel_shape = kernel_tensor->GetShape();
    const std::vector<int> kernel_dims = kernel_shape.GetDim();

    Tensor* output_tensor = node->GetOutputTensor(0);
    TShape& output_shape = output_tensor->GetShape();
    std::vector<int> out_dims = output_shape.GetDim();
    int output_c = output_shape.GetC();
    int output_w = output_shape.GetW();
    int output_h = output_shape.GetH();

    int trans_ker_size = output_c * input_c * 36 * sizeof(float);
    float* kernel_org = ( float* )get_tensor_mem(kernel_tensor);
    float* kernel_wino = ( float* )mem_alloc(trans_ker_size);

    int TILE = 4;
    int block_h = (output_h + TILE - 1) / TILE;
    int block_w = (output_w + TILE - 1) / TILE;
    int block = block_h * block_w;

    int padded_inh = TILE * block_h + 2 * pad_h;
    int padded_inw = TILE * block_w + 2 * pad_w;
    int pad_inhw = padded_inh * padded_inw;
    float* input_pad = ( float* )mem_alloc(input_c * pad_inhw * sizeof(float));
    memset(input_pad, 0, input_c * pad_inhw * sizeof(float));
    float* dot_block = ( float* )mem_alloc(36 * block * output_c * sizeof(float));
    float* transform_input = ( float* )mem_alloc(36 * block * input_c * sizeof(float));
    float* output_bordered = nullptr;
    int outw = block_w * TILE;
    int outh = block_h * TILE;
    if(outw != output_w || outh != output_h)
    {
        output_bordered = ( float* )mem_alloc(outw * outh * output_c * sizeof(float));
    }

    // transform 3x3 kernel to winograd 6x6 kernel
    conv3x3s1_winograd43_transform_kernel_sse(kernel_org, kernel_wino, input_c, output_c);

    (*node)["kernel_wino"] = kernel_wino;
    (*node)["input_pad"] = input_pad;
    (*node)["dot_block"] = dot_block;
    (*node)["transform_input"] = transform_input;
    (*node)["output_bordered"] = output_bordered;

    return true;
}

bool ConvolutionWionOps::Reshape(Node* node)
{
    if(node->ExistAttr("kernel_wino"))
    {
        float* buffer = any_cast<float*>(node->GetAttr("kernel_wino"));
        free(buffer);
        node->RemoveAttr("kernel_wino");
    }
    if(node->ExistAttr("input_pad"))
    {
        float* buffer = any_cast<float*>(node->GetAttr("input_pad"));
        free(buffer);
        node->RemoveAttr("input_pad");
    }
    if(node->ExistAttr("dot_block"))
    {
        float* buffer = any_cast<float*>(node->GetAttr("dot_block"));
        free(buffer);
        node->RemoveAttr("dot_block");
    }
    if(node->ExistAttr("transform_input"))
    {
        float* buffer = any_cast<float*>(node->GetAttr("transform_input"));
        free(buffer);
        node->RemoveAttr("transform_input");
    }
    if(node->ExistAttr("output_bordered"))
    {
        float* buffer = any_cast<float*>(node->GetAttr("output_bordered"));
        free(buffer);
        node->RemoveAttr("output_bordered");
    }

    return Prerun(node);
}

// run
bool ConvolutionWionOps::Run(Node* node)
{
    int TILE = 4;
    bool debug_conv = false;
    const char* debug_env = std::getenv("DEBUG_CONV");
    if((debug_env) && (debug_env[0] == '1'))
    {
        debug_conv = true;
    }

    Convolution* conv_op = dynamic_cast<Convolution*>(node->GetOp());
    ConvParam* param = conv_op->GetParam();
    int activation = param->activation;
    int pad_h = param->pad_h0;
    int pad_w = param->pad_w0;

    /* input */
    Tensor* input_tensor = node->GetInputTensor(0);
    float* input = ( float* )get_tensor_mem(input_tensor);
    const TShape& input_shape = input_tensor->GetShape();
    int input_c = input_shape.GetC();
    int input_h = input_shape.GetH();
    int input_w = input_shape.GetW();
    int batch_number = input_shape.GetN();
    int inp_chw = input_c * input_h * input_w;
    /* output */
    Tensor* output_tensor = node->GetOutputTensor(0);
    float* output = ( float* )get_tensor_mem(output_tensor);
    const TShape& output_shape = output_tensor->GetShape();
    int output_h = output_shape.GetH();
    int output_w = output_shape.GetW();
    int output_c = output_shape.GetC();
    int out_hw = output_h * output_w;
    int out_chw = out_hw * output_c;
    int output_n = output_shape.GetN();
    /* weight */
    float* kernel_wino = any_cast<float*>(node->GetAttr("kernel_wino"));
    Tensor* weight_tensor = node->GetInputTensor(1);
    float* kernel = ( float* )get_tensor_mem(weight_tensor);

    float* dot_block = any_cast<float*>(node->GetAttr("dot_block"));
    float* transform_input = any_cast<float*>(node->GetAttr("transform_input"));
    float* output_bordered = any_cast<float*>(node->GetAttr("output_bordered"));

    int ksize_h = param->kernel_h;
    int ksize_w = param->kernel_w;
    int stride_w = param->stride_w;
    int stride_h = param->stride_h;
    int dilation_w = param->dilation_w;
    int dilation_h = param->dilation_h;
    int group = param->group;

    int input_c_g = input_c / group;
    int output_c_g = output_c / group;
    int in_chw = input_c * input_h * input_w;
    int in_chw_g = input_h * input_w * input_c_g;
    int out_chw_g = output_c_g * out_hw;
    int kernel_size_g = output_c_g * ksize_h * ksize_w * input_c_g;

    bool have_biases = (node->GetInputNum() > 2);
    float* biases = nullptr;
    if(have_biases)
    {
        biases = ( float* )get_tensor_mem(node->GetInputTensor(2));
    }

    int block_h = (output_h + TILE - 1) / TILE;
    int block_w = (output_w + TILE - 1) / TILE;
    int block_hw = block_h * block_w;
    int padded_inh = TILE * block_h + 2 * pad_h;
    int padded_inw = TILE * block_w + 2 * pad_w;

    float* input_pad = any_cast<float*>(node->GetAttr("input_pad"));
    pad_0_align_3D(input_pad, input, input_h, input_w, padded_inh, padded_inw, input_c, pad_h, pad_w);

    if(debug_conv)
    {
        std::cout << input_c << " " << input_h << " " << input_w << "\tksp dg: " << ksize_h << " " << stride_h << " "
                  << pad_h << " " << dilation_w << " " << group << "\t" << output_c << " " << output_h << " "
                  << output_w << "\t";
    }
    for(int i = 0; i < batch_number; i++)
    {
        for(int g = 0; g < group; g++)
        {
            conv3x3s1_winograd43_sse(input_pad + i * in_chw + g * in_chw_g, output, kernel_wino, dot_block,
                                     transform_input, output_bordered, biases, padded_inw, padded_inh, input_c,
                                     output_w, output_h, output_c);
        }
    }

    if(activation >= 0)
    {
        relu(output, batch_number * out_chw, activation);
    }
    if(debug_conv)
    {
        std::cout << output[0] << " " << output[10] << "\n";
    }

    return true;
}

// postrun
bool ConvolutionWionOps::Postrun(Node* node)
{
    float* addr;
    if(node->ExistAttr("kernel_wino"))
    {
        addr = any_cast<float*>(node->GetAttr("kernel_wino"));
        mem_free(addr);
        node->RemoveAttr("kernel_wino");
    }
    if(node->ExistAttr("input_pad"))
    {
        addr = any_cast<float*>(node->GetAttr("input_pad"));
        mem_free(addr);
        node->RemoveAttr("input_pad");
    }
    if(node->ExistAttr("dot_block"))
    {
        addr = any_cast<float*>(node->GetAttr("dot_block"));
        mem_free(addr);
        node->RemoveAttr("dot_block");
    }
    if(node->ExistAttr("transform_input"))
    {
        addr = any_cast<float*>(node->GetAttr("transform_input"));
        mem_free(addr);
        node->RemoveAttr("transform_input");
    }
    if(node->ExistAttr("output_bordered"))
    {
        addr = any_cast<float*>(node->GetAttr("output_bordered"));
        if(addr != nullptr)
            mem_free(addr);
        node->RemoveAttr("output_bordered");
    }
    return true;
}

void ConvolutionWionOps::relu(float* data, int size, int activation)
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
void ConvolutionWionOps::add_bias(float* output, float* bias, int c_out, int hw)
{
    for(int c = 0; c < c_out; ++c)
    {
        for(int i = 0; i < hw; ++i)
        {
            output[c * hw + i] += bias[c];
        }
    }
}

void ConvolutionWionOps::conv3x3s1_winograd43_transform_kernel_sse(const float* kernel, float* kernel_wino, int inch,
                                                                   int outch)
{
    float* kernel_tm = ( float* )malloc(6 * 6 * inch * outch * sizeof(float));

    // G
    const float ktm[6][3] = {
        {1.0f / 4, 0.0f, 0.0f},           {-1.0f / 6, -1.0f / 6, -1.0f / 6}, {-1.0f / 6, 1.0f / 6, -1.0f / 6},
        {1.0f / 24, 1.0f / 12, 1.0f / 6}, {1.0f / 24, -1.0f / 12, 1.0f / 6}, {0.0f, 0.0f, 1.0f}};

#pragma omp parallel for
    for(int p = 0; p < outch; p++)
    {
        for(int q = 0; q < inch; q++)
        {
            const float* kernel0 = kernel + p * inch * 9 + q * 9;
            float* kernel_tm0 = kernel_tm + p * inch * 36 + q * 36;

            // transform kernel
            const float* k0 = kernel0;
            const float* k1 = kernel0 + 3;
            const float* k2 = kernel0 + 6;

            // h
            float tmp[6][3] = {0};
            for(int i = 0; i < 6; i++)
            {
                tmp[i][0] = k0[0] * ktm[i][0] + k0[1] * ktm[i][1] + k0[2] * ktm[i][2];
                tmp[i][1] = k1[0] * ktm[i][0] + k1[1] * ktm[i][1] + k1[2] * ktm[i][2];
                tmp[i][2] = k2[0] * ktm[i][0] + k2[1] * ktm[i][1] + k2[2] * ktm[i][2];
            }

            // U
            for(int j = 0; j < 6; j++)
            {
                float* tmpp = &tmp[j][0];

                for(int i = 0; i < 6; i++)
                {
                    kernel_tm0[j * 6 + i] = tmpp[0] * ktm[i][0] + tmpp[1] * ktm[i][1] + tmpp[2] * ktm[i][2];
                }
            }
        }
    }

    float* kernel_tm_test = kernel_wino;
    for(int r = 0; r < 9; r++)
    {
        int p = 0;
        for(; p + 7 < outch; p += 8)
        {
            const float* kernel0 = ( const float* )kernel_tm + p * inch * 36;
            const float* kernel1 = ( const float* )kernel_tm + (p + 1) * inch * 36;
            const float* kernel2 = ( const float* )kernel_tm + (p + 2) * inch * 36;
            const float* kernel3 = ( const float* )kernel_tm + (p + 3) * inch * 36;
            const float* kernel4 = ( const float* )kernel_tm + (p + 4) * inch * 36;
            const float* kernel5 = ( const float* )kernel_tm + (p + 5) * inch * 36;
            const float* kernel6 = ( const float* )kernel_tm + (p + 6) * inch * 36;
            const float* kernel7 = ( const float* )kernel_tm + (p + 7) * inch * 36;

            float* ktmp = kernel_tm_test + p / 8 * inch * 32;

            for(int q = 0; q < inch; q++)
            {
                ktmp[0] = kernel0[r * 4 + 0];
                ktmp[1] = kernel0[r * 4 + 1];
                ktmp[2] = kernel0[r * 4 + 2];
                ktmp[3] = kernel0[r * 4 + 3];

                ktmp[4] = kernel1[r * 4 + 0];
                ktmp[5] = kernel1[r * 4 + 1];
                ktmp[6] = kernel1[r * 4 + 2];
                ktmp[7] = kernel1[r * 4 + 3];

                ktmp[8] = kernel2[r * 4 + 0];
                ktmp[9] = kernel2[r * 4 + 1];
                ktmp[10] = kernel2[r * 4 + 2];
                ktmp[11] = kernel2[r * 4 + 3];

                ktmp[12] = kernel3[r * 4 + 0];
                ktmp[13] = kernel3[r * 4 + 1];
                ktmp[14] = kernel3[r * 4 + 2];
                ktmp[15] = kernel3[r * 4 + 3];

                ktmp[16] = kernel4[r * 4 + 0];
                ktmp[17] = kernel4[r * 4 + 1];
                ktmp[18] = kernel4[r * 4 + 2];
                ktmp[19] = kernel4[r * 4 + 3];

                ktmp[20] = kernel5[r * 4 + 0];
                ktmp[21] = kernel5[r * 4 + 1];
                ktmp[22] = kernel5[r * 4 + 2];
                ktmp[23] = kernel5[r * 4 + 3];

                ktmp[24] = kernel6[r * 4 + 0];
                ktmp[25] = kernel6[r * 4 + 1];
                ktmp[26] = kernel6[r * 4 + 2];
                ktmp[27] = kernel6[r * 4 + 3];

                ktmp[28] = kernel7[r * 4 + 0];
                ktmp[29] = kernel7[r * 4 + 1];
                ktmp[30] = kernel7[r * 4 + 2];
                ktmp[31] = kernel7[r * 4 + 3];

                ktmp += 32;
                kernel0 += 36;
                kernel1 += 36;
                kernel2 += 36;
                kernel3 += 36;
                kernel4 += 36;
                kernel5 += 36;
                kernel6 += 36;
                kernel7 += 36;
            }
        }

        for(; p + 3 < outch; p += 4)
        {
            const float* kernel0 = ( const float* )kernel_tm + p * inch * 36;
            const float* kernel1 = ( const float* )kernel_tm + (p + 1) * inch * 36;
            const float* kernel2 = ( const float* )kernel_tm + (p + 2) * inch * 36;
            const float* kernel3 = ( const float* )kernel_tm + (p + 3) * inch * 36;

            float* ktmp = kernel_tm_test + (p / 8 + (p % 8) / 4) * inch * 32;
            for(int q = 0; q < inch; q++)
            {
                ktmp[0] = kernel0[r * 4 + 0];
                ktmp[1] = kernel0[r * 4 + 1];
                ktmp[2] = kernel0[r * 4 + 2];
                ktmp[3] = kernel0[r * 4 + 3];

                ktmp[4] = kernel1[r * 4 + 0];
                ktmp[5] = kernel1[r * 4 + 1];
                ktmp[6] = kernel1[r * 4 + 2];
                ktmp[7] = kernel1[r * 4 + 3];

                ktmp[8] = kernel2[r * 4 + 0];
                ktmp[9] = kernel2[r * 4 + 1];
                ktmp[10] = kernel2[r * 4 + 2];
                ktmp[11] = kernel2[r * 4 + 3];

                ktmp[12] = kernel3[r * 4 + 0];
                ktmp[13] = kernel3[r * 4 + 1];
                ktmp[14] = kernel3[r * 4 + 2];
                ktmp[15] = kernel3[r * 4 + 3];

                ktmp += 16;
                kernel0 += 36;
                kernel1 += 36;
                kernel2 += 36;
                kernel3 += 36;
            }
        }

        for(; p < outch; p++)
        {
            const float* kernel0 = ( const float* )kernel_tm + p * inch * 36;
            float* ktmp = kernel_tm_test + (p / 8 + (p % 8) / 4 + p % 4) * inch * 32;

            for(int q = 0; q < inch; q++)
            {
                ktmp[0] = kernel0[r * 4 + 0];
                ktmp[1] = kernel0[r * 4 + 1];
                ktmp[2] = kernel0[r * 4 + 2];
                ktmp[3] = kernel0[r * 4 + 3];

                ktmp += 4;
                kernel0 += 36;
            }
        }
        kernel_tm_test += 4 * inch * outch;
    }
    free(kernel_tm);
}

static bool isWinogradSupported(const ConvParam* param, const TShape& input_shape, const TShape& output_shape,
                                int cpu_number)
{
    int input_c = input_shape.GetC();
    int input_h = input_shape.GetH();
    int input_w = input_shape.GetW();
    int output_c = output_shape.GetC();
    int group = param->group;
    int kernel_h = param->kernel_h;
    int kernel_w = param->kernel_w;
    int stride_h = param->stride_h;
    int stride_w = param->stride_w;
    int dilation_h = param->dilation_h;
    int dilation_w = param->dilation_w;

    if((input_h <= 10) && (input_w <= 10))
        return false;

    if(group != 1 || kernel_h != 3 || kernel_w != 3 || stride_h != 1 || stride_w != 1 || dilation_h != 1 ||
       dilation_w != 1 || input_c < 16 || output_c < 16)
        return false;

    return true;
}

NodeOps* SelectFunc(const CPUInfo* cpu_info, Node* node)
{
    const char* wino_config = std::getenv("NO_WINO");
    if(wino_config)
        return nullptr;
    Tensor* input = node->GetInputTensor(0);
    const int data_type = input->GetDataType();
    const ExecAttr* exec_attr = any_cast<const ExecAttr*>(node->GetAttr(ATTR_EXEC_ATTR));
    if(data_type != TENGINE_DT_FP32 || exec_attr->graph_layout != TENGINE_LAYOUT_NCHW)
        return nullptr;

    Operator* op = node->GetOp();
    Convolution* conv_op = dynamic_cast<Convolution*>(op);
    ConvParam* param = conv_op->GetParam();
    const TShape& output_shape = node->GetOutputTensor(0)->GetShape();
    const TShape& input_shape = node->GetInputTensor(0)->GetShape();
    int cpu_number = cpu_info->GetCPUNumber();
    if(!isWinogradSupported(param, input_shape, output_shape, cpu_number))
        return nullptr;

    ConvolutionWionOps* ops = new ConvolutionWionOps();
    ops->activation = param->activation;
    ops->cpu_number = cpu_number;
    return ops;
}
}    // namespace ConvolutionWinoImpl

void RegisterConvWinoNodeExec_x86(void)
{
    if(!NodeOpsRegistryManager::RegisterOPImplementor("x86", "Convolution", ConvolutionWinoImpl::SelectFunc, 50))
        LOG_ERROR() << __FUNCTION__ << " :Regist OP failed for prio \n";
}

}    // namespace TEngine
