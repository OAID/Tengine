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
 * Copyright (c) 2018, Open AI Lab
 * Author: chunyinglv@openailab.com
 */
#include <iostream>
#include <functional>
#include <stdlib.h>

#include "logger.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "graph.hpp"
#include "operator/pooling.hpp"
#include "data_type.hpp"

namespace TEngine {

namespace PoolingRef {

struct PoolOps : public NodeOps
{
    void Generic_AvgPool(const float* input, float* output, int inc, int inh, int inw, int outh, int outw, int k_h,
                         int k_w, int stride_h, int stride_w, int pad_h, int pad_w, bool caffe_flavor)
    {
        int in_hw = inh * inw;
        int out_hw = outh * outw;
        for(int c = 0; c < inc; c++)
        {
            int c_skip = c * in_hw;
            int oc_skip = c * out_hw;

            for(int ph = 0; ph < outh; ph++)
            {
                for(int pw = 0; pw < outw; pw++)
                {
                    int h_start = ph * stride_h - pad_h;
                    int h_end = std::min(h_start + k_h, inh + pad_h);
                    int w_start = pw * stride_w - pad_w;
                    int w_end = std::min(w_start + k_w, inw + pad_w);

                    int pool_size;

                    if(caffe_flavor)
                        pool_size = (h_end - h_start) * (w_end - w_start);

                    h_start = std::max(h_start, 0);
                    w_start = std::max(w_start, 0);
                    h_end = std::min(h_end, inh);
                    w_end = std::min(w_end, inw);

                    if(!caffe_flavor)
                        pool_size = (h_end - h_start) * (w_end - w_start);

                    const int out_index = oc_skip + ph * outw + pw;
                    output[out_index] = 0.f;
                    for(int h = h_start; h < h_end; h++)
                    {
                        for(int w = w_start; w < w_end; w++)
                        {
                            output[out_index] += input[c_skip + h * inw + w];
                        }
                    }    // end ksize_h,ksize_w
                    output[out_index] /= pool_size;
                }
            }
        }
    }

    template <typename type>
    void Generic_AvgPool_nhwc(const void* input_, void* output_, int inc, int inh, int inw, int outh, int outw, int k_h,
                              int k_w, int stride_h, int stride_w, int pad_h, int pad_w)
    {
        type* input = ( type* )input_;
        type* output = ( type* )output_;
        for(int c = 0; c < inc; c++)
        {
            for(int ph = 0; ph < outh; ph++)
            {
                for(int pw = 0; pw < outw; pw++)
                {
                    int index = ph * outw * inc + pw * inc + c;

                    int h_start = ph * stride_h - pad_h;
                    int w_start = pw * stride_w - pad_w;
                    int h_end = std::min(h_start + k_h, inh);
                    int w_end = std::min(w_start + k_w, inw);
                    h_start = std::max(h_start, 0);
                    w_start = std::max(w_start, 0);
                    int pool_size = (h_end - h_start) * (w_end - w_start);

                    float tmp = 0.0f;
                    for(int h = h_start; h < h_end; h++)
                    {
                        for(int w = w_start; w < w_end; w++)
                        {
                            tmp += input[h * inc * inw + w * inc + c];
                        }
                    }    // end ksize_h,ksize_w
                    output[index] = (type)(tmp / pool_size);
                }
            }
        }
    }

    void Generic_MaxPool(const float* input, float* output, int inc, int inh, int inw, int outh, int outw, int k_h,
                         int k_w, int stride_h, int stride_w, int pad_h, int pad_w)
    {
        int in_hw = inh * inw;
        int out_hw = outh * outw;
        for(int c = 0; c < inc; c++)
        {
            int c_skip = c * in_hw;
            int oc_skip = c * out_hw;

            for(int ph = 0; ph < outh; ph++)
            {
                int h_start = ph * stride_h - pad_h;
                int h_end = std::min(h_start + k_h, inh);
                h_start = std::max(h_start, 0);

                for(int pw = 0; pw < outw; pw++)
                {
                    int w_start = pw * stride_w - pad_w;
                    int w_end = std::min(w_start + k_w, inw);
                    w_start = std::max(w_start, 0);

                    const int out_index = oc_skip + ph * outw + pw;
                    output[out_index] = input[c_skip + h_start * inw + w_start];
                    for(int h = h_start; h < h_end; h++)
                    {
                        for(int w = w_start; w < w_end; w++)
                        {
                            int in_index = c_skip + h * inw + w;

                            if(input[in_index] > output[out_index])
                            {
                                output[out_index] = input[in_index];
                            }
                        }
                    }    // end ksize_h,ksize_w
                }
            }
        }
    }

    void Global_MaxPool(float* input, float* output, int inc, int in_hw)
    {
        float* out_ptr = output;
        float* in_ptr = input;
        for(int c = 0; c < inc; c++)
        {
            float max_ = in_ptr[0];
            for(int j = 0; j < in_hw; j++)
            {
                max_ = std::max(max_, in_ptr[0]);
                in_ptr++;
            }
            *out_ptr = max_;
            out_ptr++;
        }
    }

    void Global_AvgPool(float* input, float* output, int inc, int in_hw)
    {
        float* out_ptr = output;
        float* in_ptr = input;
        for(int c = 0; c < inc; c++)
        {
            float sum = 0.f;
            for(int j = 0; j < in_hw; j++)
            {
                sum += in_ptr[0];
                in_ptr++;
            }
            *out_ptr = sum / in_hw;
            out_ptr++;
        }
    }

    template <typename type> void Global_AvgPool_nhwc(void* input_, void* output_, int inc, int in_hw)
    {
        type* input = ( type* )input_;
        type* output = ( type* )output_;

        for(int c = 0; c < inc; c++)
        {
            type* in_ptr = input;
            float sum = 0.f;
            for(int j = 0; j < in_hw; j++)
            {
                sum += in_ptr[c];
                in_ptr += inc;
            }
            *output = (type)(sum / in_hw);
            output++;
        }
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
        int input_c = ishape.GetC();
        int input_h = ishape.GetH();
        int input_w = ishape.GetW();
        int input_n = ishape.GetN();
        int elem_size = DataType::GetTypeSize(itensor->GetDataType());

        int output_h = oshape.GetH();
        int output_w = oshape.GetW();

        int in_hw = input_w * input_h;
        int in_chw = input_c * in_hw;

        int out_hw = output_h * output_w;
        int out_chw = input_c * out_hw;

        // data
        uint8_t* input_data = ( uint8_t* )get_tensor_mem(itensor);
        uint8_t* output_data = ( uint8_t* )get_tensor_mem(otensor);

        if(exec_attr->graph_layout == TENGINE_LAYOUT_NCHW)
        {
            if(param_->alg == kPoolMax)
            {
                if(param_->global)
                {
                    for(int n = 0; n < input_n; n++)
                        Global_MaxPool(( float* )input_data + n * in_chw, ( float* )output_data + n * out_chw, input_c,
                                       in_hw);
                }
                else
                {
                    for(int n = 0; n < input_n; n++)
                    {
                        Generic_MaxPool(( float* )input_data + n * in_chw, ( float* )output_data + n * out_chw, input_c,
                                        input_h, input_w, output_h, output_w, param_->kernel_h,
                                        param_->kernel_w, param_->stride_h, param_->stride_w,
                                        param_->pad_h0, param_->pad_w0);
                    }
                }
            }
            else if(param_->alg == kPoolAvg)
            {
                if(param_->global)
                {
                    for(int n = 0; n < input_n; n++)
                        Global_AvgPool(( float* )input_data + n * in_chw, ( float* )output_data + n * out_chw, input_c,
                                       in_hw);
                }
                else
                {
                    for(int n = 0; n < input_n; n++)
                    {
                        Generic_AvgPool(( float* )input_data + n * in_chw, ( float* )output_data + n * out_chw, input_c,
                                        input_h, input_w, output_h, output_w, param_->kernel_h,
                                        param_->kernel_w, param_->stride_h, param_->stride_w,
                                        param_->pad_h0, param_->pad_w0, param_->caffe_flavor);
                    }
                }
            }
            else
            {
                std::cout << " Pooling type Error\n";
                return false;
            }
        }
        else
        {
            if(param_->alg == kPoolAvg)
            {
                if(param_->global)
                {
                    for(int n = 0; n < input_n; n++)
                    {
                        if(elem_size == 4)
                            Global_AvgPool_nhwc<float>(input_data + n * in_chw * elem_size,
                                                       output_data + n * out_chw * elem_size, input_c, in_hw);
                        if(elem_size == 1)
                            Global_AvgPool_nhwc<uint8_t>(input_data + n * in_chw * elem_size,
                                                         output_data + n * out_chw * elem_size, input_c, in_hw);
                    }
                }
                else
                {
                    for(int n = 0; n < input_n; n++)
                    {
                        if(elem_size == 4)
                            Generic_AvgPool_nhwc<float>(
                                input_data + n * in_chw * 4, output_data + n * out_chw * 4, input_c, input_h, input_w,
                                output_h, output_w, param_->kernel_h, param_->kernel_w,
                                param_->stride_h, param_->stride_w, param_->pad_h0, param_->pad_w0);
                        if(elem_size == 1)
                            Generic_AvgPool_nhwc<uint8_t>(
                                input_data + n * in_chw, output_data + n * out_chw * 1, input_c, input_h, input_w,
                                output_h, output_w, param_->kernel_h, param_->kernel_w,
                                param_->stride_h, param_->stride_w, param_->pad_h0, param_->pad_w0);
                    }
                }
            }
            else
            {
                std::cout << " Pooling type Error\n";
                return false;
            }
        }

        return true;
    }
};

NodeOps* SelectFunc(const CPUInfo* cpu_info, Node* node)
{
    Tensor* input = node->GetInputTensor(0);
    const int data_type = input->GetDataType();
    const ExecAttr* exec_attr = any_cast<const ExecAttr*>(node->GetAttr(ATTR_EXEC_ATTR));
    if(data_type != TENGINE_DT_FP32 || exec_attr->graph_layout != TENGINE_LAYOUT_NCHW)
        return nullptr;

    PoolOps* ops = new PoolOps();

    return ops;
}

}    // namespace PoolingRef

using namespace PoolingRef;

void RegisterPooling_NodeExec(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor("common", "Pooling", PoolingRef::SelectFunc, 1000);
}

}    // namespace TEngine
