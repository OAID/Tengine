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
 *         chunyinglv@openailab.com
 */
#include "operator/pooling.hpp"
#include "static_graph.hpp"
#include <cmath>

namespace TEngine {

static int calc_output_size(int input, int kernel, int stride, int pad, int caffe)
{
    int output = 1;
    if(pad >= 0)
    {
        if(1 == caffe)
        {
            output = 1 + std::ceil((( float )(input - kernel + 2 * pad)) / stride);
            if(pad > 0 && ((output - 1) * stride >= input + pad))
                output--;
        }
        else if(2 == caffe)
        {
            output = 1 +  (input - kernel + pad) / stride;
         }
        else
            output = 1 + (input - kernel + 2 * pad) / stride;
    }
    else
    {
        output = 1 + (input - 1) / stride;
    }
    return output;
}

static void calc_real_pads(int out, int in, int kernel, int stride, int pad, int* pad0, int* pad1)
{
    int total = (out - 1) * stride + kernel;
    int pad_num = total - in;

    if(pad_num < 0)
        pad_num = 0;

    /* for same */
    if(pad < 0)
    {
        *pad0 = pad_num / 2;
        *pad1 = pad_num - *pad0;
    }
    else
    {
        *pad0 = pad;
        *pad1 = pad_num - *pad0;
    }
}

bool Pooling::InferShape(const std::vector<TEngine::TShape>& ishape, std::vector<TEngine::TShape>& oshape, int layout)
{
    const TShape& input_shape = ishape[0];
    int input_h = input_shape.GetH();
    int input_w = input_shape.GetW();

    if(param_.kernel_h == input_h && param_.kernel_w == input_w && param_.pad_w0 <= 0)
        param_.global = 1;

    int output_h;
    int output_w;

    if(param_.global)
    {
        param_.stride_h = 1;
        param_.stride_w = 1;
        param_.kernel_h = input_h;
        param_.kernel_w = input_w;
        param_.pad_h0 = param_.pad_w0 = param_.pad_h1 = param_.pad_w1 = 0;
        output_h = 1;
        output_w = 1;
    }
    else
    {
        int caffe = param_.caffe_flavor & ~(COUNT_INCLUDE_PAD_MSK);
        output_h = calc_output_size(input_h, param_.kernel_h, param_.stride_h, param_.pad_h0, caffe);
        output_w = calc_output_size(input_w, param_.kernel_w, param_.stride_w, param_.pad_w0, caffe);
        if(2 != caffe)
        {
            calc_real_pads(output_h, input_h, param_.kernel_h, param_.stride_h, param_.pad_h0, &param_.pad_h0,
                        &param_.pad_h1);
            calc_real_pads(output_w, input_w, param_.kernel_w, param_.stride_w, param_.pad_w0, &param_.pad_w0,
                       &param_.pad_w1);
        }
        else
        {
            int pad_w0 = param_.pad_w0;
            int pad_h0 = param_.pad_h0;
            param_.pad_w0 = pad_w0/2;
            param_.pad_h0 = pad_h0/2;
            param_.pad_w1 = pad_w0 - pad_w0/2;
            param_.pad_h1 = pad_h0 - pad_h0/2;
        }
    }

    TShape shape;
    if(layout == TENGINE_LAYOUT_NCHW)
    {
        std::vector<int> dim = {input_shape.GetN(), input_shape.GetC(), output_h, output_w};

        shape.SetDim(dim);
        shape.SetDataLayout(TENGINE_LAYOUT_NCHW);
    }
    else
    {
        std::vector<int> dim = {input_shape.GetN(), output_h, output_w, input_shape.GetC()};

        shape.SetDim(dim);
        shape.SetDataLayout(TENGINE_LAYOUT_NHWC);
    }
    oshape[0] = shape;
    return true;
}

float Pooling::GetFops(const std::vector<TShape>& inputs, const std::vector<TShape>& outputs)
{
    float patch_fops = param_.kernel_h * param_.kernel_w;

    return (patch_fops * outputs[0].GetSize());
}

void Pooling::SetSchema(void)
{
    Input({"input:float32"})
        .Output({"output:float32"})
        .SetAttr("alg", 0)
        .SetAttr("kernel_h", 2)
        .SetAttr("kernel_w", 2)
        .SetAttr("stride_h", 1)
        .SetAttr("stride_w", 1)
        .SetAttr("global", 0)
        .SetAttr("caffe_flavor", 0)
        .SetAttr("pad_h0", 0)
        .SetAttr("pad_w0", 0)
        .SetAttr("pad_h1", 0)
        .SetAttr("pad_w1", 0)
        .SetDoc(R"DOC(Pooling Layer)DOC");
}

}    // namespace TEngine
