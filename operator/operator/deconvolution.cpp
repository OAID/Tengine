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
#include "operator/deconvolution.hpp"
#include "static_graph.hpp"

namespace TEngine {

bool Deconvolution::InferShape(const std::vector<TShape>& ishape, std::vector<TShape>& oshape, int layout)

{
    const TShape& input_shape = ishape[0];
    //const TShape& weight_shape = ishape[1];
    int input_n = input_shape.GetN();
    int input_h = input_shape.GetH();
    int input_w = input_shape.GetW();

    //int output_c = weight_shape.Shape(1);
    if(param_.pad_h0 < 0)
    {
        int n = (input_h - 1) / param_.stride_h + 1;
        int total_len = (n - 1) * param_.stride_h + param_.kernel_h;
        int pad_num = total_len - input_h;

        if(param_.pad_h0 == -1)    // TF or SAME_UPPER in ONNX
        {
            param_.pad_h0 = pad_num / 2;
            param_.pad_h1 = pad_num - pad_num / 2;
        }
        else
        {
            // SAME_LOWER in ONNX
            param_.pad_h0 = pad_num - pad_num / 2;
            param_.pad_h1 = pad_num / 2;
        }
    }

    if(param_.pad_w0 < 0)
    {
        int n = (input_w - 1) / param_.stride_w + 1;
        int total_len = (n - 1) * param_.stride_w + param_.kernel_w;
        int pad_num = total_len - input_w;

        if(param_.pad_w0 == -1)    // TF or SAME_UPPER in ONNX
        {
            param_.pad_w0 = pad_num / 2;
            param_.pad_w1 = pad_num - pad_num / 2;
        }
        else
        {
            // SAME_LOWER in ONNX
            param_.pad_w0 = pad_num - pad_num / 2;
            param_.pad_w1 = pad_num / 2;
        }
    }
    int kernel_extent_w = param_.dilation_w * (param_.kernel_w - 1) + 1;
    int kernel_extent_h = param_.dilation_h * (param_.kernel_h - 1) + 1;

    int output_h = (input_h - 1) * param_.stride_h + kernel_extent_h - param_.pad_h0 - param_.pad_h1;
    int output_w = (input_w - 1) * param_.stride_w + kernel_extent_w - param_.pad_w0 - param_.pad_w1;

    // std::vector<int> dim = {input_n, param_.num_output, output_h, output_w};

    TShape result;

    if(layout == TENGINE_LAYOUT_NHWC)
    {
        std::vector<int> dim = {input_n, output_h, output_w, param_.num_output};
        result.SetDim(dim);
        result.SetDataLayout(TENGINE_LAYOUT_NHWC);
    }
    else
    {
        std::vector<int> dim = {input_n, param_.num_output, output_h, output_w};
        result.SetDim(dim);
        result.SetDataLayout(TENGINE_LAYOUT_NCHW);
    }

    oshape[0] = result;
    return true;
}

float Deconvolution::GetFops(const std::vector<TShape>& inputs, const std::vector<TShape>& outputs)
{
    float ops = 1.0f * param_.num_output * param_.kernel_h * param_.kernel_w * inputs[0].GetSize() * 2;

    return ops;
}

void Deconvolution::SetSchema(void)
{
    Input({"input:float32", "weight:float32", "bias:float32"})
        .Output({"output:float32"})
        .SetAttr("kernel_h", 1)
        .SetAttr("kernel_w", 1)
        .SetAttr("stride_h", 1)
        .SetAttr("stride_w", 1)
        .SetAttr("pad_h0", 0)
        .SetAttr("pad_w0", 0)
        .SetAttr("pad_h1", 0)
        .SetAttr("pad_w1", 0)
        .SetAttr("dilation_h", 1)
        .SetAttr("dilation_w", 1)
        .SetAttr("num_output", 1)
        .SetAttr("group", 1)
        .SetAttr("activation", -1)

        .SetDoc(R"DOC(Deconvolution Layer)DOC");
}

}    // namespace TEngine
