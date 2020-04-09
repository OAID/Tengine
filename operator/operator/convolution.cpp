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
#include "operator/convolution.hpp"
#include "static_graph.hpp"

namespace TEngine {

/*

The TensorFlow Convolution example gives an overview about the difference between SAME and VALID :

For the SAME padding, the output height and width are computed as:

out_height = ceil(float(in_height) / float(strides[1]))

out_width = ceil(float(in_width) / float(strides[2]))

And

For the VALID padding, the output height and width are computed as:

out_height = ceil(float(in_height - filter_height + 1) / float(strides1))

out_width = ceil(float(in_width - filter_width + 1) / float(strides[2]))

*/

bool Convolution::InferShape(const std::vector<TShape>& ishape, std::vector<TShape>& oshape, int layout)
{
    if(ishape.size() < 2){
        return false;
    }

    const TShape& input_shape = ishape[0];
    const TShape& weight_shape = ishape[1];

    if(input_shape.GetDim().size() != 4 || weight_shape.GetDim().size() != 4){
        return false;
    }

    int input_n = input_shape.GetN();
    int input_c = input_shape.GetC();
    int input_h = input_shape.GetH();
    int input_w = input_shape.GetW();

    int output_c = weight_shape.GetN();
    int weight_c = weight_shape.GetC();
    int weight_h = weight_shape.GetH();
    int weight_w = weight_shape.GetW();

    if((input_c != weight_c * param_.group) || (output_c != param_.output_channel) || (param_.kernel_h != weight_h) ||
       (param_.kernel_w != weight_w))
    {
        return false;
    }

    param_.input_channel = input_c;

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

    int dilation_h = param_.dilation_h;
    int dilation_w = param_.dilation_w;

    int output_h =
        (input_h - dilation_h * (param_.kernel_h - 1) - 1 + param_.pad_h0 + param_.pad_h1) / param_.stride_h + 1;
    int output_w =
        (input_w - dilation_w * (param_.kernel_w - 1) - 1 + param_.pad_w0 + param_.pad_w1) / param_.stride_w + 1;

    TShape result;

    if(layout == TENGINE_LAYOUT_NHWC)
    {
        std::vector<int> dim = {input_n, output_h, output_w, output_c};
        result.SetDim(dim);
        result.SetDataLayout(TENGINE_LAYOUT_NHWC);
    }
    else
    {
        std::vector<int> dim = {input_n, output_c, output_h, output_w};
        result.SetDim(dim);
        result.SetDataLayout(TENGINE_LAYOUT_NCHW);
    }

    oshape[0] = result;

    return true;
}

float Convolution::GetFops(const std::vector<TShape>& inputs, const std::vector<TShape>& outputs)
{
    const std::vector<int>& input_dims = inputs[0].GetDim();

    int layout = inputs[0].GetDataLayout();
    int per_input_c;

    if(layout == TENGINE_LAYOUT_NCHW)
        per_input_c = input_dims[1] / param_.group;
    else
        per_input_c = input_dims[3] / param_.group;

    float ops = 1.0f * per_input_c * param_.kernel_h * param_.kernel_w * outputs[0].GetSize() * 2;

    if(ops < 0)
    {
        std::cout << "input_c: " << per_input_c << " kernel_h: " << param_.kernel_h << " kernel_w: " << param_.kernel_w;
        std::cout << "output: " << outputs[0].GetSize() << "\n";
    }

    return ops;
}

void Convolution::SetSchema(void)
{
    Input({"input:float32", "weight:float32", "bias:float32"})
        .Output({"output:float32"})
        .SetAttr("kernel_h", 1)
        .SetAttr("kernel_w", 1)
        .SetAttr("stride_h", 1)
        .SetAttr("stride_w", 1)
        .SetAttr("dilation_h", 1)
        .SetAttr("dilation_w", 1)
        .SetAttr("input_channel", 1)
        .SetAttr("output_channel", 1)
        .SetAttr("group", 1)
        .SetAttr("activation", -1)
        .SetAttr("pad_h0", 0)
        .SetAttr("pad_w0", 0)
        .SetAttr("pad_h1", 0)
        .SetAttr("pad_w1", 0)
        .SetDoc(R"DOC(Convolution Layer)DOC");
}

}    // namespace TEngine
