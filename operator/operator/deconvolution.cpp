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

    int input_n = input_shape.GetN();
    int input_h = input_shape.GetH();
    int input_w = input_shape.GetW();

    int kernel_extent = param_.dilation * (param_.kernel_size - 1) + 1;

    int output_h = (input_h - 1) * param_.stride + kernel_extent - 2 * param_.pad;
    int output_w = (input_w - 1) * param_.stride + kernel_extent - 2 * param_.pad;

    std::vector<int> dim = {input_n, param_.num_output, output_h, output_w};
    TShape result;

    result.SetDim(dim);
    result.SetDataLayout("NCHW");

    oshape[0] = result;

    return true;
}

float Deconvolution::GetFops(const std::vector<TShape>& inputs, const std::vector<TShape>& outputs)
{
    float ops = 1.0f * param_.num_output * param_.kernel_size * param_.kernel_size * inputs[0].GetSize() * 2;

    return ops;
}

void Deconvolution::SetSchema(void)
{
    Input({"input:float32", "weight:float32", "bias:float32"})
        .Output({"output:float32"})
        .SetLayout("NCHW")
        .SetAttr("kernel_size", 1)
        .SetAttr("stride", 1)
        .SetAttr("pad", 1)
        .SetAttr("num_output", 1)
        .SetAttr("dilation", 1)

        .SetDoc(R"DOC(Deconvolution Layer)DOC");
}

}    // namespace TEngine
