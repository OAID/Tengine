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
 * Author: lmzhang@openailab.com
 */
#include "operator/l2pool.hpp"

namespace TEngine {

bool L2Pool::InferShape(const std::vector<TEngine::TShape>& ishape, std::vector<TEngine::TShape>& oshape,
                    int layout)
{
    const TShape& input_shape = ishape[0];

    int input_h = input_shape.GetH();
    int input_w = input_shape.GetW();
    int output_h = 0;
    int output_w = 0;

    if(param_.padding == PaddingType::kSame)
    {
        output_h = (input_h + param_.stride_h - 1) / param_.stride_h;
        output_w = (input_w + param_.stride_w - 1) / param_.stride_w;
    }
    else
    {
        output_h = (input_h + param_.stride_h - param_.kernel_h) / param_.stride_h;
        output_w = (input_w + param_.stride_w - param_.kernel_w) / param_.stride_w;
    }
    
    TShape shape;
    std::vector<int> dim = {input_shape.GetN(), output_h, output_w, input_shape.GetC()};
    shape.SetDim(dim);
    shape.SetDataLayout(TENGINE_LAYOUT_NHWC);

    oshape[0] = shape;
    return true;
}

float L2Pool::GetFops(const std::vector<TShape>& inputs, const std::vector<TShape>& outputs)
{
    float patch_fops = param_.kernel_h * param_.kernel_w;

    return (patch_fops * outputs[0].GetSize());
}

void L2Pool::SetSchema(void)
{
    Input({"input:float32"})
        .Output({"output:float32"})
        .SetAttr("padding", PaddingType::kNone)
        .SetAttr("kernel_h", 0)
        .SetAttr("kernel_w", 0)
        .SetAttr("stride_h", 0)
        .SetAttr("stride_w", 0)
        .SetDoc(R"DOC(tflite L2Pooling Layer)DOC");


}

}
