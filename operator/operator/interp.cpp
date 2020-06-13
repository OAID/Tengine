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
 * Author: ddzhao@openailab.com
 */
#include "operator/interp.hpp"

namespace TEngine {

bool Interp::InferShape(const std::vector<TEngine::TShape>& ishape, std::vector<TEngine::TShape>& oshape, int layout)
{

    const TShape& input = ishape[0];
    int in_n = input.GetN();
    int in_c = input.GetC();
    int in_h = input.GetH();
    int in_w = input.GetW();
    if(param_.width_scale != 0.0 && param_.height_scale != 0.0)
    {
        param_.output_height = in_h * param_.height_scale;
        param_.output_width = in_w * param_.width_scale;
    }
    else{
        param_.height_scale = static_cast<float>(param_.output_height) / static_cast<float>(in_h);
        param_.width_scale = static_cast<float>(param_.output_width) / static_cast<float>(in_w);
    }
    TShape out_shape;

    std::vector<int> dim(4);

    dim[0] = in_n;
    dim[1] = in_c;
    dim[2] = param_.output_height;
    dim[3] = param_.output_width;
    out_shape.SetDim(dim);
    out_shape.SetDataLayout(input.GetDataLayout());

    oshape[0] = out_shape;
    
    return true;
}

void Interp::SetSchema(void)
{
    Input({"input:float32"})
    .Output({"output:float32"})
    .SetAttr("resize_type", 1)
    .SetAttr("width_scale", 1.0)
    .SetAttr("height_scale", 1.0)
    .SetAttr("output_width", 1)
    .SetAttr("output_height", 1)
    .SetDoc(R"DOC(Interp Operator)DOC");
}

}    // namespace TEngine
