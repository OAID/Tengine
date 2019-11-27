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
 * Author: zpluo@openailab.com
 */
#include "operator/pad.hpp"

namespace TEngine {

bool Pad::InferShape(const std::vector<TEngine::TShape>& ishape, std::vector<TEngine::TShape>& oshape, int layout)
{
    const TShape& input = ishape[0];
    // TShape& output = oshape[0];
    const std::vector<int>& in_dim = input.GetDim();
    std::vector<int> o_dim(4);
    if(param_.pad_0_h != -1 && param_.pad_0_w != -1 && param_.pad_1_h != -1 && param_.pad_1_w != -1 &&
       param_.pad_2_h != -1 && param_.pad_2_w != -1 && param_.pad_3_h != -1 && param_.pad_3_w != -1)
    {
        o_dim[0] = in_dim[0] + param_.pad_0_h + param_.pad_0_w;
        o_dim[1] = in_dim[1] + param_.pad_1_h + param_.pad_1_w;
        o_dim[2] = in_dim[2] + param_.pad_2_h + param_.pad_2_w;
        o_dim[3] = in_dim[3] + param_.pad_3_h + param_.pad_3_w;
    }
    else
    {
        return false;
    }
    TShape shape;
    shape.SetDim(o_dim);
    shape.SetDataLayout(input.GetDataLayout());
    oshape[0] = shape;
    return true;
}

void Pad::SetSchema(void)
{
    Input({"input:float32"})
        .Output({"output:float32"})
        .SetAttr("mode", 0)
        .SetAttr("pad_0_h", -1)
        .SetAttr("pad_0_w", -1)
        .SetAttr("pad_1_h", -1)
        .SetAttr("pad_1_w", -1)
        .SetAttr("pad_2_h", -1)
        .SetAttr("pad_2_w", -1)
        .SetAttr("pad_3_h", -1)
        .SetAttr("pad_3_w", -1)
        .SetAttr("value", 0)
        .SetDoc(R"DOC(Pad Layer)DOC");
}

}    // namespace TEngine
