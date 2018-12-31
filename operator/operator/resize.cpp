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
#include "operator/resize.hpp"

namespace TEngine {
bool Resize::InferShape(const std::vector<TEngine::TShape>& ishape, std::vector<TEngine::TShape>& oshape, int layout)
{
    const TShape& input = ishape[0];
    const std::vector<int>& in_dim = input.GetDim();

    int out_h = ( int )(in_dim[2] * param_.scale_h);
    int out_w = ( int )(in_dim[3] * param_.scale_w);

    TShape shape;

    std::vector<int> dim = {in_dim[0], in_dim[1], out_h, out_w};

    shape.SetDim(dim);
    shape.SetDataLayout("NCHW");

    oshape[0] = shape;

    return true;
}

void Resize::SetSchema(void)
{
    Input({"input:float32"})
        .Output({"output:float32"})
        .SetLayout("NCHW")
        .SetAttr("scale_h", 1.f)
        .SetAttr("scale_w", 1.f)

        .SetDoc(R"DOC(Resize Layer)DOC");
}

}    // namespace TEngine
