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
#include "operator/permute.hpp"

namespace TEngine {

bool Permute::InferShape(const std::vector<TEngine::TShape>& ishape, std::vector<TEngine::TShape>& oshape, int layout)
{
    const TShape& input = ishape[0];
    const std::vector<int> dims = input.GetDim();

    // only support for 0231[bhwc]
    if((param_.order0 == 0) && (param_.order1 == 2) && (param_.order2 == 3) && (param_.order3 == 1))
    {
        int n = input.GetN();
        int c = input.GetC();
        int h = input.GetH();
        int w = input.GetW();
        TShape shape;
        std::vector<int> dim = {n, h, w, c};
        shape.SetDim(dim);
        shape.SetDataLayout(TENGINE_LAYOUT_NHWC);
        oshape[0] = shape;
        return true;
    }
    else if((param_.order0 == 1) && (param_.order1 == 0) && (param_.order2 == 2) && dims.size() == 3)
    {
        // int n = input.GetN();
        int c = input.Shape(0);
        int h = input.Shape(1);
        int w = input.Shape(2);
        TShape shape;
        std::vector<int> dim = {h, c, w};
        shape.SetDim(dim);
        oshape[0] = shape;
        return true;
    }
    else
    {
        return false;
    }
}

void Permute::SetSchema(void)
{
    Input({"input:float32"})
        .Output({"output:float32"})
        .SetAttr("flag", 0)
        .SetAttr("order0", 0)
        .SetAttr("order1", 1)
        .SetAttr("order2", 2)
        .SetAttr("order3", 3)
        .SetDoc(R"DOC(Permute Layer)DOC");
}

}    // namespace TEngine
