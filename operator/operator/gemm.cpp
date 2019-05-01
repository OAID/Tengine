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
#include "operator/gemm.hpp"
#include "static_graph.hpp"

namespace TEngine {

bool Gemm::InferShape(const std::vector<TEngine::TShape>& ishape, std::vector<TEngine::TShape>& oshape, int layout)
{
    int m, n;

    const TShape& input = ishape[0];
    const TShape& weight = ishape[1];

    if(param_.transA)
        m = input.Shape(1);
    else
        m = input.Shape(0);

    if(param_.transB)
        n = weight.Shape(0);
    else
        n = weight.Shape(1);

    TShape out_shape;

    std::vector<int> dim(2);

    dim[0] = m;
    dim[1] = n;

    out_shape.SetDim(dim);
    out_shape.SetDataLayout(input.GetDataLayout());

    oshape[0] = out_shape;

    return true;
}

void Gemm::SetSchema(void)
{
    Input({"input:float32", "weight: float32", "bias: float32"})
        .Output({"output:float32"})
        .SetAttr("alpha", 1.0f)
        .SetAttr("beta", 1.0f)
        .SetAttr("transA", 0)
        .SetAttr("transB", 0)
        .SetDoc(R"DOC(Gemm Operator)DOC");
}

}    // namespace TEngine
