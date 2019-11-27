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
 * Author: bhu@openailab.com
 */
#include "operator/topkv2.hpp"
#include "static_graph.hpp"
#include "tengine_errno.hpp"

namespace TEngine {

bool TopKV2::InferShape(const std::vector<TShape>& ishape, std::vector<TShape>& oshape, int layout)
{
    const TShape shape = ishape[0];
    std::vector<int> input_dim = shape.GetDim();

    if(param_.k > input_dim.back())
    {
        printf("#Error: K-%d is valid , must be less than %d \n", param_.k, input_dim.back());
        set_tengine_errno(ENOENT);
        return false;
    }

    input_dim.back() = param_.k;
    oshape[0].SetDim(input_dim);
    oshape[0].SetDataLayout(shape.GetDataLayout());

    oshape[1].SetDim(input_dim);
    oshape[1].SetDataLayout(shape.GetDataLayout());

    return true;
}
void TopKV2::SetSchema(void)
{
    Input({"input:float32"})
        .Output({"output:float32"})
        .SetAttr("k", 1)
        .SetAttr("sorted", false)
        .SetDoc(R"DOC(TopKV2 Operator)DOC");
}
}    // namespace TEngine
