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
 * Author: haoluo@openailab.com
 */
#include "operator/swap_axis.hpp"

namespace TEngine {

bool SwapAxis::InferShape(const std::vector<TEngine::TShape>& ishape, std::vector<TEngine::TShape>& oshape, int layout)
{
    if(param_.dim_0 == param_.dim_1)
    {
        return false;
    }
    if(ishape.size() != 1 || oshape.size() != 1)
        return false;

    const std::vector<int>& in_dim = ishape[0].GetDim();
    int in_dim_size = in_dim.size();

    if(param_.dim_0 >= in_dim_size || param_.dim_1 >= in_dim_size)
        return false;

    std::vector<int> new_dim;
    new_dim.resize(in_dim_size);
    for(int i = 0; i < in_dim_size; i++)
        new_dim[i] = in_dim[i];
    new_dim[param_.dim_0] = in_dim[param_.dim_1];
    new_dim[param_.dim_1] = in_dim[param_.dim_0];

    TShape new_shape;
    new_shape.SetDim(new_dim);

    new_shape.SetDataLayout(layout);
    oshape[0] = new_shape;
    return true;
}

void SwapAxis::SetSchema(void)
{
    Input({"input:float32"})
        .Output({"output:float32"})
        .SetAttr("dim_0", 0)
        .SetAttr("dim_1", 1)
        .SetDoc(R"DOC(SwapAxis Layer)DOC");
}

}    // namespace TEngine
