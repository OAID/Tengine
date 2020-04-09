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
 * Author: bingzhang@openailab.com
 */
#include "operator/batchToSpaceND.hpp"
#include "static_graph.hpp"

namespace TEngine {

bool BatchToSpaceND::InferShape(const std::vector<TEngine::TShape>& ishape, std::vector<TEngine::TShape>& oshape, int layout)
{
    const TShape& input = ishape[0];
    
    std::vector<int> in_dims = input.GetDim();
    std::vector<int> out_dims;

    int batch = in_dims[0] / (param_.dilation_x * param_.dilation_y);
    int height = in_dims[1] * param_.dilation_y - param_.crop_top - param_.crop_bottom;
    int width = in_dims[2] * param_.dilation_x - param_.crop_left - param_.crop_right;
    int depth = in_dims[3];

    out_dims.push_back(batch);
    out_dims.push_back(height);
    out_dims.push_back(width);
    out_dims.push_back(depth);

    TShape shape;
    shape.SetDim(out_dims);
    shape.SetDataLayout(input.GetDataLayout());
    oshape[0] = shape;   

    return true;
}

void BatchToSpaceND::SetSchema(void)
{
    Input({"input:float32"})
        .Output({"output:float32"})
        .SetAttr("dilation_x", 1)
        .SetAttr("dilation_y", 1)     
        .SetAttr("crop_top", 0)
        .SetAttr("crop_bottom", 0)
        .SetAttr("crop_left", 0)
        .SetAttr("crop_right", 0)
        .SetDoc(R"DOC(BatchToSpaceND Operator)DOC");
}

}    // namespace TEngine
