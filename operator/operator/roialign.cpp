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
#include "operator/roialign.hpp"
#include "static_graph.hpp"

namespace TEngine {

bool Roialign::InferShape(const std::vector<TEngine::TShape>& ishape, std::vector<TEngine::TShape>& oshape, int layout)
{
    //printf("Infershape for roi align\n");
    const TShape& input = ishape[0];

    int out_h = param_.pooled_height;
    int out_w = param_.pooled_width;

    TShape shape;
    if(layout == TENGINE_LAYOUT_NHWC)
    {
       std::vector<int> dim = {input.GetN(), out_h, out_w, input.GetC()};
       shape.SetDim(dim);
       shape.SetDataLayout(TENGINE_LAYOUT_NHWC);
    }
    else
    {
       std::vector<int> dim = {input.GetN(), input.GetC(), out_h, out_w};

       shape.SetDim(dim);
       shape.SetDataLayout(TENGINE_LAYOUT_NCHW);
    }
    oshape[0] = shape;  
    //printf("Infershape for roi align\n");
    return true;
}

void Roialign::SetSchema(void)
{
    Input({"input:float32"})
        .Output({"output:float32"})
        .SetAttr("pooled_width", 0)
        .SetAttr("pooled_height", 0)        
        .SetAttr("spatial_scale", 0)
        .SetDoc(R"DOC(Roialign Operator)DOC");
}

}    // namespace TEngine
