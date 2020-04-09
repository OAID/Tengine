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
#include "operator/priorbox.hpp"

namespace TEngine {

bool PriorBox::InferShape(const std::vector<TEngine::TShape>& ishape, std::vector<TEngine::TShape>& oshape, int layout)
{
    const TShape& input = ishape[0];
    const std::vector<int>& feat_dim = input.GetDim();

    // out shape [feat_width,feat_height,num_priors_ * 4,2]
    int len_aspect_ratio = 1;
    if(param_.flip)
        len_aspect_ratio += 1;
    int len_max = 0;
    if(param_.max_size.size() > 0)
    {
        if(param_.max_size.size() == param_.min_size.size())
        {
            len_max += 1;
        }
        else
        {
            // max_size_len must equal min_size_len
            return false;
        }
    }
    param_.num_priors_ = (param_.aspect_ratio.size() * len_aspect_ratio + 1 + len_max) * param_.min_size.size();

    param_.out_dim_ = feat_dim[2] * feat_dim[3] * param_.num_priors_ * 4;

    TShape shape;
    std::vector<int> dim = {feat_dim[0], 2, param_.out_dim_, 1};
    shape.SetDim(dim);
    shape.SetDataLayout(input.GetDataLayout());
    oshape[0] = shape;
    return true;
}

void PriorBox::SetSchema(void)
{
    Input({"input:float32"})
        .Output({"output:float32"})
        .SetAttr("offset", 0.5)

        .SetDoc(R"DOC(PriorBox Layer)DOC");
}

}    // namespace TEngine
