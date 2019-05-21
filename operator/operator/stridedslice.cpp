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
 * Author: chunyinglv@openailab.com
 */
#include "operator/stridedslice.hpp"

namespace TEngine {

// bool StridedSlice::InferShape(const std::vector<TEngine::TShape>& ishape, std::vector<TEngine::TShape>& oshape, int layout)
// {
//     const TShape& input = ishape[0];
//     const std::vector<int>& in_dim = input.GetDim();

//     std::vector<int> o_dim=input.GetDim();

//     o_dim[0]= in_dim[0];//- param_.begin[0] + param_.end[0];
//     o_dim[1]= in_dim[1];//- param_.begin[1] + param_.end[1];
//     o_dim[2]= in_dim[2];//- param_.begin[2] + param_.end[2];
//     o_dim[3]= in_dim[3];//- param_.begin[3] + param_.end[3];

//     TShape shape;
//     shape.SetDim(o_dim);
//     shape.SetDataLayout(input.GetDataLayout());
//     oshape[0] = shape;
//     return true;
// }
bool StridedSlice::InferShape(const std::vector<TEngine::TShape>& ishape, std::vector<TEngine::TShape>& oshape, int layout)
{
    const TShape& input = ishape[0];
    const std::vector<int>& in_dim = input.GetDim();

    std::vector<int> o_dim(4);

    o_dim[0]= (in_dim[0]- param_.begin[0] + param_.end[0])/param_.stride[0];
    o_dim[1]= (in_dim[1]- param_.begin[1] + param_.end[1])/param_.stride[1];
    o_dim[2]= (in_dim[2]- param_.begin[2] + param_.end[2])/param_.stride[2];
    o_dim[3]= (in_dim[3]- param_.begin[3] + param_.end[3])/param_.stride[3];

    TShape shape;
    shape.SetDim(o_dim);
    shape.SetDataLayout(input.GetDataLayout());
    oshape[0] = shape;

    return true;
}

}    // namespace TEngine
