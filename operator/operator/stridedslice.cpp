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
#include <math.h>
namespace TEngine {

// bool StridedSlice::InferShape(const std::vector<TEngine::TShape>& ishape, std::vector<TEngine::TShape>& oshape, int
// layout)
// {
//     const TShape& input = ishape[0];
//     const std::vector<int>& in_dim = input.GetDim();

//     std::vector<int> o_dim=input.GetDim();

//     o_dim[0]= ;//- param_.begin[0] + param_.end[0];
//     o_dim[1]= in_dim[1];//- param_.begin[1] + param_.end[1];
//     o_dim[2]= in_dim[2];//- param_.begin[2] + param_.end[2];
//     o_dim[3]= in_dim[3];//- param_.begin[3] + param_.end[3];

//     TShape shape;
//     shape.SetDim(o_dim);
//     shape.SetDataLayout(input.GetDataLayout());
//     oshape[0] = shape;
//     return true;
// }
bool StridedSlice::InferShape(const std::vector<TEngine::TShape>& ishape, std::vector<TEngine::TShape>& oshape,
                              int layout)
{
    const TShape& input = ishape[0];
    const std::vector<int>& in_dim = input.GetDim();

    if(input.GetDim().size() == 4 && param_.shrink_axis_mask == 0)

    {
        std::vector<int> o_dim = input.GetDim();
        int delta_0=(-param_.begin[0] + param_.end[0])<0? param_.begin[0] -param_.end[0] :-param_.begin[0] + param_.end[0];
        int delta_1=(-param_.begin[1] + param_.end[1])<0? param_.begin[1] -param_.end[1] :-param_.begin[1] + param_.end[1];
        int delta_2=(-param_.begin[2] + param_.end[2])<0? param_.begin[2] -param_.end[2] :-param_.begin[2] + param_.end[2];
        int delta_3=(-param_.begin[3] + param_.end[3])<0? param_.begin[3] -param_.end[3] :-param_.begin[3] + param_.end[3];
        o_dim[0]= ceil(((float)in_dim[0]-(float)delta_0)/(float)param_.stride[0]);
        o_dim[1]= ceil(((float)in_dim[1]-(float)delta_1)/(float)param_.stride[1]);
        o_dim[2]= ceil(((float)in_dim[2]-(float)delta_2)/(float)param_.stride[2]);
        o_dim[3]= ceil(((float)in_dim[3]-(float)delta_3)/(float)param_.stride[3]);


        TShape shape;
        shape.SetDim(o_dim);
        shape.SetDataLayout(input.GetDataLayout());
        oshape[0] = shape;
    }
    else if(input.GetDim().size() == 3 && param_.shrink_axis_mask == 1)
    {
        std::vector<int> o_dim(2);
        o_dim[0] = in_dim[1];
        o_dim[1] = in_dim[2];
        TShape shape;
        shape.SetDim(o_dim);
        shape.SetDataLayout(input.GetDataLayout());
        oshape[0] = shape;
    }
    else if(input.GetDim().size() == 3 && param_.shrink_axis_mask == 2)
    {
        std::vector<int> o_dim(2);
        o_dim[0] = in_dim[0];
        o_dim[1] = in_dim[2];
        TShape shape;
        shape.SetDim(o_dim);
        shape.SetDataLayout(input.GetDataLayout());
        oshape[0] = shape;
    }
    else if(input.GetDim().size() == 3 && param_.shrink_axis_mask == 3)
    {
        std::vector<int> o_dim(2);
        o_dim[0] = in_dim[0];
        o_dim[1] = in_dim[1];
        TShape shape;
        shape.SetDim(o_dim);
        shape.SetDataLayout(input.GetDataLayout());
        oshape[0] = shape;
    }

    return true;
}
void StridedSlice::SetSchema(void)
{
    Input({"input:float32"})
        .Output({"output:float32"})
        .SetAttr("shrink_axis_mask", 0)
        .SetAttr("new_axis_mask", 0)
        .SetAttr("ellipsis_mask", 0)
        .SetAttr("begin_mask", 0)
        .SetAttr("end_mask", 0)
        .SetDoc(R"DOC(SwapAxis Layer)DOC");
}

}    // namespace TEngine
