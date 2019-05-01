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
#include "operator/slice.hpp"

namespace TEngine {
bool Slice::InferShape(const std::vector<TEngine::TShape>& ishape, std::vector<TEngine::TShape>& oshape, int layout)
{
    const TShape& input = ishape[0];
    std::vector<int> input_dim = input.GetDim();

    if(param_.iscaffe)
    {
        int slice_axis = param_.axis;
        if(param_.slice_point_.size()!= 0)
        {
            int prev = 0;
            int input_slice_num = input_dim[slice_axis];
        unsigned int i = 0 ;
            for (; i < param_.slice_point_.size(); ++i)
            {
                input_dim[slice_axis] = (param_.slice_point_[i] - prev);
                prev = param_.slice_point_[i];
                oshape[i].SetDim(input_dim);
                oshape[i].SetDataLayout(input.GetDataLayout());
            }
            //The last one
            input_dim[slice_axis] = (input_slice_num - prev);
            oshape[i].SetDim(input_dim);
            oshape[i].SetDataLayout(input.GetDataLayout());
        }
        else
        {
            int out_num = oshape.size();
            if(input.Shape(slice_axis) % out_num != 0)
                return false;
            if(slice_axis > (int)input_dim.size())
                return false;
            input_dim[slice_axis] = input_dim[slice_axis] / out_num;
            for(int i = 0; i < out_num; i++)
            {
                oshape[i].SetDim(input_dim);
                oshape[i].SetDataLayout(input.GetDataLayout());
            }
        }
    }
    else
    {
        std::vector<int> out_dim;
        //input shape size must be equal to begin and size's size;
        if( (param_.size_.size()!= param_.begin_.size())|| (param_.size_.size()!= input_dim.size()))
            return false;
        out_dim.reserve(input_dim.size());
        for(unsigned int i = 0; i < input_dim.size(); i++)
        {
            out_dim[i] = param_.size_[i];
        }
        oshape[0].SetDim(out_dim);
        oshape[0].SetDataLayout(input.GetDataLayout());
    }
    return true;
}
void Slice::SetSchema(void)
{
    Input({"input:float32"})
        .Output({"output:float32"})
        .SetAttr("axis", 1)
        .SetAttr("iscaffe", true)
        .SetDoc(R"DOC(Slice Operator)DOC");
}

}    // namespace TEngine
