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
 * Author: zpluo@openailab.com
 */
#include "operator/reducel2.hpp"

namespace TEngine {

//only support onnx reducel2, axis is only support one data
bool ReduceL2::InferShape(const std::vector<TEngine::TShape>& ishape, std::vector<TEngine::TShape>& oshape, int layout)
{
    const TShape& input = ishape[0];

    const std::vector<int>& in_dim = input.GetDim();
    int kd = param_.keepdim;
    int axis = param_.axis;
    
    std::vector<int> out_dim;
    if(axis < 0)
        axis = axis + (int)in_dim.size();
    

    for(unsigned int i = 0; i < in_dim.size() && i < (unsigned int)axis; i++)
    {
        out_dim.push_back(in_dim[i]);
    } 
    if(kd == 1)
    {
        for(unsigned int i = axis; i < in_dim.size();i++)
        {
            out_dim.push_back(1);
        }
    }
    TShape shape;
    shape.SetDim(out_dim);
    shape.SetDataLayout(input.GetDataLayout());
    oshape[0] = shape;
   
    return true;
}

void ReduceL2::SetSchema(void)
{
    Input({"input:float32"})
        .Output({"output:float32"})
        .SetAttr("keep_dim", 1)
        .SetAttr("axis", 0)
        .SetDoc(R"DOC(ReduceL2 Layer)DOC");
}

}    // namespace TEngine
