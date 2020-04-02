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
 * Author: bzhang@openailab.com
 */
#include "operator/transpose.hpp"

namespace TEngine {

bool Transpose::InferShape(const std::vector<TEngine::TShape>& ishape, std::vector<TEngine::TShape>& oshape, int layout)
{

    std::vector<int> out_shape;
    const TShape& input = ishape[0];
    std::vector<int> dims = input.GetDim();
    int new_shape_size = param_.tr_shape.size();
    for(int i = 0; i < new_shape_size; i++){
        out_shape.push_back(dims[param_.tr_shape[i]]);
        //printf("%d ", out_shape[i]);
    }
    //printf("\n");

/*
    old_shape.push_back(input.GetN());
    old_shape.push_back(input.GetH());
    old_shape.push_back(input.GetW());
    old_shape.push_back(input.GetC()); 

    if(param_.dim_0 != -2)
        new_shape.push_back(old_shape[param_.dim_0]);
    else
    {
        new_shape.push_back(old_shape[0]);
    }
    

    if(param_.dim_1 != -2)    
        new_shape.push_back(old_shape[param_.dim_1]);
    else
    {
        new_shape.push_back(old_shape[1]);
    }

    if(param_.dim_2 != -2)    
        new_shape.push_back(old_shape[param_.dim_2]);
    else
    {
        new_shape.push_back(old_shape[2]);
    }
    
    if(param_.dim_3 != -2)    
        new_shape.push_back(old_shape[param_.dim_3]);
    else
    {
        new_shape.push_back(old_shape[3]);
    }
*/
    TShape shape;
    shape.SetDim(out_shape);
    shape.SetDataLayout(input.GetDataLayout());
 
    oshape[0] = shape;
    return true;
}

void Transpose::SetSchema(void)
{
    Input({"input:float32"})
        .Output({"output:float32"})
        .SetAttr("dim_0", -2)
        .SetAttr("dim_1", -2)
        .SetAttr("dim_2", -2)
        .SetAttr("dim_3", -2)
        .SetDoc(R"DOC(Transpose Layer)DOC");
}

}    // namespace TEngine
