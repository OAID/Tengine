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
#include "operator/reshape.hpp"


namespace TEngine {

bool Reshape::InferShape(const std::vector<TEngine::TShape>& ishape, 
                               std::vector<TEngine::TShape>& oshape)
{
    const TShape& input=ishape[0];
    const int size=input.GetSize();
    const std::vector<int>& in_dim=input.GetDim();
    std::vector<int> new_shape(4,1); 
    int idx=-1;
    int num_axis=param_.dims.size();
    for(int i=0;i<num_axis;i++)
    {
        if(param_.dims[i]==0)
        {
            new_shape[i]=in_dim[i];
        }
        else if(param_.dims[i]==-1)
        {
            idx=i;
        }
        else
        {
             new_shape[i]=param_.dims[i];
        }
    }
    if(idx>=0)
    {
        int new_size=new_shape[0]*new_shape[1]*new_shape[2]*new_shape[3];
        new_shape[idx]=size/new_size;
    }


    TShape shape;
    shape.SetDim(new_shape);
    shape.SetDataLayout("NCHW");
    oshape[0]=shape;
    return true;    

}


void Reshape::SetSchema(void)
{
    Input({"input:float32"})
    .Output({"output:float32"})
    .SetLayout("NCHW")
    .SetDoc(R"DOC(Reshape Layer)DOC");
}


} //namespace TEngine
