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

bool Reshape::InferShape(const std::vector<TEngine::TShape>& ishape, std::vector<TEngine::TShape>& oshape, int layout)
{
    const TShape& input = ishape[0];
    const int size = input.GetSize();
    std::vector<int> new_shape;
    int new_size = 1;
    if(param_.dim_0 != -2)
        new_shape.push_back(param_.dim_0);
    if(param_.dim_1 != -2)
        new_shape.push_back(param_.dim_1);
    if(param_.dim_2 != -2)
        new_shape.push_back(param_.dim_2);
    if(param_.dim_3 != -2)
        new_shape.push_back(param_.dim_3);

    // printf("new_shape: %d, %d, %d, %d\n",new_shape[0],new_shape[1],new_shape[2],new_shape[3]);
    int dim_size = new_shape.size();
    int idx = -1;
    for(int i = 0; i < dim_size; i++)
    {
        if(new_shape[i] == 0)
            new_shape[i] = 1;
        else if(new_shape[i] == -1)
            idx = i;
        else
            new_size *= new_shape[i];
    }

    if(idx >= 0)
    {
        new_shape[idx] = size / new_size;
    }

    TShape shape;
    shape.SetDim(new_shape);
    // only support 2-D 3-D or 4-D
    if(new_shape.size() == 4)
    {
        if(layout == TENGINE_LAYOUT_NCHW)
            shape.SetDataLayout("NCHW");
        else
            shape.SetDataLayout("NHWC");
    }
    else if(new_shape.size() == 3)
    {
        shape.SetDataLayout("NHW");
    }
    else if(new_shape.size() == 2)
    {
        shape.SetDataLayout("HW");
    }
    else
        return false;
    oshape[0] = shape;
    return true;
}

void Reshape::SetSchema(void)
{
    Input({"input:float32"})
        .Output({"output:float32"})
        .SetLayout("NCHW")
        .SetAttr("dim_0", -2)
        .SetAttr("dim_1", -2)
        .SetAttr("dim_2", -2)
        .SetAttr("dim_3", -2)
        .SetAttr("dim_size", 0)
        .SetDoc(R"DOC(Reshape Layer)DOC");
}

}    // namespace TEngine
