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
#include "operator/reduction.hpp"

namespace TEngine {

bool Reduction::InferShape(const std::vector<TEngine::TShape>& ishape, std::vector<TEngine::TShape>& oshape, int layout)
{
    const TShape& input = ishape[0];

    const std::vector<int>& in_dim = input.GetDim();
    int in_size = in_dim.size();
    std::vector<int> new_shape;
    if(param_.dim_0 != -2)
        new_shape.push_back(param_.dim_0);
    if(param_.dim_1 != -2)
        new_shape.push_back(param_.dim_1);
    if(param_.dim_2 != -2)
        new_shape.push_back(param_.dim_2);
    if(param_.dim_3 != -2)
        new_shape.push_back(param_.dim_3);
    bool should_reduced[4] = {false};
    int reduceddim = 0;
    int kd = param_.keepdim;
    int newshape_size = new_shape.size();
    std::vector<int> real_shape = {0, 1, 2, 3};
    if(newshape_size)
    {
        for(int i = 0; i < newshape_size; i++)
        {
            if(new_shape[i] >= 0)
            {
                int idx = new_shape[i];
                if(input.GetDataLayout() == TENGINE_LAYOUT_NHWC)
                    idx = real_shape[idx];
                if(idx >= 0 && idx < 4)
                {
                    should_reduced[idx] = true;
                    ++reduceddim;
                }
            }
            else if(new_shape[i] < 0)
            {
                int current = in_dim.size() + new_shape[i];
                if(input.GetDataLayout() == TENGINE_LAYOUT_NHWC)
                {
                    current = real_shape[current];
                }

                should_reduced[current] = true;
                ++reduceddim;
            }
        }
    }
    else
    {
        for(int idx = 0; idx < in_size; ++idx)
        {
            should_reduced[idx] = true;
            ++reduceddim;
        }
    }
    if(in_size - reduceddim == 0)
    {
        if(kd == 0)
        {
            std::vector<int> odim = {1};
            TShape shape;
            shape.SetDim(odim);

            shape.SetDataLayout(input.GetDataLayout());
            oshape[0] = shape;
            return true;
        }
        else
        {
            std::vector<int> odim(in_size);
            for(int i_idx = 0, o_idx = 0; i_idx < in_size; i_idx++)
            {
                odim[o_idx++] = 1;
            }
            TShape shape;
            shape.SetDim(odim);

            shape.SetDataLayout(input.GetDataLayout());
            oshape[0] = shape;
            return true;
        }

        return false;
    }
    else
    {
        int o_size = 0;
        if(kd == 0)
        {
            o_size = in_size - reduceddim;
        }
        else
        {
            o_size = in_size;
        }
        std::vector<int> odim(o_size);
        for(int i_idx = 0, o_idx = 0; i_idx < in_size; i_idx++)
        {
            if(!should_reduced[i_idx])
            {
                odim[o_idx++] = in_dim[i_idx];
            }
            else if(should_reduced[i_idx] && kd == 1)
            {
                odim[o_idx++] = 1;
            }
        }
        TShape shape;
        shape.SetDim(odim);

        shape.SetDataLayout(input.GetDataLayout());
        oshape[0] = shape;
        return true;
    }

    return false;
}

void Reduction::SetSchema(void)
{
    Input({"input:float32"})
        .Output({"output:float32"})
        .SetAttr("dim_0", -2)
        .SetAttr("dim_1", -2)
        .SetAttr("dim_2", -2)
        .SetAttr("dim_3", -2)
        .SetAttr("keepdim", 0)
        .SetAttr("type", 0)
        .SetDoc(R"DOC(Squeeze Layer)DOC");
}

}    // namespace TEngine
