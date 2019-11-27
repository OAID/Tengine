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
#include "operator/tile.hpp"

namespace TEngine {

bool Tile::InferShape(const std::vector<TEngine::TShape>& ishape, std::vector<TEngine::TShape>& oshape, int layout)
{
    std::cout << "Test shape\n";
    const TShape& input_shape = ishape[0];
    // const std::vector<int>& in_dim = input_shape.GetDim();

    int param_size = param_.reps.size();
    // int input_dim_size = in_dim.size();

    if(param_size != 0)
    {
        // std::reverse(param_.reps.begin(), param_.reps.end());
        for(int i = 0; i < param_size / 2; i++)
        {
            int temp = param_.reps.at(0);
            param_.reps.at(i) = param_.reps.at(param_size - i - 1);
            param_.reps.at(param_size - i - 1) = temp;
        }
    }
    else
    {
        return false;
    }

    switch(param_size)
    {
        case 0:
            for(int i = 0; i < 4; i++)
            {
                param_.reps.push_back(1);
            }
        case 1:
            for(int i = 0; i < 3; i++)
            {
                param_.reps.push_back(1);
            }
            break;
        case 2:
            for(int i = 0; i < 2; i++)
            {
                param_.reps.push_back(1);
            }
            break;
        case 3:
            param_.reps.push_back(1);
            break;
        default:
            break;
    }

    int output_n = input_shape.GetN() * param_.reps.at(3);
    int output_c = input_shape.GetC() * param_.reps.at(2);
    int output_h = input_shape.GetH() * param_.reps.at(1);
    int output_w = input_shape.GetW() * param_.reps.at(0);

    TShape shape;
    if(layout == TENGINE_LAYOUT_NCHW)
    {
        std::vector<int> dim = {output_n, output_c, output_h, output_w};

        shape.SetDim(dim);
        shape.SetDataLayout(TENGINE_LAYOUT_NCHW);
    }
    else
    {
        std::vector<int> dim = {output_n, output_h, output_w, output_c};

        shape.SetDim(dim);
        shape.SetDataLayout(TENGINE_LAYOUT_NHWC);
    }
    oshape[0] = shape;

    return true;
}
void Tile::SetSchema(void)
{
    Input({"input:float"}).Output({"output:float"}).SetAttr("reps", 0).SetDoc(R"DOC(Tile Layer)DOC");
}

}    // namespace TEngine
