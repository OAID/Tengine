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
 * Author: bhu@openailab.com
 */
#include "operator/argmin.hpp"
#include "static_graph.hpp"
#include "tengine_errno.hpp"

namespace TEngine {

bool ArgMin::InferShape(const std::vector<TEngine::TShape>& ishape, std::vector<TEngine::TShape>& oshape, int layout)
{
    int axis = param_.axis;
    TShape shape = ishape[0];

    std::vector<int> in_dim;
    in_dim = ishape[0].GetDim();
    if(axis >= ( int )in_dim.size())
    {
        std::cout << "axis must be less than input dimension\n";
        set_tengine_errno(ENOENT);
        return false;
    }
#if 0
    printf("input shape : ");
    for (size_t i = 0; i < in_dim.size(); i++) {
        std::cout << in_dim.at(i)<<" "  ;
    }
#endif
    in_dim.erase(in_dim.begin() + axis);

    std::vector<int> out_dim;
    out_dim = in_dim;
#if 0
    printf("\noutput shape : ");    
    for (size_t i = 0; i < out_dim.size(); i++) {
        std::cout << out_dim.at(i) ;
    }
    std::cout<<std::endl;
#endif
    shape.SetDim(out_dim);
    oshape[0] = shape;

    return true;
}

void ArgMin::SetSchema(void)
{
    Input({"input:float32"}).Output({"output:int32"}).SetDoc(R"DOC(ArgMin Operator)DOC");
}

}    // namespace TEngine
