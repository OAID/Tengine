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
 * Copyright (c) 2017, Open AI Lab
 * Author: haitao@openailab.com
 */
#include "operator/matmul.hpp"
#include "static_graph.hpp"

namespace TEngine {

// only support nchw 
bool MatMul::InferShape(const std::vector<TEngine::TShape>& ishape, std::vector<TEngine::TShape>& oshape, int layout)
{
    std::vector<int> out_dims;
    
    TShape input_shape0 = ishape[0];
    TShape input_shape1 = ishape[1];
    
    auto &dim0 = input_shape0.GetDim();   
    auto &dim1 = input_shape1.GetDim(); 

    if(dim0.size() != dim1.size())
    {
        std::cout<<"dim's size of inputs must be qual for operator matmul\n";
        return false;
    }
    int size = (int)dim0.size();
    if(2 == size)
    {
        out_dims.push_back(dim0[0]);
        out_dims.push_back(dim1[1]);
    }
    else if(3 == size)
    {
        out_dims.push_back(dim0[0]);
        out_dims.push_back(dim0[1]);
        out_dims.push_back(dim1[2]); 
    }
    else if(4 == size)
    {
        out_dims.push_back(dim0[0]);
        out_dims.push_back(dim0[1]);
        out_dims.push_back(dim0[2]);
        out_dims.push_back(dim1[3]); 
    }
    
    TShape shape;
    shape.SetDim(out_dims);
    shape.SetDataLayout(layout);
    
    oshape[0] = shape;    

    return true;
}

void MatMul::SetSchema(void)
{
    Input({"input:float32"}).Output({"output:float32"}).SetDoc(R"DOC(MatMul Operator)DOC");
}

}    // namespace TEngine
