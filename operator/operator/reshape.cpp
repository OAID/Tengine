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
    std::vector<int> input_dim = input.GetDim();
    //if(param_.is_onnx == false){    
        int new_size = 1;       
        int new_shape_size = param_.re_shape.size();
        int in_idx =0;
    
        int input_dim_size = input.GetDim().size();    
        std::vector<int> in_dims = input.GetDim();
        for(int i = 0; i < new_shape_size; ++i)
        {
            if(0 == param_.re_shape[i])
            {
                new_shape.push_back(in_dims[i]);
                in_idx++;
            }
            else if(-1 == param_.re_shape[i])
            {
                new_shape.push_back(-1);
                in_idx++; 
            }
            else if(-2 == param_.re_shape[i])
            {
                for( ;in_idx < input_dim_size;++in_idx)
                {
                    new_shape.push_back(in_dims[in_idx]);
                }            
            }
            else if(-3 == param_.re_shape[i])
            {
                new_shape.push_back(in_dims[in_idx]*in_dims[in_idx+1]);
                in_idx  = in_idx + 2;
            }
            else if(-4 == param_.re_shape[i])
            {
                int muti_val = param_.re_shape[i+1];
                if(muti_val == -1)
                    muti_val = 1;
                new_shape.push_back(muti_val);
                new_shape.push_back(param_.re_shape[i+2]) ;
                i=i+2;
                in_idx++;
            }
            else
            {
                new_shape.push_back(param_.re_shape[i]);
            }
        }    

        int idx = -1;
        int dim_size = new_shape.size();
        for(int i = 0; i < dim_size; i++)
        {
            if(new_shape[i] == -1)
                idx = i;
            else
                new_size *= new_shape[i];
        }

        if(idx >= 0)
        {
            new_shape[idx] = size / new_size;
        }
        if(new_shape[0]==-1 && new_shape.size()==1)
        {
            new_shape[0]=size;
        }
        
        if(param_.reverse)
        {
            std::vector<int> tmp = new_shape;
            int j = 0;
            for(int i = dim_size-1; i>=0 ; --i)
            {
                new_shape[j++] = tmp[i];
            }    
        }
    //}
    
    /*else {
        printf("Reshape: ");
        for(int i = 0; i < (int)new_shape.size(); i++)
            printf("%d ", new_shape[i]);
        printf("\n");     
    }*/
    
    TShape shape;
    shape.SetDim(new_shape);
    shape.SetDataLayout(input.GetDataLayout());

    oshape[0] = shape;
    return true;

}

void Reshape::SetSchema(void)
{
    /*
    Input({"input:float32"})
        .Output({"output:float32"})
        .SetAttr("dim_0", -2)
        .SetAttr("dim_1", -2)
        .SetAttr("dim_2", -2)
        .SetAttr("dim_3", -2)
        .SetAttr("dim_size", 0)
        .SetDoc(R"DOC(Reshape Layer)DOC");
        */
    Input({"input:float32"})
        .Output({"output:float32"})
        .SetAttr("is_mxnet", false)
        .SetAttr("reverse", false)
        .SetAttr("is_onnx", false)
        .SetDoc(R"DOC(Reshape Layer)DOC");

}

}    // namespace TEngine
