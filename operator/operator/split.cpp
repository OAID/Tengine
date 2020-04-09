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
#include "operator/split.hpp"
#include "static_graph.hpp"

namespace TEngine {

bool Split::InferShape(const std::vector<TShape>& ishape, std::vector<TShape>& oshape, int layout)
{
    int axis = param_.axis;
    const TShape shape = ishape[0];
    std::vector<int> input_dim = shape.GetDim();

    if(param_.is_caffe)
    {
        for(unsigned int i = 0; i < oshape.size(); i++)
            oshape[i] = ishape[0];
    }else if (param_.is_onnx){
        if(param_.split_sizes_.size() != 0)
        {
            int sumcheck = 0;
            int input_slice_num = input_dim[axis];
            for(unsigned int i = 0; i < param_.split_sizes_.size(); ++i)
            {
                sumcheck += param_.split_sizes_[i];
            }
            if(sumcheck != input_slice_num)
            {
                return false;
            }
            for(unsigned int i = 0; i < param_.split_sizes_.size(); ++i)
            {
                input_dim[axis] = (param_.split_sizes_[i]);
                //for(int i = 0; i < axis; i++)
                //    input_dim[i] = (param_.split_sizes_[i]);
                //printf("%d %d %d %d \n", input_dim[0],input_dim[1],input_dim[2],input_dim[3]);
                
                oshape[i].SetDim(input_dim);
                oshape[i].SetDataLayout(shape.GetDataLayout());                 
            }           
        }       
    }
    else
    {
        if(param_.split_sizes_.size() != 0)
        {
            int sumcheck = 0;
            int input_slice_num = input_dim[axis];
            for(unsigned int i = 0; i < param_.split_sizes_.size(); ++i)
            {
                sumcheck += param_.split_sizes_[i];
            }
            if(sumcheck != input_slice_num)
            {
                return false;
            }
            for(unsigned int i = 0; i < param_.split_sizes_.size(); ++i)
            {
                input_dim[axis] = (param_.split_sizes_[i]);
                //printf("%d ", input_dim[axis]);
                oshape[i].SetDim(input_dim);
                oshape[i].SetDataLayout(shape.GetDataLayout());
            }
        }
        else
        {
            int split_dim = param_.split_dim;
            int split_shape = 0;
            std::vector<int> dim;
            dim = ishape[0].GetDim();
            if(dim[axis] % split_dim != 0)
                return false;
            split_shape = dim[axis] / split_dim;
            input_dim[axis] = split_shape;
            if( split_shape == 1)
            {
                input_dim.erase(input_dim.begin() + axis);
            }
            for(unsigned int i = 0; i < oshape.size(); i++)
            {
                oshape[i].SetDim(input_dim);
                oshape[i].SetDataLayout(shape.GetDataLayout());
            }
        }
    }

    return true;
}
void Split::SetSchema(void)
{
    Input({"input:float32"})
        .Output({"output:float32"})
        .SetAttr("axis", 0)
        .SetAttr("split_dim", 1)
        // .SetAttr("squeeze_axis", 0)
        .SetAttr("is_caffe", false)
        .SetAttr("is_onnx", false)
        .SetDoc(R"DOC(Split Operator)DOC");
}
}    // namespace TEngine
