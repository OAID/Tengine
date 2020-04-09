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
 * Author: bingzhang@openailab.com
 */
#include "operator/crop.hpp"

namespace TEngine {

bool Crop::InferShape(const std::vector<TEngine::TShape>& ishape, std::vector<TEngine::TShape>& oshape, int layout)
{
    const TShape& input_shape = ishape[1];
    int input_h = input_shape.GetH();
    int input_w = input_shape.GetW();

    int output_h = 0;
    int output_w = 0;
    // Mxnet 
    if(param_.flag == 1){
        if(param_.num_args == 2)
        {
            output_h = input_h;
            output_w = input_w;
        }
        if(param_.num_args == 1)
        {
            output_h = param_.crop_h;
            output_w = param_.crop_h;
        }
    }
    //Caffe
    if(param_.flag == 0){
        output_h = input_h;
        output_w = input_w;
    }

    TShape shape;
    if(layout == TENGINE_LAYOUT_NCHW)
    {
        std::vector<int> dim = {input_shape.GetN(), input_shape.GetC(), output_h, output_w};

        shape.SetDim(dim);
        shape.SetDataLayout(TENGINE_LAYOUT_NCHW);
    }
    else
    {
        std::vector<int> dim = {input_shape.GetN(), output_h, output_w, input_shape.GetC()};

        shape.SetDim(dim);
        shape.SetDataLayout(TENGINE_LAYOUT_NHWC);
    }
    oshape[0] = shape;

    return true;
}
void Crop::SetSchema(void)
{
    Input({"input:float"})
        .Output({"output:float"})
        .SetAttr("offset_c", 0)        
        .SetAttr("offset_h", 0)
        .SetAttr("offset_w", 0)
        .SetAttr("crop_h", 0)
        .SetAttr("crop_w", 0)
        .SetAttr("num_args", 0)
        .SetAttr("center_crop", false)
        .SetAttr("flag", 0)
        .SetAttr("axis", 2)        
        .SetDoc(R"DOC(Crop Layer)DOC");
}

}    // namespace TEngine
