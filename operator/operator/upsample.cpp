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
 * Author: zhangrui@openailab.com
 */
#include "operator/upsample.hpp"

namespace TEngine {
bool Upsample::InferShape(const std::vector<TEngine::TShape>& ishape, std::vector<TEngine::TShape>& oshape, int layout)
{
    TShape shape;
    ;
    std::vector<int> dim;
    dim = ishape[0].GetDim();
    int scale = param_.scale;
    // NCHW output_h = in_h * scale output_w = in_w * scale;
    dim[2] = dim[2] * scale;
    dim[3] = dim[3] * scale;

    shape.SetDim(dim);
    shape.SetDataLayout(TENGINE_LAYOUT_NCHW);

    oshape[0] = shape;

    return true;
}
void Upsample::SetSchema(void)
{
    Input({"input:float32"}).Output({"output:float32"}).SetAttr("scale", 1).SetDoc(R"DOC(Upsample Operator)DOC");
}

}    // namespace TEngine
