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
#include <unordered_map>

#include "tensor_shape.hpp"
#include "logger.hpp"
#include "compiler.hpp"

namespace TEngine {

void TShape::SetDim(const std::vector<int>& args)
{
    dim_ = args;
}

void TShape::DumpShape(std::ostream& os) const
{
    std::string result = "[";

    if(dim_.size() > 0)
    {
        unsigned i;

        for(i = 0; i < dim_.size() - 1; i++)
        {
            result += std::to_string(dim_[i]) + ",";
        }

        if(i == (dim_.size() - 1))
        {
            result += std::to_string(dim_[i]);
        }
    }

    result += "]";
    os << result;
}

int TShape::GetN(void) const
{
    return Shape(0);
}

int TShape::GetC(void) const
{
    if(layout_ == TENGINE_LAYOUT_NCHW)
        return Shape(1);
    else
        return Shape(3);
}

int TShape::GetH(void) const
{
    if(layout_ == TENGINE_LAYOUT_NCHW)
        return Shape(2);
    else
        return Shape(1);
}

int TShape::GetW(void) const
{
    if(layout_ == TENGINE_LAYOUT_NCHW)
        return Shape(3);
    else
        return Shape(2);
}

}    // namespace TEngine
