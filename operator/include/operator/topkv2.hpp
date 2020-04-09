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
#ifndef __TOPKV2_HPP__
#define __TOPKV2_HPP__

#include "operator.hpp"
#include "topkv2_param.hpp"

namespace TEngine {

class TopKV2 : public OperatorWithParam<TopKV2, TopKV2Param>
{
public:
    TopKV2()
    {
        name_ = "TopKV2";
    }
    TopKV2(const TopKV2& src) = default;
    virtual ~TopKV2() {}

    bool InferShape(const std::vector<TShape>& ishape, std::vector<TShape>& oshape, int layout) override;

    void SetSchema(void) override;
};
}    // namespace TEngine

#endif
