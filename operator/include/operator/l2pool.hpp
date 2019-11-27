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
 * Author: lmzhang@openailab.com
 */
#ifndef __L2_POOL_HPP__
#define __L2_POOL_HPP__

#include "operator.hpp"
#include "l2pool_param.hpp"

namespace TEngine {

class L2Pool : public OperatorWithParam<L2Pool, L2PoolParam>
{
public:
    L2Pool()
    {
        name_ = "L2Pool";
    }
    L2Pool(const L2Pool& src) = default;

    virtual ~L2Pool() {}
    bool InferShape(const std::vector<TEngine::TShape>& ishape, std::vector<TEngine::TShape>& oshape,
                    int layout) override;
    float GetFops(const std::vector<TEngine::TShape>& ishape, const std::vector<TEngine::TShape>& oshape) override;
    void SetSchema(void) override;

};

}

#endif
