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
 * Author: zpluo@openailab.com
 */
#ifndef __FLOOR_HPP__
#define __FLOOR_HPP__

#include "operator.hpp"

namespace TEngine {

class Floor : public OperatorNoParam<Floor>
{
public:
    Floor()
    {
        name_ = "Floor";
    }
    Floor(const Floor& src) = default;
    virtual ~Floor(){};

    float GetFops(const std::vector<TShape>& inputs, const std::vector<TShape>& outputs) override;

    void SetSchema(void) override;
};

}    // namespace TEngine

#endif