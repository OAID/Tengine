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
#ifndef __LAYERNORMLSTM_HPP__
#define __LAYERNORMLSTM_HPP__

#include "layernormlstm_param.hpp"
#include "operator.hpp"

namespace TEngine{

class LayerNormLSTM : public OperatorWithParam<LayerNormLSTM, LayerNormLSTMParam>
{
public:
    LayerNormLSTM(void)
    {
        name_ = "LayerNormLSTM";
    }
    LayerNormLSTM(const LayerNormLSTM& src)= default;
    
    virtual ~LayerNormLSTM() {}
    bool InferShape(const std::vector<TShape>&, std::vector<TShape>&, int layout) override;    
    void SetSchema(void) override;
};
    
}

#endif
