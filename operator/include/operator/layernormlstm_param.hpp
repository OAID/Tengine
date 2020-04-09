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
#ifndef __LAYERNORMLSTM_PARAM_HPP__
#define __LAYERNORMLSTM_PARAM_HPP__

#include "operator.hpp"

namespace TEngine{

enum class KernelType{
  kFullKernel = 0,
  kBasicKernel
};

enum class FusedActivation{
  kNone = 0,
  kRelu,
  kRelu1,
  kRelu6,
  kTanh,
  kSignBit,
  kSigmoid,
};

class LayerNormLSTMParam : public NamedParam
{
public:  
    float cell_clip;
    float proj_clip;
    int hidden_size;
    int output_size;
    KernelType kernel_type;
    FusedActivation fused_activation;

    DECLARE_PARSER_STRUCTURE(LayerNormLSTMParam)
    {
        DECLARE_PARSER_ENTRY(cell_clip);
        DECLARE_PARSER_ENTRY(proj_clip);
        DECLARE_PARSER_ENTRY(hidden_size);
        DECLARE_PARSER_ENTRY(output_size);
        DECLARE_PARSER_ENTRY(kernel_type);
        DECLARE_PARSER_ENTRY(fused_activation);
    };


};

}
#endif
