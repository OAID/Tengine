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
#ifndef __GEMM_PARAM_HPP__
#define __GEMM_PARAM_HPP__

#include "parameter.hpp"

namespace TEngine {

struct GemmParam : public NamedParam
{
    float alpha;
    float beta;
    int transA;
    int transB;

    DECLARE_PARSER_STRUCTURE(GemmParam)
    {
        DECLARE_PARSER_ENTRY(alpha);
        DECLARE_PARSER_ENTRY(beta);
        DECLARE_PARSER_ENTRY(transA);
        DECLARE_PARSER_ENTRY(transB);
    }
};

}    // namespace TEngine

#endif
