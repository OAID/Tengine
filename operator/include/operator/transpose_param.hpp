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
 * Author: bzhang@openailab.com
 */
#ifndef __TRANSPOSE_PARAM_HPP__
#define __TRANSPOSE_PARAM_HPP__

#include "parameter.hpp"

namespace TEngine {

struct TransposeParam : public NamedParam
{
    int dim_0;
    int dim_1;
    int dim_2;
    int dim_3;

    std::vector<int> tr_shape;

    DECLARE_PARSER_STRUCTURE(TransposeParam)
    {
        DECLARE_PARSER_ENTRY(dim_0);
        DECLARE_PARSER_ENTRY(dim_1);
        DECLARE_PARSER_ENTRY(dim_2);
        DECLARE_PARSER_ENTRY(dim_3);
    };
};

}    // namespace TEngine

#endif
