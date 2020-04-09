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
 * Author: chunyinglv@openailab.com
 */
#ifndef __PERMUTE_PARAM_HPP__
#define __PERMUTE_PARAM_HPP__

#include "parameter.hpp"

namespace TEngine {

struct PermuteParam : public NamedParam
{
    int flag;
    int order0;
    int order1;
    int order2;
    int order3;

    DECLARE_PARSER_STRUCTURE(PermuteParam)
    {
        DECLARE_PARSER_ENTRY(flag);
        DECLARE_PARSER_ENTRY(order0);
        DECLARE_PARSER_ENTRY(order1);
        DECLARE_PARSER_ENTRY(order2);
        DECLARE_PARSER_ENTRY(order3);
    };
};

}    // namespace TEngine

#endif
