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
#ifndef __ELTWISE_PARAM_HPP__
#define __ELTWISE_PARAM_HPP__

#include "parameter.hpp"

enum EltType
{
    ELT_PROD,
    ELT_PROD_SCALAR,
    ELT_SUM,
    ELT_SUM_SCALAR,
    ELT_SUB,
    ELT_SUB_SCALAR,
    ELT_MAX,
    ELT_RSQRT,
    ELT_MIN_SCALAR,
    ELT_LAST,
    ELT_DIV,
    ELT_LOG,
    ELT_EXP,
    ELT_SQRT,
    ELT_FLOOR,
    ELT_SQUARE,
    ELT_POW,
    ELT_POWER
};

namespace TEngine {

struct EltwiseParam : public NamedParam
{
    // std::string method;
    // EltType type;
    int type;
    int caffe_flavor;
    float shift;
    float power;
    float scale;

    DECLARE_PARSER_STRUCTURE(EltwiseParam)
    {
        DECLARE_PARSER_ENTRY(type);
        DECLARE_PARSER_ENTRY(caffe_flavor);
        DECLARE_PARSER_ENTRY(shift);
        DECLARE_PARSER_ENTRY(power);
        DECLARE_PARSER_ENTRY(scale);
    };
};

}    // namespace TEngine

#endif
