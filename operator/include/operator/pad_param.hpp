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
#ifndef __PAD_PARAM_HPP__
#define __PAD_PARAM_HPP__

#include "parameter.hpp"

namespace TEngine {

struct PadParam : public NamedParam
{
    // mode : 0: CONSTANT; 1: REFLECT; 2: SYMMETRIC.
    int mode;
    int pad_0_h;
    int pad_0_w;
    int pad_1_h;
    int pad_1_w;
    int pad_2_h;
    int pad_2_w;
    int pad_3_h;
    int pad_3_w;
    float value;

    DECLARE_PARSER_STRUCTURE(PadParam)
    {
        DECLARE_PARSER_ENTRY(mode);
        DECLARE_PARSER_ENTRY(pad_0_h);
        DECLARE_PARSER_ENTRY(pad_0_w);
        DECLARE_PARSER_ENTRY(pad_1_h);
        DECLARE_PARSER_ENTRY(pad_1_w);
        DECLARE_PARSER_ENTRY(pad_2_h);
        DECLARE_PARSER_ENTRY(pad_2_w);
        DECLARE_PARSER_ENTRY(pad_3_h);
        DECLARE_PARSER_ENTRY(pad_3_w);
        DECLARE_PARSER_ENTRY(value);
    };
};
}    // namespace TEngine

#endif