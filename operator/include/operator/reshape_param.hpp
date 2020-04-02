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
#ifndef __RESHAPE_PARAM_HPP__
#define __RESHAPE_PARAM_HPP__

#include "parameter.hpp"

namespace TEngine {

struct ReshapeParam : public NamedParam
{
    /*
    int dim_0;
    int dim_1;
    int dim_2;
    int dim_3;
    int dim_size;
    int axis;
    DECLARE_PARSER_STRUCTURE(ReshapeParam)
    {
        DECLARE_PARSER_ENTRY(axis);
        DECLARE_PARSER_ENTRY(dim_0);
        DECLARE_PARSER_ENTRY(dim_1);
        DECLARE_PARSER_ENTRY(dim_2);
        DECLARE_PARSER_ENTRY(dim_3);
        DECLARE_PARSER_ENTRY(dim_size);
    };
    */
    std::vector<int> re_shape;
    bool reverse;
    bool is_mxnet;
    bool is_onnx;
    int dim_size;
    DECLARE_PARSER_STRUCTURE(ReshapeParam)
    {
        DECLARE_PARSER_ENTRY(reverse);
        DECLARE_PARSER_ENTRY(is_mxnet);
        DECLARE_PARSER_ENTRY(is_onnx);
    };

};

}    // namespace TEngine

#endif
