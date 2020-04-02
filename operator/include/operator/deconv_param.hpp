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
#ifndef __DECONVOLUTION_PARAM_HPP__
#define __DECONVOLUTION_PARAM_HPP__

#include "parameter.hpp"

namespace TEngine {

struct DeconvParam : public NamedParam
{
    int num_output;
    int kernel_h;
    int kernel_w;
    int stride_h;
    int stride_w;
    int pad_h0;
    int pad_w0;
    int pad_h1;
    int pad_w1;
    int dilation_h;
    int dilation_w;
    int group;
    int activation;

    DECLARE_PARSER_STRUCTURE(DeconvParam)
    {
        DECLARE_PARSER_ENTRY(num_output);
        DECLARE_PARSER_ENTRY(kernel_h);
        DECLARE_PARSER_ENTRY(kernel_w);
        DECLARE_PARSER_ENTRY(stride_h);
        DECLARE_PARSER_ENTRY(stride_w);
        DECLARE_PARSER_ENTRY(pad_h0);
        DECLARE_PARSER_ENTRY(pad_w0);
        DECLARE_PARSER_ENTRY(pad_h1);
        DECLARE_PARSER_ENTRY(pad_w1);
        DECLARE_PARSER_ENTRY(dilation_h);
        DECLARE_PARSER_ENTRY(dilation_w);
        DECLARE_PARSER_ENTRY(group);
        DECLARE_PARSER_ENTRY(activation);
    };
};

}    // namespace TEngine

#endif
