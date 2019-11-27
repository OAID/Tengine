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
#ifndef __POOLING_PARAM_HPP__
#define __POOLING_PARAM_HPP__

#include "parameter.hpp"

enum PoolArg
{
    kPoolMax,
    kPoolAvg,
    kPoolRand
};

enum PoolingSize
{
    POOL_GENERIC,
    POOL_K2S2,
    POOL_K3S2,
    POOL_K3S1
};

namespace TEngine {

#define COUNT_INCLUDE_PAD_MSK 0x010

struct PoolParam : public NamedParam
{
    int alg;
    int kernel_h;
    int kernel_w;
    int stride_h;
    int stride_w;
    int global;
    int caffe_flavor;
    int pad_h0;    // top padding rows
    int pad_w0;    // left padding columns
    int pad_h1;    // bottom padding rows
    int pad_w1;    // right padding columns

    DECLARE_PARSER_STRUCTURE(PoolParam)
    {
        DECLARE_PARSER_ENTRY(alg);
        DECLARE_PARSER_ENTRY(kernel_h);
        DECLARE_PARSER_ENTRY(kernel_w);
        DECLARE_PARSER_ENTRY(stride_h);
        DECLARE_PARSER_ENTRY(stride_w);
        DECLARE_PARSER_ENTRY(global);
        DECLARE_PARSER_ENTRY(caffe_flavor);
        DECLARE_PARSER_ENTRY(pad_h0);
        DECLARE_PARSER_ENTRY(pad_w0);
        DECLARE_PARSER_ENTRY(pad_h1);
        DECLARE_PARSER_ENTRY(pad_w1);
    };
};

}    // namespace TEngine

#endif
