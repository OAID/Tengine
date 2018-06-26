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

enum PoolArg {
  kPoolMax,
  kPoolAvg,
  kPoolRand
};


namespace TEngine {

struct PoolParam : public NamedParam {

    std::string method;
    PoolArg alg;
    int kernel_h;
    int kernel_w;
    int pad_h;
    int pad_w;
    int stride_h;
    int stride_w;
    int global;
    int caffe_flavor;
    std::vector<int> kernel_shape;   ///> The size of the kernel along each axis (H, W).   
    std::vector<int> strides;        ///> stride along each axis (H, W).     
    std::vector<int> pads;       ///> [x1_begin, x2_begin...x1_end, x2_end,...] for each axis. 

    DECLARE_PARSER_STRUCTURE(PoolParam) {
       DECLARE_PARSER_ENTRY(method);
       DECLARE_PARSER_ENTRY(kernel_h);
       DECLARE_PARSER_ENTRY(kernel_w);
       DECLARE_PARSER_ENTRY(stride_h);
       DECLARE_PARSER_ENTRY(stride_w);
       DECLARE_PARSER_ENTRY(pad_h);
       DECLARE_PARSER_ENTRY(pad_w);
       DECLARE_PARSER_ENTRY(global);
       DECLARE_PARSER_ENTRY(caffe_flavor);
    };

};


} //namespace TEngine


#endif
