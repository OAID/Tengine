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
#ifndef __PRIORBOX_PARAM_HPP__
#define __PRIORBOX_PARAM_HPP__


#include "parameter.hpp"


namespace TEngine {


struct PriorBoxParam : public NamedParam {

    std::vector<float> min_size;
    std::vector<float> max_size;
    std::vector<float> variance;
    std::vector<float> aspect_ratio;
    int flip;
    int clip;
    int img_size;
    int img_h;
    int img_w;
    float step_w;
    float step_h;
    float offset;

    int num_priors_;
    int out_dim_;

    DECLARE_PARSER_STRUCTURE(PriorBoxParam) 
    {
       DECLARE_PARSER_ENTRY(offset);
    };

};


} //namespace TEngine


#endif
