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
#ifndef __LRN_PARAM_HPP__
#define __LRN_PARAM_HPP__

#include "parameter.hpp"

namespace TEngine {

#define LRN_ACROSS_CHANNELS 0
#define LRN_WITHIN_CHANNEL  1

struct LRNParam : public NamedParam {
   int   local_size;
   float alpha;
   float beta;
   int   norm_region;
   float k;

   DECLARE_PARSER_STRUCTURE(LRNParam) {
        DECLARE_PARSER_ENTRY(local_size);
        DECLARE_PARSER_ENTRY(alpha);
        DECLARE_PARSER_ENTRY(beta);
        DECLARE_PARSER_ENTRY(norm_region);
        DECLARE_PARSER_ENTRY(k);
   };
   
};


} //namespace TEngine


#endif
