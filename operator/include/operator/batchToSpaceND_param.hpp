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
 * Author: bingzhang@openailab.com
 */
#ifndef __BATCHTOSPACEND_PARAM_HPP__
#define __BATCHTOSPACEND_PARAM_HPP__

#include "parameter.hpp"

namespace TEngine {

struct BatchToSpaceNDParam : public NamedParam
{
    int dilation_x;
    int dilation_y;
    int crop_top;
    int crop_bottom;
    int crop_left;
    int crop_right;
    DECLARE_PARSER_STRUCTURE(BatchToSpaceNDParam)
    {
        DECLARE_PARSER_ENTRY(dilation_x);
        DECLARE_PARSER_ENTRY(dilation_y);
        DECLARE_PARSER_ENTRY(crop_top);
        DECLARE_PARSER_ENTRY(crop_bottom);
        DECLARE_PARSER_ENTRY(crop_left);
        DECLARE_PARSER_ENTRY(crop_right);
    }
};

}    // namespace TEngine

#endif
