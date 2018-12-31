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
 * Author: haitao@openailab.com
 */
#ifndef __SRC_TM_SERIALIZER_HPP__
#define __SRC_TM_SERIALIZER_HPP__

#include "patch_serializer.hpp"

namespace TEngine {

class SrcTmSerializer : public PatchSerializer
{
public:
    SrcTmSerializer(void);
    int GetPatchNumber(void) final;
    uint32_t GetVendorId(void) final;
    uint32_t GetNNId(void) final;
};

}    // namespace TEngine

#endif
