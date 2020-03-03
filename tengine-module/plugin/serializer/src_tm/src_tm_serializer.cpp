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
#include "src_tm_serializer.hpp"
#include "logger.hpp"

namespace TEngine {

SrcTmSerializer::SrcTmSerializer(void)
{
    name_ = "SrcTmSerializer";

    // using tm serializer as its backend
    // this require tm serializer must be registered already

    SerializerPtr serializer;

    if(!SerializerManager::SafeGet("tengine", serializer))
    {
        LOG_ERROR() << "please register " << name_ << " after tengine serializer\n";
        backend_ = NULL;
    }
    else
        backend_ = serializer.get();
}

int SrcTmSerializer::GetPatchNumber(void)
{
    const char* str_patch_number = getenv("MODLE_SRC_NUMBER");

    if(!str_patch_number)
        return 4;

    int patch_number = strtoul(str_patch_number, NULL, 10);

    return patch_number;
}

uint32_t SrcTmSerializer::GetVendorId(void)
{
    // MODEL_VENDOR_ID MUST be in hex
    const char* vendor_str = getenv("MODEL_VENDOR_ID");

    if(!vendor_str)
        return 0xdeadbeaf;

    uint32_t vendor_id = strtoul(vendor_str, NULL, 16);

    return vendor_id;
}
uint32_t SrcTmSerializer::GetNNId(void)
{
    // MODEL_NN_ID MUST be in hex
    const char* nn_id_str = getenv("MODLE_NN_ID");

    if(!nn_id_str)
        return 0x55aa55aa;

    uint32_t nn_id = strtoul(nn_id_str, NULL, 16);

    return nn_id;
}
}    // namespace TEngine
