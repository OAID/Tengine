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
#ifndef __PATCH_SERIALIZER_HPP__
#define __PATCH_SERIALIZER_HPP__

#include "src_serializer.hpp"

namespace TEngine {

class PatchSerializer : public SrcSerializer
{
public:
    virtual bool encrypt(const void* text, int text_size, void** crypt_addr, int* crypt_size) override;
    virtual bool decrypt(const void* crypt_addr, int crypt_size, uint32_t vendor_id, uint32_t nn_id, void** text_addr,
                         int* text_size) override;

    virtual bool SaveToSource(const char* model_name, void* addr, int size) override;

    virtual std::vector<struct model_patch> GetModelByName(const std::string& model_name) override;

    virtual int GetPatchNumber(void) = 0;
    virtual void GenerateIOV(void* buf, uint32_t vendor_id, uint32_t nn_id);
    virtual uint32_t GetVendorId(void) = 0;
    virtual uint32_t GetNNId(void) = 0;
    virtual const char* GetRegisterFuncName(void);

    virtual ~PatchSerializer(void){};
};

}    // namespace TEngine

#endif
