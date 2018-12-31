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
#include <malloc.h>
#include <string.h>

#include "src_serializer.hpp"

namespace TEngine {

bool SrcSerializer::LoadModel(const std::vector<std::string>& file_list, StaticGraph* static_graph)
{
    if(file_list.size() != 1)
        return false;

    if(backend_->GetFileNum() != 1)
        return false;

    const std::string& model_name = file_list[0];

    std::vector<struct model_patch> patch_list = GetModelByName(model_name);

    if(patch_list.empty())
        return false;

    /*concat all patches first */

    unsigned int total_patch_size = 0;

    for(unsigned int i = 0; i < patch_list.size(); i++)
    {
        total_patch_size += patch_list[i].patch_size;
    }

    if(total_patch_size != patch_list[0].total_size)
        return false;

    void* plain_mem = ( void* )malloc(total_patch_size);

    for(unsigned int i = 0; i < patch_list.size(); i++)
    {
        uint32_t offset = patch_list[i].patch_off;

        memcpy(( char* )plain_mem + offset, patch_list[i].addr, patch_list[i].patch_size);
    }

    /*do decryption */
    void* text_addr;
    int text_len;

    bool ret =
        decrypt(plain_mem, total_patch_size, patch_list[0].vendor_id, patch_list[0].nn_id, &text_addr, &text_len);
    free(plain_mem);

    if(!ret)
        return false;

    /* decompress */

    void* origin_addr;
    int origin_len;

    ret = decompress(text_addr, text_len, &origin_addr, &origin_len);

    if(ret)
        free(text_addr);
    else
    {
        origin_addr = text_addr;
        origin_len = text_len;
    }

    /* real load */

    std::vector<const void*> addr_list;
    std::vector<int> size_list;

    addr_list.push_back(origin_addr);
    size_list.push_back(origin_len);

    /* note: the static graph  MUST free the origin_addr when it is destructed */
    ret = backend_->LoadModel(addr_list, size_list, static_graph);

    if(!ret)
        free(origin_addr);

    return ret;
}

bool SrcSerializer::SaveModel(const std::vector<std::string>& file_list, Graph* graph)
{
    if(file_list.size() != 1)
        return false;

    if(backend_->GetFileNum() != 1)
        return false;

    std::vector<void*> addr_list;
    std::vector<int> size_list;

    bool ret = backend_->SaveModel(addr_list, size_list, graph);

    if(!ret)
        return false;

    // do compress

    void* compressed_addr;
    int compressed_size;

    ret = compress(addr_list[0], size_list[0], &compressed_addr, &compressed_size);

    if(ret)
    {
        free(addr_list[0]);
    }
    else
    {
        compressed_addr = addr_list[0];
        compressed_size = size_list[0];
    }

    // do encrypt
    void* crypt_addr;
    int crypt_size;

    ret = encrypt(compressed_addr, compressed_size, &crypt_addr, &crypt_size);

    free(compressed_addr);

    if(!ret)
        return false;

    ret = SaveToSource(file_list[0].c_str(), crypt_addr, crypt_size);

    free(crypt_addr);

    return ret;
}

}    // namespace TEngine
