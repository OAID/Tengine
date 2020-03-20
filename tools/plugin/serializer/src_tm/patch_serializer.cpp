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
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "patch_serializer.hpp"

namespace TEngine {

static std::vector<struct model_patch> g_patch_list;

const char* PatchSerializer::GetRegisterFuncName(void)
{
    return "patch_serializer_register";
}

// implement a simple encryption for demo case
bool PatchSerializer::encrypt(const void* text, int text_size, void** crypt_addr, int* crypt_size)
{
    uint32_t vendor_id = GetVendorId();
    uint32_t nn_id = GetNNId();

    int new_len = ((text_size + 31) & ~31) + 32;
    void* new_addr = malloc(new_len);
    unsigned int iov[8];
    unsigned int* p;

    struct timeval tv;

    gettimeofday(&tv, NULL);

    srandom(tv.tv_usec);

    /* create iv */
    for(int i = 0; i < 8; i++)
    {
        iov[i] = random();
    }

    p = ( unsigned int* )new_addr;

    memcpy(p, iov, 32);

    /*apply vendor_id and nn_id */
    GenerateIOV(p, vendor_id, nn_id);

    unsigned int* crypt_base = p + 8;

    memcpy(crypt_base, text, text_size);

    for(int i = text_size + 32; i < new_len; i++)
    {
        unsigned char* p = ( unsigned char* )new_addr;
        p[i] = random();
    }

    for(int i = (new_len / 32) - 1; i > 0; i--)
    {
        unsigned int* base = ( unsigned int* )new_addr;

        base += i * 8;

        for(int j = 0; j < 8; j++)
            base[j] = base[j] ^ base[j - 8];
    }

    // restore un-changed iov
    memcpy(new_addr, iov, 32);

    *crypt_addr = new_addr;
    *crypt_size = new_len;

    return true;
}

bool PatchSerializer::decrypt(const void* crypt_addr, int crypt_size, uint32_t vendor_id, uint32_t nn_id,
                              void** text_addr, int* text_size)
{
    // must be 32 byte aligned
    if(crypt_size & 31)
        return false;

    int new_len = crypt_size - 32;
    unsigned int* new_addr = ( unsigned int* )malloc(new_len);

    unsigned int* p = ( unsigned int* )crypt_addr;

    *text_addr = new_addr;
    *text_size = new_len;

    GenerateIOV(p, vendor_id, nn_id);

    for(int j = 0; j < 8; j++)
    {
        new_addr[j] = p[j] ^ p[j + 8];
    }

    p += 8;

    for(int i = 1; i < new_len / 32; i++)
    {
        new_addr += 8;
        p += 8;

        for(int j = 0; j < 8; j++)
            new_addr[j] = new_addr[j - 8] ^ p[j];
    }

    return true;
}

static int create_patch_data_blob(FILE* fp, char* addr, int size, const char* prefix)
{
    unsigned char* p = ( unsigned char* )addr;

    fprintf(fp, "\nstatic char %s_data[]={\n", prefix);

    for(int i = 0; i < size - 1; i++)
    {
        if((i % 16) == 0)
            fprintf(fp, "\n");
        fprintf(fp, "0x%02x,", p[i]);
    }

    fprintf(fp, "0x%02x\n};\n", p[size - 1]);

    return 0;
}

static int create_patch_header(FILE* fp, struct model_patch* patch, const char* prefix)
{
    fprintf(fp, "\n#include <stdint.h>\n");
    fprintf(fp, "\n#define MAX_MODEL_NAME_LEN (64-1)\n");
    fprintf(fp, "\nstruct model_patch {\n");
    fprintf(fp, "\tchar model_name[MAX_MODEL_NAME_LEN+1];\n");
    fprintf(fp, "\tuint32_t  vendor_id;\n");
    fprintf(fp, "\tuint32_t  nn_id;\n");
    fprintf(fp, "\tuint32_t  total_size;\n");
    fprintf(fp, "\tuint32_t  patch_off;\n");
    fprintf(fp, "\tuint32_t  patch_size;\n");
    fprintf(fp, "\tvoid *    addr;\n");
    fprintf(fp, "};\n\n");

    fprintf(fp, "\nstatic struct model_patch %s_patch={\n", prefix);
    fprintf(fp, "\t\t.model_name=\"%s\",\n", patch->model_name);
    fprintf(fp, "\t\t.vendor_id=0x%08x,\n", patch->vendor_id);
    fprintf(fp, "\t\t.nn_id=0x%08x,\n", patch->nn_id);
    fprintf(fp, "\t\t.total_size=%d,\n", patch->total_size);
    fprintf(fp, "\t\t.patch_off=%u,\n", patch->patch_off);
    fprintf(fp, "\t\t.patch_size=%u,\n", patch->patch_size);
    fprintf(fp, "\t\t.addr=%s_data};\n", prefix);

    return 0;
}

static int create_patch_registry(FILE* fp, const char* prefix, const char* reg_func_name)
{
    unsigned int rand = random();

    fprintf(fp, "\nextern void %s (struct model_patch *);\n", reg_func_name);
    fprintf(fp, "\n\nstatic void __attribute__((constructor)) %s_reg_%u (void)\n", prefix, rand);
    fprintf(fp, "{\n");
    fprintf(fp, "\t%s (&%s_patch);\n", reg_func_name, prefix);
    fprintf(fp, "}\n");

    return 0;
}

static int create_c_file(const char* model_name, int patch_idx, char* addr, int offset, int size, int total_size,
                         uint32_t vendor_id, uint32_t nn_id, const char* reg_func)
{
    char* start = addr + offset;
    FILE* fp;
    const char* dir_name = "./model_src";

    char* file_name = ( char* )malloc(strlen(dir_name) + strlen(model_name) + 64);

    sprintf(file_name, "%s/%s_%d.c", dir_name, model_name, patch_idx);

    // open file
    fp = fopen(file_name, "w");

    if(fp == NULL)
    {
        LOG_ERROR() << "cannot create file: " << file_name << "\n";
        free(file_name);
        return -1;
    }

    LOG_INFO() << "file: " << file_name << " created\n";

    struct model_patch header;

    strncpy(header.model_name, model_name, MAX_MODEL_NAME_LEN);

    header.model_name[MAX_MODEL_NAME_LEN] = 0x0;

    header.vendor_id = vendor_id;
    header.nn_id = nn_id;
    header.total_size = total_size;
    header.patch_off = offset;
    header.patch_size = size;

    sprintf(file_name, "%s_%d", model_name, patch_idx);

    create_patch_data_blob(fp, start, size, file_name);
    create_patch_header(fp, &header, file_name);
    create_patch_registry(fp, file_name, reg_func);

    fclose(fp);

    free(file_name);

    return 0;
}

void PatchSerializer::GenerateIOV(void* buf, uint32_t vendor_id, uint32_t nn_id)
{
    unsigned int* p = ( unsigned int* )buf;

    for(int i = 0; i < 8; i++)
    {
        p[i] = p[i] ^ ((i & 0x1) ? vendor_id : nn_id);
    }
}

bool PatchSerializer::SaveToSource(const char* model_name, void* addr, int size)
{
    unsigned int patch_number = GetPatchNumber();
    const char* reg_func = GetRegisterFuncName();
    uint32_t vendor_id = GetVendorId();
    uint32_t nn_id = GetNNId();

    const char* p = model_name + strlen(model_name) - 1;

    while(p != model_name)
    {
        if(*p != '/')
            p--;
        else
        {
            p++;
        }
    }

    unsigned int i;
    int step = size / patch_number;

    for(i = 0; i < patch_number; i++)
    {
        int patch_offset = i * step;
        int patch_size = step;

        if(i == patch_number - 1)
            patch_size = size - patch_offset;

        if(create_c_file(p, i, ( char* )addr, patch_offset, patch_size, size, vendor_id, nn_id, reg_func) < 0)
            break;
    }

    if(i == patch_number)
        return true;
    else
        return false;
}

std::vector<struct model_patch> PatchSerializer::GetModelByName(const std::string& model_name)
{
    std::vector<struct model_patch> ret;

    for(unsigned int i = 0; i < g_patch_list.size(); i++)
    {
        const struct model_patch& patch = g_patch_list[i];

        if(model_name == patch.model_name)
            ret.emplace_back(patch);
    }

    return ret;
}

}    // namespace TEngine

extern "C" void patch_serializer_register(struct model_patch*);

using namespace TEngine;

void patch_serializer_register(struct model_patch* patch)
{
    TEngine::g_patch_list.emplace_back(*patch);
}
