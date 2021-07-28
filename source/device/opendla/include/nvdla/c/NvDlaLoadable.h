/*
 * Copyright (c) 2017-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef NVDLA_LOADABLE_C_H
#define NVDLA_LOADABLE_C_H

#include "nvdla/c/NvDlaType.h"

#define NVDLA_LOADABLE_INTERFACE_NONE 0U
#define NVDLA_LOADABLE_INTERFACE_DLA1 1U
#define NVDLA_LOADABLE_INTERFACE_EMU1 2U

#define NVDLA_LOADABLE_SUB_INTERFACE_DLA1_NONE  0U
#define NVDLA_LOADABLE_SUB_INTERFACE_DLA1_ADDR0 1U
#define NVDLA_LOADABLE_SUB_INTERFACE_DLA1_DEPS  2U
#define NVDLA_LOADABLE_SUB_INTERFACE_DLA1_OPS   3U
#define NVDLA_LOADABLE_SUB_INTERFACE_DLA1_SURFS 4U
#define NVDLA_LOADABLE_SUB_INTERFACE_DLA1_LUTS  5U

#define NVDLA_LOADABLE_SUB_INTERFACE_EMU1_NONE  0U
#define NVDLA_LOADABLE_SUB_INTERFACE_EMU1_ADDR0 1U
/* #define NVDLA_LOADABLE_SUB_INTERFACE_EMU1_DEPS  2U */
#define NVDLA_LOADABLE_SUB_INTERFACE_EMU1_OPS   3U
#define NVDLA_LOADABLE_SUB_INTERFACE_EMU1_SURFS 4U
/* #define NVDLA_LOADABLE_SUB_INTERFACE_EMU1_LUTS  5U */


#define NVDLA_LOADABLE_MEMORY_DOMAIN_SYSMEM 0U
#define NVDLA_LOADABLE_MEMORY_DOMAIN_SRAM 1U

#define NVDLA_LOADABLE_MEMORY_FLAGS_NONE    0U
#define NVDLA_LOADABLE_MEMORY_FLAGS_ALLOC   1U
#define NVDLA_LOADABLE_MEMORY_FLAGS_SET     2U
#define NVDLA_LOADABLE_MEMORY_FLAGS_INPUT   4U
#define NVDLA_LOADABLE_MEMORY_FLAGS_OUTPUT  8U
#define NVDLA_LOADABLE_MEMORY_FLAGS_DEBUG  16U

#define NVDLA_LOADABLE_EVENT_OP_WAIT   0U
#define NVDLA_LOADABLE_EVENT_OP_SIGNAL 1U

#define NVDLA_LOADABLE_TENSOR_DESC_NUM_STRIDES 8U    /* a little room to grow */

struct NvDlaLoadableI;
typedef struct NvDlaLoadable
{
    void *self;
    const struct NvDlaLoadableI *i;
} NvDlaLoadable;

typedef struct NvDlaLoadableVersion
{
    NvU8 major;
    NvU8 minor;
    NvU8 subMinor;
} NvDlaLoadableVersion;

typedef struct NvDlaLoadableMemoryListEntry
{
    NvU16 id;
    NvU64 size;
    NvU32 alignment;
    NvU8  domain;
    NvU8  flags;
    NvU16 bindId;
    NvU16 tensorDescId;
    size_t numContents;
    NvU8 **contents;
    NvU32 *offsets;
} NvDlaLoadableMemoryListEntry;

typedef struct NvDlaLoadableEventListEntry
{
    NvU16 id;
    NvU16 target;
    NvU8 op;
    NvU32 val;
} NvDlaLoadableEventListEntry;

typedef struct NvDlaLoadableTaskListEntry
{
    NvU16 id;
    NvU32 interface;
    NvS16 instance;
    size_t numPreActions;
    NvU16 *preActions;   // [event id]...
    size_t numPostActions;
    NvU16 *postActions;  // [event id]...
    NvU16 numAddressList;
    NvU16 *addressList; // [addr list id]...[addr list id]
} NvDlaLoadableTaskListEntry;

typedef struct NvDlaLoadableAddressListEntry
{
    NvU16 id;
    NvU16 memId;
    NvU64 size;
    NvU64 offset;
} NvDlaLoadableAddressListEntry;

typedef struct NvDlaLoadableBlob
{
    char *name;
    NvU64 size;
    NvU32 interface;
    NvU32 sub_interface;
    NvDlaLoadableVersion version;
} NvDlaLoadableBlob;


struct NvDlaLoadableI
{
    const char * (*getName)(NvDlaLoadable);
};

#ifdef __cplusplus
extern "C" {
#endif


#ifdef __cplusplus
}
#endif

#endif /* NVDLA_LOADABLE_C_H */
