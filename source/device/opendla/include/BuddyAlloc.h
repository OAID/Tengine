/*
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef NVDLA_UTILS_BUDDY_ALLOC_H
#define NVDLA_UTILS_BUDDY_ALLOC_H

#include <stdint.h>
#include "BitBinaryTree.h"

typedef struct NvDlaFreeBlock NvDlaFreeBlock;

struct NvDlaFreeBlock
{
    NvDlaFreeBlock* prev;
    NvDlaFreeBlock* next;
};

#define NVDLA_UTILS_BUDDY_ALLOC_MAX_NUM_BLOCKTYPE 32

typedef struct NvDlaBuddyAllocClass NvDlaBuddyAllocClass;
typedef struct NvDlaBuddyAllocInst NvDlaBuddyAllocInst;

struct NvDlaBuddyAllocClass
{
    // static methods
    NvDlaError (*construct)(NvDlaBuddyAllocInst* self,
            const void* poolData, NvU32 poolSize,
            NvU8 minElementSizeLog2);
    NvDlaError (*destruct)(NvDlaBuddyAllocInst* self);

    void* (*allocate)(NvDlaBuddyAllocInst* self, NvU32 size);
    NvDlaError (*deallocate)(NvDlaBuddyAllocInst* self, void* ptr);
};

struct NvDlaBuddyAllocInst
{
    // instance members
    const void* poolData;
    NvU32 poolSize;
    NvU8 maxElementSizeLog2;
    NvU8 minElementSizeLog2;

    NvDlaFreeBlock* freeHead[NVDLA_UTILS_BUDDY_ALLOC_MAX_NUM_BLOCKTYPE];

    NvDlaBitBinaryTreeInst* fxfData;
    NvDlaBitBinaryTreeInst* splitData;

    // debug info
    //NvDlaBitBinaryTree* allocData;
};


extern const NvDlaBuddyAllocClass NvDlaBuddyAlloc;

#endif // NVDLA_UTILS_BUDDY_ALLOC_H
