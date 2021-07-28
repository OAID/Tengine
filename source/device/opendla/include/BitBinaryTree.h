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

#ifndef NVDLA_UTILS_BIT_BINARY_TREE_H
#define NVDLA_UTILS_BIT_BINARY_TREE_H

#include <stdbool.h>
#include "dlaerror.h"
#include "dlatypes.h"

typedef struct NvDlaBitBinaryTreeClass NvDlaBitBinaryTreeClass;
typedef struct NvDlaBitBinaryTreeInst NvDlaBitBinaryTreeInst;

struct NvDlaBitBinaryTreeClass
{
    // static methods
    NvDlaError (*construct)(NvDlaBitBinaryTreeInst* self, NvU8 numLevels);
    NvDlaError (*destruct)(NvDlaBitBinaryTreeInst* self);

    bool (*get)(const NvDlaBitBinaryTreeInst* self, NvU8 level, NvU32 index);
    void (*set)(NvDlaBitBinaryTreeInst* self, NvU8 level, NvU32 index, bool value);
    bool (*flip)(NvDlaBitBinaryTreeInst* self, NvU8 level, NvU32 index);
    NvDlaError (*print)(const NvDlaBitBinaryTreeInst* self);
};

struct NvDlaBitBinaryTreeInst
{
    // instance members
    NvU8 numLevels;
    NvU8* treeStorage;
};


extern const NvDlaBitBinaryTreeClass NvDlaBitBinaryTree;

#endif // NVDLA_UTILS_BIT_BINARY_TREE_H
