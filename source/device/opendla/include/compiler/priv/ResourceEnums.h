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

#ifndef NVDLA_PRIV_RESOURCE_ENUMS_H
#define NVDLA_PRIV_RESOURCE_ENUMS_H

//
// the category to which a tensor surface desc belongs:
// EXTERNAL: network i/p and network o/p
// GLOBAL:   common tensors across images: weights/bias/etc
// LOCAL:    temp tensors passed between engines 'through mem'
// STREAM:   temp tensors passed between engines 'over the wire'
//
// used in dla-engine-ast's tensor surface desc context
//
#define TENSOR_CATEGORY_ENUMS(op)  \
    op(UNKNOWN_TENSOR,  0U)    \
    op(EXTERNAL_TENSOR, 1U)    \
    op(GLOBAL_TENSOR,   2U)    \
    op(LOCAL_TENSOR,    3U)    \
    op(STREAM_TENSOR,   4U)

//
// the type of data a memory buffer represents
// Types:
//  - IMAGE_DATA:       image input data
//  - FEATURE_DATA:     feature data
//  - KERNEL_WEIGHTS:   kernel weights
//  - BIAS_WEIGHTS:     bias weights
//
// used in the dla-engine-ast's mem mgmnt context
//
#define MEMORY_BUFFER_TYPE_ENUMS(op) \
    op(bIMAGE_DATA,      0U)       \
    op(bFEATURE_DATA,    1U)       \
    op(bKERNEL_WEIGHTS,  2U)       \
    op(bBIAS_WEIGHTS,    3U)

//
// the memory destination where a buffer resides
// Types:
//  - DRAM:   aka sysmem
//  - CVSRAM: dla/pva shared fast local mem
//  - CBUFF:  convolution buffer
//  - STREAM: stream/on-the-fly
//
// used in the dla-engine-ast context
//
#define MEMORY_LOCATION_ENUMS(op) \
    op(lUNKNOWN, 0U)       \
    op(lDRAM,    1U)       \
    op(lCVSRAM,  2U)       \
    op(lCBUFF,   3U)       \
    op(lSTREAM,  4U)

//
// the type of resource-pool a RM manages that contains all buffers
// and their descriptors; along with the description of the various surfaces
// looking into different sections of those buffers.
//
// Types:
//  - GLOBAL_DRAM_POOL:     pool managing kernel/bias weight buffers in DRAM
//  - LOCAL_DRAM_POOL:      pool managing per task temporary data buffers in DRAM
//  - LOCAL_CVSRAM_POOL:    pool managing weights and feature data buffers in CV-SRAM
//
// !!!                                                       !!!
// !!! NOTE: we make use of the fact that global is := 0     !!!
// !!! and that the others form array indices to follow...   !!!
// !!!                                                       !!!
//
// used in the dla-engine-ast context
//
#define POOL_TYPE_ENUMS(op)  \
    op(GLOBAL_DRAM_POOL,    0U) \
    op(LOCAL_DRAM_POOL,     1U) \
    op(LOCAL_CVSRAM_POOL,   2U)


#endif  /* NVDLA_PRIV_RESOURCE_ENUMS_H */
