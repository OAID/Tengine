/*
 * Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#ifndef _NVDLA_INF_H_
#define _NVDLA_INF_H_

#include "dlaerror.h"
#include "dlatypes.h"

#define NVDLA_MAX_BUFFERS_PER_TASK (6144)

struct NvDlaMemDescRec{
    void *handle;
    NvU32 offset;
};
typedef struct NvDlaMemDescRec NvDlaMemDesc;

struct NvDlaTaskRec {
    NvU64 task_id;
    NvU32 num_addresses;
    NvDlaMemDesc address_list[NVDLA_MAX_BUFFERS_PER_TASK];
};
typedef struct NvDlaTaskRec NvDlaTask;

typedef enum NvDlaHeap {
    NvDlaHeap_System,
    NvDlaHeap_SRAM,
} NvDlaHeap;

#ifdef __cplusplus
extern "C" {
#endif

NvDlaError NvDlaInitialize(void **session_handle);
void NvDlaDestroy(void *session_handle);

NvDlaError NvDlaOpen(void *session_handle, NvU32 instance, void **device_handle);
void NvDlaClose(void *device_handle);

NvDlaError NvDlaSubmit(void *session_handle, void *device_handle, NvDlaTask *tasks, NvU32 num_tasks);

NvDlaError NvDlaAllocMem(void *session_handle, void *device_handle,
                         void **mem_handle, void **pData, NvU32 size,
                         NvDlaHeap heap);
NvDlaError NvDlaFreeMem(void *session_handle, void *device_handle, void *mem_handle,
                        void *pData, NvU32 size);

#ifdef __cplusplus
}
#endif

#endif /* end of _NVDLA_INF_H_ */
