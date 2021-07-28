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

#ifndef _NVDLA_OS_INF_H_
#define _NVDLA_OS_INF_H_

#include <stddef.h>
#include <dirent.h>

#include "dlaerror.h"
#include "dlatypes.h"

#define NVDLA_OPEN_WRITE (0x1)
#define NVDLA_OPEN_READ (0x2)
#define NVDLA_OPEN_CREATE (0x4)
#define NVDLA_OPEN_APPEND (0x8)

/*
 * Thread structures
 */
struct NvDlaThreadRec {
    void *handle;
};
typedef struct NvDlaThreadRec NvDlaThread;
typedef struct NvDlaThreadRec* NvDlaThreadHandle;

typedef void (*NvDlaThreadFunction)(void *args);

/*
 * Files and directory structures.
 */
typedef enum {
    NvDlaFileType_Unknown = 0,
    NvDlaFileType_File,
    NvDlaFileType_Directory,
    NvDlaFileType_Fifo,
    NvDlaFileType_CharacterDevice,
    NvDlaFileType_BlockDevice,

    NvDlaFileType_Force32 = 0x7FFFFFFF
} NvDlaFileType;

struct NvDlaStatTypeRec
{
    NvU64 size;
    NvDlaFileType type;

    NvU64 mtime;
};
typedef struct NvDlaStatTypeRec NvDlaStatType;

enum NvDlaSeek {
    NvDlaSeek_Set,
    NvDlaSeek_Cur,
    NvDlaSeek_End
};
typedef enum NvDlaSeek NvDlaSeekEnum;

struct NvDlaFileRec {
    int fd;
};
typedef struct NvDlaFileRec NvDlaFile;
typedef struct NvDlaFileRec* NvDlaFileHandle;

struct NvDlaDirRec {
    DIR *dir;
};
typedef struct NvDlaDirRec NvDlaDir;
typedef struct NvDlaDirRec* NvDlaDirHandle;

#ifdef __cplusplus
extern "C" {
#endif

/*
 * General OS related calls.
 */
void *NvDlaAlloc(size_t size);
void NvDlaFree(void *ptr);

void NvDlaDebugPrintf( const char *format, ... );

NvU32 NvDlaGetTimeMS(void);
void NvDlaSleepMS(NvU32 msec);

/*
 * Thread related functions
 */
NvDlaError
NvDlaThreadCreate( NvDlaThreadFunction function, void *args,
    NvDlaThreadHandle *thread);
void NvDlaThreadJoin(NvDlaThreadHandle thread);
void NvDlaThreadYield(void);

/*
 * File and directory operations
 */
NvDlaError NvDlaStat(const char *filename, NvDlaStatType *stat);
NvDlaError NvDlaMkdir(char *dirname);
NvDlaError NvDlaFremove(const char *filename);

NvDlaError NvDlaFopen(const char *path, NvU32 flags,
    NvDlaFileHandle *file);
void NvDlaFclose(NvDlaFileHandle stream);
NvDlaError NvDlaFwrite(NvDlaFileHandle stream, const void *ptr, size_t size);
NvDlaError NvDlaFread(NvDlaFileHandle stream, void *ptr,
                size_t size, size_t *bytes);
NvDlaError NvDlaFseek(NvDlaFileHandle file, NvS64 offset, NvDlaSeekEnum whence);
NvDlaError NvDlaFstat(NvDlaFileHandle file, NvDlaStatType *stat);
NvU64 NvDlaStatGetSize(NvDlaStatType *stat);
NvDlaError NvDlaFgetc(NvDlaFileHandle stream, NvU8 *c);
void NvDlaMemset(void *s, NvU8 c, size_t size);

NvDlaError NvDlaOpendir(const char *path, NvDlaDirHandle *dir);
NvDlaError NvDlaReaddir(NvDlaDirHandle dir, char *name, size_t size);
void NvDlaClosedir(NvDlaDirHandle dir);
#ifdef __cplusplus
}
#endif

#endif /* end of _NVDLA_OS_INF_H_ */
