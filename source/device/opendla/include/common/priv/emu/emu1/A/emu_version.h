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

#ifndef NVDLA_PRIV_EMU_EMU1_A_EMU_VERSION_H
#define NVDLA_PRIV_EMU_EMU1_A_EMU_VERSION_H

#define EMULATOR_VERSION_MAJOR      0x00
#define EMULATOR_VERSION_MINOR      0x00
#define EMULATOR_VERSION_SUBMINOR   0x01

static inline NvU32 emu_version(void)
{
    return (NvU32)(((EMULATOR_VERSION_MAJOR & 0xff) << 16) |
                   ((EMULATOR_VERSION_MINOR & 0xff) << 8) |
                   ((EMULATOR_VERSION_SUBMINOR & 0xff)));
}

//
// gerrit change representing delivery of the emulator and associated headers
//
inline const std::string emu_gerrit_change() { return std::string("Id2fcf23860dd79ff52420ff2661918b43b58dc32"); }
inline const std::string emu_gerrit_review() { return std::string("http://git-master.nvidia.com/r/1313572");    }


#endif // NVDLA_PRIV_EMU_EMU1_A_EMU_VERSION_H
