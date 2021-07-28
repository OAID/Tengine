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

#ifndef NVDLA_UTILS_DLAIMAGE_H
#define NVDLA_UTILS_DLAIMAGE_H

#include <stdbool.h>
#include "dlaerror.h"
#include "dlatypes.h"

#include <sstream>

class NvDlaImage
{
public:
    typedef enum _PixelFormat
    {
        T_R8 = 0,
        T_R8_I = 1,
        T_R10 = 2,
        T_R12 = 3,
        T_R16 = 4,
        T_R16_I = 5,
        T_R16_F = 6,
        T_A16B16G16R16 = 7,
        T_A16B16G16R16_F = 8,
        T_X16B16G16R16 = 9,
        T_A16Y16U16V16 = 10,
        T_A16Y16U16V16_F = 11,
        T_V16U16Y16A16 = 12,
        T_A8B8G8R8 = 13,
        T_A8R8G8B8 = 14,
        T_B8G8R8A8 = 15,
        T_R8G8B8A8 = 16,
        T_X8B8G8R8 = 17,
        T_X8R8G8B8 = 18,
        T_B8G8R8X8 = 19,
        T_R8G8B8X8 = 20,
        T_A2B10G10R10 = 21,
        T_A2R10G10B10 = 22,
        T_B10G10R10A2 = 23,
        T_R10G10B10A2 = 24,
        T_A2Y10U10V10 = 25,
        T_V10U10Y10A2 = 26,
        T_A8Y8U8V8 = 27,
        T_V8U8Y8A8 = 28,
        T_Y8___U8V8_N444 = 29,
        T_Y8___V8U8_N444 = 30,
        T_Y10___U10V10_N444 = 31,
        T_Y10___V10U10_N444 = 32,
        T_Y12___U12V12_N444 = 33,
        T_Y12___V12U12_N444 = 34,
        T_Y16___U16V16_N444 = 35,
        T_Y16___V16U16_N444 = 36,

        D_F8_CHW_I = 37,
        D_F16_CHW_I = 38,
        D_F16_CHW_F = 39,

        D_F8_CxHWx_x32_I =  40,
        D_F8_CxHWx_x8_I =  41,
        D_F16_CxHWx_x16_I = 42,
        D_F16_CxHWx_x16_F = 43,

        D_F32_CHW_F = 44,
        D_F32_CxHWx_x8_F = 45,

        T_R8G8B8 = 46,
        T_B8G8R8 = 47,
    } PixelFormat;

    typedef enum _PixelFormatType
    {
        UINT = 0,
        INT = 1,
        IEEEFP = 2,
        UNKNOWN = 3
    } PixelFormatType;

    static const NvU32 ms_version;

    struct Metadata
    {
        PixelFormat surfaceFormat;
        NvU32 width;
        NvU32 height;
        NvU32 channel;

        NvU32 lineStride;
        NvU32 surfaceStride;
        NvU32 size;
    } m_meta;

    void* m_pData;

    NvS8 getBpe() const;
    PixelFormatType getPixelFormatType() const;
    NvS32 getAddrOffset(NvU32 w, NvU32 h, NvU32 c) const;
    NvDlaError printInfo() const;
    NvDlaError printBuffer(bool showBorders) const;

    NvDlaError serialize(std::stringstream& sstream, bool stableHash) const;
    NvDlaError deserialize(std::stringstream& sstream);
    NvDlaError packData(std::stringstream& sstream, bool stableHash) const;
    NvDlaError unpackData(std::stringstream& sstream);
};

#endif // NVDLA_UTILS_DLAIMAGE_H
