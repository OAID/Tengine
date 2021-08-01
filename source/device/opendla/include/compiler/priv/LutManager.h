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

#ifndef NVDLA_PRIV_LUT_MANAGER_H
#define NVDLA_PRIV_LUT_MANAGER_H

#include "EngineAST.h"

namespace nvdla {
namespace priv {

class LutManager
{
public:
    LutManager();
    virtual ~LutManager();

    typedef NvS16 LutHandle;

protected:
    enum LutTypeEnum
    {
        LUT_TYPE_LRN,
        LUT_TYPE_SIGMOID,
        LUT_TYPE_TANH
    };

    struct LRNParams
    {
        NvU32 localSize;
        NvF32 alpha;
        NvF32 beta;
        NvF32 k;
    };

    struct LutParams
    {
        LutTypeEnum type;
        surface::SurfacePrecisionEnum precision;
        LRNParams lrnParams;
    };

    NvDlaError genKey(const LutParams* params, std::string* key) const;
    NvDlaError getHandle(const LutParams* lutParams, LutHandle* hLut);

public:
    NvDlaError registerLRN(surface::SurfacePrecisionEnum precision, NvU32 localSize, NvF32 alpha, NvF32 beta, NvF32 k, LutHandle* hLut);
    NvDlaError registerSigmoid(surface::SurfacePrecisionEnum precision, LutHandle* hLut);
    NvDlaError registerTanh(surface::SurfacePrecisionEnum precision, LutHandle* hLut);

    NvS16 getIndex(LutHandle hLut) const;
    NvS16 getNumRegisteredLuts();

    NvDlaError writeLutData(NvU16 lutSlot, DLALUTParamAccessor lutAcc);
    NvDlaError writeLRNData(const LutParams* lutParams, DLALUTParamAccessor lutAcc);
    NvDlaError writeSigmoidData(const LutParams* lutParams, DLALUTParamAccessor lutAcc);
    NvDlaError writeTanhData(const LutParams* lutParams, DLALUTParamAccessor lutAcc);

protected:
    std::map<std::string, LutHandle> m_lutHandles;
    std::map<LutHandle, LutParams> m_lutParams;
    LutHandle m_hNextFree;
};

}; // namespace priv
}; // namespace nvdla

#endif // NVDLA_PRIV_LUT_MANAGER_H
