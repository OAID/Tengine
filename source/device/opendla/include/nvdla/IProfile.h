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

#ifndef NVDLA_I_PROFILE_H
#define NVDLA_I_PROFILE_H

#include "nvdla/IType.h"

namespace nvdla
{

class IWisdom;
class ILoadable;

class IProfile
{
public:
    virtual const char* getName() const = 0;

    virtual NvDlaError getNumLoadables(int *) const =  0;

    // if name.length() == 0 or index < 0 then they are ignored for the purpose of lookup
    virtual NvDlaError getLoadable(const std::string &name, int index, ILoadable **) = 0;

    virtual NvDlaError setNetworkInputSurfaceFormat(nvdla::PixelFormat) = 0;
    virtual NvDlaError setNetworkOutputSurfaceFormat(nvdla::PixelFormat) = 0;
    virtual NvDlaError setNetworkInputPixelMapping(nvdla::PixelMapping) = 0;
    virtual NvDlaError setNetworkInputDataFormat(const nvdla::DataFormat) = 0;
    virtual NvDlaError setNetworkOutputDataFormat(const nvdla::DataFormat) = 0;
    virtual NvDlaError setNetworkInputPixelOffX(NvU32) = 0;
    virtual NvDlaError setNetworkInputPixelOffY(NvU32) = 0;

    virtual NvDlaError setCanCompressWeights(bool) = 0;
    virtual NvDlaError setCanWinograd(bool) = 0;
    virtual NvDlaError setCanSDPMergeMathOps(bool) = 0;
    virtual NvDlaError setCanSDPFuseSubEngineOps(bool) = 0;
    virtual NvDlaError setCanSDPBustNOPs(bool) = 0;
    virtual NvDlaError setCanSDPFuseVerticalOps(bool) = 0;
    virtual NvDlaError setCanSDPPDPOnFly(bool) = 0;
    virtual NvDlaError setUseCVSRAMAllocate(bool) = 0;
    virtual NvDlaError setUseMemPool(bool) = 0;
    virtual NvDlaError setUseReusePooledMemory(bool) = 0;
    virtual NvDlaError setCopyOutDebugSurfaces(bool) = 0;
    virtual NvDlaError setUseGreedyEviction(bool) = 0;
    virtual NvDlaError setCONVDataBanksAllotted(NvU32) = 0;
    virtual NvDlaError setCONVWeightBanksAllotted(NvU32) = 0;
    virtual NvDlaError setGlobalDRAMSize(NvU64) = 0;
    virtual NvDlaError setLocalDRAMSize(NvU64) = 0;
    virtual NvDlaError setLocalCVSRAMSize(NvU64) = 0;
    virtual NvDlaError setMultiBatchSize(NvU32) = 0;
    virtual NvDlaError setCanIMGPostChnlExtend(bool) = 0;
    virtual NvDlaError setComputePrecision(nvdla::DataType) = 0;
    virtual NvDlaError setTensorScalingMode(nvdla::TensorScalingMode) = 0;
    virtual NvDlaError setQuantizationMode(nvdla::QuantizationMode) = 0;


    struct IGlobalParams {
        NvU32 pixelOffsetX;
        NvU32 pixelOffsetY;
        nvdla::PixelFormat  inputPixelFormat;
        nvdla::DataFormat   inputDataFormat;     // NCHW default
        nvdla::PixelMapping inputPixelMapping;
        nvdla::PixelFormat  outputPixelFormat;
        nvdla::DataFormat   outputDataFormat;    // NCHW default

        IGlobalParams() :
            pixelOffsetX(0),
            pixelOffsetY(0),
            inputPixelFormat(nvdla::PixelFormat::FEATURE),
            inputDataFormat(nvdla::DataFormat::NCHW),
            inputPixelMapping(nvdla::PixelMapping::PITCH_LINEAR),
            outputPixelFormat(nvdla::PixelFormat::FEATURE),
            outputDataFormat(nvdla::DataFormat::NCHW)
        { }
    };

    struct ICompileParams {
        bool    canCompressWeights;
        bool    canWinograd;
        NvU32   convWeightBanksAllotted;
        NvU32   convDataBanksAllotted;
        bool    canSdpPdpOnFly;
        bool    canSdpMergeMathOps;
        bool    canSdpFuseSubEngineOps;
        bool    canSdpBustNOPs;
        bool    canSdpFuseVerticalOps;
        bool    useCvsramAllocate;
        bool    useMemPool;
        bool    useReusePooledMemory;
        bool    greedyEviction;
        bool    copyOutDebugSurfaces;
        NvU64   globalDramSize;
        NvU64   localDramSize;
        NvU64   localCvsramSize;
        NvU32   multiBatchSize;
        bool    canImgPostChnlExtend;
        nvdla::DataType computePrecision;
        nvdla::TensorScalingMode tensorScalingMode;
        nvdla::QuantizationMode quantizationMode;

        ICompileParams() :
            canCompressWeights(false),
            canWinograd(false),
            convWeightBanksAllotted(8),
            convDataBanksAllotted(8),
            canSdpPdpOnFly(false),
            canSdpMergeMathOps(false),
            canSdpFuseSubEngineOps(false),
            canSdpBustNOPs(false),
            canSdpFuseVerticalOps(false),
            useCvsramAllocate(false),
            useMemPool(false),
            useReusePooledMemory(false),
            greedyEviction(false),
            copyOutDebugSurfaces(false),
            globalDramSize(1LLU << 28),
            localDramSize(1LLU << 30),
            localCvsramSize(1LLU << 20),
            multiBatchSize(0),
            canImgPostChnlExtend(false),
            computePrecision(nvdla::DataType::HALF),
            tensorScalingMode(nvdla::TensorScalingMode::NONE),
            quantizationMode(nvdla::QuantizationMode::NONE)
        { }
    };

    virtual NvDlaError initGlobalParams(IGlobalParams*) = 0;
    virtual NvDlaError initCompileParams(ICompileParams*) = 0;
    virtual void initWithDefaultProfile() = 0;

protected:
    IProfile();
    virtual ~IProfile();
};


} // nvdla


#endif // NVDLA_I_PROFILER_H
