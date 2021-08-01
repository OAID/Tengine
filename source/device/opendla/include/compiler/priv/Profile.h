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

#ifndef NVDLA_PRIV_PROFILE_H
#define NVDLA_PRIV_PROFILE_H

#include <map>
#include <vector>

#include "nvdla/IProfile.h"

#include "Surface.h"
#include "Type.h"

namespace nvdla {

class ILayerProfile;
class ITensorProfile;

namespace priv {

class WisdomContainerEntry;

class Profile;

class ProfileFactory
{
public:
    typedef PrivPair<IProfile*, Profile*> ProfilePrivPair;
    static ProfilePrivPair newProfile();
    static Profile* priv(IProfile*);
    static IProfile* i(Profile*);
    static IProfile* self(void*);
    static IProfile* deserializeFrom(WisdomContainerEntry*);

protected:
    static BiMap<IProfile*, Profile*> s_priv;
    static BiMap<void*, IProfile*> s_self;
    static IProfile* deserializeProfile(WisdomContainerEntry*);
};

class Profile : public IProfile
{
public: // externally facing
    Profile()
    {
    }
    virtual ~Profile()
    {
    }

    virtual const char* getName() const;

    virtual NvDlaError getNumLoadables(int*) const;
    virtual NvDlaError getLoadable(const std::string& name, int index, ILoadable**);

public: // internally facing
    void setBasicProfile();
    void setDefaultProfile();
    void setPerformanceProfile();
    void setFastMathProfile();
    virtual void setName(const char*);

    virtual NvU16 getFactoryType() const;
    virtual bool serializeTo(WisdomContainerEntry*) const;
    virtual bool deserializeFrom(WisdomContainerEntry*);

    virtual NvDlaError insertLoadable(const std::string& name, int index, ILoadable* i_loadable);

    virtual NvDlaError setNetworkInputSurfaceFormat(nvdla::PixelFormat);
    surface::SurfaceFormat networkInputSurfaceFormat() const
    {
        return m_globalParams.m_NwInSurfFormat;
    }

    virtual NvDlaError setNetworkOutputSurfaceFormat(nvdla::PixelFormat);
    surface::SurfaceFormat networkOutputSurfaceFormat() const
    {
        return m_globalParams.m_NwOutSurfFormat;
    }

    virtual NvDlaError setNetworkInputPixelMapping(nvdla::PixelMapping);
    surface::PixelMapping networkInputPixelMapping() const
    {
        return m_globalParams.m_NwInPixelMapping;
    }

    virtual NvDlaError setNetworkInputDataFormat(const nvdla::DataFormat nidf)
    {
        m_globalParams.m_NwInDataFormat = nidf;
        return NvDlaSuccess;
    }
    nvdla::DataFormat networkInputDataFormat() const
    {
        return m_globalParams.m_NwInDataFormat;
    }

    virtual NvDlaError setNetworkOutputDataFormat(const nvdla::DataFormat nodf)
    {
        m_globalParams.m_NwOutDataFormat = nodf;
        return NvDlaSuccess;
    }
    nvdla::DataFormat networkOutputDataFormat() const
    {
        return m_globalParams.m_NwOutDataFormat;
    }

    virtual NvDlaError setNetworkInputPixelOffX(NvU32 offX)
    {
        m_globalParams.m_NwInPixelOffX = offX;
        return NvDlaSuccess;
    }
    NvU32 networkInputPixelOffX() const
    {
        return m_globalParams.m_NwInPixelOffX;
    }

    virtual NvDlaError setNetworkInputPixelOffY(NvU32 offY)
    {
        m_globalParams.m_NwInPixelOffY = offY;
        return NvDlaSuccess;
    }
    NvU32 networkInputPixelOffY() const
    {
        return m_globalParams.m_NwInPixelOffY;
    }

    virtual NvDlaError setCanCompressWeights(bool ccp)
    {
        m_compileParams.m_canCompressWeights = ccp;
        return NvDlaSuccess;
    }
    bool canCompressWeights() const
    {
        return m_compileParams.m_canCompressWeights;
    }

    virtual NvDlaError setCanWinograd(bool cw)
    {
        m_compileParams.m_canWinograd = cw;
        return NvDlaSuccess;
    }
    bool canWinograd() const
    {
        return m_compileParams.m_canWinograd;
    }

    virtual NvDlaError setCanSDPFuseVerticalOps(bool csfv)
    {
        m_compileParams.m_canSDPFuseVerticalOps = csfv;
        return NvDlaSuccess;
    }
    bool canSDPFuseVerticalOps() const
    {
        return m_compileParams.m_canSDPFuseVerticalOps;
    }

    virtual NvDlaError setCanSDPMergeMathOps(bool csmm)
    {
        m_compileParams.m_canSDPMergeMathOps = csmm;
        return NvDlaSuccess;
    }
    bool canSDPMergeMathOps() const
    {
        return m_compileParams.m_canSDPMergeMathOps;
    }

    virtual NvDlaError setCanSDPFuseSubEngineOps(bool csmm)
    {
        m_compileParams.m_canSDPFuseSubEngineOps = csmm;
        return NvDlaSuccess;
    }
    bool canSDPFuseSubEngineOps() const
    {
        return m_compileParams.m_canSDPFuseSubEngineOps;
    }

    virtual NvDlaError setCanSDPBustNOPs(bool csbn)
    {
        m_compileParams.m_canSDPBustNOPs = csbn;
        return NvDlaSuccess;
    }
    bool canSDPBustNOPs() const
    {
        return m_compileParams.m_canSDPBustNOPs;
    }

    virtual NvDlaError setCanSDPPDPOnFly(bool cspo)
    {
        m_compileParams.m_canSDPPDPOnFly = cspo;
        return NvDlaSuccess;
    }
    bool canSDPPDPOnFly() const
    {
        return m_compileParams.m_canSDPPDPOnFly;
    }

    virtual NvDlaError setUseCVSRAMAllocate(bool uca)
    {
        m_compileParams.m_useCVSRAMAllocate = uca;
        return NvDlaSuccess;
    }
    bool useCVSRAMAllocate() const
    {
        return m_compileParams.m_useCVSRAMAllocate;
    }

    virtual NvDlaError setUseMemPool(bool ump)
    {
        m_compileParams.m_useMemPool = ump;
        return NvDlaSuccess;
    }
    bool useMemPool() const
    {
        return m_compileParams.m_useMemPool;
    }

    virtual NvDlaError setUseReusePooledMemory(bool urpm)
    {
        m_compileParams.m_useReusePooledMemory = urpm;
        return NvDlaSuccess;
    }
    bool useReusePooledMemory() const
    {
        return m_compileParams.m_useReusePooledMemory;
    }

    virtual NvDlaError setCopyOutDebugSurfaces(bool cods)
    {
        m_compileParams.m_copyOutDebugSurfaces = cods;
        return NvDlaSuccess;
    }
    bool copyOutDebugSurfaces() const
    {
        return m_compileParams.m_copyOutDebugSurfaces;
    }

    virtual NvDlaError setUseGreedyEviction(bool uge)
    {
        m_compileParams.m_useGreedyEviction = uge;
        return NvDlaSuccess;
    }
    bool useGreedyEviction() const
    {
        return m_compileParams.m_useGreedyEviction;
    }

    virtual NvDlaError setCONVDataBanksAllotted(NvU32 db)
    {
        m_compileParams.m_CONVDataBanksAllotted = db;
        return NvDlaSuccess;
    }
    NvU32 dataBanksAlloted() const
    {
        return m_compileParams.m_CONVDataBanksAllotted;
    }

    virtual NvDlaError setCONVWeightBanksAllotted(NvU32 wb)
    {
        m_compileParams.m_CONVWeightBanksAllotted = wb;
        return NvDlaSuccess;
    }
    NvU32 weightBanksAlloted() const
    {
        return m_compileParams.m_CONVWeightBanksAllotted;
    }

    virtual NvDlaError setGlobalDRAMSize(NvU64 gds)
    {
        m_compileParams.m_globalDRAMSize = gds;
        return NvDlaSuccess;
    }
    NvU64 globalDRAMPoolSize() const
    {
        return m_compileParams.m_globalDRAMSize;
    }

    virtual NvDlaError setLocalDRAMSize(NvU64 lds)
    {
        m_compileParams.m_localDRAMSize = lds;
        return NvDlaSuccess;
    }
    NvU64 localDRAMPoolSize() const
    {
        return m_compileParams.m_localDRAMSize;
    }

    virtual NvDlaError setLocalCVSRAMSize(NvU64 lcs)
    {
        m_compileParams.m_localCVSRAMSize = lcs;
        return NvDlaSuccess;
    }
    NvU64 localCVSRAMPoolSize() const
    {
        return m_compileParams.m_localCVSRAMSize;
    }

    virtual NvDlaError setMultiBatchSize(NvU32 mbs)
    {
        m_compileParams.m_multiBatchSize = mbs;
        return NvDlaSuccess;
    }
    NvU32 multiBatchSize() const
    {
        return m_compileParams.m_multiBatchSize;
    }

    virtual NvDlaError setCanIMGPostChnlExtend(bool cipce)
    {
        m_compileParams.m_canIMGPostChnlExtend = cipce;
        return NvDlaSuccess;
    }
    bool canIMGPostChnlExtend() const
    {
        return m_compileParams.m_canIMGPostChnlExtend;
    }

    virtual NvDlaError setComputePrecision(nvdla::DataType cp);
    surface::SurfacePrecision computePrecision() const
    {
        return m_compileParams.m_computePrecision;
    }

    virtual NvDlaError setTensorScalingMode(nvdla::TensorScalingMode tsm);
    nvdla::TensorScalingMode tensorScalingMode() const
    {
        return m_compileParams.m_tensorScalingMode;
    }

    virtual NvDlaError setQuantizationMode(nvdla::QuantizationMode qm);
    nvdla::QuantizationMode quantizationMode() const
    {
        return m_compileParams.m_quantizationMode;
    }

    inline bool debug() const
    {
        return true;
    }

    virtual NvDlaError initGlobalParams(IGlobalParams* gp);
    virtual NvDlaError initCompileParams(ICompileParams* cp);
    virtual void initWithDefaultProfile();

    struct GlobalParams
    {
        NvU32 m_NwInPixelOffX;
        NvU32 m_NwInPixelOffY;
        nvdla::DataFormat m_NwInDataFormat;  // NCHW default
        nvdla::DataFormat m_NwOutDataFormat; // NCHW default
        surface::SurfaceFormat m_NwInSurfFormat;
        surface::SurfaceFormat m_NwOutSurfFormat;
        surface::PixelMapping m_NwInPixelMapping;

        GlobalParams()
            : m_NwInPixelOffX(0),
              m_NwInPixelOffY(0),
              m_NwInDataFormat(nvdla::DataFormat::NCHW),
              m_NwOutDataFormat(nvdla::DataFormat::NCHW),
              m_NwInSurfFormat(surface::SurfaceFormatEnum::NVDLA_FEATURE_DATA_FP16),
              m_NwOutSurfFormat(surface::SurfaceFormatEnum::NVDLA_FEATURE_DATA_FP16),
              m_NwInPixelMapping(surface::PixelMappingEnum::PITCH_LINEAR)
        {
        }
    };

    struct CompileParams
    {
        bool m_canCompressWeights;
        bool m_canWinograd;
        NvU32 m_CONVWeightBanksAllotted;
        NvU32 m_CONVDataBanksAllotted;
        bool m_canSDPPDPOnFly;
        bool m_canSDPMergeMathOps;
        bool m_canSDPFuseSubEngineOps;
        bool m_canSDPBustNOPs;
        bool m_canSDPFuseVerticalOps;
        bool m_useCVSRAMAllocate;
        bool m_useMemPool;
        bool m_useReusePooledMemory;
        bool m_copyOutDebugSurfaces;
        bool m_useGreedyEviction;
        NvU64 m_globalDRAMSize;
        NvU64 m_localDRAMSize;
        NvU64 m_localCVSRAMSize;
        NvU32 m_multiBatchSize;
        bool m_canIMGPostChnlExtend;
        surface::SurfacePrecision m_computePrecision;
        nvdla::TensorScalingMode m_tensorScalingMode;
        nvdla::QuantizationMode m_quantizationMode;

        CompileParams()
            : m_canCompressWeights(false),
              m_canWinograd(false),
              m_CONVWeightBanksAllotted(8),
              m_CONVDataBanksAllotted(8),
              m_canSDPPDPOnFly(false),
              m_canSDPMergeMathOps(false),
              m_canSDPFuseSubEngineOps(false),
              m_canSDPBustNOPs(false),
              m_canSDPFuseVerticalOps(false),
              m_useCVSRAMAllocate(false),
              m_useMemPool(false),
              m_useReusePooledMemory(false),
              m_copyOutDebugSurfaces(false),
              m_useGreedyEviction(false),
              m_globalDRAMSize(1LLU << 29),
              m_localDRAMSize(1LLU << 30),
              m_localCVSRAMSize(1LLU << 20),
              m_multiBatchSize(0),
              m_canIMGPostChnlExtend(true),
              m_computePrecision(surface::SurfacePrecisionEnum::NVDLA_PRECISION_FP16),
              m_tensorScalingMode(nvdla::TensorScalingMode::NONE),
              m_quantizationMode(nvdla::QuantizationMode::NONE)
        {
        }
    };

protected:
    std::string m_name;
    std::map<std::string, ILoadable*> m_loadablesByName;
    std::vector<ILoadable*> m_loadables;

    inline bool isBasicProfile()
    {
        return m_name == std::string("basic");
    }
    inline bool isDefaultProfile()
    {
        return m_name == std::string("default");
    }
    inline bool isPerformanceProfile()
    {
        return m_name == std::string("performance");
    }
    inline bool isFastMathProfile()
    {
        return m_name == std::string("fast-math");
    }

    GlobalParams m_globalParams;
    CompileParams m_compileParams;
};

} // namespace priv

} // namespace nvdla

#endif // NVDLA_PRIV_PROFILE_H
