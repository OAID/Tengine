/*
 * Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef NVDLA_PRIV_TARGETCONFIG_H
#define NVDLA_PRIV_TARGETCONFIG_H

#include <map>
#include <vector>

#include "nvdla/ITargetConfig.h"

#include "Surface.h"
#include "Type.h"

namespace nvdla {

namespace priv {

class WisdomContainerEntry;

class TargetConfig;

class TargetConfigFactory
{
public:
    typedef PrivPair<ITargetConfig*, TargetConfig*> TargetConfigPrivPair;
    static TargetConfigPrivPair newTargetConfig();
    static TargetConfig* priv(ITargetConfig*);
    static ITargetConfig* i(TargetConfig*);
    static ITargetConfig* self(void*);

protected:
    static BiMap<ITargetConfig*, TargetConfig*> s_priv;
    static BiMap<void*, ITargetConfig*> s_self;
};

class TargetConfig : public ITargetConfig
{
public: // externally facing
    TargetConfig()
    {
    }
    virtual ~TargetConfig()
    {
    }

    virtual const char* getName() const;

public:
    virtual void setName(const char*);

    NvU32 atomicCSize() const
    {
        return m_targetConfigParams.m_atomicCSize;
    }
    NvU32 atomicKSize() const
    {
        return m_targetConfigParams.m_atomicKSize;
    }
    NvU32 memoryAtomicSize() const
    {
        return m_targetConfigParams.m_memoryAtomicSize;
    }
    NvU32 bufBankAllotted() const
    {
        return m_targetConfigParams.m_numConvBufBankAllotted;
    }
    NvU32 bufEntriesPerBank() const
    {
        return m_targetConfigParams.m_numConvBufEntriesPerBank;
    }
    NvU32 bufEntryWidth() const
    {
        return m_targetConfigParams.m_numConvBufEntryWidth;
    }
    NvU32 maxBatchSize() const
    {
        return m_targetConfigParams.m_maxBatchSize;
    }

    bool isWinogradCapable() const
    {
        return m_targetConfigParams.m_isWinogradCapable;
    }
    bool isCompressWeightsCapable() const
    {
        return m_targetConfigParams.m_isCompressWeightsCapable;
    }
    bool isBatchModeCapable() const
    {
        return m_targetConfigParams.m_isBatchModeCapable;
    }
    bool isPDPCapable() const
    {
        return m_targetConfigParams.m_isPDPCapable;
    }
    bool isCDPCapable() const
    {
        return m_targetConfigParams.m_isCDPCapable;
    }
    bool isSDPBiasCapable() const
    {
        return m_targetConfigParams.m_isSDPBiasCapable;
    }
    bool isSDPBatchNormCapable() const
    {
        return m_targetConfigParams.m_isSDPBatchNormCapable;
    }
    bool isSDPEltWiseCapable() const
    {
        return m_targetConfigParams.m_isSDPEltWiseCapable;
    }
    bool isSDPLutCapable() const
    {
        return m_targetConfigParams.m_isSDPLutCapable;
    }
    bool isBDMACapable() const
    {
        return m_targetConfigParams.m_isBDMACapable;
    }
    bool isRubikCapable() const
    {
        return m_targetConfigParams.m_isRubikCapable;
    }

    virtual NvDlaError initTargetConfigParams(ITargetConfigParams* cp);

    struct TargetConfigParams
    {
        NvU32 m_atomicCSize;
        NvU32 m_atomicKSize;
        NvU32 m_memoryAtomicSize;
        NvU32 m_numConvBufBankAllotted;
        NvU32 m_numConvBufEntriesPerBank;
        NvU32 m_numConvBufEntryWidth;
        NvU32 m_maxBatchSize;
        bool m_isWinogradCapable;
        bool m_isCompressWeightsCapable;
        bool m_isBatchModeCapable;
        bool m_isPDPCapable;
        bool m_isCDPCapable;
        bool m_isSDPBiasCapable;
        bool m_isSDPBatchNormCapable;
        bool m_isSDPEltWiseCapable;
        bool m_isSDPLutCapable;
        bool m_isBDMACapable;
        bool m_isRubikCapable;

        TargetConfigParams()
            : m_atomicCSize(64),
              m_atomicKSize(32),
              m_memoryAtomicSize(32),
              m_numConvBufBankAllotted(16),
              m_numConvBufEntriesPerBank(256),
              m_numConvBufEntryWidth(128),
              m_maxBatchSize(32),
              m_isWinogradCapable(false),
              m_isCompressWeightsCapable(false),
              m_isBatchModeCapable(false),
              m_isPDPCapable(false),
              m_isCDPCapable(false),
              m_isSDPBiasCapable(false),
              m_isSDPBatchNormCapable(false),
              m_isSDPEltWiseCapable(false),
              m_isSDPLutCapable(false),
              m_isBDMACapable(false),
              m_isRubikCapable(false)
        {
        }
    };

    inline bool isFullConfig()
    {
        return m_instance_name == std::string("nv_full");
    }
    inline bool isLargeConfig()
    {
        return m_instance_name == std::string("nv_large");
    }
    inline bool isSmallConfig()
    {
        return m_instance_name == std::string("nv_small");
    }

protected:
    std::string m_instance_name;
    TargetConfigParams m_targetConfigParams;
};

} // namespace priv

} // namespace nvdla

#endif // NVDLA_PRIV_CONFIG_H
