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

#ifndef NVDLA_I_WISDOM_H
#define NVDLA_I_WISDOM_H

#include <nvdla/IType.h>

namespace nvdla
{
class IRuntime;
class IProfiler;
class IProfile;
class ICompiler;
class INetwork;
class ILayer;
class ILoadable;
class ITensor;
class IWisdomContainerEntry;

class IWisdom
{
public:

    virtual bool open(const std::string &uri) = 0;

    //    int getNumTestPoints();
    //    void save(INetwork *, NVDLANetwork *);
    //    void save(const OpList &, const WisdomTestPoint &);
    //    TestPoint &getTestPoint(int);

    virtual void close() = 0 ;

    virtual IWisdomContainerEntry *getRootEntry() = 0;

    // global parameters associated with any result within the wisdom context
    virtual NvDlaError setDataType(DataType::UnderlyingType d) = 0;
    virtual NvDlaError getDataType(DataType::UnderlyingType *) const = 0;

    virtual bool setNetwork(INetwork *) = 0;
    virtual bool setNetworkTransient(INetwork *) = 0;

    virtual INetwork *getNetwork() = 0;

    virtual IProfiler *getProfiler() = 0;

    virtual ICompiler *getCompiler() = 0;

    //
    // Dictionary/symbol table interfaces.
    //
    virtual bool insertNetworkSymbol(INetwork *, const std::string &) = 0;
    virtual INetwork *findNetworkSymbol(const std::string &) = 0;

    virtual bool insertLayerSymbol(ILayer *, const std::string &) = 0;
    virtual ILayer *findLayerSymbol(const std::string &) = 0;

    virtual bool insertTensorSymbol(ITensor *, const std::string &) = 0;
    virtual ITensor *findTensorSymbol(const std::string &) = 0;

    virtual bool insertLoadableSymbol(ILoadable *, const std::string &) = 0;
    virtual ILoadable *findLoadableSymbol(const std::string &) = 0;


protected:

    IWisdom();
    virtual ~IWisdom();
    //    friend static void deleteWisdom(IWisdom *);
};

IWisdom *createWisdom();
NvDlaError destroyWisdom(IWisdom *wisdom);

} // nvdla

#endif // NVDLA_I_WISDOM_H
