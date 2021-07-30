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

#ifndef NVDLA_PRIV_LOADABLE_H
#define NVDLA_PRIV_LOADABLE_H

#include <map>

#include <nvdla/ILoadable.h>

#include "priv/Type.h"
#include "priv/loadable_generated.h"

namespace nvdla {

namespace priv {

class Loadable;

class LoadableFactory
{
public:
    typedef PrivPair<ILoadable*, Loadable*> LoadablePrivPair;

    static LoadablePrivPair newLoadable();
    static void deleteLoadable(ILoadable* loadable);

    static Loadable* priv(ILoadable*);
    static ILoadable* i(Loadable*);
    static ILoadable* self(void* s);

protected:
    static BiMap<ILoadable*, Loadable*> s_priv;
    static BiMap<void*, ILoadable*> s_self;

    friend class Runtime;
    static ILoadable* deserializeLoadable(NvU8*);
};

class Loadable : public ILoadable
{
public: // externally facing
    virtual std::string getName() const;

    virtual int getNumMemoryListEntries() const;
    virtual MemoryListEntry getMemoryListEntry(NvU16 pool_id) const;

    virtual int getNumEventListEntries() const;
    virtual EventListEntry getEventListEntry(NvU16 event_id) const;

    virtual int getNumTaskListEntries() const;
    virtual TaskListEntry getTaskListEntry(NvU16 task_id) const;

    virtual int getNumSubmitListEntries() const;
    virtual SubmitListEntry getSubmitListEntry(NvU16 submit_id) const;

    virtual int getNumAddressListEntries() const;
    virtual AddressListEntry getAddressListEntry(NvU16 address_list_index) const;

    virtual int getNumTensorDescListEntries() const;
    virtual TensorDescListEntry getTensorDescListEntry(NvU16 tensor_desc_list_index) const;

    virtual NvDlaError getNetworkDataType(DataType::UnderlyingType*) const;

    virtual NvDlaError getNumInputTensors(int*) const;
    virtual NvDlaError getInputTensorDesc(NvU16 id, TensorDescListEntry*) const;

    virtual NvDlaError getNumOutputTensors(int*) const;
    virtual NvDlaError getOutputTensorDesc(NvU16 id, TensorDescListEntry*) const;

    virtual int getNumRelocEntries() const;
    virtual RelocEntry getRelocEntry(NvU16 i) const;

public: // internally facing
    virtual int setSymbolContent(std::string name, const ILoadable::Blob&, NvU8* data);
    virtual bool getSymbolContent(std::string name, ILoadable::Blob&, NvU8*&);

    virtual NvU16 getFactoryType() const;

    void setMemoryListEntries(const std::vector<MemoryListEntry>&);
    void setEventListEntries(const std::vector<EventListEntry>&);
    void setTaskListEntries(const std::vector<TaskListEntry>&);
    void setSubmitListEntries(const std::vector<SubmitListEntry>&);
    void setAddressListEntries(const std::vector<AddressListEntry>&);
    void setTensorDescListEntries(const std::vector<TensorDescListEntry>&);
    void setRelocEntries(const std::vector<RelocEntry>&);

    const std::vector<TaskListEntry>& getTaskListEntries() const;
    const std::vector<SubmitListEntry>& getSubmitListEntries() const;
    const std::vector<MemoryListEntry>& getMemoryListEntries() const;
    const std::vector<AddressListEntry>& getAddressListEntries() const;
    const std::vector<EventListEntry>& getEventListEntries() const;
    const std::vector<TensorDescListEntry>& getTensorDescListEntries() const;
    const std::vector<RelocEntry>& getRelocEntries() const;

    Loadable();
    virtual ~Loadable();

    virtual bool serialize();
    virtual NvDlaError getSerializedData(NvU8* buffer);
    virtual NvDlaError getSerializedDataSize(NvU64* size);
    virtual bool deserializeFrom(NvU8*);

    struct Symbol
    {
        std::string name;
        ILoadable::Interface interface;
        NvU32 subInterface;
        ILoadable::Version version;
        NvU64 size;
        NvU8* data;
        Symbol()
        {
        }
    };

    inline bool debugSymbolContent()
    {
        return true;
    }

protected:
    friend class Runtime;
    std::map<std::string, Symbol> mSymbols;
    std::vector<MemoryListEntry> mMemoryListEntries;
    std::vector<TaskListEntry> mTaskListEntries;
    std::vector<SubmitListEntry> mSubmitListEntries;
    std::vector<EventListEntry> mEventListEntries;
    std::vector<AddressListEntry> mAddressListEntries;
    std::vector<TensorDescListEntry> mTensorDescListEntries;
    std::vector<RelocEntry> mRelocEntries;

    std::string mName;

private:
    flatbuffers::FlatBufferBuilder mFbb;
};

} // namespace priv

} // namespace nvdla

#endif
