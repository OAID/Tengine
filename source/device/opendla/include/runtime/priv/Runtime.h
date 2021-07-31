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

#ifndef NVDLA_PRIV_RUNTIME_H
#define NVDLA_PRIV_RUNTIME_H

#include <string>
#include <map>
#include <vector>
#include <memory>
#include <algorithm>

#include "priv/Type.h"

#include "nvdla/ILoadable.h"
#include "nvdla/IRuntime.h"

#include "priv/EMUInterface.h"

namespace nvdla
{
class ITensor;

namespace priv
{

class Loadable;
class Runtime;


class RuntimeFactory
{
public:
    typedef PrivPair<IRuntime *, Runtime*> RuntimePrivPair;

    static RuntimePrivPair newRuntime();
    static void deleteRuntime(IRuntime *);

    static Runtime *priv(IRuntime *);
    static IRuntime *i(Runtime *);
    static IRuntime *self(void *s);

protected:
    static BiMap<IRuntime *, Runtime *> s_priv;
    static BiMap<void *, IRuntime *> s_self;

};

class Runtime : public IRuntime
{
public: // externally facing

    // device interfaces
    virtual NvU16 getMaxDevices();
    virtual NvU16 getNumDevices();

    virtual bool initEMU(void);
    virtual void stopEMU(void);

    virtual bool load(NvU8 *buf, int instance);
    virtual void unload(void);
    virtual NvDlaError allocateSystemMemory(void **h_mem, NvU64 size, void **pData);
    virtual void freeSystemMemory(void *phMem, NvU64 size);

    virtual bool bindInputTensor (int index, void *hMem);
    virtual bool bindOutputTensor(int index, void *hMem);

    virtual NvDlaError getNetworkDataType(uint8_t *) const;

    virtual NvDlaError getNumInputTensors(int *);
    virtual NvDlaError getInputTensorDesc(int id, IRuntime::NvDlaTensor *);
    virtual NvDlaError setInputTensorDesc(int id, const IRuntime::NvDlaTensor *);

    virtual NvDlaError getNumOutputTensors(int *);
    virtual NvDlaError getOutputTensorDesc(int id, IRuntime::NvDlaTensor *);
    virtual NvDlaError setOutputTensorDesc(int id, const IRuntime::NvDlaTensor *);

    virtual bool submit();

public: // internally facing
    Runtime();

    virtual NvU16 getFactoryType() const;

protected:

    friend class RuntimeFactory;

    virtual ~Runtime();

    inline bool debugMemoryLayout() const  { return true; }
    inline bool debugTasks() const { return true; }
    inline bool debugVersions() const { return true; }
    inline bool debugLoadables() const { return true; }
    inline bool debugBinding() const { return true; }
    inline bool debugStrideRewrite() const { return true; }

    NvDlaError submitInternal(void);

    virtual void *getDLADeviceContext(size_t sel_i);
    size_t getMaxDLADevices() { return 1; }

    void *m_dla_handle;
    void *m_dla_device_handles[2];
    Emulator *m_emu_engine;

    void *h_network_desc_mem;
    void *h_op_desc_mem;
    void *h_surf_desc_mem;
    void *h_dependency_list_mem;

    std::vector<ILoadable::TaskListEntry> m_task_entries;
    std::vector<ILoadable::SubmitListEntry> m_submit_entries;
    std::vector<ILoadable::MemoryListEntry> m_memory_entries;
    std::vector<ILoadable::AddressListEntry> m_address_entries;
    std::vector<ILoadable::EventListEntry> m_event_entries;
    std::vector<ILoadable::TensorDescListEntry> m_tensor_desc_entries;
    std::vector<ILoadable::RelocEntry> m_reloc_entries;

    class Task  {
    public:
        Task() { }
        Task(const ILoadable::TaskListEntry &e) : mEntry(e)        { }
        Task(const Task &o)                     : mEntry(o.mEntry) { }
        NvU16 id() const { return mEntry.id; }
        NvU32 interface() const { return mEntry.interface; }
        NvS16 instance()  const { return mEntry.instance; }
        std::vector<NvU16> &address_list() { return mEntry.address_list; }
        std::vector<NvU16> &preactions() { return mEntry.preactions; }
        std::vector<NvU16> &postactions() { return mEntry.postactions; }
    protected:
        friend class Runtime;
        ILoadable::TaskListEntry mEntry;

    };

    class Submit {
    public:
        Submit() { }
        Submit(const ILoadable::SubmitListEntry &e) : mEntry(e) { }
        Submit(const Submit &o) : mEntry(o.mEntry) { }
        NvU16 id() const { return mEntry.id; }
        std::vector<NvU16> &tasks() { return mEntry.tasks; }
    protected:
        friend class Runtime;
        ILoadable::SubmitListEntry mEntry;
    };

    class Memory {
    public:
        Memory() : hMem(0), pVirtAddr(0) { }
        Memory(const ILoadable::MemoryListEntry &e) : hMem(0), pVirtAddr(0), mEntry(e) { }
        Memory(const Memory &o)                     : hMem(o.hMem), pVirtAddr(0), mEntry(o.mEntry) { }
        inline NvU16 id() { return mEntry.id; }
        inline NvU64 size() { return mEntry.size; }
        inline NvU32 alignment() { return mEntry.alignment; }
        inline NvU8 domain() { return mEntry.domain; }
        inline bool bound() { return hMem != 0; }
        inline NvU8 flags() { return mEntry.flags; }
        inline void setHandle(void *h) { hMem = h; }
        inline void *getHandle() const { return hMem; }
        inline void setVirtAddr(void *addr) { pVirtAddr = addr; }
        inline void *getVirtAddr() const { return pVirtAddr; }
        inline std::vector<std::string> & contents() { return mEntry.contents; }
        inline std::vector<uint64_t> & offsets() { return mEntry.offsets; }
        inline int inputBindId() const {
            if ( mEntry.flags & mEntry.flags_input() ) {
                return (int) mEntry.bind_id;
            }
            return -1;
        }
        inline int outputBindId() const {
            if ( mEntry.flags & mEntry.flags_output() ) {
                return (int) mEntry.bind_id;
            }
            return -1;
        }
        inline bool bindable() const {
            return !!(mEntry.flags & (mEntry.flags_input() | mEntry.flags_output()));
        }
        inline int bindId(IOD &which) const
        {
            // there should be only one valid.  but if not take in order of
            // input, output
            if ( mEntry.flags & mEntry.flags_input() ) {
                which = IOD_Input;
                return (int) mEntry.bind_id;
            } else if ( mEntry.flags & mEntry.flags_output() ) {
                which = IOD_Output;
                return (int) mEntry.bind_id;
            }
            which = IOD_Max;
            return -1;
        };
        inline int tensorDescId() const {
            if ( mEntry.flags & ( mEntry.flags_input() | mEntry.flags_output() ) ) {
                return (int) mEntry.tensor_desc_id;
            }
            return -1;
        }

    protected:
        friend class Runtime;
        void *hMem;
        void *pVirtAddr;
        ILoadable::MemoryListEntry mEntry;
    };

    class Event {
    public:
        NvU16 id()const { return mEntry.id; }
        NvU8  op() const { return mEntry.op; }
        NvU16 target()const { return mEntry.target; }
        NvU32 val() const { return mEntry.val; }
    protected:
        friend class Runtime;
        ILoadable::EventListEntry mEntry;
    };

    class Address {
    public:
        Address() { }
        Address(const ILoadable::AddressListEntry &e) : mEntry(e) { }
        Address(const Address &o) : mEntry(o.mEntry) { }
        NvU16 id() const { return mEntry.id; }
        NvU16 mem_id() const { return mEntry.mem_id; }
        NvU64 offset() const { return mEntry.offset; }
    public:
        friend class Runtime;
        ILoadable::AddressListEntry mEntry;
    };

public:
    class TensorDesc : public ILoadable::TensorDescListEntry {
    public:
        TensorDesc();
        TensorDesc(const ILoadable::TensorDescListEntry &e);
        IRuntime::NvDlaTensor bindTensorDesc() const;
    };

    bool bindTensorMemory(ITensor *, void *hMem);
    bool unbindTensorMemory(ITensor *, void *hMem);

protected:
    std::vector<Task> m_task;
    std::vector<Submit> m_submit;
    std::vector<Memory> m_memory;
    std::vector<Event> m_event;
    std::vector<Address> m_address;
    std::vector<TensorDesc> m_tensor_desc;

    Loadable * m_loaded;
    size_t m_loaded_instance;

    bool versionsCompatible(const ILoadable::Version &, const ILoadable::Version &);

    size_t m_numDLATasks;

    NvDlaError loadMemory(Loadable *, Memory *);
    void unloadMemory(Memory *);
    bool fillTaskAddressList(Task *task, NvDlaTask *);

    bool fillEMUTaskAddressList(Task *task, EMUTaskDescAccessor taskDescAcc);

    //
    // maintenance of ids/lookups for bind ids, associated memory, tensor descs
    //
    NvDlaError initBindableMemory();
    NvDlaError getMemoryFromBindId(IOD w, int id, Memory * &bound_mem);
    std::vector<std::vector<Memory *>> m_bindable_memory; // indexed on [iod][bind_id]
    std::map<void *, void *> m_hmem_memory_map; // maintains 1-1 relation between hmem and mapped memory.

    class MemoryId_BindId_Is // helper predicate
    {
    public:
        MemoryId_BindId_Is(int find_id) : m_find_id(find_id) { }
        bool operator () (Memory *&mem) const { IOD na; return m_find_id == mem->bindId(na); }
    protected:
        int m_find_id;
    };

    class Memory_BindId_LT_Compare // helper comparator
    {
    public:
        bool operator() (Memory* const& i, Memory* const& j) { IOD na; return i->bindId(na) < j->bindId(na); }
    };

    NvDlaError mergeSetTensorDesc(IOD w, int bindID, int tensorDescId, const IRuntime::NvDlaTensor *tdl);
    NvDlaError rewriteStrides(IOD w, int bindID, int tensorDescId, const NvU32 *);

};


} // nvdla::priv
} // nvdla

#endif // NVDLA_PRIV_RUNTIME_H
