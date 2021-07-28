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

#ifndef NVDLA_I_LOADABLE_H
#define NVDLA_I_LOADABLE_H

#include <string>
#include <vector>

#include "nvdla/c/NvDlaType.h"
#include "nvdla/c/NvDlaLoadable.h"

#include "nvdla/IType.h"
#include "nvdla/IRuntime.h"



// some gnu stuff is defining these in host mode... evil.
#ifdef major
#undef major
#endif
#ifdef minor
#undef minor
#endif

//
// ILoadable synopsis
//
//     storage/manipulation of
//         . memory object w/content
//         . blank memory object
//         . content w/engine rev check
//
//     task submission model
//         two engine types: dla and cpu
//         describe then submit a set (small ish?) of tasks
//         global
//             memory objects -> placed in address list
//             event objects  -> placed in event lists
//             io objects     -> tensor bind point list ?? actually needed ??
//         per-task
//             engine type
//             address list content (addr0, any ref'd cmd buffers, ++)
//             preaction event list
//             postaction event list
//
//
//     task set load (all at once)
//         global setup operations
//             resolve alloc'd memory objects
//             resolve alloc'd event objects
//         for each task inspect its address list
//             move mem as needed for content setup
//             mark any still unsatisfied
//         for each task inspect its event list
//             mark any which continue to be unsatisfied
//
//     task set exec (all at once)
//         check for unbound memory
//         check for unbond events
//         submit each task

//

namespace nvdla
{

class ILoadable
{
public:

    enum Interface {
        Interface_NONE = NVDLA_LOADABLE_INTERFACE_NONE,
        Interface_DLA1 = NVDLA_LOADABLE_INTERFACE_DLA1,
        Interface_EMU1 = NVDLA_LOADABLE_INTERFACE_EMU1,
    };

    enum MemoryDomain {
        MemoryDomain_SYSMEM = NVDLA_LOADABLE_MEMORY_DOMAIN_SYSMEM,
        MemoryDomain_SRAM = NVDLA_LOADABLE_MEMORY_DOMAIN_SRAM,
    };

    enum MemoryFlags {
        MemoryFlags_NONE  = NVDLA_LOADABLE_MEMORY_FLAGS_NONE,
        MemoryFlags_ALLOC  = NVDLA_LOADABLE_MEMORY_FLAGS_ALLOC,
        MemoryFlags_SET    = NVDLA_LOADABLE_MEMORY_FLAGS_SET,
        MemoryFlags_INPUT  = NVDLA_LOADABLE_MEMORY_FLAGS_INPUT,
        MemoryFlags_OUTPUT = NVDLA_LOADABLE_MEMORY_FLAGS_OUTPUT,
        MemoryFlags_DEBUG  = NVDLA_LOADABLE_MEMORY_FLAGS_DEBUG
    };

    enum EventOp {
        EventOp_WAIT   = NVDLA_LOADABLE_EVENT_OP_WAIT,
        EventOp_SIGNAL = NVDLA_LOADABLE_EVENT_OP_SIGNAL
    };

    struct Version
    {
        NvU8 major;
        NvU8 minor;
        NvU8 sub_minor;
        Version(NvU8 maj, NvU8 min, NvU8 sub) : major(maj), minor(min), sub_minor(sub) { }
        Version() : major(0), minor(0), sub_minor(0) { }

        void toC(NvDlaLoadableVersion &c) const
        {
            c.major = major;
            c.minor = minor;
            c.subMinor = sub_minor;
        }
    };

    struct MemoryListEntry
    {
        NvU16 id;
        NvU64 size;
        NvU32 alignment; // 0 for n/a, otherwise byte alignment
        NvU8  domain;
        static inline NvU8 domain_sysmem() { return MemoryDomain_SYSMEM; }
        static inline NvU8 domain_sram() { return MemoryDomain_SRAM; }
        NvU8  flags; // alloc or alloc_content or is-input or is-output
        static inline NvU8  flags_alloc()  { return MemoryFlags_ALLOC;  }
        static inline NvU8  flags_set()    { return MemoryFlags_SET;    }
        static inline NvU8  flags_input()  { return MemoryFlags_INPUT;  }
        static inline NvU8  flags_output() { return MemoryFlags_OUTPUT; }
        static inline NvU8  flags_debug()  { return MemoryFlags_DEBUG;  }
        NvU16 bind_id;  // valid iff flag_{input|output|debug}()  is set
        NvU16 tensor_desc_id; // valid iff bind_id is valid ( != -1 )
        std::vector<std::string> contents;  // symbolic reference to content blob
        std::vector<uint64_t>    offsets;   // associated offset for contents

        MemoryListEntry() : id(0), size(0), alignment(0), domain(0), flags(0),
                            bind_id(0), tensor_desc_id(0), contents(), offsets() { }
        MemoryListEntry(const MemoryListEntry &o) : id(o.id), size(o.size), alignment(o.alignment), domain(o.domain), flags(o.flags),
                                                    bind_id(o.bind_id),
                                                    tensor_desc_id(o.tensor_desc_id),
                                                    contents(o.contents),
                                                    offsets(o.offsets) { }
        MemoryListEntry(NvU16 i, NvU64 s, NvU32 a, NvU8 d, NvU8 f, std::string sym = std::string(), uint64_t o = 0) :
            id(i), size(s), alignment(a), domain(d), flags(f), bind_id(0), tensor_desc_id(0)
        {
            if ( sym.size() )
            {
                contents.push_back(sym);
                offsets.push_back(o);
            }
        }
    };

    struct EventListEntry
    {
        NvU16 id;
        NvU16 target;
        NvU8 op;
        static inline NvU8 op_wait() { return EventOp_WAIT; }
        static inline NvU8 op_signal() { return EventOp_SIGNAL; }
        NvU32 val;
        void toC(NvDlaLoadableEventListEntry &c) const
        {
            c.id = id;
            c.target = target;
            c.op = op;
            c.val = val;
        }
    };

    struct TaskListEntry
    {
        NvU16 id;
        NvU32 interface; // DLA interface id
        static inline NvU32 interface_NONE() { return Interface_NONE; }
        static inline NvU32 interface_DLA1() { return Interface_DLA1; }
        static inline NvU32 interface_EMU1() { return Interface_EMU1; }

        NvS16 instance; // -1 := for any available
        static inline NvS16 instance_ANY() { return -1; }

        std::vector<NvU16> preactions;   // [event id]...
        std::vector<NvU16> postactions;  // [event id]...
        std::vector<NvU16> address_list; // [addr list id]...[addr list id]
        TaskListEntry(const TaskListEntry &o) :
            id(o.id),
            interface(o.interface),
            instance(o.instance),
            preactions(o.preactions),
            postactions(o.postactions),
            address_list(o.address_list) { }

        TaskListEntry() : id(0),
                          interface(Interface_NONE),
                          instance(-1),
                          preactions(),
                          postactions(),
                          address_list() { }
    };

    struct SubmitListEntry
    {
        NvU16 id;
        std::vector<NvU16> tasks;
    };

    struct AddressListEntry
    {
        NvU16 id;     // all possible address list entries are given an id
        NvU16 mem_id; // determines hRm (+offset from below)
        NvU64 size;   // assert size <= memory[mem_id].size
        NvU64 offset; // assert (offset + size) <= memory[mem_id].size
        AddressListEntry() : id(0), mem_id(0), size(0), offset(0) { }
        AddressListEntry(NvU16 i, NvU16 m, NvU64 s, NvU64 o = 0) : id(i), mem_id(m), size(s), offset(o) { }
        AddressListEntry(const AddressListEntry &o) : id(o.id), mem_id(o.mem_id), size(o.size), offset(o.offset) { }
        void toC(NvDlaLoadableAddressListEntry &c) const {
            c.id = id;
            c.memId = mem_id;
            c.size = size;
            c.offset = offset;
        }
    };

    struct TensorDescListEntry
    {
        std::string name;
        NvU16 id;
        NvU16 memId;
        NvU64 size;
        NvU64 offset;
        NvDlaDims4 dims;
        NvU8 dataFormat;
        NvU8 dataType;
        NvU8 dataCategory;
        NvU8 pixelFormat;
        NvU8 pixelMapping;
        NvU32 stride[NVDLA_LOADABLE_TENSOR_DESC_NUM_STRIDES];
    };

    struct RelocEntry
    {
        NvU16 addressListId; // fix vs. this addr list item
        NvU16 writeId;   // fix *within this* memory id given offset below
        NvU64 offset;    // buffer offset to the fixup
        NvU32 interface; // dla1, emu1, etc.
        NvU32 subInterface; //  dla1-surf_desc, etc.
        NvU8  relocType; // stride0..7 (aka line, surf)

        RelocEntry(const RelocEntry &o) :
            addressListId(o.addressListId),
            writeId(o.writeId),
            offset(o.offset),
            interface(o.interface),
            subInterface(o.subInterface),
            relocType(o.relocType) { }

        RelocEntry(NvS16 a, NvU64 o, NvU32 i, NvU32 s, NvU8 r) :
            addressListId(a),
            writeId(0), // invalid
            offset(o),
            interface(i),
            subInterface(s),
            relocType(r) { }

        RelocEntry(NvS16 a, NvS16 w, NvU64 o, NvU32 i, NvU32 s, NvU8 r) :
            addressListId(a),
            writeId(w),
            offset(o),
            interface(i),
            subInterface(s),
            relocType(r) { }
    };

    struct Blob
    {
        std::string name;
        NvU64 size;
        Interface interface;
        NvU32 subInterface;
        Version version;

        Blob() :
            size(0),
            interface(Interface_NONE),
            subInterface(0) { }

        Blob(const std::string &n, NvU64 s, Interface i, NvU32 si, Version v) :
            name(n),
            size(s),
            interface(i),
            subInterface(si),
            version(v) { }
    };

    virtual std::string getName() const = 0;

    virtual int getNumMemoryListEntries() const = 0;
    virtual MemoryListEntry getMemoryListEntry(NvU16 mem_id) const = 0;

    virtual int getNumEventListEntries() const = 0;
    virtual EventListEntry getEventListEntry(NvU16 event_id) const = 0;

    virtual int getNumTaskListEntries() const = 0;
    virtual TaskListEntry getTaskListEntry(NvU16 task_id) const = 0;

    virtual int getNumAddressListEntries() const = 0;
    virtual AddressListEntry getAddressListEntry(NvU16 i) const = 0;

    virtual int getNumTensorDescListEntries() const = 0;
    virtual TensorDescListEntry getTensorDescListEntry(NvU16 i) const = 0;

    virtual NvDlaError getNetworkDataType(DataType::UnderlyingType *) const = 0;

    virtual NvDlaError getNumInputTensors(int *) const = 0;
    virtual NvDlaError getInputTensorDesc(NvU16 id, ILoadable::TensorDescListEntry *) const = 0;

    virtual NvDlaError getNumOutputTensors(int *) const = 0;
    virtual NvDlaError getOutputTensorDesc(NvU16 id, ILoadable::TensorDescListEntry *) const = 0;

protected:
    ILoadable();
    virtual ~ILoadable();
};

} // nvdla
#endif // NVDLA_I_LOADABLE_H
