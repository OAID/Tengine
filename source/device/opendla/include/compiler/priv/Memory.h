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

#ifndef NVDLA_PRIV_MEMORY_H
#define NVDLA_PRIV_MEMORY_H

#include <set>
#include <string>
#include <vector>

#include "BuddyAlloc.h"
#include "Check.h"
#include "MultiBatch.h"
#include "ResourceEnums.h" // for memory enums
#include "Type.h"
//#include "priv/EngineAST.h"

namespace nvdla {

namespace priv {

namespace engine_ast {
class Node;
}

namespace surface {
class TensorSurfaceDesc;
}

namespace memory {
class TensorBufferDesc;

enum PoolTypeEnum
{
    POOL_TYPE_ENUMS(GEN_ENUM)
};
typedef SequenceEnum<PoolTypeEnum, NvU8> PoolType;

enum LocationEnum
{
    MEMORY_LOCATION_ENUMS(GEN_ENUM)
};
typedef SequenceEnum<LocationEnum, NvU8> Location;

enum MemoryBufferTypeEnum
{
    MEMORY_BUFFER_TYPE_ENUMS(GEN_ENUM)
};
typedef SequenceEnum<MemoryBufferTypeEnum, NvU8> MemoryBufferType;

enum TensorCategoryEnum
{
    TENSOR_CATEGORY_ENUMS(GEN_ENUM)
};
typedef SequenceEnum<TensorCategoryEnum, NvU8> TensorCategory;

//
// used by the compiler to partition memory pools.
//
class Pool
{
public:
    Pool()
        : m_base_addr(NULL),
          m_size(0),
          m_min_buffer_size(0),
          m_addr_mgr(NULL),
          m_name(""),
          m_memoryId(-1),
          m_addressId(-1),
          m_type(PoolTypeEnum::LOCAL_DRAM_POOL),
          m_size_used(0)
    {
    }

    Pool(const Pool& o)
        : m_base_addr(o.m_base_addr),
          m_size(o.m_size),
          m_min_buffer_size(o.m_min_buffer_size),
          m_addr_mgr(o.m_addr_mgr), // xxx clone behavior?
          m_name(o.m_name),
          m_memoryId(o.m_memoryId),
          m_addressId(o.m_addressId),
          m_type(o.m_type),
          m_contents(o.m_contents), // xxx clone behavior?
          m_size_used(o.m_size_used)
    {
    }
    ~Pool();

    const std::string name() const
    {
        return m_name;
    }
    PoolType type() const
    {
        return m_type;
    }

    NvDlaError init(PoolType pt, NvU64 poolSize, NvU32 minBufferSize);
    NvU64 size() const
    {
        return m_size;
    }
    NvU64 sizeUsed() const
    {
        return m_size_used;
    }

    Location location();
    NvDlaError allocate(TensorBufferDesc* tbd, NvU16 batchId = 0);
    NvDlaError deallocate(TensorBufferDesc* tbd, NvU16 batchId = 0);

    void setMemoryId(NvS16 id)
    {
        m_memoryId = id;
    }
    NvS16 memoryId() const
    {
        return m_memoryId;
    }

    void setAddressId(NvS16 id)
    {
        m_addressId = id;
    }
    NvS16 addressId() const
    {
        return m_addressId;
    }

    void insertContent(surface::TensorSurfaceDesc* tsd)
    {
        m_contents.insert(tsd);
    }
    std::set<surface::TensorSurfaceDesc*>& contents()
    {
        return m_contents;
    }

    static inline bool debug()
    {
        return true;
    }
    static inline bool debugBinding()
    {
        return true;
    }

protected:
    void* m_base_addr;
    NvU64 m_size;
    NvU32 m_min_buffer_size;
    NvDlaBuddyAllocInst* m_addr_mgr;
    std::string m_name;
    NvS16 m_memoryId;
    NvS16 m_addressId;
    PoolType m_type;
    std::set<surface::TensorSurfaceDesc*> m_contents;
    NvU64 m_size_used;
};

class TensorBufferDesc
{
public:
    class BufferDescState
    {
    public:
        BufferDescState()
            : m_address(0),
              m_is_allocated(false),
              m_memoryId(-1),
              m_pool_offset(0),
              m_mem_loc(LocationEnum::lUNKNOWN),
              m_pool(NULL)
        {
        }
        virtual ~BufferDescState()
        {
        }

        void setBufferAddress(void* addr)
        {
            m_address = addr;
        }
        template<typename T>
        T* bufferAddress() const
        {
            return (T*)m_address;
        }

        void setAllocated()
        {
            m_is_allocated = true;
        }
        void clearAllocated()
        {
            m_is_allocated = false;
        }
        bool allocated()
        {
            return m_is_allocated;
        }

        void setMemoryId(NvS16 mid)
        {
            m_memoryId = mid;
        }
        NvS16 memoryId() const
        {
            return m_memoryId;
        }

        NvU64 poolOffset() const
        {
            return m_pool_offset;
        }
        void setPoolOffset(NvU64 poolOffset)
        {
            m_pool_offset = poolOffset;
        }

        void setMemoryLoc(Location loc)
        {
            m_mem_loc = loc;
        }
        Location memoryLoc() const
        {
            return m_mem_loc;
        }

        Pool* pool() const
        {
            return m_pool;
        }
        void setPool(Pool* pool)
        {
            m_pool = pool;
        }

    protected:
        void* m_address;
        bool m_is_allocated;
        NvS16 m_memoryId;
        NvU64 m_pool_offset;
        Location m_mem_loc; // DRAM/CV-SRAM/CBuff/Stream
        Pool* m_pool;
    };

    TensorBufferDesc(NvU16 numBatches)
        : m_id(""),
          m_align_bytes(32),
          m_size(0)
    {
        m_mb_tbd_state = new MultiBatchState<BufferDescState>(numBatches);
    }

    ~TensorBufferDesc();

    static inline bool debugBinding()
    {
        return true;
    }

    void setId(const std::string id)
    {
        m_id = id;
    }
    const std::string id() const
    {
        return m_id;
    }

    void setMemoryId(NvS16 mid, NvU16 batchId = 0)
    {
        if (m_mb_tbd_state->batch(batchId).pool())
        {
            gLogError << "tried to assign a memory id to a buffer (" << id() << ") within a pool" << std::endl;
        }
        m_mb_tbd_state->batch(batchId).setMemoryId(mid);
    }
    NvS16 memoryId(NvU16 batchId = 0)
    {
        // if buffer is within pool, it uses same memId as the pool's and doesn't have its own
        if (m_mb_tbd_state->batch(batchId).pool())
        {
            return m_mb_tbd_state->batch(batchId).pool()->memoryId();
        }
        return m_mb_tbd_state->batch(batchId).memoryId();
    }

    void setAddress(void* addr, NvU16 batchId = 0)
    {
        m_mb_tbd_state->batch(batchId).setBufferAddress(addr);
    }
    template<typename T>
    T* address(NvU16 batchId = 0)
    {
        return m_mb_tbd_state->batch(batchId).bufferAddress<T>();
    }

    void setPoolOffset(NvU64 off, NvU16 batchId = 0)
    {
        m_mb_tbd_state->batch(batchId).setPoolOffset(off);
    }
    NvU64 poolOffset(NvU16 batchId = 0)
    {
        return m_mb_tbd_state->batch(batchId).poolOffset();
    }

    void setMemoryLoc(Location loc, NvU16 batchId = 0)
    {
        m_mb_tbd_state->batch(batchId).setMemoryLoc(loc);
    }
    Location memoryLoc(NvU16 batchId = 0) const
    {
        return m_mb_tbd_state->batch(batchId).memoryLoc();
    }

    Pool* pool(NvU16 batchId = 0) const
    {
        return m_mb_tbd_state->batch(batchId).pool();
    }
    void setPool(Pool* pool, NvU16 batchId = 0)
    {
        m_mb_tbd_state->batch(batchId).setPool(pool);
    }

    void setAllocated(NvU16 batchId = 0)
    {
        m_mb_tbd_state->batch(batchId).setAllocated();
    }
    void clearAllocated(NvU16 batchId = 0)
    {
        m_mb_tbd_state->batch(batchId).clearAllocated();
    }
    bool allocated(NvU16 batchId = 0)
    {
        return m_mb_tbd_state->batch(batchId).allocated();
    }

    void setAlignment(NvU16 align)
    {
        m_align_bytes = align;
    }
    NvU16 alignBytes()
    {
        return m_align_bytes;
    }

    NvU64 size()
    {
        return m_size;
    }
    void setSize(NvU64 size)
    {
        m_size = size;
    }
    void resetSize()
    {
        m_size = 0;
    }

    NvDlaError addSurface(nvdla::priv::surface::TensorSurfaceDesc* tsd);
    std::set<nvdla::priv::surface::TensorSurfaceDesc*>& surfaces();

    bool bindable() const;
    NvS16 bindId(enum IOD& bindDomain) const;
    surface::TensorSurfaceDesc* boundSurface(size_t) const;

    // void setContent(const void* data) { m_content = true;  std::memcpy(address<void>(), data, size()); }
    bool content() const;

protected:
    std::string m_id;
    NvU16 m_align_bytes;
    NvU64 m_size;
    std::set<surface::TensorSurfaceDesc*> m_surfaces;
    // tbd state for each of the batches in multi-batch case
    MultiBatchState<BufferDescState>* m_mb_tbd_state;
};

} // namespace memory
} // namespace priv
} // namespace nvdla

#endif // NVDLA_PRIV_MEMORY_H
