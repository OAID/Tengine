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

#ifndef NVDLA_PRIV_DLA_RESOURCE_MANAGER_H
#define NVDLA_PRIV_DLA_RESOURCE_MANAGER_H

#include <set>
#include <string>

#include "Memory.h"
#include "Network.h"
#include "ResourceEnums.h" // for memory enums
#include "Surface.h"
#include "Type.h"

namespace nvdla {

namespace priv {

namespace memory {

class Pool;
class TensorBufferDesc;

class DLAResourceManager
{
public:
    DLAResourceManager()
        : m_next_buffer_id(0),
          m_next_surface_desc_id(0)
    {
        m_pools = std::vector<Pool>(PoolType::num_elements(), Pool());
    }
    DLAResourceManager(const DLAResourceManager& o);
    ~DLAResourceManager();

    std::string nextSurfaceDescId()
    {
        return std::string("tsd-") + toString(m_next_surface_desc_id++);
    }

    std::string nextBufferId()
    {
        return std::string("tb-") + toString(m_next_buffer_id++);
    }

    std::vector<Pool>* memoryPools()
    {
        return &m_pools;
    }
    std::vector<TensorBufferDesc*> getBufferDescs();
    std::vector<surface::TensorSurfaceDesc*> getSurfaceDescs();

    /* TENSOR-BUFFER MANAGEMENT */
    // register a new tensor buffer desc and add it to the directory
    // and the memory pool
    // numBatches = number of batches this BD represents
    TensorBufferDesc* regTensorBufferDesc(NvU16 numBatches);

    // unregister a tensor buffer desc and remove it from the directory
    // and the memory pool
    bool unregTensorBufferDesc(TensorBufferDesc*);

    // reserve size #bytes for a registered tensor buffer desc
    // return -1 on error
    //    int reserveSize(TensorBufferDesc *, NvU64 size);

    // free the supplied tensor buffer, thereby reclaiming its memory
    // return -1 on error
    // int freeTensorBuffer(TensorBufferDesc *);

    /* TENSOR-SURFACE_DESCRIPTOR MANAGEMENT */
    // register a new tensor surface descriptor for the given type of tensor
    // and add it to the directory
    // surface type = kernel/bias/feature-data/image
    // numBatches = number of batches this SD represents
    surface::TensorSurfaceDesc* regTensorSurfaceDesc(TensorType type, NvU16 numBatches);

    // unregister a tensor surface desc and remove it from the directory
    bool unregTensorSurfaceDesc(surface::TensorSurfaceDesc*);

protected:
    std::vector<Pool> m_pools;
    std::string m_name;

    int m_next_buffer_id;
    int m_next_surface_desc_id;

    // for keeping directories ordered by (string) id
    template<class Tp>
    struct CompareById
    {
        bool operator()(const Tp& lhs, const Tp& rhs) const
        {
            return lhs->id() < rhs->id();
        }
    };

    // just sets with specific ordering
    typedef std::set<surface::TensorSurfaceDesc*, CompareById<surface::TensorSurfaceDesc*> > TensorSurfaceDirectory;
    typedef std::set<TensorBufferDesc*, CompareById<TensorBufferDesc*> > TensorBufferDirectory;

    TensorBufferDirectory m_buffer_desc_directory;
    TensorSurfaceDirectory m_surface_desc_directory;

    typedef TensorSurfaceDirectory::iterator TensorSurfaceDirectoryIter;
    typedef TensorBufferDirectory::iterator TensorBufferDirectoryIter;
};

} // namespace memory
} // namespace priv
} // namespace nvdla

#endif // NVDLA_PRIV_MEMORY_MANAGER_H
