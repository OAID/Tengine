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

#ifndef NVDLA_PRIV_MULTIBATCH_H
#define NVDLA_PRIV_MULTIBATCH_H

#include "ErrorMacros.h"

namespace nvdla
{
namespace priv
{

template < typename StateClass >
class MultiBatchState
{
public:
    MultiBatchState() : m_batch_size(1)
    {
        m_batch_states = std::vector< StateClass >(1);
    }
    MultiBatchState(NvU16 numBatches)
    {
        m_batch_size = numBatches;
        m_batch_states = std::vector< StateClass >(numBatches);
    }
    virtual ~MultiBatchState() { }

    StateClass& batch(NvU16 batchId)
    {
        if (batchId > m_batch_size)
        {
            REPORT_ERROR(NvDlaError_BadParameter, "batchId: %d > batch_size: %d. Returning state for batch=0", batchId, m_batch_size);
            return m_batch_states[0];
        }
        else
        {
            return m_batch_states[batchId];
        }
    }

protected:
    NvU16                     m_batch_size;
    std::vector< StateClass > m_batch_states;
};

};  // nvdla::priv

};  // nvdla::

#endif /* NVDLA_PRIV_MULTIBATCH_H */
