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

#ifndef NVDLA_PRIV_EMULATOR_H
#define NVDLA_PRIV_EMULATOR_H

#include <queue>

#include "priv/EMUInterface.h"

#include "nvdla_os_inf.h"

namespace nvdla {
class ITensor;

namespace priv {

class Emulator
{
public: // externally facing
    Emulator();
    virtual ~Emulator();

    bool ping();

    NvDlaError submit(NvU8* task_mem, bool blocking);
    NvDlaError start();
    bool stop();
    bool run();

public: // internally facing
    inline bool debugPrint()
    {
        return true;
    }
    inline bool debugOps()
    {
        return true;
    }

protected:
    static void threadFunction(void* arg);
    NvDlaError processTask(NvU8* task_mem, std::vector<NvU8*> addressList);

    NvS8 getBpe(EMUBufferDescAccessor buffer);
    NvDlaError getAddrOffset(EMUBufferDescAccessor in, NvU32 x, NvU32 y, NvU32 c, NvU32* offset);

    NvDlaError executePower(EMUPowerOpDescAccessor opDesc, EMUCommonOpDescAccessor commonOpDesc,
                            EMUPowerBufferDescsAccessor bufDescs, std::vector<NvU8*> addressList);
    NvDlaError executeSoftmax(EMUSoftmaxOpDescAccessor opDesc, EMUCommonOpDescAccessor commonOpDesc,
                              EMUSoftmaxBufferDescsAccessor bufDescs, std::vector<NvU8*> addressList);

private:
    std::queue<NvU8*> m_taskQueue;

    NvDlaThreadHandle m_thread;
    bool m_threadActive;

    bool m_signalShutdown;
};

} // namespace priv
} // namespace nvdla

#endif // NVDLA_PRIV_EMULATOR_H
