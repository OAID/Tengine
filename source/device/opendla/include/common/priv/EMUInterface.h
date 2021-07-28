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

#ifndef NVDLA_PRIV_EMU_INTERFACE_H
#define NVDLA_PRIV_EMU_INTERFACE_H

#include <stdbool.h>

#include "priv/Type.h"
#include "priv/EMUInterfaceEnums.h"

#include "dlaerror.h"
#include "dlatypes.h"

#define EMU_FORMAT_INT8     0
#define EMU_FORMAT_INT8_8   1
#define EMU_FORMAT_INT16    2
#define EMU_FORMAT_FF16     3
#define EMU_FORMAT_UINT8    4
#define EMU_FORMAT_UINT16   5

namespace nvdla
{

namespace priv
{

//
// struct emu_address
//
class EMUAddress
{
public:
    virtual size_t struct_size()  const = 0;
    virtual size_t struct_align() const = 0;

    virtual void * hMem(NvU8 *base)  const = 0;
    virtual NvU32 * offset(NvU8 *base)  const = 0;

protected:
    EMUAddress()          { }
    virtual ~EMUAddress() { }
};

class EMUAddressAccessor
{
public:
    NvU8 * struct_base()  const;
    size_t struct_size()  const;
    size_t struct_align() const;

    void * hMem()  const;
    NvU32 * offset()  const;

    EMUAddressAccessor(NvU8 *base, const EMUAddress &);

protected:
    NvU8 *_base;
    const EMUAddress &_n;
};


//
// struct emu_task_desc
//
class EMUTaskDesc
{
public:
    virtual size_t struct_size()  const = 0;
    virtual size_t struct_align() const = 0;

    virtual NvU32 * numAddresses(NvU8 *base)   const = 0;
    virtual size_t maxBuffersPerTask() const = 0;
    virtual EMUAddressAccessor addressList(NvU8 *base, size_t c) const = 0;

protected:
    EMUTaskDesc()          { }
    virtual ~EMUTaskDesc() { }
};

class EMUTaskDescAccessor
{
public:
    NvU8 * struct_base()  const;
    size_t struct_size()  const;
    size_t struct_align() const;

    NvU32 * numAddresses()   const;
    size_t maxBuffersPerTask() const;
    EMUAddressAccessor addressList(size_t c) const;

    EMUTaskDescAccessor(NvU8 *base, const EMUTaskDesc &);

protected:
    NvU8 *_base;
    const EMUTaskDesc &_n;
};

//
// struct emu_network_desc
//
class EMUNetworkDesc
{
public:
    virtual size_t struct_size()  const = 0;
    virtual size_t struct_align() const = 0;

    virtual NvS16 * operationDescIndex(NvU8 *base)   const = 0;
    virtual NvS16 * operationBufferDescIndex(NvU8 *base)     const = 0;
    virtual NvU16 * numOperations(NvU8 *base)        const = 0;

protected:
    EMUNetworkDesc()          { }
    virtual ~EMUNetworkDesc() { }
};

class EMUNetworkDescAccessor
{
public:
    NvU8 * struct_base()  const;
    size_t struct_size()  const;
    size_t struct_align() const;

    NvS16 * operationDescIndex()   const;
    NvS16 * operationBufferDescIndex()     const;
    NvU16 * numOperations()        const;

    EMUNetworkDescAccessor(NvU8 *base, const EMUNetworkDesc &);

protected:
    NvU8 *_base;
    const EMUNetworkDesc &_n;
};

//
// struct emu_common_op_desc
//
class EMUCommonOpDesc
{
public:
    virtual size_t struct_size()  const = 0;
    virtual size_t struct_align() const = 0;

    virtual NvU8 * op_type(NvU8 *base) const = 0;
    virtual NvF32 * input_scale_factor(NvU8 *base) const = 0;
    virtual NvF32 * output_scale_factor(NvU8 *base) const = 0;

protected:
    EMUCommonOpDesc()          { }
    virtual ~EMUCommonOpDesc() { }
};

class EMUCommonOpDescAccessor
{
public:
    NvU8 * struct_base()  const;
    size_t struct_size()  const;
    size_t struct_align() const;

    NvU8 * op_type() const;
    NvF32 * input_scale_factor() const;
    NvF32 * output_scale_factor() const;

    EMUCommonOpDescAccessor(NvU8 *base, const EMUCommonOpDesc &);

protected:
    NvU8 *_base;
    const EMUCommonOpDesc &_n;
};


//
// struct emu_power_op_desc
//
class EMUPowerOpDesc
{
public:
    virtual size_t struct_size()  const = 0;
    virtual size_t struct_align() const = 0;

    virtual EMUCommonOpDescAccessor commonOpDescAccessor(NvU8 *base) const = 0;
    virtual NvF32 * power(NvU8 *base) const = 0;
    virtual NvF32 * scale(NvU8 *base) const = 0;
    virtual NvF32 * shift(NvU8 *base) const = 0;

protected:
    EMUPowerOpDesc()          { }
    virtual ~EMUPowerOpDesc() { }
};

class EMUPowerOpDescAccessor
{
public:
    NvU8 * struct_base()  const;
    size_t struct_size()  const;
    size_t struct_align() const;

    EMUCommonOpDescAccessor commonOpDescAccessor() const;
    NvF32 * power() const;
    NvF32 * scale() const;
    NvF32 * shift() const;

    EMUPowerOpDescAccessor(NvU8 *base, const EMUPowerOpDesc &);

protected:
    NvU8 *_base;
    const EMUPowerOpDesc &_n;
};


//
// struct emu_softmax_op_desc
//
class EMUSoftmaxOpDesc
{
public:
    virtual size_t struct_size()  const = 0;
    virtual size_t struct_align() const = 0;

    virtual EMUCommonOpDescAccessor commonOpDescAccessor(NvU8 *base) const = 0;
    virtual NvU8 * axis(NvU8 *base) const = 0;

protected:
    EMUSoftmaxOpDesc()          { }
    virtual ~EMUSoftmaxOpDesc() { }
};

class EMUSoftmaxOpDescAccessor
{
public:
    NvU8 * struct_base()  const;
    size_t struct_size()  const;
    size_t struct_align() const;

    EMUCommonOpDescAccessor commonOpDescAccessor() const;
    NvU8 * axis() const;

    EMUSoftmaxOpDescAccessor(NvU8 *base, const EMUSoftmaxOpDesc &);

protected:
    NvU8 *_base;
    const EMUSoftmaxOpDesc &_n;
};


//
// union emu_operation_container
//
class EMUOperationContainer
{
public:
    virtual size_t struct_size()  const = 0;
    virtual size_t struct_align() const = 0;

    virtual EMUPowerOpDescAccessor powerOpDescAccessor(NvU8 *base, size_t c) const = 0;
    virtual EMUSoftmaxOpDescAccessor softmaxOpDescAccessor(NvU8 *base, size_t c) const = 0;

protected:
    EMUOperationContainer()          { }
    virtual ~EMUOperationContainer() { }
};

class EMUOperationContainerAccessor
{
public:
    NvU8 * struct_base()  const;
    size_t struct_size()  const;
    size_t struct_align() const;

    EMUPowerOpDescAccessor powerOpDescAccessor(size_t c) const;
    EMUSoftmaxOpDescAccessor softmaxOpDescAccessor(size_t c) const;

    EMUOperationContainerAccessor(NvU8 *base, const EMUOperationContainer &);

protected:
    NvU8 *_base;
    const EMUOperationContainer &_n;
};


//
// struct emu_buffer_desc
//
class EMUBufferDesc
{
public:
    virtual size_t struct_size()  const = 0;
    virtual size_t struct_align() const = 0;

    virtual NvS16 * addressIndex(NvU8 *base)   const = 0;
    virtual NvU32 * addressIndexOffset(NvU8 *base)   const = 0;
    virtual NvU32 * size(NvU8 *base)       const = 0;
    virtual NvU16 * format(NvU8 *base)     const = 0;
    virtual NvU16   format_FF16()          const = 0;
    virtual NvU16   format_INT8()          const = 0;
    virtual NvU16   format_INT8_8()        const = 0;
    virtual NvU16   format_UINT8()         const = 0;
    virtual NvU16   format_INT16()         const = 0;
    virtual NvU16   format_UINT16()        const = 0;
    virtual NvU16 * width(NvU8 *base)      const = 0;
    virtual NvU16 * height(NvU8 *base)     const = 0;
    virtual NvU16 * channel(NvU8 *base)    const = 0;
    virtual NvU32 * lineStride(NvU8 *base) const = 0;
    virtual NvU32 * surfStride(NvU8 *base) const = 0;

protected:
    EMUBufferDesc()          { }
    virtual ~EMUBufferDesc() { }
};

class EMUBufferDescAccessor
{
public:
    NvU8 * struct_base()  const;
    size_t struct_size()  const;
    size_t struct_align() const;

    NvS16 * addressIndex()  const;
    NvU32 * addressIndexOffset() const;
    NvU32 * size()          const;
    NvU16 * format()        const;
    NvU16   format_FF16()   const;
    NvU16   format_INT8()   const;
    NvU16   format_INT8_8() const;
    NvU16   format_UINT8()  const;
    NvU16   format_INT16()  const;
    NvU16   format_UINT16() const;
    NvU16 * width()      const;
    NvU16 * height()     const;
    NvU16 * channel()    const;
    NvU32 * lineStride() const;
    NvU32 * surfStride() const;

    EMUBufferDescAccessor(NvU8 *base, const EMUBufferDesc &);

protected:
    NvU8 *_base;
    const EMUBufferDesc &_n;
};


//
// struct emu_power_buffer_descs
//
class EMUPowerBufferDescs
{
public:
    virtual size_t struct_size()  const = 0;
    virtual size_t struct_align() const = 0;

    virtual EMUBufferDescAccessor srcDataAccessor(NvU8 *base) const = 0;
    virtual EMUBufferDescAccessor dstDataAccessor(NvU8 *base) const = 0;

protected:
    EMUPowerBufferDescs()          { }
    virtual ~EMUPowerBufferDescs() { }
};

class EMUPowerBufferDescsAccessor
{
public:
    NvU8 * struct_base()  const;
    size_t struct_size()  const;
    size_t struct_align() const;

    EMUBufferDescAccessor srcDataAccessor() const;
    EMUBufferDescAccessor dstDataAccessor() const;

    EMUPowerBufferDescsAccessor(NvU8 *base, const EMUPowerBufferDescs &);

protected:
    NvU8 *_base;
    const EMUPowerBufferDescs &_n;
};


//
// struct emu_softmax_buffer_descs
//
class EMUSoftmaxBufferDescs
{
public:
    virtual size_t struct_size()  const = 0;
    virtual size_t struct_align() const = 0;

    virtual EMUBufferDescAccessor srcDataAccessor(NvU8 *base) const = 0;
    virtual EMUBufferDescAccessor dstDataAccessor(NvU8 *base) const = 0;

protected:
    EMUSoftmaxBufferDescs()          { }
    virtual ~EMUSoftmaxBufferDescs() { }
};

class EMUSoftmaxBufferDescsAccessor
{
public:
    NvU8 * struct_base()  const;
    size_t struct_size()  const;
    size_t struct_align() const;

    EMUBufferDescAccessor srcDataAccessor() const;
    EMUBufferDescAccessor dstDataAccessor() const;

    EMUSoftmaxBufferDescsAccessor(NvU8 *base, const EMUSoftmaxBufferDescs &);

protected:
    NvU8 *_base;
    const EMUSoftmaxBufferDescs &_n;
};


//
// union emu_operation_buffer_container
//
class EMUOperationBufferContainer
{
public:
    virtual size_t struct_size()  const = 0;
    virtual size_t struct_align() const = 0;

    virtual EMUPowerBufferDescsAccessor powerBufferDescsAccessor(NvU8 *base, size_t c) const = 0;
    virtual EMUSoftmaxBufferDescsAccessor softmaxBufferDescsAccessor(NvU8 *base, size_t c) const = 0;

protected:
    EMUOperationBufferContainer()          { }
    virtual ~EMUOperationBufferContainer() { }
};

class EMUOperationBufferContainerAccessor
{
public:
    NvU8 * struct_base()  const;
    size_t struct_size()  const;
    size_t struct_align() const;

    EMUPowerBufferDescsAccessor powerBufferDescsAccessor(size_t c) const;
    EMUSoftmaxBufferDescsAccessor softmaxBufferDescsAccessor(size_t c) const;

    EMUOperationBufferContainerAccessor(NvU8 *base, const EMUOperationBufferContainer &);

protected:
    NvU8 *_base;
    const EMUOperationBufferContainer &_n;
};


class EMUInterface
{
public:
    virtual ~EMUInterface() { }

    // these are the targeted versions
    virtual NvU8 emulatorTargetVersionMajor()    const = 0;
    virtual NvU8 emulatorTargetVersionMinor()    const = 0;
    virtual NvU8 emulatorTargetVersionSubminor() const = 0;
    virtual NvU32 emulatorTargetVersion()         const = 0;

    virtual const std::string emulatorTargetGerritChange() const = 0;
    virtual const std::string emulatorTargetGerritReview() const = 0;

    // this is what was found to be running
    virtual NvU8 emulatorVersionMajor()    const = 0;
    virtual NvU8 emulatorVersionMinor()    const = 0;
    virtual NvU8 emulatorVersionSubminor() const = 0;
    virtual NvU32 emulatorVersion()         const = 0;

    EMUTaskDescAccessor      taskDescAccessor(NvU8 *base)  const;
    EMUNetworkDescAccessor   networkDescAccessor(NvU8 *base)  const;
    EMUOperationContainerAccessor operationContainerAccessor(NvU8 *base)  const;
    EMUBufferDescAccessor bufferDescAccessor(NvU8 *base)  const;
    EMUOperationBufferContainerAccessor operationBufferContainerAccessor(NvU8 *base)  const;

protected:
    virtual const EMUTaskDesc     & taskDesc()  const = 0;
    virtual const EMUNetworkDesc  & networkDesc()  const = 0;
    virtual const EMUOperationContainer & operationContainer()  const = 0;
    virtual const EMUBufferDesc & bufferDesc()  const = 0;
    virtual const EMUOperationBufferContainer & operationBufferContainer()  const = 0;
};


class EMUInterfaceA : public EMUInterface
{
public:
    virtual ~EMUInterfaceA() { }

    // these are the targeted versions
    NvU8 emulatorTargetVersionMajor()    const;
    NvU8 emulatorTargetVersionMinor()    const;
    NvU8 emulatorTargetVersionSubminor() const;
    NvU32 emulatorTargetVersion()         const;

    const std::string emulatorTargetGerritChange() const;
    const std::string emulatorTargetGerritReview() const;

    // this is what was found to be running
    NvU8 emulatorVersionMajor()    const;
    NvU8 emulatorVersionMinor()    const;
    NvU8 emulatorVersionSubminor() const;
    NvU32 emulatorVersion()         const;

protected:
    const EMUTaskDesc     & taskDesc()  const;
    const EMUNetworkDesc  & networkDesc()  const;
    const EMUOperationContainer & operationContainer()  const;
    const EMUBufferDesc & bufferDesc()  const;
    const EMUOperationBufferContainer & operationBufferContainer()  const;
    const EMUAddress & address()  const;
};


} // nvdla::priv
} // nvdla

#endif
