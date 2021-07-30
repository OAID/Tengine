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

#ifndef NVDLA_PRIV_DLA_INTERFACE_H
#define NVDLA_PRIV_DLA_INTERFACE_H

#include <string>

#include "Type.h"

namespace nvdla {
namespace priv {

//
// struct dla_network_desc
//
class DLANetworkDesc
{
public:
    virtual size_t struct_size() const = 0;
    virtual size_t struct_align() const = 0;

    virtual size_t op_BDMA() const = 0;
    virtual size_t op_CONV() const = 0;
    virtual size_t op_SDP() const = 0;
    virtual size_t op_PDP() const = 0;
    virtual size_t op_CDP() const = 0;
    virtual size_t op_RUBIK() const = 0;
    virtual size_t numOpHeads() const = 0;

    virtual int16_t* operationDescIndex(NvU8* base) const = 0;
    virtual int16_t* surfaceDescIndex(NvU8* base) const = 0;
    virtual int16_t* dependencyGraphIndex(NvU8* base) const = 0;
    virtual int16_t* LUTDataIndex(NvU8* base) const = 0;
    virtual int16_t* ROIArrayIndex(NvU8* base) const = 0;
    virtual int16_t* surfaceIndex(NvU8* base) const = 0;
    virtual int16_t* statListIndex(NvU8* base) const = 0;
    virtual int16_t* reserved1(NvU8* base) const = 0;
    virtual int16_t* opHead(NvU8* base, size_t h) const = 0;
    virtual uint16_t* numROIs(NvU8* base) const = 0;
    virtual uint16_t* numOperations(NvU8* base) const = 0;
    virtual uint16_t* numLUTs(NvU8* base) const = 0;
    virtual uint16_t* numAddresses(NvU8* base) const = 0;
    virtual int16_t* inputLayer(NvU8* base) const = 0;
    virtual uint8_t* dynamicROI(NvU8* base) const = 0;
    virtual uint8_t* reserved0(NvU8* base) const = 0;

protected:
    DLANetworkDesc()
    {
    }
    virtual ~DLANetworkDesc()
    {
    }
};

class DLANetworkDescAccessor
{
public:
    NvU8* struct_base() const;
    size_t struct_size() const;
    size_t struct_align() const;

    size_t op_BDMA() const;
    size_t op_CONV() const;
    size_t op_SDP() const;
    size_t op_PDP() const;
    size_t op_CDP() const;
    size_t op_RUBIK() const;
    size_t numOpHeads() const;

    int16_t* operationDescIndex() const;
    int16_t* surfaceDescIndex() const;
    int16_t* dependencyGraphIndex() const;
    int16_t* LUTDataIndex() const;
    int16_t* ROIArrayIndex() const;
    int16_t* surfaceIndex() const;
    int16_t* statListIndex() const;
    int16_t* reserved1() const;
    int16_t* opHead(size_t h) const;
    uint16_t* numROIs() const;
    uint16_t* numOperations() const;
    uint16_t* numLUTs() const;
    uint16_t* numAddresses() const;
    int16_t* inputLayer() const;
    uint8_t* dynamicROI() const;
    uint8_t* reserved0() const;

    DLANetworkDescAccessor(NvU8* base, const DLANetworkDesc&);

protected:
    NvU8* _base;
    const DLANetworkDesc& _n;
};

//
// struct dla_consumer
//
class DLAConsumer
{
public:
    virtual size_t struct_size() const = 0;
    virtual size_t struct_align() const = 0;

    virtual int16_t* index(NvU8* base) const = 0;
    virtual uint8_t* event(NvU8* base) const = 0;
    virtual uint8_t event_OpCompleted() const = 0;
    virtual uint8_t event_OpProgrammed() const = 0;
    virtual uint8_t event_OpEnabled() const = 0;
    virtual uint8_t event_OpCDMAWeightDone() const = 0;
    virtual uint8_t event_OpCDMADataDone() const = 0;
    virtual uint8_t* res(NvU8* base) const = 0;

protected:
    DLAConsumer()
    {
    }
    virtual ~DLAConsumer()
    {
    }
};

class DLAConsumerAccessor
{
public:
    NvU8* struct_base() const;
    size_t struct_size() const;
    size_t struct_align() const;

    int16_t* index() const;
    uint8_t* event() const;
    uint8_t event_OpCompleted() const;
    uint8_t event_OpProgrammed() const;
    uint8_t event_OpEnabled() const;
    uint8_t event_OpCDMAWeightDone() const;
    uint8_t event_OpCDMADataDone() const;
    uint8_t* res() const;

    DLAConsumerAccessor(NvU8* base, const DLAConsumer&);

protected:
    NvU8* _base;
    const DLAConsumer& _c;
};

//
// struct dla_common_op_desc
//
class DLACommonOpDesc
{
public:
    virtual size_t struct_size() const = 0;
    virtual size_t struct_align() const = 0;

    virtual int16_t* index(NvU8* base) const = 0;
    virtual int8_t* roiIndex(NvU8* base) const = 0;
    virtual uint8_t* opType(NvU8* base) const = 0;
    virtual uint8_t opType_BDMA() const = 0;
    virtual uint8_t opType_CONV() const = 0;
    virtual uint8_t opType_SDP() const = 0;
    virtual uint8_t opType_PDP() const = 0;
    virtual uint8_t opType_CDP() const = 0;
    virtual uint8_t opType_RUBIK() const = 0;
    virtual uint8_t* dependencyCount(NvU8* base) const = 0;
    virtual uint8_t* reserved_xxx(NvU8* base) const = 0;
    virtual uint8_t* reserved0(NvU8* base, size_t i) const = 0;
    virtual size_t numConsumers() const = 0;
    virtual DLAConsumerAccessor consumerAccessor(NvU8* base, size_t c) const = 0;
    virtual DLAConsumerAccessor fusedParentAccessor(NvU8* base) const = 0;

protected:
    DLACommonOpDesc()
    {
    }
    virtual ~DLACommonOpDesc()
    {
    }
};

class DLACommonOpDescAccessor
{
public:
    NvU8* struct_base() const;
    size_t struct_size() const;
    size_t struct_align() const;

    int16_t* index() const;
    int8_t* roiIndex() const;
    uint8_t* opType() const;
    uint8_t opType_BDMA() const;
    uint8_t opType_CONV() const;
    uint8_t opType_SDP() const;
    uint8_t opType_PDP() const;
    uint8_t opType_CDP() const;
    uint8_t opType_RUBIK() const;

    uint8_t* dependencyCount() const;
    uint8_t* reserved_xxx() const;
    uint8_t* reserved0(size_t i) const;
    size_t numConsumers() const;
    DLAConsumerAccessor consumerAccessor(size_t c) const;
    DLAConsumerAccessor fusedParentAccessor() const;

    DLACommonOpDescAccessor(NvU8* base, const DLACommonOpDesc&);

protected:
    NvU8* _base;
    const DLACommonOpDesc& _c;
};

//
// struct dla_bdma_transfer_desc
//
class DLABDMATransferDesc
{
public:
    virtual size_t struct_size() const = 0;
    virtual size_t struct_align() const = 0;

    virtual int16_t* srcAddress(NvU8* base) const = 0;
    virtual int16_t* dstAddress(NvU8* base) const = 0;
    virtual uint32_t* lineSize(NvU8* base) const = 0;
    virtual uint32_t* lineRepeat(NvU8* base) const = 0;
    virtual uint32_t* srcLine(NvU8* base) const = 0;
    virtual uint32_t* dstLine(NvU8* base) const = 0;
    virtual uint32_t* surfaceRepeat(NvU8* base) const = 0;
    virtual uint32_t* srcSurface(NvU8* base) const = 0;
    virtual uint32_t* dstSurface(NvU8* base) const = 0;

protected:
    DLABDMATransferDesc()
    {
    }
    virtual ~DLABDMATransferDesc()
    {
    }
};

class DLABDMATransferDescAccessor
{
public:
    NvU8* struct_base() const;
    size_t struct_size() const;
    size_t struct_align() const;

    int16_t* srcAddress() const;
    int16_t* dstAddress() const;
    uint32_t* lineSize() const;
    uint32_t* lineRepeat() const;
    uint32_t* srcLine() const;
    uint32_t* dstLine() const;
    uint32_t* surfaceRepeat() const;
    uint32_t* srcSurface() const;
    uint32_t* dstSurface() const;

    DLABDMATransferDescAccessor(NvU8* base, const DLABDMATransferDesc&);

protected:
    NvU8* _base;
    const DLABDMATransferDesc& _t;
};

//
// struct dla_bdma_surface_desc
//
class DLABDMASurfaceDesc
{
public:
    virtual size_t struct_size() const = 0;
    virtual size_t struct_align() const = 0;

    virtual uint8_t* srcType(NvU8* base) const = 0;
    virtual uint8_t* dstType(NvU8* base) const = 0;
    virtual uint8_t type_MC() const = 0;
    virtual uint8_t type_CV() const = 0;
    virtual uint8_t type_HW() const = 0;

    virtual uint16_t* numTransfers(NvU8* base) const = 0;
    virtual uint16_t maxNumTransfers() const = 0;
    virtual DLABDMATransferDescAccessor transferAccessor(NvU8* base, size_t c) const = 0;

protected:
    DLABDMASurfaceDesc()
    {
    }
    virtual ~DLABDMASurfaceDesc()
    {
    }
};

class DLABDMASurfaceDescAccessor
{
public:
    NvU8* struct_base() const;
    size_t struct_size() const;
    size_t struct_align() const;

    uint8_t type_MC() const;
    uint8_t type_CV() const;
    uint8_t type_HW() const;
    uint16_t maxNumTransfers() const;

    uint8_t* srcType() const;
    uint8_t* dstType() const;
    uint16_t* numTransfers() const;
    DLABDMATransferDescAccessor transferAccessor(size_t c) const;

    DLABDMASurfaceDescAccessor(NvU8* base, const DLABDMASurfaceDesc&);

protected:
    NvU8* _base;
    const DLABDMASurfaceDesc& _s;
};

//
// struct dla_bdma_op_desc
//
class DLABDMAOpDesc
{
public:
    virtual size_t struct_size() const = 0;
    virtual size_t struct_align() const = 0;

    virtual uint16_t* numTransfers(NvU8* base) const = 0;
    virtual uint16_t* reserved0(NvU8* base) const = 0;

protected:
    DLABDMAOpDesc()
    {
    }
    virtual ~DLABDMAOpDesc()
    {
    }
};

class DLABDMAOpDescAccessor
{
public:
    NvU8* struct_base() const;
    size_t struct_size() const;
    size_t struct_align() const;

    uint16_t* numTransfers() const;
    uint16_t* reserved0() const;

    DLABDMAOpDescAccessor(NvU8* base, const DLABDMAOpDesc&);

protected:
    NvU8* _base;
    const DLABDMAOpDesc& _s;
};

//
// struct dla_bdma_stat_desc
//
class DLABDMAStatDesc
{
public:
    virtual size_t struct_size() const = 0;
    virtual size_t struct_align() const = 0;

    virtual uint32_t* readStall(NvU8* base) const = 0;
    virtual uint32_t* writeStall(NvU8* base) const = 0;
    virtual uint32_t* runtime(NvU8* base) const = 0;

protected:
    DLABDMAStatDesc()
    {
    }
    virtual ~DLABDMAStatDesc()
    {
    }
};

class DLABDMAStatDescAccessor
{
public:
    NvU8* struct_base() const;
    size_t struct_size() const;
    size_t struct_align() const;

    uint32_t* readStall() const;
    uint32_t* writeStall() const;
    uint32_t* runtime() const;

    DLABDMAStatDescAccessor(NvU8* base, const DLABDMAStatDesc&);

protected:
    NvU8* _base;
    const DLABDMAStatDesc& _s;
};

//
// struct dla_cvt_param
//
class DLACVTParam
{
public:
    virtual size_t struct_size() const = 0;
    virtual size_t struct_align() const = 0;

    virtual int16_t* scale(NvU8* base) const = 0;
    virtual uint8_t* truncate(NvU8* base) const = 0;
    virtual int32_t* offset(NvU8* base) const = 0;
    virtual uint8_t* enable(NvU8* base) const = 0;
    virtual uint16_t* reserved_xxx(NvU8* base) const = 0;

protected:
    DLACVTParam()
    {
    }
    virtual ~DLACVTParam()
    {
    }
};

class DLACVTParamAccessor
{
public:
    NvU8* struct_base() const;
    size_t struct_size() const;
    size_t struct_align() const;

    int16_t* scale() const;
    uint8_t* truncate() const;
    int32_t* offset() const;
    uint8_t* enable() const;
    uint16_t* reserved_xxx() const;

    DLACVTParamAccessor(NvU8* base, const DLACVTParam&);

protected:
    NvU8* _base;
    const DLACVTParam& _l;
};

//
// struct dla_data_cube
//
class DLADataCube
{
public:
    virtual size_t struct_size() const = 0;
    virtual size_t struct_align() const = 0;

    virtual uint8_t* type_xxx(NvU8* base) const = 0;
    virtual uint8_t type_MC_xxx() const = 0;
    virtual uint8_t type_CV_xxx() const = 0;
    virtual uint8_t type_HW_xxx() const = 0;
    virtual uint16_t* type(NvU8* base) const = 0;
    virtual uint16_t type_MC() const = 0;
    virtual uint16_t type_CV() const = 0;
    virtual uint16_t type_HW() const = 0;
    virtual int16_t* address(NvU8* base) const = 0;
    virtual uint32_t* offset(NvU8* base) const = 0;
    virtual uint32_t* size(NvU8* base) const = 0;
    virtual uint16_t* width(NvU8* base) const = 0;
    virtual uint16_t* height(NvU8* base) const = 0;
    virtual uint16_t* channel(NvU8* base) const = 0;
    virtual uint16_t* reserved0(NvU8* base) const = 0;
    virtual uint32_t* lineStride(NvU8* base) const = 0;
    virtual uint32_t* surfStride(NvU8* base) const = 0;
    virtual uint32_t* planeStride(NvU8* base) const = 0;

protected:
    DLADataCube()
    {
    }
    virtual ~DLADataCube()
    {
    }
};

class DLADataCubeAccessor
{
public:
    NvU8* struct_base() const;
    size_t struct_size() const;
    size_t struct_align() const;

    uint8_t* type_xxx() const;
    uint8_t type_MC_xxx() const;
    uint8_t type_CV_xxx() const;
    uint8_t type_HW_xxx() const;
    uint16_t* type() const;
    uint16_t type_MC() const;
    uint16_t type_CV() const;
    uint16_t type_HW() const;
    int16_t* address() const;
    uint32_t* offset() const;
    uint32_t* size() const;
    uint16_t* width() const;
    uint16_t* height() const;
    uint16_t* channel() const;
    uint16_t* reserved0() const;
    uint32_t* lineStride() const;
    uint32_t* surfStride() const;
    uint32_t* planeStride() const;

    DLADataCubeAccessor(NvU8* base, const DLADataCube&);

protected:
    NvU8* _base;
    const DLADataCube& _l;
};

//
// struct dla_conv_surface_desc
//
class DLAConvSurfaceDesc
{
public:
    virtual size_t struct_size() const = 0;
    virtual size_t struct_align() const = 0;

    virtual DLADataCubeAccessor weightDataAccessor(NvU8* base) const = 0;
    virtual DLADataCubeAccessor meanDataAccessor(NvU8* base) const = 0;
    virtual DLADataCubeAccessor wmbDataAccessor(NvU8* base) const = 0;
    virtual DLADataCubeAccessor wgsDataAccessor(NvU8* base) const = 0;
    virtual DLADataCubeAccessor srcDataAccessor(NvU8* base) const = 0;
    virtual DLADataCubeAccessor dstDataAccessor(NvU8* base) const = 0;
    virtual uint64_t* offsetU_xxx(NvU8* base) const = 0;
    virtual int64_t* offsetU(NvU8* base) const = 0;
    virtual uint32_t* offsetV(NvU8* base) const = 0;
    virtual uint32_t* inLineUVStride(NvU8* base) const = 0;

protected:
    DLAConvSurfaceDesc()
    {
    }
    virtual ~DLAConvSurfaceDesc()
    {
    }
};

class DLAConvSurfaceDescAccessor
{
public:
    NvU8* struct_base() const;
    size_t struct_size() const;
    size_t struct_align() const;

    DLADataCubeAccessor weightDataAccessor() const;
    DLADataCubeAccessor meanDataAccessor() const;
    DLADataCubeAccessor wmbDataAccessor() const;
    DLADataCubeAccessor wgsDataAccessor() const;
    DLADataCubeAccessor srcDataAccessor() const;
    DLADataCubeAccessor dstDataAccessor() const;
    uint64_t* offsetU_xxx() const;
    int64_t* offsetU() const;
    uint32_t* offsetV() const;
    uint32_t* inLineUVStride() const;

    DLAConvSurfaceDescAccessor(NvU8* base, const DLAConvSurfaceDesc&);

protected:
    NvU8* _base;
    const DLAConvSurfaceDesc& _l;
};

//
// struct dla_conv_op_desc
//
class DLAConvOpDesc
{
public:
    virtual size_t struct_size() const = 0;
    virtual size_t struct_align() const = 0;

    virtual uint8_t* inPrecision(NvU8* base) const = 0;
    virtual uint8_t inPrecision_Int8() const = 0;
    virtual uint8_t inPrecision_Int16() const = 0;
    virtual uint8_t inPrecision_FP16() const = 0;
    virtual uint8_t* outPrecision(NvU8* base) const = 0;
    virtual uint8_t outPrecision_Int8() const = 0;
    virtual uint8_t outPrecision_Int16() const = 0;
    virtual uint8_t outPrecision_FP16() const = 0;
    virtual DLACVTParamAccessor inCVTAccessor(NvU8* base) const = 0;
    virtual DLACVTParamAccessor outCVTAccessor(NvU8* base) const = 0;
    virtual int16_t* padVal(NvU8* base) const = 0;
    virtual uint8_t* convMode(NvU8* base) const = 0;
    virtual uint8_t convMode_Direct() const = 0;
    virtual uint8_t convMode_Winograd() const = 0;
    virtual uint8_t* dataReuse(NvU8* base) const = 0;
    virtual uint8_t* weightReuse(NvU8* base) const = 0;
    virtual uint8_t* skipDataRls(NvU8* base) const = 0;
    virtual uint8_t* skipWeightRls(NvU8* base) const = 0;
    virtual uint8_t* reserved0(NvU8* base) const = 0;
    virtual uint16_t* entryPerSlice(NvU8* base) const = 0;
    virtual uint16_t* fetchGrain(NvU8* base) const = 0;
    virtual uint8_t* dataFormat(NvU8* base) const = 0;
    virtual uint8_t dataFormat_T_R8() const = 0;
    virtual uint8_t dataFormat_T_R10() const = 0;
    virtual uint8_t dataFormat_T_R12() const = 0;
    virtual uint8_t dataFormat_T_R16() const = 0;
    virtual uint8_t dataFormat_T_R16_I() const = 0;
    virtual uint8_t dataFormat_T_R16_F() const = 0;
    virtual uint8_t dataFormat_T_A16B16G16R16() const = 0;
    virtual uint8_t dataFormat_T_X16B16G16R16() const = 0;
    virtual uint8_t dataFormat_T_A16B16G16R16_F() const = 0;
    virtual uint8_t dataFormat_T_A16Y16U16V16() const = 0;
    virtual uint8_t dataFormat_T_V16U16Y16A16() const = 0;
    virtual uint8_t dataFormat_T_A16Y16U16V16_F() const = 0;
    virtual uint8_t dataFormat_T_A8B8G8R8() const = 0;
    virtual uint8_t dataFormat_T_A8R8G8B8() const = 0;
    virtual uint8_t dataFormat_T_B8G8R8A8() const = 0;
    virtual uint8_t dataFormat_T_R8G8B8A8() const = 0;
    virtual uint8_t dataFormat_T_X8B8G8R8() const = 0;
    virtual uint8_t dataFormat_T_X8R8G8B8() const = 0;
    virtual uint8_t dataFormat_T_B8G8R8X8() const = 0;
    virtual uint8_t dataFormat_T_R8G8B8X8() const = 0;
    virtual uint8_t dataFormat_T_A2B10G10R10() const = 0;
    virtual uint8_t dataFormat_T_A2R10G10B10() const = 0;
    virtual uint8_t dataFormat_T_B10G10R10A2() const = 0;
    virtual uint8_t dataFormat_T_R10G10B10A2() const = 0;
    virtual uint8_t dataFormat_T_Y8___U8V8_N444() const = 0;
    virtual uint8_t dataFormat_T_Y8___V8U8_N444() const = 0;
    virtual uint8_t dataFormat_T_Y10___U10V10_N444() const = 0;
    virtual uint8_t dataFormat_T_Y10___V10U10_N444() const = 0;
    virtual uint8_t dataFormat_T_Y12___U12V12_N444() const = 0;
    virtual uint8_t dataFormat_T_Y12___V12U12_N444() const = 0;
    virtual uint8_t dataFormat_T_Y16___U16V16_N444() const = 0;
    virtual uint8_t dataFormat_T_Y16___V16U16_N444() const = 0;
    virtual uint8_t dataFormat_T_Y16___U8V8_N444() const = 0;
    virtual uint8_t dataFormat_T_Y16___V8U8_N444() const = 0;
    virtual uint8_t dataFormat_T_Y8___U8___V8_N444() const = 0;
    virtual uint8_t dataFormat_T_Y10___U10___V10_N444() const = 0;
    virtual uint8_t dataFormat_T_Y12___U12___V12_N444() const = 0;
    virtual uint8_t dataFormat_T_Y16___U16___V16_N444() const = 0;
    virtual uint8_t dataFormat_T_Y16___U8___V8_N444() const = 0;
    virtual uint8_t dataFormat_T_A2Y10U10V10() const = 0;
    virtual uint8_t dataFormat_T_V10U10Y10A2() const = 0;
    virtual uint8_t dataFormat_T_A8Y8U8V8() const = 0;
    virtual uint8_t dataFormat_T_V8U8Y8A8() const = 0;
    virtual uint8_t dataFormat_FEATURE() const = 0;
    virtual uint8_t* pixelMapping(NvU8* base) const = 0;
    virtual uint8_t pixelMapping_PitchLinear() const = 0;
    virtual uint8_t* batch(NvU8* base) const = 0;
    virtual uint8_t* weightFormat(NvU8* base) const = 0;
    virtual uint8_t weightFormat_Uncompressed() const = 0;
    virtual uint8_t weightFormat_Compressed() const = 0;
    virtual uint8_t* dataBank(NvU8* base) const = 0;
    virtual uint8_t* weightBank(NvU8* base) const = 0;
    virtual uint32_t* batchStride(NvU8* base) const = 0;
    virtual uint16_t* release(NvU8* base) const = 0;
    virtual uint8_t* postExtension(NvU8* base) const = 0;
    virtual uint8_t* reserved1_xxx(NvU8* base) const = 0;
    virtual uint8_t* pixelOverride(NvU8* base) const = 0;
    virtual uint8_t pixelOverride_UINT() const = 0;
    virtual uint8_t pixelOverride_INT() const = 0;
    virtual uint8_t* meanFormat(NvU8* base) const = 0;
    virtual uint8_t meanFormat_None() const = 0;
    virtual uint8_t meanFormat_Global() const = 0;
    virtual uint8_t meanFormat_PerPixel() const = 0;
    virtual uint8_t meanFormat_Disable() const = 0;
    virtual uint8_t meanFormat_Enable() const = 0;
    virtual int16_t* meanRY(NvU8* base) const = 0;
    virtual int16_t* meanGU(NvU8* base) const = 0;
    virtual int16_t* meanBV(NvU8* base) const = 0;
    virtual int16_t* meanAX(NvU8* base) const = 0;
    virtual uint8_t* convStrideX(NvU8* base) const = 0;
    virtual uint8_t* convStrideY(NvU8* base) const = 0;
    virtual uint8_t* padXLeft(NvU8* base) const = 0;
    virtual uint8_t* padXRight(NvU8* base) const = 0;
    virtual uint8_t* padYTop(NvU8* base) const = 0;
    virtual uint8_t* padYBottom(NvU8* base) const = 0;
    virtual uint8_t* dilationX(NvU8* base) const = 0;
    virtual uint8_t* dilationY(NvU8* base) const = 0;
    virtual uint8_t* reserved2(NvU8* base, size_t i) const = 0;
    virtual uint8_t* praTruncate(NvU8* base) const = 0;
    virtual uint16_t* inputWidthCSC(NvU8* base) const = 0;
    virtual uint16_t* inputHeightCSC(NvU8* base) const = 0;
    virtual uint16_t* inputChannelCSC(NvU8* base) const = 0;
    virtual uint16_t* kernelWidthCSC(NvU8* base) const = 0;
    virtual uint16_t* kernelHeightCSC(NvU8* base) const = 0;
    virtual uint16_t* kernelChannelCSC(NvU8* base) const = 0;
    virtual uint16_t* inputWidthCMAC(NvU8* base) const = 0;
    virtual uint16_t* inputHeightCMAC(NvU8* base) const = 0;
    virtual uint32_t* bytesPerKernel(NvU8* base) const = 0;

protected:
    DLAConvOpDesc()
    {
    }
    virtual ~DLAConvOpDesc()
    {
    }
};

class DLAConvOpDescAccessor
{
public:
    NvU8* struct_base() const;
    size_t struct_size() const;
    size_t struct_align() const;

    uint8_t* inPrecision() const;
    uint8_t inPrecision_Int8() const;
    uint8_t inPrecision_Int16() const;
    uint8_t inPrecision_FP16() const;
    uint8_t* outPrecision() const;
    uint8_t outPrecision_Int8() const;
    uint8_t outPrecision_Int16() const;
    uint8_t outPrecision_FP16() const;
    DLACVTParamAccessor inCVTAccessor() const;
    DLACVTParamAccessor outCVTAccessor() const;
    int16_t* padVal() const;
    uint8_t* convMode() const;
    uint8_t convMode_Direct() const;
    uint8_t convMode_Winograd() const;
    uint8_t* dataReuse() const;
    uint8_t* weightReuse() const;
    uint8_t* skipDataRls() const;
    uint8_t* skipWeightRls() const;
    uint8_t* reserved0() const;
    uint16_t* entryPerSlice() const;
    uint16_t* fetchGrain() const;
    uint8_t* dataFormat() const;
    uint8_t dataFormat_T_R8() const;
    uint8_t dataFormat_T_R10() const;
    uint8_t dataFormat_T_R12() const;
    uint8_t dataFormat_T_R16() const;
    uint8_t dataFormat_T_R16_I() const;
    uint8_t dataFormat_T_R16_F() const;
    uint8_t dataFormat_T_A16B16G16R16() const;
    uint8_t dataFormat_T_X16B16G16R16() const;
    uint8_t dataFormat_T_A16B16G16R16_F() const;
    uint8_t dataFormat_T_A16Y16U16V16() const;
    uint8_t dataFormat_T_V16U16Y16A16() const;
    uint8_t dataFormat_T_A16Y16U16V16_F() const;
    uint8_t dataFormat_T_A8B8G8R8() const;
    uint8_t dataFormat_T_A8R8G8B8() const;
    uint8_t dataFormat_T_B8G8R8A8() const;
    uint8_t dataFormat_T_R8G8B8A8() const;
    uint8_t dataFormat_T_X8B8G8R8() const;
    uint8_t dataFormat_T_X8R8G8B8() const;
    uint8_t dataFormat_T_B8G8R8X8() const;
    uint8_t dataFormat_T_R8G8B8X8() const;
    uint8_t dataFormat_T_A2B10G10R10() const;
    uint8_t dataFormat_T_A2R10G10B10() const;
    uint8_t dataFormat_T_B10G10R10A2() const;
    uint8_t dataFormat_T_R10G10B10A2() const;
    uint8_t dataFormat_T_Y8___U8V8_N444() const;
    uint8_t dataFormat_T_Y8___V8U8_N444() const;
    uint8_t dataFormat_T_Y10___U10V10_N444() const;
    uint8_t dataFormat_T_Y10___V10U10_N444() const;
    uint8_t dataFormat_T_Y12___U12V12_N444() const;
    uint8_t dataFormat_T_Y12___V12U12_N444() const;
    uint8_t dataFormat_T_Y16___U16V16_N444() const;
    uint8_t dataFormat_T_Y16___V16U16_N444() const;
    uint8_t dataFormat_T_Y16___U8V8_N444() const;
    uint8_t dataFormat_T_Y16___V8U8_N444() const;
    uint8_t dataFormat_T_Y8___U8___V8_N444() const;
    uint8_t dataFormat_T_Y10___U10___V10_N444() const;
    uint8_t dataFormat_T_Y12___U12___V12_N444() const;
    uint8_t dataFormat_T_Y16___U16___V16_N444() const;
    uint8_t dataFormat_T_Y16___U8___V8_N444() const;
    uint8_t dataFormat_T_A2Y10U10V10() const;
    uint8_t dataFormat_T_V10U10Y10A2() const;
    uint8_t dataFormat_T_A8Y8U8V8() const;
    uint8_t dataFormat_T_V8U8Y8A8() const;
    uint8_t dataFormat_FEATURE() const;
    uint8_t* pixelMapping() const;
    uint8_t pixelMapping_PitchLinear() const;
    uint8_t* batch() const;
    uint8_t* weightFormat() const;
    uint8_t weightFormat_Uncompressed() const;
    uint8_t weightFormat_Compressed() const;
    uint8_t* dataBank() const;
    uint8_t* weightBank() const;
    uint32_t* batchStride() const;
    uint16_t* release() const;
    uint8_t* postExtension() const;
    uint8_t* reserved1_xxx() const;
    uint8_t* pixelOverride() const;
    uint8_t pixelOverride_UINT() const;
    uint8_t pixelOverride_INT() const;
    uint8_t* meanFormat() const;
    uint8_t meanFormat_None() const;
    uint8_t meanFormat_Global() const;
    uint8_t meanFormat_PerPixel() const;
    uint8_t meanFormat_Disable() const;
    uint8_t meanFormat_Enable() const;
    int16_t* meanRY() const;
    int16_t* meanGU() const;
    int16_t* meanBV() const;
    int16_t* meanAX() const;
    uint8_t* convStrideX() const;
    uint8_t* convStrideY() const;
    uint8_t* padXLeft() const;
    uint8_t* padXRight() const;
    uint8_t* padYTop() const;
    uint8_t* padYBottom() const;
    uint8_t* dilationX() const;
    uint8_t* dilationY() const;
    uint8_t* reserved2(size_t) const;
    uint8_t* praTruncate() const;
    uint16_t* inputWidthCSC() const;
    uint16_t* inputHeightCSC() const;
    uint16_t* inputChannelCSC() const;
    uint16_t* kernelWidthCSC() const;
    uint16_t* kernelHeightCSC() const;
    uint16_t* kernelChannelCSC() const;
    uint16_t* inputWidthCMAC() const;
    uint16_t* inputHeightCMAC() const;
    uint32_t* bytesPerKernel() const;

    DLAConvOpDescAccessor(NvU8* base, const DLAConvOpDesc&);

protected:
    NvU8* _base;
    const DLAConvOpDesc& _l;
};

//
// struct dla_conv_stat_desc
//
class DLAConvStatDesc
{
public:
    virtual size_t struct_size() const = 0;
    virtual size_t struct_align() const = 0;

    virtual uint32_t* dataReadStall(NvU8* base) const = 0;
    virtual uint32_t* weightReadStall(NvU8* base) const = 0;
    virtual uint32_t* dataReadLatency(NvU8* base) const = 0;
    virtual uint32_t* weightReadLatency(NvU8* base) const = 0;
    virtual uint32_t* saturationCount(NvU8* base) const = 0;
    virtual uint32_t* nanDataNum(NvU8* base) const = 0;
    virtual uint32_t* nanWeightNum(NvU8* base) const = 0;
    virtual uint32_t* infDataNum(NvU8* base) const = 0;
    virtual uint32_t* infWeightNum(NvU8* base) const = 0;
    virtual uint32_t* runtime(NvU8* base) const = 0;

protected:
    DLAConvStatDesc()
    {
    }
    virtual ~DLAConvStatDesc()
    {
    }
};

class DLAConvStatDescAccessor
{
public:
    NvU8* struct_base() const;
    size_t struct_size() const;
    size_t struct_align() const;

    uint32_t* dataReadStall() const;
    uint32_t* weightReadStall() const;
    uint32_t* dataReadLatency() const;
    uint32_t* weightReadLatency() const;
    uint32_t* saturationCount() const;
    uint32_t* nanDataNum() const;
    uint32_t* nanWeightNum() const;
    uint32_t* infDataNum() const;
    uint32_t* infWeightNum() const;
    uint32_t* runtime() const;

    DLAConvStatDescAccessor(NvU8* base, const DLAConvStatDesc&);

protected:
    NvU8* _base;
    const DLAConvStatDesc& _l;
};

//
//  union dla_lut_offset
//
class DLALUTOffset
{
public:
    virtual size_t struct_size() const = 0;
    virtual size_t struct_align() const = 0;

    virtual uint8_t* expOffset_xxx(NvU8* base) const = 0;
    virtual int8_t* expOffset(NvU8* base) const = 0;
    virtual uint8_t* fracBits_xxx(NvU8* base) const = 0;
    virtual int8_t* fracBits(NvU8* base) const = 0;
    virtual uint16_t* reserved0(NvU8* base) const = 0;

protected:
    DLALUTOffset()
    {
    }
    virtual ~DLALUTOffset()
    {
    }
};

class DLALUTOffsetAccessor
{
public:
    NvU8* struct_base() const;
    size_t struct_size() const;
    size_t struct_align() const;

    uint8_t* expOffset_xxx() const;
    int8_t* expOffset() const;
    uint8_t* fracBits_xxx() const;
    int8_t* fracBits() const;
    uint16_t* reserved0() const;

    DLALUTOffsetAccessor(NvU8* base, const DLALUTOffset&);

protected:
    NvU8* _base;
    const DLALUTOffset& _l;
};

//
// struct dla_float_data
//
class DLAFloatData
{
public:
    virtual size_t struct_size() const = 0;
    virtual size_t struct_align() const = 0;

    virtual int16_t* scale(NvU8* base) const = 0;
    virtual uint8_t* shifter_xxx(NvU8* base) const = 0;
    virtual int8_t* shifter(NvU8* base) const = 0;
    virtual uint8_t* reserved0(NvU8* base) const = 0;

protected:
    DLAFloatData()
    {
    }
    virtual ~DLAFloatData()
    {
    }
};

class DLAFloatDataAccessor
{
public:
    NvU8* struct_base() const;
    size_t struct_size() const;
    size_t struct_align() const;

    int16_t* scale() const;
    uint8_t* shifter_xxx() const;
    int8_t* shifter() const;
    uint8_t* reserved0() const;

    DLAFloatDataAccessor(NvU8* base, const DLAFloatData&);

protected:
    NvU8* _base;
    const DLAFloatData& _l;
};

//
// struct dla_slope
//
class DLASlope
{
public:
    virtual size_t struct_size() const = 0;
    virtual size_t struct_align() const = 0;

    virtual DLAFloatDataAccessor dataIAccessor(NvU8* base) const = 0;
    virtual uint16_t* dataF(NvU8* base) const = 0;

protected:
    DLASlope()
    {
    }
    virtual ~DLASlope()
    {
    }
};

class DLASlopeAccessor
{
public:
    NvU8* struct_base() const;
    size_t struct_size() const;
    size_t struct_align() const;

    DLAFloatDataAccessor dataIAccessor() const;
    uint16_t* dataF() const;

    DLASlopeAccessor(NvU8* base, const DLASlope&);

protected:
    NvU8* _base;
    const DLASlope& _l;
};

//
// struct dla_lut_param
//
class DLALUTParam
{
public:
    virtual size_t struct_size() const = 0;
    virtual size_t struct_align() const = 0;

    virtual int16_t* linearExpTable(NvU8* base, size_t i) const = 0;
    virtual size_t numLinearExpTable() const = 0;
    virtual int16_t* linearOnlyTable(NvU8* base, size_t i) const = 0;
    virtual size_t numLinearOnlyTable() const = 0;

    virtual uint8_t* method(NvU8* base) const = 0;
    virtual uint8_t method_Exponential() const = 0;
    virtual uint8_t method_Linear() const = 0;
    virtual DLALUTOffsetAccessor linearExpOffsetAccessor(NvU8* base) const = 0;
    virtual DLALUTOffsetAccessor linearOnlyOffsetAccessor(NvU8* base) const = 0;

    virtual uint64_t* linearExpStart(NvU8* base) const = 0;
    virtual uint64_t* linearExpEnd(NvU8* base) const = 0;
    virtual uint64_t* linearOnlyStart(NvU8* base) const = 0;
    virtual uint64_t* linearOnlyEnd(NvU8* base) const = 0;

    virtual DLASlopeAccessor linearExpUnderflowSlopeAccessor(NvU8* base) const = 0;
    virtual DLASlopeAccessor linearExpOverflowSlopeAccessor(NvU8* base) const = 0;
    virtual DLASlopeAccessor linearOnlyUnderflowSlopeAccessor(NvU8* base) const = 0;
    virtual DLASlopeAccessor linearOnlyOverflowSlopeAccessor(NvU8* base) const = 0;

    virtual uint8_t* hybridPriority(NvU8* base) const = 0;
    virtual uint8_t* underflowPriority(NvU8* base) const = 0;
    virtual uint8_t* overflowPriority(NvU8* base) const = 0;
    virtual uint8_t priority_LinearExp() const = 0;
    virtual uint8_t priority_LinearOnly() const = 0;
    virtual int8_t* inputScaleLog2(NvU8* base) const = 0;

protected:
    DLALUTParam()
    {
    }
    virtual ~DLALUTParam()
    {
    }
};

class DLALUTParamAccessor
{
public:
    NvU8* struct_base() const;
    size_t struct_size() const;
    size_t struct_align() const;

    int16_t* linearExpTable(size_t i) const;
    size_t numLinearExpTable() const;
    int16_t* linearOnlyTable(size_t i) const;
    size_t numLinearOnlyTable() const;
    uint8_t* method() const;
    uint8_t method_Exponential() const;
    uint8_t method_Linear() const;
    DLALUTOffsetAccessor linearExpOffsetAccessor() const;
    DLALUTOffsetAccessor linearOnlyOffsetAccessor() const;
    uint64_t* linearExpStart() const;
    uint64_t* linearExpEnd() const;
    uint64_t* linearOnlyStart() const;
    uint64_t* linearOnlyEnd() const;
    DLASlopeAccessor linearExpUnderflowSlopeAccessor() const;
    DLASlopeAccessor linearExpOverflowSlopeAccessor() const;
    DLASlopeAccessor linearOnlyUnderflowSlopeAccessor() const;
    DLASlopeAccessor linearOnlyOverflowSlopeAccessor() const;
    uint8_t* hybridPriority() const;
    uint8_t* underflowPriority() const;
    uint8_t* overflowPriority() const;
    uint8_t priority_LinearExp() const;
    uint8_t priority_LinearOnly() const;
    int8_t* inputScaleLog2() const;

    DLALUTParamAccessor(NvU8* base, const DLALUTParam&);

protected:
    NvU8* _base;
    const DLALUTParam& _l;
};

//
// struct dla_sdp_surface_desc
//
class DLASDPSurfaceDesc
{
public:
    virtual size_t struct_size() const = 0;
    virtual size_t struct_align() const = 0;

    virtual DLADataCubeAccessor srcDataAccessor(NvU8* base) const = 0;
    virtual DLADataCubeAccessor x1DataAccessor(NvU8* base) const = 0;
    virtual DLADataCubeAccessor x2DataAccessor(NvU8* base) const = 0;
    virtual DLADataCubeAccessor yDataAccessor(NvU8* base) const = 0;
    virtual DLADataCubeAccessor dstDataAccessor(NvU8* base) const = 0;

protected:
    DLASDPSurfaceDesc()
    {
    }
    virtual ~DLASDPSurfaceDesc()
    {
    }
};

class DLASDPSurfaceDescAccessor
{
public:
    NvU8* struct_base() const;
    size_t struct_size() const;
    size_t struct_align() const;

    DLADataCubeAccessor srcDataAccessor() const;
    DLADataCubeAccessor x1DataAccessor() const;
    DLADataCubeAccessor x2DataAccessor() const;
    DLADataCubeAccessor yDataAccessor() const;
    DLADataCubeAccessor dstDataAccessor() const;

    DLASDPSurfaceDescAccessor(NvU8* base, const DLASDPSurfaceDesc&);

protected:
    NvU8* _base;
    const DLASDPSurfaceDesc& _l;
};

class DLASDPCVT
{
public:
    virtual size_t struct_size() const = 0;
    virtual size_t struct_align() const = 0;

    virtual DLACVTParamAccessor aluCVTAccessor(NvU8* base) const = 0;
    virtual DLACVTParamAccessor mulCVTAccessor(NvU8* base) const = 0;

protected:
    DLASDPCVT()
    {
    }
    virtual ~DLASDPCVT()
    {
    }
};

class DLASDPCVTAccessor
{
public:
    NvU8* struct_base() const;
    size_t struct_size() const;
    size_t struct_align() const;

    DLACVTParamAccessor aluCVTAccessor() const;
    DLACVTParamAccessor mulCVTAccessor() const;

    DLASDPCVTAccessor(NvU8* base, const DLASDPCVT&);

protected:
    NvU8* _base;
    const DLASDPCVT& _l;
};

//
// dla_sdp_op
//
class DLASDPOp
{
public:
    virtual size_t struct_size() const = 0;
    virtual size_t struct_align() const = 0;

    virtual uint8_t* enable(NvU8* base) const = 0;
    virtual uint8_t* ALUType(NvU8* base) const = 0;
    virtual uint8_t ALUType_Max() const = 0;
    virtual uint8_t ALUType_Min() const = 0;
    virtual uint8_t ALUType_Sum() const = 0;
    virtual uint8_t ALUType_Eql() const = 0;
    virtual uint8_t* type(NvU8* base) const = 0;
    virtual uint8_t type_None() const = 0;
    virtual uint8_t type_Mul() const = 0;
    virtual uint8_t type_Add() const = 0;
    virtual uint8_t type_Both() const = 0;
    virtual uint8_t* mode(NvU8* base) const = 0;
    virtual uint8_t mode_PerLayer() const = 0;
    virtual uint8_t mode_PerKernel() const = 0;
    virtual uint8_t mode_PerPoint() const = 0;
    virtual uint8_t* act(NvU8* base) const = 0;
    virtual uint8_t act_None() const = 0;
    virtual uint8_t act_RelU() const = 0;
    virtual uint8_t act_LUT() const = 0;
    virtual uint8_t* shiftValue(NvU8* base) const = 0;
    virtual int16_t* ALUOperand_xxx(NvU8* base) const = 0;
    virtual int16_t* MulOperand_xxx(NvU8* base) const = 0;
    virtual int32_t* ALUOperand(NvU8* base) const = 0;
    virtual int32_t* MulOperand(NvU8* base) const = 0;
    virtual uint8_t* truncate(NvU8* base) const = 0;
    virtual uint8_t* precision(NvU8* base) const = 0;
    virtual DLASDPCVTAccessor cvt(NvU8* base) const = 0;

protected:
    DLASDPOp()
    {
    }
    virtual ~DLASDPOp()
    {
    }
};

class DLASDPOpAccessor
{
public:
    NvU8* struct_base() const;
    size_t struct_size() const;
    size_t struct_align() const;

    uint8_t* enable() const;
    uint8_t* ALUType() const;
    uint8_t ALUType_Max() const;
    uint8_t ALUType_Min() const;
    uint8_t ALUType_Sum() const;
    uint8_t ALUType_Eql() const;
    uint8_t* type() const;
    uint8_t type_None() const;
    uint8_t type_Mul() const;
    uint8_t type_Add() const;
    uint8_t type_Both() const;
    uint8_t* mode() const;
    uint8_t mode_PerLayer() const;
    uint8_t mode_PerKernel() const;
    uint8_t mode_PerPoint() const;
    uint8_t* act() const;
    uint8_t act_None() const;
    uint8_t act_RelU() const;
    uint8_t act_LUT() const;
    uint8_t* shiftValue() const;
    int16_t* ALUOperand_xxx() const;
    int16_t* MulOperand_xxx() const;
    int32_t* ALUOperand() const;
    int32_t* MulOperand() const;
    uint8_t* truncate() const;
    uint8_t* precision() const;
    DLASDPCVTAccessor cvt() const;

    DLASDPOpAccessor(NvU8* base, const DLASDPOp&);

protected:
    NvU8* _base;
    const DLASDPOp& _l;
};

//
// struct dla_sdp_op_desc
//
class DLASDPOpDesc
{
public:
    virtual size_t struct_size() const = 0;
    virtual size_t struct_align() const = 0;

    virtual uint8_t* srcPrecision(NvU8* base) const = 0;
    virtual uint8_t srcPrecision_Int8() const = 0;
    virtual uint8_t srcPrecision_Int16() const = 0;
    virtual uint8_t srcPrecision_FP16() const = 0;
    virtual uint8_t* dstPrecision(NvU8* base) const = 0;
    virtual uint8_t dstPrecision_Int8() const = 0;
    virtual uint8_t dstPrecision_Int16() const = 0;
    virtual uint8_t dstPrecision_FP16() const = 0;
    virtual int16_t* LUTIndex(NvU8* base) const = 0;
    virtual DLACVTParamAccessor outCVTAccessor(NvU8* base) const = 0;
    virtual uint8_t* convMode(NvU8* base) const = 0;
    virtual uint8_t convMode_Direct() const = 0;
    virtual uint8_t convMode_Winograd() const = 0;
    virtual uint8_t* batchNum(NvU8* base) const = 0;
    virtual uint16_t* reserved0(NvU8* base) const = 0;
    virtual uint32_t* batchStride(NvU8* base) const = 0;
    virtual DLASDPOpAccessor x1OpAccessor(NvU8* base) const = 0;
    virtual DLASDPOpAccessor x2OpAccessor(NvU8* base) const = 0;
    virtual DLASDPOpAccessor yOpAccessor(NvU8* base) const = 0;

protected:
    DLASDPOpDesc()
    {
    }
    virtual ~DLASDPOpDesc()
    {
    }
};

class DLASDPOpDescAccessor
{
public:
    NvU8* struct_base() const;
    size_t struct_size() const;
    size_t struct_align() const;

    uint8_t* srcPrecision() const;
    uint8_t srcPrecision_Int8() const;
    uint8_t srcPrecision_Int16() const;
    uint8_t srcPrecision_FP16() const;
    uint8_t* dstPrecision() const;
    uint8_t dstPrecision_Int8() const;
    uint8_t dstPrecision_Int16() const;
    uint8_t dstPrecision_FP16() const;
    int16_t* LUTIndex() const;
    DLACVTParamAccessor outCVTAccessor() const;
    uint8_t* convMode() const;
    uint8_t convMode_Direct() const;
    uint8_t convMode_Winograd() const;
    uint8_t* batchNum() const;
    uint16_t* reserved0() const;
    uint32_t* batchStride() const;
    DLASDPOpAccessor x1OpAccessor() const;
    DLASDPOpAccessor x2OpAccessor() const;
    DLASDPOpAccessor yOpAccessor() const;

    DLASDPOpDescAccessor(NvU8* base, const DLASDPOpDesc&);

protected:
    NvU8* _base;
    const DLASDPOpDesc& _l;
};

//
// struct dla_sdp_stat_desc
//
class DLASDPStatDesc
{
public:
    virtual size_t struct_size() const = 0;
    virtual size_t struct_align() const = 0;

    virtual uint32_t* nanInputNum(NvU8* base) const = 0;
    virtual uint32_t* infInputNum(NvU8* base) const = 0;
    virtual uint32_t* nanOutputNum(NvU8* base) const = 0;
    virtual uint32_t* wdmaWriteStall(NvU8* base) const = 0;
    virtual uint32_t* lutUnderflow(NvU8* base) const = 0;
    virtual uint32_t* lutOverflow(NvU8* base) const = 0;
    virtual uint32_t* lutHybrid(NvU8* base) const = 0;
    virtual uint32_t* lutLEHit(NvU8* base) const = 0;
    virtual uint32_t* lutLOHit(NvU8* base) const = 0;
    virtual uint32_t* saturationCount(NvU8* base) const = 0;
    virtual uint32_t* runtime(NvU8* base) const = 0;

protected:
    DLASDPStatDesc()
    {
    }
    virtual ~DLASDPStatDesc()
    {
    }
};

class DLASDPStatDescAccessor
{
public:
    NvU8* struct_base() const;
    size_t struct_size() const;
    size_t struct_align() const;

    uint32_t* nanInputNum() const;
    uint32_t* infInputNum() const;
    uint32_t* nanOutputNum() const;
    uint32_t* wdmaWriteStall() const;
    uint32_t* lutUnderflow() const;
    uint32_t* lutOverflow() const;
    uint32_t* lutHybrid() const;
    uint32_t* lutLEHit() const;
    uint32_t* lutLOHit() const;
    uint32_t* saturationCount() const;
    uint32_t* runtime() const;

    DLASDPStatDescAccessor(NvU8* base, const DLASDPStatDesc&);

protected:
    NvU8* _base;
    const DLASDPStatDesc& _l;
};

//
// struct dla_pdp_surface_desc
//
class DLAPDPSurfaceDesc
{
public:
    virtual size_t struct_size() const = 0;
    virtual size_t struct_align() const = 0;

    virtual DLADataCubeAccessor srcDataAccessor(NvU8* base) const = 0;
    virtual DLADataCubeAccessor dstDataAccessor(NvU8* base) const = 0;

protected:
    DLAPDPSurfaceDesc()
    {
    }
    virtual ~DLAPDPSurfaceDesc()
    {
    }
};

class DLAPDPSurfaceDescAccessor
{
public:
    NvU8* struct_base() const;
    size_t struct_size() const;
    size_t struct_align() const;

    DLADataCubeAccessor srcDataAccessor() const;
    DLADataCubeAccessor dstDataAccessor() const;

    DLAPDPSurfaceDescAccessor(NvU8* base, const DLAPDPSurfaceDesc&);

protected:
    NvU8* _base;
    const DLAPDPSurfaceDesc& _l;
};

//
// dla_pdp_op_desc
//
class DLAPDPOpDesc
{
public:
    virtual size_t struct_size() const = 0;
    virtual size_t struct_align() const = 0;

    virtual uint8_t* precision(NvU8* base) const = 0;
    virtual uint8_t precision_Int8() const = 0;
    virtual uint8_t precision_Int16() const = 0;
    virtual uint8_t precision_FP16() const = 0;
    virtual uint8_t* reserved0_xxx(NvU8* base, size_t i) const = 0;
    virtual uint8_t* reserved0(NvU8* base) const = 0;
    virtual int16_t* paddingValue_xxx(NvU8* base, size_t i) const = 0;
    virtual int32_t* paddingValue(NvU8* base, size_t i) const = 0;
    virtual uint8_t* splitNum(NvU8* base) const = 0;
    virtual uint8_t* reserved1_xxx(NvU8* base, size_t i) const = 0;
    virtual uint16_t* partialInWidthFirst(NvU8* base) const = 0;
    virtual uint16_t* partialInWidthMid(NvU8* base) const = 0;
    virtual uint16_t* partialInWidthLast(NvU8* base) const = 0;

    virtual uint16_t* partialWidthFirst(NvU8* base) const = 0;
    virtual uint16_t* partialWidthMid(NvU8* base) const = 0;
    virtual uint16_t* partialWidthLast(NvU8* base) const = 0;

    virtual uint8_t* poolMode(NvU8* base) const = 0;
    virtual uint8_t poolMode_AVG() const = 0;
    virtual uint8_t poolMode_MAX() const = 0;
    virtual uint8_t poolMode_MIN() const = 0;
    virtual uint8_t* poolWidth(NvU8* base) const = 0;
    virtual uint8_t* poolHeight(NvU8* base) const = 0;
    virtual uint8_t* reserved2_xxx(NvU8* base) const = 0;

    virtual uint8_t* strideX(NvU8* base) const = 0;
    virtual uint8_t* strideY(NvU8* base) const = 0;
    virtual uint16_t* strideX_xxx(NvU8* base) const = 0;
    virtual uint16_t* strideY_xxx(NvU8* base) const = 0;
    virtual uint16_t* reserved3_xxx(NvU8* base) const = 0;

    virtual uint8_t* padLeft(NvU8* base) const = 0;
    virtual uint8_t* padRight(NvU8* base) const = 0;
    virtual uint8_t* padTop(NvU8* base) const = 0;
    virtual uint8_t* padBottom(NvU8* base) const = 0;

protected:
    DLAPDPOpDesc()
    {
    }
    virtual ~DLAPDPOpDesc()
    {
    }
};

class DLAPDPOpDescAccessor
{
public:
    NvU8* struct_base() const;
    size_t struct_size() const;
    size_t struct_align() const;

    uint8_t* precision() const;
    uint8_t precision_Int8() const;
    uint8_t precision_Int16() const;
    uint8_t precision_FP16() const;
    uint8_t* reserved0_xxx(size_t i) const;
    uint8_t* reserved0() const;

    int16_t* paddingValue_xxx(size_t i) const;
    int32_t* paddingValue(size_t i) const;
    uint8_t* splitNum() const;
    uint8_t* reserved1_xxx(size_t i) const;
    uint16_t* partialInWidthFirst() const;
    uint16_t* partialInWidthMid() const;
    uint16_t* partialInWidthLast() const;

    uint16_t* partialWidthFirst() const;
    uint16_t* partialWidthMid() const;
    uint16_t* partialWidthLast() const;

    uint8_t* poolMode() const;
    uint8_t poolMode_AVG() const;
    uint8_t poolMode_MAX() const;
    uint8_t poolMode_MIN() const;
    uint8_t* poolWidth() const;
    uint8_t* poolHeight() const;
    uint8_t* reserved2_xxx() const;

    uint8_t* strideX() const;
    uint8_t* strideY() const;
    uint16_t* strideX_xxx() const;
    uint16_t* strideY_xxx() const;
    uint16_t* reserved3_xxx() const;

    uint8_t* padLeft() const;
    uint8_t* padRight() const;
    uint8_t* padTop() const;
    uint8_t* padBottom() const;

    DLAPDPOpDescAccessor(NvU8* base, const DLAPDPOpDesc&);

protected:
    NvU8* _base;
    const DLAPDPOpDesc& _l;
};

//
// struct dla_pdp_stat_desc
//
class DLAPDPStatDesc
{
public:
    virtual size_t struct_size() const = 0;
    virtual size_t struct_align() const = 0;

    virtual uint32_t* infInputNum(NvU8* base) const = 0;
    virtual uint32_t* nanInputNum(NvU8* base) const = 0;
    virtual uint32_t* nanOutputNum(NvU8* base) const = 0;
    virtual uint32_t* writeStall(NvU8* base) const = 0;
    virtual uint32_t* runtime(NvU8* base) const = 0;

protected:
    DLAPDPStatDesc()
    {
    }
    virtual ~DLAPDPStatDesc()
    {
    }
};

class DLAPDPStatDescAccessor
{
public:
    NvU8* struct_base() const;
    size_t struct_size() const;
    size_t struct_align() const;

    uint32_t* infInputNum() const;
    uint32_t* nanInputNum() const;
    uint32_t* nanOutputNum() const;
    uint32_t* writeStall() const;
    uint32_t* runtime() const;

    DLAPDPStatDescAccessor(NvU8* base, const DLAPDPStatDesc&);

protected:
    NvU8* _base;
    const DLAPDPStatDesc& _l;
};

//
// dla_cdp_surface_desc
//
class DLACDPSurfaceDesc
{
public:
    virtual size_t struct_size() const = 0;
    virtual size_t struct_align() const = 0;

    virtual DLADataCubeAccessor srcDataAccessor(NvU8* base) const = 0;
    virtual DLADataCubeAccessor dstDataAccessor(NvU8* base) const = 0;

protected:
    DLACDPSurfaceDesc()
    {
    }
    virtual ~DLACDPSurfaceDesc()
    {
    }
};

class DLACDPSurfaceDescAccessor
{
public:
    NvU8* struct_base() const;
    size_t struct_size() const;
    size_t struct_align() const;

    DLADataCubeAccessor srcDataAccessor() const;
    DLADataCubeAccessor dstDataAccessor() const;

    DLACDPSurfaceDescAccessor(NvU8* base, const DLACDPSurfaceDesc&);

protected:
    NvU8* _base;
    const DLACDPSurfaceDesc& _l;
};

//
// dla_cdp_op_desc
//
class DLACDPOpDesc
{
public:
    virtual size_t struct_size() const = 0;
    virtual size_t struct_align() const = 0;

    virtual uint8_t* inPrecision(NvU8* base) const = 0;
    virtual uint8_t inPrecision_Int8() const = 0;
    virtual uint8_t inPrecision_Int16() const = 0;
    virtual uint8_t inPrecision_FP16() const = 0;
    virtual uint8_t* outPrecision(NvU8* base) const = 0;
    virtual uint8_t outPrecision_Int8() const = 0;
    virtual uint8_t outPrecision_Int16() const = 0;
    virtual uint8_t outPrecision_FP16() const = 0;
    virtual int16_t* LUTIndex(NvU8* base) const = 0;
    virtual DLACVTParamAccessor inCVTAccessor(NvU8* base) const = 0;
    virtual DLACVTParamAccessor outCVTAccessor(NvU8* base) const = 0;
    virtual uint8_t* localSize(NvU8* base) const = 0;
    virtual uint8_t* bypassSquareSum(NvU8* base) const = 0;
    virtual uint8_t* bypassOutMul(NvU8* base) const = 0;
    virtual uint8_t* reserved0(NvU8* base) const = 0;

protected:
    DLACDPOpDesc()
    {
    }
    virtual ~DLACDPOpDesc()
    {
    }
};

class DLACDPOpDescAccessor
{
public:
    NvU8* struct_base() const;
    size_t struct_size() const;
    size_t struct_align() const;

    uint8_t* inPrecision() const;
    uint8_t inPrecision_Int8() const;
    uint8_t inPrecision_Int16() const;
    uint8_t inPrecision_FP16() const;
    uint8_t* outPrecision() const;
    uint8_t outPrecision_Int8() const;
    uint8_t outPrecision_Int16() const;
    uint8_t outPrecision_FP16() const;
    int16_t* LUTIndex() const;
    DLACVTParamAccessor inCVTAccessor() const;
    DLACVTParamAccessor outCVTAccessor() const;
    uint8_t* localSize() const;
    uint8_t* bypassSquareSum() const;
    uint8_t* bypassOutMul() const;
    uint8_t* reserved0() const;

    DLACDPOpDescAccessor(NvU8* base, const DLACDPOpDesc&);

protected:
    NvU8* _base;
    const DLACDPOpDesc& _l;
};

//
// struct dla_cdp_stat_desc
//
class DLACDPStatDesc
{
public:
    virtual size_t struct_size() const = 0;
    virtual size_t struct_align() const = 0;

    virtual uint32_t* nanInputNum(NvU8* base) const = 0;
    virtual uint32_t* infInputNum(NvU8* base) const = 0;
    virtual uint32_t* nanOutputNum(NvU8* base) const = 0;
    virtual uint32_t* writeStall(NvU8* base) const = 0;
    virtual uint32_t* lutUflow(NvU8* base) const = 0;
    virtual uint32_t* lutOflow(NvU8* base) const = 0;
    virtual uint32_t* lutHybrid(NvU8* base) const = 0;
    virtual uint32_t* lutLEHit(NvU8* base) const = 0;
    virtual uint32_t* lutLOHit(NvU8* base) const = 0;
    virtual uint32_t* saturationCount(NvU8* base) const = 0;
    virtual uint32_t* runtime(NvU8* base) const = 0;

protected:
    DLACDPStatDesc()
    {
    }
    virtual ~DLACDPStatDesc()
    {
    }
};

class DLACDPStatDescAccessor
{
public:
    NvU8* struct_base() const;
    size_t struct_size() const;
    size_t struct_align() const;

    uint32_t* nanInputNum() const;
    uint32_t* infInputNum() const;
    uint32_t* nanOutputNum() const;
    uint32_t* writeStall() const;
    uint32_t* lutUflow() const;
    uint32_t* lutOflow() const;
    uint32_t* lutHybrid() const;
    uint32_t* lutLEHit() const;
    uint32_t* lutLOHit() const;
    uint32_t* saturationCount() const;
    uint32_t* runtime() const;

    DLACDPStatDescAccessor(NvU8* base, const DLACDPStatDesc&);

protected:
    NvU8* _base;
    const DLACDPStatDesc& _l;
};

//
// struct dla_rubik_surface_desc
//
class DLARubikSurfaceDesc
{
public:
    virtual size_t struct_size() const = 0;
    virtual size_t struct_align() const = 0;

    virtual DLADataCubeAccessor srcDataAccessor(NvU8* base) const = 0;
    virtual DLADataCubeAccessor dstDataAccessor(NvU8* base) const = 0;

protected:
    DLARubikSurfaceDesc()
    {
    }
    virtual ~DLARubikSurfaceDesc()
    {
    }
};

class DLARubikSurfaceDescAccessor
{
public:
    NvU8* struct_base() const;
    size_t struct_size() const;
    size_t struct_align() const;

    DLADataCubeAccessor srcDataAccessor() const;
    DLADataCubeAccessor dstDataAccessor() const;

    DLARubikSurfaceDescAccessor(NvU8* base, const DLARubikSurfaceDesc&);

protected:
    NvU8* _base;
    const DLARubikSurfaceDesc& _l;
};

//
// dla_rubik_op_desc
//
class DLARubikOpDesc
{
public:
    virtual size_t struct_size() const = 0;
    virtual size_t struct_align() const = 0;

    virtual uint8_t* mode(NvU8* base) const = 0;
    virtual uint8_t mode_Contract() const = 0;
    virtual uint8_t mode_Split() const = 0;
    virtual uint8_t mode_Merge() const = 0;
    virtual uint8_t* precision(NvU8* base) const = 0;
    virtual uint8_t precision_Int8() const = 0;
    virtual uint8_t precision_Int16() const = 0;
    virtual uint8_t precision_FP16() const = 0;
    virtual uint8_t* strideX(NvU8* base) const = 0;
    virtual uint8_t* strideY(NvU8* base) const = 0;

protected:
    DLARubikOpDesc()
    {
    }
    virtual ~DLARubikOpDesc()
    {
    }
};

class DLARubikOpDescAccessor
{
public:
    NvU8* struct_base() const;
    size_t struct_size() const;
    size_t struct_align() const;

    uint8_t* mode() const;
    uint8_t mode_Contract() const;
    uint8_t mode_Split() const;
    uint8_t mode_Merge() const;
    uint8_t* precision() const;
    uint8_t precision_Int8() const;
    uint8_t precision_Int16() const;
    uint8_t precision_FP16() const;
    uint8_t* strideX() const;
    uint8_t* strideY() const;

    DLARubikOpDescAccessor(NvU8* base, const DLARubikOpDesc&);

protected:
    NvU8* _base;
    const DLARubikOpDesc& _l;
};

//
// struct dla_rubik_stat_desc
//
class DLARubikStatDesc
{
public:
    virtual size_t struct_size() const = 0;
    virtual size_t struct_align() const = 0;

    virtual uint32_t* readStall(NvU8* base) const = 0;
    virtual uint32_t* writeStall(NvU8* base) const = 0;
    virtual uint32_t* runtime(NvU8* base) const = 0;

protected:
    DLARubikStatDesc()
    {
    }
    virtual ~DLARubikStatDesc()
    {
    }
};

class DLARubikStatDescAccessor
{
public:
    NvU8* struct_base() const;
    size_t struct_size() const;
    size_t struct_align() const;

    uint32_t* readStall() const;
    uint32_t* writeStall() const;
    uint32_t* runtime() const;

    DLARubikStatDescAccessor(NvU8* base, const DLARubikStatDesc&);

protected:
    NvU8* _base;
    const DLARubikStatDesc& _l;
};

//
// union dla_surface_container
//
class DLASurfaceContainer
{
public:
    virtual size_t struct_size() const = 0;
    virtual size_t struct_align() const = 0;

    virtual DLABDMASurfaceDescAccessor bdmaSurfaceDescAccessor(NvU8* b, size_t c) const = 0;
    virtual DLAConvSurfaceDescAccessor convSurfaceDescAccessor(NvU8* b, size_t c) const = 0;
    virtual DLASDPSurfaceDescAccessor sdpSurfaceDescAccessor(NvU8* b, size_t c) const = 0;
    virtual DLAPDPSurfaceDescAccessor pdpSurfaceDescAccessor(NvU8* b, size_t c) const = 0;
    virtual DLACDPSurfaceDescAccessor cdpSurfaceDescAccessor(NvU8* b, size_t c) const = 0;
    virtual DLARubikSurfaceDescAccessor rubikSurfaceDescAccessor(NvU8* b, size_t c) const = 0;

protected:
    DLASurfaceContainer()
    {
    }
    virtual ~DLASurfaceContainer()
    {
    }
};

class DLASurfaceContainerAccessor
{
public:
    NvU8* struct_base() const;
    size_t struct_size() const;
    size_t struct_align() const;

    DLABDMASurfaceDescAccessor bdmaSurfaceDescAccessor(size_t c) const;
    DLAConvSurfaceDescAccessor convSurfaceDescAccessor(size_t c) const;
    DLASDPSurfaceDescAccessor sdpSurfaceDescAccessor(size_t c) const;
    DLAPDPSurfaceDescAccessor pdpSurfaceDescAccessor(size_t c) const;
    DLACDPSurfaceDescAccessor cdpSurfaceDescAccessor(size_t c) const;
    DLARubikSurfaceDescAccessor rubikSurfaceDescAccessor(size_t c) const;

    DLASurfaceContainerAccessor(NvU8* base, const DLASurfaceContainer&);

protected:
    NvU8* _base;
    const DLASurfaceContainer& _l;
};

//
// union dla_operation_container
//
class DLAOperationContainer
{
public:
    virtual size_t struct_size() const = 0;
    virtual size_t struct_align() const = 0;

    virtual DLABDMAOpDescAccessor bdmaOpDescAccessor(NvU8* b, size_t c) const = 0;
    virtual DLAConvOpDescAccessor convOpDescAccessor(NvU8* b, size_t c) const = 0;
    virtual DLASDPOpDescAccessor sdpOpDescAccessor(NvU8* b, size_t c) const = 0;
    virtual DLAPDPOpDescAccessor pdpOpDescAccessor(NvU8* b, size_t c) const = 0;
    virtual DLACDPOpDescAccessor cdpOpDescAccessor(NvU8* b, size_t c) const = 0;
    virtual DLARubikOpDescAccessor rubikOpDescAccessor(NvU8* b, size_t c) const = 0;

protected:
    DLAOperationContainer()
    {
    }
    virtual ~DLAOperationContainer()
    {
    }
};

class DLAOperationContainerAccessor
{
public:
    NvU8* struct_base() const;
    size_t struct_size() const;
    size_t struct_align() const;

    DLABDMAOpDescAccessor bdmaOpDescAccessor(size_t c) const;
    DLAConvOpDescAccessor convOpDescAccessor(size_t c) const;
    DLASDPOpDescAccessor sdpOpDescAccessor(size_t c) const;
    DLAPDPOpDescAccessor pdpOpDescAccessor(size_t c) const;
    DLACDPOpDescAccessor cdpOpDescAccessor(size_t c) const;
    DLARubikOpDescAccessor rubikOpDescAccessor(size_t c) const;

    DLAOperationContainerAccessor(NvU8* base, const DLAOperationContainer&);

protected:
    NvU8* _base;
    const DLAOperationContainer& _l;
};

//
// union dla_stat_container
//
class DLAStatContainer
{
public:
    virtual size_t struct_size() const = 0;
    virtual size_t struct_align() const = 0;

    virtual DLABDMAStatDescAccessor bdmaStatDescAccessor(NvU8* b, size_t c) const = 0;
    virtual DLAConvStatDescAccessor convStatDescAccessor(NvU8* b, size_t c) const = 0;
    virtual DLASDPStatDescAccessor sdpStatDescAccessor(NvU8* b, size_t c) const = 0;
    virtual DLAPDPStatDescAccessor pdpStatDescAccessor(NvU8* b, size_t c) const = 0;
    virtual DLACDPStatDescAccessor cdpStatDescAccessor(NvU8* b, size_t c) const = 0;
    virtual DLARubikStatDescAccessor rubikStatDescAccessor(NvU8* b, size_t c) const = 0;

protected:
    DLAStatContainer()
    {
    }
    virtual ~DLAStatContainer()
    {
    }
};

class DLAStatContainerAccessor
{
public:
    NvU8* struct_base() const;
    size_t struct_size() const;
    size_t struct_align() const;

    DLABDMAStatDescAccessor bdmaStatDescAccessor(size_t c) const;
    DLAConvStatDescAccessor convStatDescAccessor(size_t c) const;
    DLASDPStatDescAccessor sdpStatDescAccessor(size_t c) const;
    DLAPDPStatDescAccessor pdpStatDescAccessor(size_t c) const;
    DLACDPStatDescAccessor cdpStatDescAccessor(size_t c) const;
    DLARubikStatDescAccessor rubikStatDescAccessor(size_t c) const;

    DLAStatContainerAccessor(NvU8* base, const DLAStatContainer& l);

protected:
    NvU8* _base;
    const DLAStatContainer& _l;
};

class DLAInterface
{
public:
    virtual ~DLAInterface()
    {
    }

    // these are the targetted versions
    virtual NvU8 firmwareTargetVersionMajor() const = 0;
    virtual NvU8 firmwareTargetVersionMinor() const = 0;
    virtual NvU8 firmwareTargetVersionSubminor() const = 0;
    virtual NvU32 firmwareTargetVersion() const = 0;

    virtual NvU8 firmwareVersionMajor() const = 0;
    virtual NvU8 firmwareVersionMinor() const = 0;
    virtual NvU8 firmwareVersionSubminor() const = 0;
    virtual NvU32 firmwareVersion() const = 0;

    DLANetworkDescAccessor networkDescAccessor(NvU8* b) const;
    DLAConsumerAccessor consumerAccessor(NvU8* b) const;
    DLACommonOpDescAccessor commonOpDescAccessor(NvU8* b) const;
    DLACVTParamAccessor cvtParamAccessor(NvU8* b) const;
    DLADataCubeAccessor dataCubeAccessor(NvU8* b) const;
    DLAConvSurfaceDescAccessor convSurfaceDescAccessor(NvU8* b) const;
    DLAConvOpDescAccessor convOpDescAccessor(NvU8* b) const;
    DLALUTOffsetAccessor lutOffsetAccessor(NvU8* b) const;
    DLAFloatDataAccessor floatDataAccessor(NvU8* b) const;
    DLASlopeAccessor slopeAccessor(NvU8* b) const;
    DLALUTParamAccessor lutParamAccessor(NvU8* b) const;
    DLASDPSurfaceDescAccessor sdpSurfaceDescAccessor(NvU8* b) const;
    DLASDPOpAccessor sdpOpAccessor(NvU8* b) const;
    DLASDPOpDescAccessor sdpOpDescAccessor(NvU8* b) const;
    DLASDPCVTAccessor sdpCVTAccessor(NvU8* b) const;
    DLAPDPSurfaceDescAccessor pdpSurfaceDescAccessor(NvU8* b) const;
    DLAPDPOpDescAccessor pdpOpDescAccessor(NvU8* b) const;
    DLACDPSurfaceDescAccessor cdpSurfaceDescAccessor(NvU8* b) const;
    DLACDPOpDescAccessor cdpOpDescAccessor(NvU8* b) const;
    DLARubikSurfaceDescAccessor rubikSurfaceDescAccessor(NvU8* b) const;
    DLARubikOpDescAccessor rubikOpDescAccessor(NvU8* b) const;
    DLASurfaceContainerAccessor surfaceContainerAccessor(NvU8* b) const;
    DLAOperationContainerAccessor operationContainerAccessor(NvU8* b) const;

protected:
    virtual const DLANetworkDesc& networkDesc() const = 0;
    virtual const DLAConsumer& consumer() const = 0;
    virtual const DLACommonOpDesc& commonOpDesc() const = 0;
    virtual const DLACVTParam& cvtParam() const = 0;
    virtual const DLADataCube& dataCube() const = 0;
    virtual const DLAConvSurfaceDesc& convSurfaceDesc() const = 0;
    virtual const DLAConvOpDesc& convOpDesc() const = 0;
    virtual const DLALUTOffset& lutOffset() const = 0;
    virtual const DLAFloatData& floatData() const = 0;
    virtual const DLASlope& slope() const = 0;
    virtual const DLALUTParam& lutParam() const = 0;
    virtual const DLASDPSurfaceDesc& sdpSurfaceDesc() const = 0;
    virtual const DLASDPOp& sdpOp() const = 0;
    virtual const DLASDPOpDesc& sdpOpDesc() const = 0;
    virtual const DLASDPCVT& sdpCVT() const = 0;
    virtual const DLAPDPSurfaceDesc& pdpSurfaceDesc() const = 0;
    virtual const DLAPDPOpDesc& pdpOpDesc() const = 0;
    virtual const DLACDPSurfaceDesc& cdpSurfaceDesc() const = 0;
    virtual const DLACDPOpDesc& cdpOpDesc() const = 0;
    virtual const DLARubikSurfaceDesc& rubikSurfaceDesc() const = 0;
    virtual const DLARubikOpDesc& rubikOpDesc() const = 0;
    virtual const DLASurfaceContainer& surfaceContainer() const = 0;
    virtual const DLAOperationContainer& operationContainer() const = 0;
};

class DLAInterfaceA : public DLAInterface
{
public:
    virtual ~DLAInterfaceA()
    {
    }

    virtual NvU8 firmwareTargetVersionMajor() const;
    virtual NvU8 firmwareTargetVersionMinor() const;
    virtual NvU8 firmwareTargetVersionSubminor() const;
    virtual NvU32 firmwareTargetVersion() const;

    virtual NvU8 firmwareVersionMajor() const;
    virtual NvU8 firmwareVersionMinor() const;
    virtual NvU8 firmwareVersionSubminor() const;
    virtual NvU32 firmwareVersion() const;

    DLANetworkDescAccessor networkDescAccessor(NvU8* b) const;
    DLAConsumerAccessor consumerAccessor(NvU8* b) const;
    DLACommonOpDescAccessor commonOpDescAccessor(NvU8* b) const;
    DLACVTParamAccessor cvtParamAccessor(NvU8* b) const;
    DLADataCubeAccessor dataCubeAccessor(NvU8* b) const;
    DLAConvSurfaceDescAccessor convSurfaceDescAccessor(NvU8* b) const;
    DLAConvOpDescAccessor convOpDescAccessor(NvU8* b) const;
    DLALUTOffsetAccessor lutOffsetAccessor(NvU8* b) const;
    DLAFloatDataAccessor floatDataAccessor(NvU8* b) const;
    DLASlopeAccessor slopeAccessor(NvU8* b) const;
    DLALUTParamAccessor lutParamAccessor(NvU8* b) const;
    DLASDPSurfaceDescAccessor sdpSurfaceDescAccessor(NvU8* b) const;
    DLASDPOpAccessor sdpOpAccessor(NvU8* b) const;
    DLASDPOpDescAccessor sdpOpDescAccessor(NvU8* b) const;
    DLAPDPSurfaceDescAccessor pdpSurfaceDescAccessor(NvU8* b) const;
    DLAPDPOpDescAccessor pdpOpDescAccessor(NvU8* b) const;
    DLACDPSurfaceDescAccessor cdpSurfaceDescAccessor(NvU8* b) const;
    DLACDPOpDescAccessor cdpOpDescAccessor(NvU8* b) const;
    DLARubikSurfaceDescAccessor rubikSurfaceDescAccessor(NvU8* b) const;
    DLARubikOpDescAccessor rubikOpDescAccessor(NvU8* b) const;
    DLASurfaceContainerAccessor surfaceContainerAccessor(NvU8* b) const;
    DLAOperationContainerAccessor operationContainerAccessor(NvU8* b) const;

protected:
    virtual const DLANetworkDesc& networkDesc() const;
    virtual const DLAConsumer& consumer() const;
    virtual const DLACommonOpDesc& commonOpDesc() const;
    virtual const DLACVTParam& cvtParam() const;
    virtual const DLADataCube& dataCube() const;
    virtual const DLAConvSurfaceDesc& convSurfaceDesc() const;
    virtual const DLAConvOpDesc& convOpDesc() const;
    virtual const DLALUTOffset& lutOffset() const;
    virtual const DLAFloatData& floatData() const;
    virtual const DLASlope& slope() const;
    virtual const DLALUTParam& lutParam() const;
    virtual const DLASDPSurfaceDesc& sdpSurfaceDesc() const;
    virtual const DLASDPOp& sdpOp() const;
    virtual const DLASDPOpDesc& sdpOpDesc() const;
    virtual const DLASDPCVT& sdpCVT() const;
    virtual const DLAPDPSurfaceDesc& pdpSurfaceDesc() const;
    virtual const DLAPDPOpDesc& pdpOpDesc() const;
    virtual const DLACDPSurfaceDesc& cdpSurfaceDesc() const;
    virtual const DLACDPOpDesc& cdpOpDesc() const;
    virtual const DLARubikSurfaceDesc& rubikSurfaceDesc() const;
    virtual const DLARubikOpDesc& rubikOpDesc() const;
    virtual const DLASurfaceContainer& surfaceContainer() const;
    virtual const DLAOperationContainer& operationContainer() const;
    //    virtual const       & ()  const;
    //    virtual const       & ()  const;
};

class DLAInterfaceB : public DLAInterface
{
public:
    virtual ~DLAInterfaceB()
    {
    }

    virtual NvU8 firmwareTargetVersionMajor() const;
    virtual NvU8 firmwareTargetVersionMinor() const;
    virtual NvU8 firmwareTargetVersionSubminor() const;
    virtual NvU32 firmwareTargetVersion() const;

    virtual NvU8 firmwareVersionMajor() const;
    virtual NvU8 firmwareVersionMinor() const;
    virtual NvU8 firmwareVersionSubminor() const;
    virtual NvU32 firmwareVersion() const;

    DLANetworkDescAccessor networkDescAccessor(NvU8* b) const;
    DLAConsumerAccessor consumerAccessor(NvU8* b) const;
    DLACommonOpDescAccessor commonOpDescAccessor(NvU8* b) const;
    DLACVTParamAccessor cvtParamAccessor(NvU8* b) const;
    DLADataCubeAccessor dataCubeAccessor(NvU8* b) const;
    DLAConvSurfaceDescAccessor convSurfaceDescAccessor(NvU8* b) const;
    DLAConvOpDescAccessor convOpDescAccessor(NvU8* b) const;
    DLALUTOffsetAccessor lutOffsetAccessor(NvU8* b) const;
    DLAFloatDataAccessor floatDataAccessor(NvU8* b) const;
    DLASlopeAccessor slopeAccessor(NvU8* b) const;
    DLALUTParamAccessor lutParamAccessor(NvU8* b) const;
    DLASDPSurfaceDescAccessor sdpSurfaceDescAccessor(NvU8* b) const;
    DLASDPOpAccessor sdpOpAccessor(NvU8* b) const;
    DLASDPOpDescAccessor sdpOpDescAccessor(NvU8* b) const;
    DLAPDPSurfaceDescAccessor pdpSurfaceDescAccessor(NvU8* b) const;
    DLAPDPOpDescAccessor pdpOpDescAccessor(NvU8* b) const;
    DLACDPSurfaceDescAccessor cdpSurfaceDescAccessor(NvU8* b) const;
    DLACDPOpDescAccessor cdpOpDescAccessor(NvU8* b) const;
    DLARubikSurfaceDescAccessor rubikSurfaceDescAccessor(NvU8* b) const;
    DLARubikOpDescAccessor rubikOpDescAccessor(NvU8* b) const;
    DLASurfaceContainerAccessor surfaceContainerAccessor(NvU8* b) const;
    DLAOperationContainerAccessor operationContainerAccessor(NvU8* b) const;

protected:
    virtual const DLANetworkDesc& networkDesc() const;
    virtual const DLAConsumer& consumer() const;
    virtual const DLACommonOpDesc& commonOpDesc() const;
    virtual const DLACVTParam& cvtParam() const;
    virtual const DLADataCube& dataCube() const;
    virtual const DLAConvSurfaceDesc& convSurfaceDesc() const;
    virtual const DLAConvOpDesc& convOpDesc() const;
    virtual const DLALUTOffset& lutOffset() const;
    virtual const DLAFloatData& floatData() const;
    virtual const DLASlope& slope() const;
    virtual const DLALUTParam& lutParam() const;
    virtual const DLASDPSurfaceDesc& sdpSurfaceDesc() const;
    virtual const DLASDPOp& sdpOp() const;
    virtual const DLASDPOpDesc& sdpOpDesc() const;
    virtual const DLASDPCVT& sdpCVT() const;
    virtual const DLAPDPSurfaceDesc& pdpSurfaceDesc() const;
    virtual const DLAPDPOpDesc& pdpOpDesc() const;
    virtual const DLACDPSurfaceDesc& cdpSurfaceDesc() const;
    virtual const DLACDPOpDesc& cdpOpDesc() const;
    virtual const DLARubikSurfaceDesc& rubikSurfaceDesc() const;
    virtual const DLARubikOpDesc& rubikOpDesc() const;
    virtual const DLASurfaceContainer& surfaceContainer() const;
    virtual const DLAOperationContainer& operationContainer() const;
    //    virtual const       & ()  const;
    //    virtual const       & ()  const;
};

class DLAInterfaceC : public DLAInterface
{
public:
    virtual ~DLAInterfaceC()
    {
    }

    virtual NvU8 firmwareTargetVersionMajor() const;
    virtual NvU8 firmwareTargetVersionMinor() const;
    virtual NvU8 firmwareTargetVersionSubminor() const;
    virtual NvU32 firmwareTargetVersion() const;

    virtual NvU8 firmwareVersionMajor() const;
    virtual NvU8 firmwareVersionMinor() const;
    virtual NvU8 firmwareVersionSubminor() const;
    virtual NvU32 firmwareVersion() const;

    DLANetworkDescAccessor networkDescAccessor(NvU8* b) const;
    DLAConsumerAccessor consumerAccessor(NvU8* b) const;
    DLACommonOpDescAccessor commonOpDescAccessor(NvU8* b) const;
    DLACVTParamAccessor cvtParamAccessor(NvU8* b) const;
    DLADataCubeAccessor dataCubeAccessor(NvU8* b) const;
    DLAConvSurfaceDescAccessor convSurfaceDescAccessor(NvU8* b) const;
    DLAConvOpDescAccessor convOpDescAccessor(NvU8* b) const;
    DLALUTOffsetAccessor lutOffsetAccessor(NvU8* b) const;
    DLAFloatDataAccessor floatDataAccessor(NvU8* b) const;
    DLASlopeAccessor slopeAccessor(NvU8* b) const;
    DLALUTParamAccessor lutParamAccessor(NvU8* b) const;
    DLASDPSurfaceDescAccessor sdpSurfaceDescAccessor(NvU8* b) const;
    DLASDPOpAccessor sdpOpAccessor(NvU8* b) const;
    DLASDPOpDescAccessor sdpOpDescAccessor(NvU8* b) const;
    DLAPDPSurfaceDescAccessor pdpSurfaceDescAccessor(NvU8* b) const;
    DLAPDPOpDescAccessor pdpOpDescAccessor(NvU8* b) const;
    DLACDPSurfaceDescAccessor cdpSurfaceDescAccessor(NvU8* b) const;
    DLACDPOpDescAccessor cdpOpDescAccessor(NvU8* b) const;
    DLARubikSurfaceDescAccessor rubikSurfaceDescAccessor(NvU8* b) const;
    DLARubikOpDescAccessor rubikOpDescAccessor(NvU8* b) const;
    DLASurfaceContainerAccessor surfaceContainerAccessor(NvU8* b) const;
    DLAOperationContainerAccessor operationContainerAccessor(NvU8* b) const;

protected:
    virtual const DLANetworkDesc& networkDesc() const;
    virtual const DLAConsumer& consumer() const;
    virtual const DLACommonOpDesc& commonOpDesc() const;
    virtual const DLACVTParam& cvtParam() const;
    virtual const DLADataCube& dataCube() const;
    virtual const DLAConvSurfaceDesc& convSurfaceDesc() const;
    virtual const DLAConvOpDesc& convOpDesc() const;
    virtual const DLALUTOffset& lutOffset() const;
    virtual const DLAFloatData& floatData() const;
    virtual const DLASlope& slope() const;
    virtual const DLALUTParam& lutParam() const;
    virtual const DLASDPSurfaceDesc& sdpSurfaceDesc() const;
    virtual const DLASDPOp& sdpOp() const;
    virtual const DLASDPOpDesc& sdpOpDesc() const;
    virtual const DLASDPCVT& sdpCVT() const;
    virtual const DLAPDPSurfaceDesc& pdpSurfaceDesc() const;
    virtual const DLAPDPOpDesc& pdpOpDesc() const;
    virtual const DLACDPSurfaceDesc& cdpSurfaceDesc() const;
    virtual const DLACDPOpDesc& cdpOpDesc() const;
    virtual const DLARubikSurfaceDesc& rubikSurfaceDesc() const;
    virtual const DLARubikOpDesc& rubikOpDesc() const;
    virtual const DLASurfaceContainer& surfaceContainer() const;
    virtual const DLAOperationContainer& operationContainer() const;
    //    virtual const       & ()  const;
    //    virtual const       & ()  const;
};

class DLAInterfaceD : public DLAInterface
{
public:
    virtual ~DLAInterfaceD()
    {
    }

    virtual NvU8 firmwareTargetVersionMajor() const;
    virtual NvU8 firmwareTargetVersionMinor() const;
    virtual NvU8 firmwareTargetVersionSubminor() const;
    virtual NvU32 firmwareTargetVersion() const;

    virtual NvU8 firmwareVersionMajor() const;
    virtual NvU8 firmwareVersionMinor() const;
    virtual NvU8 firmwareVersionSubminor() const;
    virtual NvU32 firmwareVersion() const;

    DLANetworkDescAccessor networkDescAccessor(NvU8* b) const;
    DLAConsumerAccessor consumerAccessor(NvU8* b) const;
    DLACommonOpDescAccessor commonOpDescAccessor(NvU8* b) const;
    DLACVTParamAccessor cvtParamAccessor(NvU8* b) const;
    DLADataCubeAccessor dataCubeAccessor(NvU8* b) const;
    DLAConvSurfaceDescAccessor convSurfaceDescAccessor(NvU8* b) const;
    DLAConvOpDescAccessor convOpDescAccessor(NvU8* b) const;
    DLALUTOffsetAccessor lutOffsetAccessor(NvU8* b) const;
    DLAFloatDataAccessor floatDataAccessor(NvU8* b) const;
    DLASlopeAccessor slopeAccessor(NvU8* b) const;
    DLALUTParamAccessor lutParamAccessor(NvU8* b) const;
    DLASDPSurfaceDescAccessor sdpSurfaceDescAccessor(NvU8* b) const;
    DLASDPOpAccessor sdpOpAccessor(NvU8* b) const;
    DLASDPOpDescAccessor sdpOpDescAccessor(NvU8* b) const;
    DLAPDPSurfaceDescAccessor pdpSurfaceDescAccessor(NvU8* b) const;
    DLAPDPOpDescAccessor pdpOpDescAccessor(NvU8* b) const;
    DLACDPSurfaceDescAccessor cdpSurfaceDescAccessor(NvU8* b) const;
    DLACDPOpDescAccessor cdpOpDescAccessor(NvU8* b) const;
    DLARubikSurfaceDescAccessor rubikSurfaceDescAccessor(NvU8* b) const;
    DLARubikOpDescAccessor rubikOpDescAccessor(NvU8* b) const;
    DLASurfaceContainerAccessor surfaceContainerAccessor(NvU8* b) const;
    DLAOperationContainerAccessor operationContainerAccessor(NvU8* b) const;

protected:
    virtual const DLANetworkDesc& networkDesc() const;
    virtual const DLAConsumer& consumer() const;
    virtual const DLACommonOpDesc& commonOpDesc() const;
    virtual const DLACVTParam& cvtParam() const;
    virtual const DLADataCube& dataCube() const;
    virtual const DLAConvSurfaceDesc& convSurfaceDesc() const;
    virtual const DLAConvOpDesc& convOpDesc() const;
    virtual const DLALUTOffset& lutOffset() const;
    virtual const DLAFloatData& floatData() const;
    virtual const DLASlope& slope() const;
    virtual const DLALUTParam& lutParam() const;
    virtual const DLASDPSurfaceDesc& sdpSurfaceDesc() const;
    virtual const DLASDPOp& sdpOp() const;
    virtual const DLASDPOpDesc& sdpOpDesc() const;
    virtual const DLASDPCVT& sdpCVT() const;
    virtual const DLAPDPSurfaceDesc& pdpSurfaceDesc() const;
    virtual const DLAPDPOpDesc& pdpOpDesc() const;
    virtual const DLACDPSurfaceDesc& cdpSurfaceDesc() const;
    virtual const DLACDPOpDesc& cdpOpDesc() const;
    virtual const DLARubikSurfaceDesc& rubikSurfaceDesc() const;
    virtual const DLARubikOpDesc& rubikOpDesc() const;
    virtual const DLASurfaceContainer& surfaceContainer() const;
    virtual const DLAOperationContainer& operationContainer() const;
    //    virtual const       & ()  const;
    //    virtual const       & ()  const;
};

class DLAInterfaceE : public DLAInterface
{
public:
    virtual ~DLAInterfaceE()
    {
    }

    virtual NvU8 firmwareTargetVersionMajor() const;
    virtual NvU8 firmwareTargetVersionMinor() const;
    virtual NvU8 firmwareTargetVersionSubminor() const;
    virtual NvU32 firmwareTargetVersion() const;

    virtual NvU8 firmwareVersionMajor() const;
    virtual NvU8 firmwareVersionMinor() const;
    virtual NvU8 firmwareVersionSubminor() const;
    virtual NvU32 firmwareVersion() const;

    DLANetworkDescAccessor networkDescAccessor(NvU8* b) const;
    DLAConsumerAccessor consumerAccessor(NvU8* b) const;
    DLACommonOpDescAccessor commonOpDescAccessor(NvU8* b) const;
    DLACVTParamAccessor cvtParamAccessor(NvU8* b) const;
    DLADataCubeAccessor dataCubeAccessor(NvU8* b) const;
    DLAConvSurfaceDescAccessor convSurfaceDescAccessor(NvU8* b) const;
    DLAConvOpDescAccessor convOpDescAccessor(NvU8* b) const;
    DLALUTOffsetAccessor lutOffsetAccessor(NvU8* b) const;
    DLAFloatDataAccessor floatDataAccessor(NvU8* b) const;
    DLASlopeAccessor slopeAccessor(NvU8* b) const;
    DLALUTParamAccessor lutParamAccessor(NvU8* b) const;
    DLASDPSurfaceDescAccessor sdpSurfaceDescAccessor(NvU8* b) const;
    DLASDPOpAccessor sdpOpAccessor(NvU8* b) const;
    DLASDPOpDescAccessor sdpOpDescAccessor(NvU8* b) const;
    DLAPDPSurfaceDescAccessor pdpSurfaceDescAccessor(NvU8* b) const;
    DLAPDPOpDescAccessor pdpOpDescAccessor(NvU8* b) const;
    DLACDPSurfaceDescAccessor cdpSurfaceDescAccessor(NvU8* b) const;
    DLACDPOpDescAccessor cdpOpDescAccessor(NvU8* b) const;
    DLARubikSurfaceDescAccessor rubikSurfaceDescAccessor(NvU8* b) const;
    DLARubikOpDescAccessor rubikOpDescAccessor(NvU8* b) const;
    DLASurfaceContainerAccessor surfaceContainerAccessor(NvU8* b) const;
    DLAOperationContainerAccessor operationContainerAccessor(NvU8* b) const;

protected:
    virtual const DLANetworkDesc& networkDesc() const;
    virtual const DLAConsumer& consumer() const;
    virtual const DLACommonOpDesc& commonOpDesc() const;
    virtual const DLACVTParam& cvtParam() const;
    virtual const DLADataCube& dataCube() const;
    virtual const DLAConvSurfaceDesc& convSurfaceDesc() const;
    virtual const DLAConvOpDesc& convOpDesc() const;
    virtual const DLALUTOffset& lutOffset() const;
    virtual const DLAFloatData& floatData() const;
    virtual const DLASlope& slope() const;
    virtual const DLALUTParam& lutParam() const;
    virtual const DLASDPSurfaceDesc& sdpSurfaceDesc() const;
    virtual const DLASDPOp& sdpOp() const;
    virtual const DLASDPOpDesc& sdpOpDesc() const;
    virtual const DLASDPCVT& sdpCVT() const;
    virtual const DLAPDPSurfaceDesc& pdpSurfaceDesc() const;
    virtual const DLAPDPOpDesc& pdpOpDesc() const;
    virtual const DLACDPSurfaceDesc& cdpSurfaceDesc() const;
    virtual const DLACDPOpDesc& cdpOpDesc() const;
    virtual const DLARubikSurfaceDesc& rubikSurfaceDesc() const;
    virtual const DLARubikOpDesc& rubikOpDesc() const;
    virtual const DLASurfaceContainer& surfaceContainer() const;
    virtual const DLAOperationContainer& operationContainer() const;
    //    virtual const       & ()  const;
    //    virtual const       & ()  const;
};

} // namespace priv
} // namespace nvdla

#endif // NVDLA_PRIV_DLA_INTERFACE_H
