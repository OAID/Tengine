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

#ifndef NVDLA_PRIV_WEIGHT_TRNS_UNIT_H
#define NVDLA_PRIV_WEIGHT_TRNS_UNIT_H

#include <fstream>
#include <iostream>
#include <sstream>
#include "half.h"
#include "nvdla/IType.h"
#include "priv/Check.h"
#include "priv/EngineAST.h"

#define ENABLE_PRECISION_RANGE_CHECK 0

#define WG_FULL_CHANNELS_PER_ATOM  4
#define WG_BYTES_PER_ATOM          32
#define WG_WTS_SIZE_ALIGN          128

#define PRECISION_SWITCH(modelPrec, computePrec, retVal, func, ...)     \
    switch(modelPrec) {                                                 \
        case nvdla::DataType::INT8:                                     \
        switch(computePrec) {                                           \
            case surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT8:   \
                retVal = func<NvS8, NvS8>(__VA_ARGS__); break;          \
            case surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT16:  \
                retVal = func<NvS8, NvS16>(__VA_ARGS__); break;         \
            case surface::SurfacePrecisionEnum::NVDLA_PRECISION_FP16:   \
                retVal = func<NvS8, half_float::half>(__VA_ARGS__); break;          \
            default:                                                    \
            REPORT_ERROR(NvDlaError_NotSupported, "Don't support %d", computePrec);      \
        }; break;                                                       \
        case nvdla::DataType::INT16:                                    \
        switch(computePrec) {                                           \
            case surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT8:   \
                retVal = func<NvS16, NvS8>(__VA_ARGS__); break;         \
            case surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT16:  \
                retVal = func<NvS16, NvS16>(__VA_ARGS__); break;        \
            case surface::SurfacePrecisionEnum::NVDLA_PRECISION_FP16:   \
                retVal = func<NvS16, half_float::half>(__VA_ARGS__); break;         \
            default:                                                    \
            REPORT_ERROR(NvDlaError_NotSupported, "Don't support %d", computePrec);      \
        }; break;                                                       \
        case nvdla::DataType::HALF:                                     \
        switch(computePrec) {                                           \
            case surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT8:   \
                retVal = func<half_float::half, NvS8>(__VA_ARGS__); break;          \
            case surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT16:  \
                retVal = func<half_float::half, NvS16>(__VA_ARGS__); break;         \
            case surface::SurfacePrecisionEnum::NVDLA_PRECISION_FP16:   \
                retVal = func<half_float::half, half_float::half>(__VA_ARGS__); break;          \
            default:                                                    \
            REPORT_ERROR(NvDlaError_NotSupported, "Don't support %d", computePrec);      \
        }; break;                                                       \
        case nvdla::DataType::FLOAT:                                    \
        switch(computePrec) {                                           \
            case surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT8:   \
                retVal = func<NvF32, NvS8>(__VA_ARGS__); break;         \
            case surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT16:  \
                retVal = func<NvF32, NvS16>(__VA_ARGS__); break;        \
            case surface::SurfacePrecisionEnum::NVDLA_PRECISION_FP16:   \
                retVal = func<NvF32, half_float::half>(__VA_ARGS__); break;         \
            default:                                                    \
            REPORT_ERROR(NvDlaError_NotSupported, "Don't support %d", computePrec);      \
        }; break;                                                       \
        default:                                                        \
        REPORT_ERROR(NvDlaError_NotSupported, "Don't support %d", modelPrec);            \
    }

namespace nvdla
{
namespace priv
{

class WeightTrns
{
    static bool debugMath() { return false; }
    template <typename T>
    static std::string toString(T val)
    {
        std::stringstream stream;
        stream << val;
        return stream.str();
    }

public:
    struct WeightDims {
        NvS64 wtSize;       //!<  total size of conv layer weights
        int numKernels;     //!<  total kernels in the weight
        int numChannels;    //!<  total channels in each kernel
        int width;      //!<  width of the weight
        int height;     //!<  height of the height
        int strideX;        //!<  kernel stride along width
        int strideY;        //!<  kernel stride along height

        WeightDims(NvS64 size, int k, int c, int w, int h, int cx, int cy) :
            wtSize(size), numKernels(k), numChannels(c),
            width(w), height(h),
            strideX(cx), strideY(cy) {};

        WeightDims() :
            wtSize(0), numKernels(0), numChannels(0),
            width(0), height(0),
            strideX(0), strideY(0){};
    };

    /* Check if given value fits within the data format precision range.
     *
     *         <--|----------|-----|-----|----------|-->
     *         -ve min   -ve max   0   +ve min    +ve max
     * api:    lowest()  -1*min()       min()      max()
     * fp16:   -65504    -6.10e-05    6.10e-05     65504
     *
     * note:
     *   - lowest() is c++11 api
     *   - there is no standard api to get -ve max value for a data format,
     *     hence -1 * min() is used.
     *
     */
    template<typename IT, typename RT>
    static inline bool isWithinNumericalLimits(IT val)
    {
        bool retval = true;
#if ENABLE_PRECISION_RANGE_CHECK
        if ( (val > std::numeric_limits<RT>::max())
                || (val < std::numeric_limits<RT>::lowest())
                || (val !=0 && val < std::numeric_limits<RT>::min() && val > (-1.0f*std::numeric_limits<RT>::min())) )
            {
                    if ( debugMath() )
                    {
                        gLogInfo << val << " is beyond "
                                 << "Pmax("  << float(std::numeric_limits<RT>::max())
                                 << ")/Nmin("<< float(std::numeric_limits<RT>::lowest())
                                 << ") OR is within "
                                 << "Pmin("<< float(std::numeric_limits<RT>::min())
                                 << ")<->Nmax("<< float(-1.0f*std::numeric_limits<RT>::min())
                                 << ") limits of compute precision" << std::endl;
                    }
                    retval = false;
            }
#endif
        return retval;
    }

    //! Quantize weights per-kernel (1 scaling factor for the entire KCRS blob)
    template <typename MP, typename CP>
    static std::vector<NvF32> perKernelQuantizeWts
        (
            Weights highPrecWts,
            NvS32 G, NvS32 K, NvS32 C, NvS32 RS, NvS32 kStride, NvS32 cStride,
            NvS8* quantizedWts
        )
    {
        std::vector<NvF32> filterScales;
        NvF32 max = std::numeric_limits<NvF32>::lowest(); // 先把 max 初始化为 FLoat 32 的最小值
        const MP* origWts = reinterpret_cast<const MP*>(const_cast<void*>(highPrecWts.values));
        // 获取所有Tensor的最大值
        for (NvS32 g = 0; g < G; g++)
        {
            NvS32 gOffset = g * K * C * RS;
            for (NvS32 k = 0; k < K; k++)
            {
                for (NvS32 c = 0; c < C; c++)
                {
                    for (NvS32 rs = 0; rs < RS; rs++)
                        // max = std::max(max, std::abs(origWts[gOffset + k * kStride + c * cStride + rs] * inScale));
                        max = std::max<NvF32>(max, std::fabs(origWts[gOffset + k * kStride + c * cStride + rs]));
                }
            }
        }
        std::cout << "max value is " << std::fabs(max)  << std::endl;

        NvF32 scale = (std::fabs(max) < FLT_EPSILON)? 1 : max / 127, invScale = 1 / scale;

        // 开始量化
        for (NvS32 g = 0; g < G; g++)
        {
            NvS32 gOffset = g * K * C * RS;
            for (NvS32 k = 0; k < K; k++)
            {
                for (NvS32 c = 0; c < C; c++)
                {
                    for (NvS32 rs = 0; rs < RS; rs++)
                    {
                        NvS32 index = gOffset + k * kStride + c * cStride + rs;
                        // quantizedWts[index] = int8_t(std::floor(origWts[index] * inScale * invScale + 0.5f));

                        // quantizedWts[index] = static_cast<NvS8>(std::floor(origWts[index] * invScale + 0.5f));
                        NvS32 int32Weight = static_cast<NvS32>(std::floor(origWts[index] * invScale + 0.5f));
                        quantizedWts[index] = static_cast<NvS8>(std::max(std::min(int32Weight, static_cast<NvS32>(std::numeric_limits<NvS8>::max())),
                                                                         static_cast<NvS32>(std::numeric_limits<NvS8>::lowest())));
                    }
                }
                filterScales.push_back(scale);
            }
        }
        return filterScales;
    }

    //! Quantize weights per-filter (1 scaling factor for each CRS blob)
    template <typename MP, typename CP>
    static std::vector<NvF32> perFilterQuantizeWts
        (
            Weights highPrecWts,
            NvS32 G, NvS32 K, NvS32 C, NvS32 RS, NvS32 kStride, NvS32 cStride,
            NvS8* quantizedWts
        )
    {
        std::vector<NvF32> filterScales;
        const MP* origWts = reinterpret_cast<const MP*>(const_cast<void*>(highPrecWts.values));

        for (NvS32 g = 0; g < G; g++)
        {
            NvS32 gOffset = g * K * C * RS;
            for (NvS32 k = 0; k < K; k++)
            {
                NvF32 max = std::numeric_limits<NvF32>::lowest();
                for (NvS32 c = 0; c < C; c++)
                {
                    for (NvS32 rs = 0; rs < RS; rs++)
                        // max = std::max(max, std::abs(origWts[gOffset + k * kStride + c * cStride + rs] * inScale));
                        max = std::max<NvF32>(max, std::fabs(origWts[gOffset + k * kStride + c * cStride + rs]));
                }
                std::cout << "max value is " << std::fabs(max)  << std::endl;

                NvF32 scale = (std::fabs(max) < FLT_EPSILON)? 1 : max / 127, invScale = 1 / scale;


                for (NvS32 c = 0; c < C; c++)
                {
                    for (NvS32 rs = 0; rs < RS; rs++)
                    {
                        NvS32 index = gOffset + k * kStride + c * cStride + rs;
                        // quantizedWts[index] = int8_t(std::floor(origWts[index] * inScale * invScale + 0.5f));

                        // quantizedWts[index] = static_cast<NvS8>(std::floor(origWts[index] * invScale + 0.5f));
                        NvS32 int32Weight = static_cast<NvS32>(std::floor(origWts[index] * invScale + 0.5f));
                        quantizedWts[index] = static_cast<NvS8>(std::max(std::min(int32Weight, static_cast<NvS32>(std::numeric_limits<NvS8>::max())),
                                                                         static_cast<NvS32>(std::numeric_limits<NvS8>::lowest())));
                    }
                }

                filterScales.push_back(scale);
            }
        }
        return filterScales;
    }

    //!<  Zero pad caffe wts to match its #chnls with IMG input
    template<typename IT, typename RT>
    static Weights zeroPadWtsForIMG
        (
            WeightDims  origWDims,              //!<  dims of orig caffe wt blob
            WeightDims  zeroPadWDims,           //!<  dims of wt blob after zero padding
            Weights     srcWts                  //!<  pts to orig caffe wt blob
        )
    {
        NvS64 trnsSize = 0;
        NvS64 zpCnt = 0;
        Weights IMG_ZP_wts = Weights(nvdla::DataType::FLOAT, NULL, 0);
        API_CHECK_WEIGHTS_RETVAL(srcWts, IMG_ZP_wts);

        IT* pSrcWts = reinterpret_cast<IT*>(const_cast<void*>(srcWts.values));

        int zpR = zeroPadWDims.height;
        int zpS = zeroPadWDims.width;
        int zpC = zeroPadWDims.numChannels;
        int zpK = zeroPadWDims.numKernels;
        int zpSize = zeroPadWDims.wtSize;

        IT* pIMGZPWts = reinterpret_cast<IT*>(engine_ast::MemoryCollector::getInstance()->allocateMemory(zpSize * sizeof(IT)));
        memset(pIMGZPWts, 0, zpSize * sizeof(IT));
        IT* pIMGZPWtsCopy = pIMGZPWts;

        for (int ind_k = 0; ind_k < zpK; ++ind_k)
            for (int ind_c = 0; ind_c < zpC; ++ind_c)
                for (int ind_r = 0; ind_r < zpR; ++ind_r)
                    for (int ind_s = 0; ind_s < zpS; ++ind_s)
                    {
                        IT* dest = getAddrOffset<IT>(ind_k, ind_c, ind_r, ind_s,
                                                     zeroPadWDims, pIMGZPWts);
                        IT* src  = getAddrOffset<IT>(ind_k, ind_c, ind_r,
                                                     ind_s, origWDims, pSrcWts);
                        if (src != NULL) {
                            *dest = *src;
                        }
                        else {
                            *dest = 0.0f;
                            zpCnt++;
                        }
                        trnsSize++;
                    }

        if (trnsSize != zeroPadWDims.wtSize)
        {
            return IMG_ZP_wts;
        }

        IMG_ZP_wts.type   = getEnumFromType<IT>();
        IMG_ZP_wts.values = pIMGZPWtsCopy;
        IMG_ZP_wts.count  = zeroPadWDims.wtSize;

        return IMG_ZP_wts;
    }

    //!< Zero pad for per-element data >
    template<typename IT, typename RT>
    static Weights zeroPadFeatureData
        (
            WeightDims origDims,        // dims of data (bias/scale)
            WeightDims zeroPadDims,     // dims after padding
            Weights srcData            // source data
        )
    {
        /* Support only for numKernel == 1 */
        if (zeroPadDims.numKernels != 1) {
            return Weights(nvdla::DataType::FLOAT, NULL, 0);
        }

        return zeroPadWtsForIMG<IT,RT>(origDims, zeroPadDims, srcData);
    }

    //!< Split weights into multiple kernels if number of groups greater than 1 >
    template<typename IT, typename RT>
    static Weights padGroupWeights
        (
            WeightDims  origWDims,              //!<  dims of orig caffe wt blob
            WeightDims  groupWDims,             //!<  dims of wt blob after padding
            Weights     srcWts,                 //!<  pts to orig caffe wt blob
            NvU32       numGroups               //!<  number of groups
        )
    {
        Weights Group_wts = Weights(nvdla::DataType::FLOAT, NULL, 0);
        API_CHECK_WEIGHTS_RETVAL(srcWts, Group_wts);

        IT* pSrcWts = reinterpret_cast<IT*>(const_cast<void*>(srcWts.values));

        unsigned int oR = origWDims.height;
        unsigned int oS = origWDims.width;
        unsigned int oC = origWDims.numChannels;
        unsigned int oK = origWDims.numKernels;

        unsigned int groupWSize = groupWDims.wtSize;

        IT* pGroupWts = reinterpret_cast<IT*>(engine_ast::MemoryCollector::getInstance()->allocateMemory(groupWSize * sizeof(IT)));
        memset(pGroupWts, 0, groupWSize * sizeof(IT));
        IT* pGroupWtsCopy = pGroupWts;

        NvU32 perGroupK = oK/numGroups;

        for (unsigned int ind_g = 0; ind_g < numGroups; ind_g++)
        {
            for (unsigned int ind_k = 0; ind_k < perGroupK; ind_k++)
            {
                for (unsigned int ind_c = 0; ind_c < oC; ind_c++)
                {
                    for (unsigned int ind_r = 0; ind_r < oR; ind_r++)
                    {
                        for (unsigned int ind_s = 0; ind_s < oS; ind_s++)
                        {
                            unsigned int ind_kk = ind_g * perGroupK + ind_k;
                            IT* src = getAddrOffset<IT>(ind_kk, ind_c, ind_r, ind_s,
                                                        origWDims, pSrcWts);
                            IT* dest  = getAddrOffsetForGroup<IT>(ind_kk, ind_c, ind_r,
                                                                  ind_s, ind_g, oC, groupWDims, pGroupWts);
                            *dest = *src;
                        }
                    }
                }
            }
        }

        Group_wts.type   = getEnumFromType<IT>();
        Group_wts.values = pGroupWtsCopy;
        Group_wts.count  = groupWDims.wtSize;

        return Group_wts;
    }

    //!<  Do channel pre-extension on raw caffe wts for IMG convolution
    template<typename IT, typename RT>
    static Weights preChnlExtWtsForIMG
        (
            WeightDims    origWDims,            //!<  dims of orig caffe wt blob
            WeightDims    preCEWDims,           //!<  dims of wt blob after pre Chnl Ext
            Weights       srcWts                //!<  ptr to orig caffe wt blob
        )
    {
        NvS64 trnsCnt = 0;
        Weights IMG_Pre_CE_Wts = Weights(nvdla::DataType::FLOAT, NULL, 0);
        API_CHECK_WEIGHTS_RETVAL(srcWts, IMG_Pre_CE_Wts);

        IT* pSrcWts = reinterpret_cast<IT*>(const_cast<void*>(srcWts.values));

        int origK = origWDims.numKernels;

        int ceS = preCEWDims.width;
        int ceR = preCEWDims.height;
        int ceC = preCEWDims.numChannels;
        int ceK = preCEWDims.numKernels;
        NVDLA_UNUSED(ceK);

        if (preCEWDims.wtSize != origWDims.wtSize)
            return IMG_Pre_CE_Wts;

        IT* pIMGPreCEWts =
            reinterpret_cast<IT*>(engine_ast::MemoryCollector::getInstance()->allocateMemory(preCEWDims.wtSize * sizeof(IT)));
        memset(pIMGPreCEWts, 0, preCEWDims.wtSize * sizeof(IT));
        IT* pIMGPreCEWtsCopy = pIMGPreCEWts;

        for (int ind_k = 0; ind_k < origK; ++ind_k)
            for (int ind_c = 0; ind_c < ceC; ++ind_c)
                for (int ind_r = 0; ind_r < ceR; ++ind_r)
                    for (int ind_s = 0; ind_s < ceS; ++ind_s)
                    {
                        IT* dest = getAddrOffset<IT>(ind_k, ind_c, ind_r, ind_s,
                                                     preCEWDims, pIMGPreCEWts);
                        IT* src  = getCaffeAddrForIMGPreChnlExt<IT>(ind_k, ind_c, ind_r,
                                                                    ind_s, origWDims, pSrcWts);
                        if (src != NULL) {
                            *dest = *src;
                        }
                        else {
                            *dest = 0.0f;
                        }
                        trnsCnt++;
                    }

        if (trnsCnt != preCEWDims.wtSize) {
            return IMG_Pre_CE_Wts;
        }

        IMG_Pre_CE_Wts.type   = getEnumFromType<IT>();
        IMG_Pre_CE_Wts.values = pIMGPreCEWtsCopy;
        IMG_Pre_CE_Wts.count  = trnsCnt;

        return IMG_Pre_CE_Wts;
    }

    //!<  Do channel post-extension on raw caffe wts for IMG convolution
    template<typename IT, typename RT>
    static Weights postChnlExtWtsForIMG
        (
            WeightDims    wDims,          //!< dims of wt blob before post Chnl Extension
            Weights       srcWts,         //!< ptr to orig caffe wt blob
            NvU32         postExtFactor,  //!< factor(1,2 or 4) upto which extension should be made
            bool          &postChnlExtWtsSuccess,
            int atomicKSize,
            int atomicCSize,
            int cbufWidth
        )
    {
        Weights IMG_Post_CE_Wts = Weights(nvdla::DataType::FLOAT, NULL, 0);

        API_CHECK_WEIGHTS0_RETVAL(srcWts, IMG_Post_CE_Wts);

        IT* pSrcWts = reinterpret_cast<IT*>(const_cast<void*>(srcWts.values));

        std::vector<IMGAtomicWtOp> vWtOps;
        vWtOps.clear();

        // postExtFactor == 1, means no change after.
        if (postExtFactor == 1)
        {
            postChnlExtWtsSuccess = false;
            return IMG_Post_CE_Wts;
        }

        //Prepare wt translation ops
        prepWtTrnsOpsForIMGPostCE<RT>(wDims, postExtFactor, vWtOps, atomicKSize, atomicCSize);


        if (vWtOps.size() == 0) {
            postChnlExtWtsSuccess = false;
            return IMG_Post_CE_Wts;
        }

        //Execute wt translation ops
        IMG_Post_CE_Wts = execWtTrnsOpsForIMGPostCE<IT, RT>(wDims, pSrcWts, vWtOps, atomicKSize, atomicCSize, cbufWidth);

        postChnlExtWtsSuccess = true;
        return IMG_Post_CE_Wts;
    }

    /* Join factors of 2 adjacent product ops into single blob iff they dont overshoot
     * the dynamic range of the compute precision
     */
    template <typename IT, typename RT>
    static Weights combineMultiplicationFactors
        (
            engine_ast::SDPMode mode,           // per-channel/layer
            WeightDims          commonDims,
            Weights&            rawF1Data,
            Weights&            rawF2Data
        )
    {
        NvDlaError e = NvDlaSuccess;
        Weights combinedData = Weights(nvdla::DataType::FLOAT, NULL, 0);
        NVDLA_UNUSED(e);

        IT* pRawF1Blob = reinterpret_cast<IT*>(const_cast<void*>(rawF1Data.values));
        IT* pRawF2Blob = reinterpret_cast<IT*>(const_cast<void*>(rawF2Data.values));
        IT* pCombinedBlob;

        API_CHECK_WEIGHTS_RETVAL(rawF1Data, combinedData);
        API_CHECK_WEIGHTS_RETVAL(rawF2Data, combinedData);

        if (rawF1Data.count != rawF2Data.count)
        {
            ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Can't combine multiplication "
                                                      "factors of 2 blobs which are of different sizes %d != %d",
                                 rawF1Data.count, rawF2Data.count);
        }

        combinedData.type   = rawF1Data.type;
        combinedData.count  = rawF1Data.count;
        pCombinedBlob       =
            reinterpret_cast<IT*>(engine_ast::MemoryCollector::getInstance()->allocateMemory(rawF1Data.count * sizeof(IT)));
        combinedData.values = NULL;

        if (mode.v() == engine_ast::SDPModeEnum::SDP_MODE_PER_LAYER)
        {
            IT combinedVal = pRawF1Blob[0] * pRawF2Blob[0];
            if (!isWithinNumericalLimits<IT, RT>(combinedVal))
            {
                engine_ast::MemoryCollector::getInstance()->freeMemory(pCombinedBlob);
                goto fail;
            }
            pCombinedBlob[0] = combinedVal;
        }
        else if (mode.v() == engine_ast::SDPModeEnum::SDP_MODE_PER_CHANNEL)
        {
            for (int cc = 0; cc < commonDims.numChannels; ++cc)
            {
                IT combinedVal = pRawF1Blob[cc] * pRawF2Blob[cc];
                if (!isWithinNumericalLimits<IT, RT>(combinedVal))
                {
                    engine_ast::MemoryCollector::getInstance()->freeMemory(pCombinedBlob);
                    goto fail;
                }
                pCombinedBlob[cc] = combinedVal;
            }
        }
        else
        {
            ORIGINATE_ERROR_FAIL(NvDlaError_NotSupported, "Don't support joining multiplication "
                                                          "factors for SDP mode %s", mode.c_str());
        }

        combinedData.values = pCombinedBlob;

        fail:
        return combinedData;
    }

    /* Join factors of 2 adjacent additive ops into single blob iff they dont overshoot
     * the dynamic range of the compute precision
     */
    template <typename IT, typename RT>
    static Weights combineAdditionFactors
        (
            engine_ast::SDPMode mode,           // per-channel/layer
            WeightDims          commonDims,
            Weights&            rawF1Data,
            Weights&            rawF2Data
        )
    {
        NvDlaError e = NvDlaSuccess;
        Weights combinedData = Weights(nvdla::DataType::FLOAT, NULL, 0);
        NVDLA_UNUSED(e);

        IT* pRawF1Blob = reinterpret_cast<IT*>(const_cast<void*>(rawF1Data.values));
        IT* pRawF2Blob = reinterpret_cast<IT*>(const_cast<void*>(rawF2Data.values));
        IT* pCombinedBlob;

        API_CHECK_WEIGHTS_RETVAL(rawF1Data, combinedData);
        API_CHECK_WEIGHTS_RETVAL(rawF2Data, combinedData);

        if (rawF1Data.count != rawF2Data.count)
        {
            ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Can't combine addition "
                                                      "factors of 2 blobs which are of different sizes %d != %d",
                                 rawF1Data.count, rawF2Data.count);
        }

        combinedData.type   = rawF1Data.type;
        combinedData.count  = rawF1Data.count;
        pCombinedBlob       =
            reinterpret_cast<IT*>(engine_ast::MemoryCollector::getInstance()->allocateMemory(rawF1Data.count * sizeof(IT)));
        combinedData.values = NULL;

        if (mode.v() == engine_ast::SDPModeEnum::SDP_MODE_PER_LAYER)
        {
            IT combinedVal = pRawF1Blob[0] + pRawF2Blob[0];
            if (!isWithinNumericalLimits<IT, RT>(combinedVal))
            {
                engine_ast::MemoryCollector::getInstance()->freeMemory(pCombinedBlob);
                goto fail;
            }
            pCombinedBlob[0] = combinedVal;
        }
        else if (mode.v() == engine_ast::SDPModeEnum::SDP_MODE_PER_CHANNEL)
        {
            for (int cc = 0; cc < commonDims.numChannels; ++cc)
            {
                IT combinedVal = pRawF1Blob[cc] + pRawF2Blob[cc];
                if (!isWithinNumericalLimits<IT, RT>(combinedVal))
                {
                    engine_ast::MemoryCollector::getInstance()->freeMemory(pCombinedBlob);
                    goto fail;
                }
                pCombinedBlob[cc] = combinedVal;
            }
        }
        else
        {
            ORIGINATE_ERROR_FAIL(NvDlaError_NotSupported, "Don't support joining addition "
                                                          "factors for SDP mode %s", mode.c_str());
        }

        combinedData.values = pCombinedBlob;

        fail:
        return combinedData;
    }

    template <typename IT, typename RT>
    Weights combineKernelWeightsAndScaleData
        (
            engine_ast::SDPMode sclMode,           // per-channel/layer
            nvdla::Dims4        krnlWtDims,
            nvdla::Dims4        sclDims,
            Weights&            krnlWts,
            Weights&            sclData
        )
    {
        NvDlaError e = NvDlaSuccess;
        Weights combinedData = Weights(nvdla::DataType::FLOAT, NULL, 0);
        NVDLA_UNUSED(e);

        IT* pKrnlWts = reinterpret_cast<IT*>(const_cast<void*>(krnlWts.values));
        IT* pSclData = reinterpret_cast<IT*>(const_cast<void*>(sclData.values));
        IT* pCombinedBlob;

        API_CHECK_WEIGHTS_RETVAL(krnlWts, combinedData);
        API_CHECK_WEIGHTS_RETVAL(sclData, combinedData);

        combinedData.type   = krnlWts.type;
        combinedData.count  = krnlWts.count;
        pCombinedBlob       =
            reinterpret_cast<IT*>(engine_ast::MemoryCollector::getInstance()->allocateMemory(krnlWts.count * sizeof(IT)));
        combinedData.values = NULL;

        if (sclMode.v() == engine_ast::SDPModeEnum::SDP_MODE_PER_LAYER)
        {
            for (int i = 0; i < krnlWts.count; ++i)
            {
                IT combinedVal = pKrnlWts[i] * pSclData[0];
                if (!isWithinNumericalLimits<IT, RT>(combinedVal))
                {
                    engine_ast::MemoryCollector::getInstance()->freeMemory(pCombinedBlob);
                    goto fail;
                }
                pCombinedBlob[i] = combinedVal;
            }
        }
        else if (sclMode.v() == engine_ast::SDPModeEnum::SDP_MODE_PER_CHANNEL)
        {
            for (int kk = 0; kk < krnlWtDims.n; ++kk)
            {
                IT perChnlScl = pSclData[kk];
                NvU64 wtsPerKrnl = krnlWtDims.c * krnlWtDims.h * krnlWtDims.w;
                for (int cc = 0; cc < (int)wtsPerKrnl; ++cc)
                {
                    IT combinedVal = pKrnlWts[(kk * wtsPerKrnl) + cc] * perChnlScl;
                    if (!isWithinNumericalLimits<IT, RT>(combinedVal))
                    {
                        engine_ast::MemoryCollector::getInstance()->freeMemory(pCombinedBlob);
                        goto fail;
                    }
                    pCombinedBlob[(kk * wtsPerKrnl) + cc] = combinedVal;
                }
            }
        }
        else
        {
            ORIGINATE_ERROR_FAIL(NvDlaError_NotSupported, "Don't support joining kernel wts and scale "
                                                          "factors for SDP mode %s", sclMode.c_str());
        }

        combinedData.values = pCombinedBlob;

        fail:
        return combinedData;
    }

    //!<  translate raw caffe wts to that suitable for direct convolution
    //!<  We only support:
    //!<      - int8 -> int8 translation
    //!<      - int16-> int16 translation
    //!<      - fp16 -> fp16 translation
    //!<      - fp32 -> fp32 translation (will deprecate soon)
    //!<      - fp32 -> fp16 translation
    template <typename IT, typename RT>
    static Weights translateWtsForDC
        (
            WeightDims                 wDims,               //!<  dims of orig caffe wt blob
            Weights&                   srcWts,              //!<  ptr to orig caffe wt blob
            int atomicKSize,
            int atomicCSize,
            int cbufWidth,
            std::map<std::string, IT>& mCaffeHash = *(new std::map<std::string, IT>()) //!<  hash of the entire caffe wt blob
        )
    {
        Weights DC_tr_wts = Weights(nvdla::DataType::FLOAT, NULL, 0);

        API_CHECK_WEIGHTS_RETVAL(srcWts, DC_tr_wts);

        bool isSanityOn = mCaffeHash.size() > 0 ? true : false;
        IT* pSrcWts  = reinterpret_cast<IT*>(const_cast<void*>(srcWts.values));
        IT* pDCCEWts = pSrcWts;
        WeightDims origWDims = wDims;
        std::vector<AtomicWtOp> vWtOps;
        vWtOps.clear();

        //Channel extend if need be
        //pDCCEWts = doSlowCEForDC<IT>(wDims, pSrcWts); // FIXME: do wt chnl extension only for IMGs in a separate API

        //if (isSanityOn)
        //    if(runSanityForDCWtChnlExt<IT>(pSrcWts, origWDims, pDCCEWts, wDims))
        //        return Weights{nvdla::DataType::FLOAT, NULL, 0};

        //Prepare wt translation ops
        prepWtTrnsOpsForDC<RT>(wDims, vWtOps, atomicKSize, atomicCSize);

        //Execute wt translation ops
        DC_tr_wts = execWtTrnsOpsForDC<IT, RT>(wDims, pDCCEWts, vWtOps, atomicKSize, atomicCSize, cbufWidth);

        if (isSanityOn)
            if (runSanityForDCWtTrns<IT, RT>(reinterpret_cast<RT*>(const_cast<void*>(DC_tr_wts.values)),
                                             origWDims,
                                             vWtOps,
                                             mCaffeHash,
                                             atomicKSize,
                                             atomicCSize))
                return Weights{nvdla::DataType::FLOAT, NULL, 0};

        if (pDCCEWts != pSrcWts)
        {
            engine_ast::MemoryCollector::getInstance()->freeMemory(pDCCEWts);
            pDCCEWts = NULL;
        }

        return DC_tr_wts;
    }

    //!<  convert weights from CKRS to KCRS format
    //!<  This function implements:
    //!<      - int8 -> fp32 translation
    //!<      - int16-> fp32 translation
    //!<      - fp16 -> fp32 translation
    template <typename IT, typename RT>
    static Weights convertWtsToKCRS
        (
            WeightDims                 wDims,               //!<  dims of orig caffe wt blob
            Weights&                   srcWts              //!<  ptr to orig caffe wt blob
        )
    {
        NvU32 trnsCount = 0;
        Weights DC_tr_wts = Weights(nvdla::DataType::FLOAT, NULL, 0);

        API_CHECK_WEIGHTS_RETVAL(srcWts, DC_tr_wts);

        IT* pSrcWts  = reinterpret_cast<IT*>(const_cast<void*>(srcWts.values));

        NvF32* pDCDestWts =
            reinterpret_cast<NvF32*>(engine_ast::MemoryCollector::getInstance()->allocateMemory(wDims.wtSize * sizeof(NvF32)));
        memset(pDCDestWts, 0, wDims.wtSize * sizeof(NvF32));
        NvF32* pDCDestWtsCopy = pDCDestWts;

        for (int k = 0; k < wDims.numKernels; ++k)
        {
            for (int c = 0; c < wDims.numChannels; ++c)
            {
                for (int r = 0; r < wDims.height; ++r)
                {
                    for (int s = 0; s < wDims.width; ++s)
                    {
                        int ckrsIndex = s + wDims.width *
                                            (r + wDims.height *
                                                 (k + wDims.numKernels *
                                                      (c)));
                        IT value = pSrcWts[ckrsIndex];
                        *pDCDestWts = NvF32(value);
                        pDCDestWts++;
                        trnsCount++;
                    }
                }
            }
        }

        DC_tr_wts.type = getEnumFromType<NvF32>();
        DC_tr_wts.values = pDCDestWtsCopy;
        DC_tr_wts.count  = trnsCount;

        return DC_tr_wts;
    }

    /**
     * Splits the raw KCRS weights into set of weights based on deconvolution strides.
     * This function returns set of splitted weights along with splitSetWtDims.
     **/
    template<typename IT, typename RT>
    static std::vector<Weights> splitWeightsForDeconv
        (
            Weights rawKCRSWts,
            Dims4   origWtDims,
            Dims2   deconvStrides,
            Dims4&   splitSetWtDims
        )
    {
        std::vector<Weights> splitSetWts;

        // Determine split dimension
        NvS32 splitSetW = (NvS32)ceilf(origWtDims.w/float(deconvStrides.w));
        NvS32 splitSetH = (NvS32)ceilf(origWtDims.h/float(deconvStrides.h));

        splitSetWtDims = Dims4(origWtDims.n, origWtDims.c, splitSetH, splitSetW);

        // Splitting weight data
        IT* pKCRS = reinterpret_cast<IT*>(const_cast<void*>(rawKCRSWts.values));

        for (NvU16 y = 0; y < deconvStrides.h; ++y)
        {
            for (NvU16 x = 0; x < deconvStrides.w; ++x)
            {
                NvU16 sStartIndex = x;
                NvU16 rStartIndex = y;

                IT* pSplitSet =
                    (IT*) engine_ast::MemoryCollector::getInstance()->allocateMemory(sizeof(IT) *
                                                                                     (splitSetWtDims.n *
                                                                                      splitSetWtDims.c *
                                                                                      splitSetWtDims.h *
                                                                                      splitSetWtDims.w));
                IT* pSplitSetCopy = pSplitSet;
                NvU64 splitSetCnt = 0;

                for (NvU16 k = 0; k < splitSetWtDims.n; ++k)
                {
                    for (NvU16 c = 0; c < splitSetWtDims.c; ++c)
                    {
                        for (NvU16 r = 0; r < splitSetWtDims.h; ++r)
                        {
                            for (NvU16 s = 0; s < splitSetWtDims.w; ++s)
                            {
                                NvU16 sJumpIndex = sStartIndex + (s * deconvStrides.w);
                                NvU16 rJumpIndex = rStartIndex + (r * deconvStrides.h);
                                if ((sJumpIndex < origWtDims.w) && (rJumpIndex < origWtDims.h))
                                {
                                    NvU32 unsplitWtIndex = sJumpIndex + origWtDims.w *
                                                                        (rJumpIndex + origWtDims.h *
                                                                                      (c + origWtDims.c * (k)));
                                    *pSplitSet = pKCRS[unsplitWtIndex];
                                }
                                else
                                {
                                    *pSplitSet = 0;
                                }
                                ++pSplitSet;
                                ++splitSetCnt;
                            }
                        }
                    }
                }

                splitSetWts.push_back(Weights(getEnumFromType<IT>(), pSplitSetCopy, splitSetCnt));
            }
        }

        return splitSetWts;
    }

    //!<  translate raw caffe wts to that suitable for Deconvolution
    //!<  We only support:
    //!<      - int8 -> int8 translation
    //!<      - int16-> int16 translation
    //!<      - fp16 -> fp16 translation
    //!<      - fp32 -> fp32 translation (will deprecate soon)
    //!<      - fp32 -> fp16 translation
    template <typename IT, typename RT>
    static Weights translateWtsForDeconv
        (
            WeightDims                 wDims,               //!<  dims of orig caffe wt blob
            Weights&                   srcWts,              //!<  ptr to orig caffe wt blob
            int atomicKSize,
            int atomicCSize,
            int cbufWidth,
            std::map<std::string, IT>& mCaffeHash = *(new std::map<std::string, IT>()) //!<  hash of the entire caffe wt blob
        )
    {
        Weights DC_tr_wts = Weights(nvdla::DataType::FLOAT, NULL, 0);

        API_CHECK_WEIGHTS_RETVAL(srcWts, DC_tr_wts);

        bool isSanityOn = mCaffeHash.size() > 0 ? true : false;
        IT* pSrcWts  = reinterpret_cast<IT*>(const_cast<void*>(srcWts.values));
        WeightDims origWDims = wDims;
        std::vector<AtomicWtOp> vWtOps;
        vWtOps.clear();

        prepWtTrnsOpsForDeconv<RT>(wDims, vWtOps, atomicKSize, atomicCSize);

        //Execute wt translation ops
        DC_tr_wts = execWtTrnsOpsForDeconv<IT, RT>(wDims, pSrcWts, vWtOps, atomicKSize, atomicCSize, cbufWidth);

        if (isSanityOn)
            if (runSanityForDeconvWtTrns<IT, RT>(reinterpret_cast<RT*>(const_cast<void*>(DC_tr_wts.values)),
                                                 origWDims,
                                                 vWtOps,
                                                 mCaffeHash,
                                                 atomicKSize,
                                                 atomicCSize))
                return Weights{nvdla::DataType::FLOAT, NULL, 0};

        return DC_tr_wts;
    }


    //!<  check if wt dims and conv stride allow WG convolution such as:
    //!<        Stride = 1, size = 3x3 or
    //!<        Stride = 2, size = 6x6 or 5x5 or
    //!<        Stride = 3, size = 7x7 or 8x8 or 9x9, etc
    static bool isWGPossible
        (
            WeightDims wDims
        )
    {
        // right now restrict only to 3x3 stride 1
//            if (wDims.width > wDims.strideX * 2 &&
//                wDims.width <= wDims.strideX * 3 &&
//                wDims.height > wDims.strideY * 2 &&
//                wDims.height <= wDims.strideY * 3)
        if (wDims.width == 3 && wDims.height == 3 &&
            wDims.strideX == 1 && wDims.strideY == 1)
            return true;
        else
            return false;
    }

    //!<  translate raw caffe wts to that suitable for winograd convolution
    template <typename IT, typename RT>
    static Weights translateWtsForWG
        (
            WeightDims       wDims,          //!<  dims of orig caffe wt blob
            Weights&         srcWts,         //!<  ptr to orig caffe wt blob
            std::map<std::string, IT>& mCaffeHash = *(new std::map<std::string, IT>())//!<  hash of the entire caffe wt blob
        )
    {
        Weights WG_tr_wts = Weights{nvdla::DataType::FLOAT, NULL, 0};
        API_CHECK_WEIGHTS_RETVAL(srcWts, WG_tr_wts);

        bool isSanityOn = mCaffeHash.size() > 0 ? true : false;
        IT* pSrcWts  = reinterpret_cast<IT*>(const_cast<void*>(srcWts.values));
        IT* pWGZPWts = pSrcWts;
        IT* pWGCEWts = pSrcWts;
        IT* pWGMTWts = pSrcWts;
        WeightDims origWDims = wDims;
        WeightDims zpWDims = wDims;
        WeightDims ceWDims = wDims;
        std::vector<AtomicWtOp> vWtOps;
        vWtOps.clear();

        if (wDims.strideX != wDims.strideY)
            return srcWts;  // WG trns is not possible for unequal X/Y strides

        int bpe = sizeof(RT);

        //Zero pad channels if not a multiple of 32
        if ((wDims.numChannels % (WG_BYTES_PER_ATOM/bpe)) != 0)
        {
            int numPadChnls = WG_BYTES_PER_ATOM/bpe -
                              (wDims.numChannels % (WG_BYTES_PER_ATOM/bpe));

            /* if padding zeroes in chnl direction, update caffeWtHash */
            mCaffeHash.clear();
            pWGZPWts = padZerosForWG<IT>(wDims, pSrcWts, numPadChnls, mCaffeHash);
            zpWDims = wDims;

            /* check sanctity of zero padding */
            if (isSanityOn)
                if (runSanityForWGWtZeroPadded<IT>(pSrcWts, origWDims, pWGZPWts, zpWDims))
                    return Weights{nvdla::DataType::FLOAT, NULL, 0};
        }

        //Channel extend if need be (stride > 1)
        if (wDims.width != 3 &&
            wDims.height != 3 &&
            wDims.strideX > 1)
        {
            /* if extending channels, update caffeWtHash */
            mCaffeHash.clear();
            pWGCEWts = doCEForWG<IT>(wDims, pWGZPWts, mCaffeHash);
            ceWDims = wDims;

            /* check sanity for chnl extn */
            if (isSanityOn)
                if (runSanityForWGWtChnlExt<IT>(pWGZPWts, zpWDims, pWGCEWts, wDims))
                    return Weights{nvdla::DataType::FLOAT, NULL, 0};
        }
        //free up mem allocated for ZP wt surface if CE also happened
        if (pWGCEWts != pSrcWts && pWGZPWts != pSrcWts)
        {
            engine_ast::MemoryCollector::getInstance()->freeMemory(pWGZPWts);
            pWGZPWts = NULL;
        }
        else if (pWGCEWts == pSrcWts)
        {
            //if CE didn't happen irrespective of ZP
            ceWDims = zpWDims;
            pWGCEWts = pWGZPWts;
        }


        //Translate the 3x3xC cube to 4x4xC
        pWGMTWts = WGMatrixTrns<IT>(wDims, pWGCEWts);

        /* check sanity of mat trns */
        if (isSanityOn)
            if (runSanityForWGWtMatrixTrns<IT>(pWGCEWts, ceWDims, pWGMTWts, wDims))
                return Weights{nvdla::DataType::FLOAT, NULL, 0};

        //free up mem allocated for CE
        if (pWGCEWts != pSrcWts)
        {
            engine_ast::MemoryCollector::getInstance()->freeMemory(pWGCEWts);
            pWGCEWts = NULL;
        }

        //Prepare wt translation ops
        prepWtTrnsOpsForWG<RT>(wDims, vWtOps);

        //Execute wt translation ops
        WG_tr_wts = execWtTrnsOpsForWG<IT, RT>(wDims, pWGMTWts, vWtOps);

        /* check sanity of wt trns */
        if (isSanityOn)
            if (runSanityForWGWtTrns<IT, RT>(reinterpret_cast<RT*>(const_cast<void*>(WG_tr_wts.values)),
                                             vWtOps,
                                             mCaffeHash))
                return Weights{nvdla::DataType::FLOAT, NULL, 0};

        engine_ast::MemoryCollector::getInstance()->freeMemory(pWGMTWts);
        pWGMTWts = NULL;

        return WG_tr_wts;
    }

    // convert CRS blob to Cpg[RSCf]Cfg[RSCf] blob
    template <typename IT, typename RT>
    static Weights translateCRSToFeatureData
        (
            WeightDims crsDims,
            Weights srcData,
            int channelsPerGroup = 0
        )
    {
        NvU32 trnsCount = 0;
        Weights featureFormatData = Weights(nvdla::DataType::FLOAT, NULL, 0);

        // use channelsPerGroup if specified
        int cf  = channelsPerGroup != 0? channelsPerGroup:
                  sizeof(RT) == 1 ? 32 : 16;   // full channels per group

        int cfg = crsDims.numChannels / cf;    // number of full channel groups
        int cp  = crsDims.numChannels % cf;    // partial channels per group
        int cpg = cp ? 1 : 0;                  // number of partial channel groups
        NvU64 ffSize = (cfg + cpg) * cf * crsDims.width * crsDims.height;

        IT *pSrcData  = reinterpret_cast<IT*>(const_cast<void*>(srcData.values));
        RT *pDestData = reinterpret_cast<RT*>(engine_ast::MemoryCollector::getInstance()->allocateMemory(ffSize * sizeof(RT)));
        memset(pDestData, 0, ffSize * sizeof(RT));

        RT *pDestDataCopy = pDestData;

        for (unsigned ind_cfg = 0; ind_cfg < (unsigned)cfg; ind_cfg++)
        {
            for (unsigned ind_r = 0; ind_r < (unsigned)crsDims.height; ind_r++)
            {
                for (unsigned ind_s = 0; ind_s < (unsigned)crsDims.width; ind_s++)
                {
                    for (unsigned ind_c = ind_cfg*cf; ind_c < (unsigned)((ind_cfg*cf) + cf); ind_c++)
                    {
                        IT value = pSrcData[ind_s +
                                            crsDims.width*(ind_r +
                                                           crsDims.height*(ind_c))];
                        *pDestData = RT(value);
                        pDestData++;
                        trnsCount++;
                    }
                }
            }
        }

        if (cpg)
        {
            for (unsigned ind_r = 0; ind_r < (unsigned)crsDims.height; ind_r++)
            {
                for (unsigned ind_s = 0; ind_s < (unsigned)crsDims.width; ind_s++)
                {
                    for (unsigned ind_c = cfg*cf; ind_c < (unsigned)((cfg*cf) + cf); ind_c++)
                    {
                        if (ind_c < (unsigned)((cfg*cf) + cp))
                        {
                            IT value = pSrcData[ind_s +
                                                crsDims.width*(ind_r +
                                                               crsDims.height*(ind_c))];
                            *pDestData = RT(value);
                        }
                        else
                        {
                            *pDestData = 0;
                        }
                        pDestData++;
                        trnsCount++;
                    }
                }
            }
        }

        if (trnsCount != ffSize)
        {
            gLogInternalError << "Problem in translating data to FF:"
                              << "copy count: " << trnsCount
                              << "dimension count:"<< ffSize
                              << std::endl;
            return featureFormatData;
        }

        featureFormatData.type   = getEnumFromType<RT>();
        featureFormatData.values = pDestDataCopy;
        featureFormatData.count  = trnsCount;
        return featureFormatData;
    }

    template <typename IT, typename RT>
    static Weights translatePEBiasToFD
        (
            WeightDims dims,
            Weights srcData,
            int channelsPerGroup = 0
        )
    {
        return translateCRSToFeatureData<IT,RT>(dims, srcData, channelsPerGroup);
    }

    // interleave/alternate between data bits from 2 data blobs
    template <typename IT, typename RT>
    static Weights interlayDataBlobs
        (
            Weights& D1Blob,
            Weights& D2Blob
        )
    {
        NvDlaError e = NvDlaSuccess;
        NVDLA_UNUSED(e);

        NvS64 trnsCount = 0;
        Weights interleavedData = Weights(nvdla::DataType::FLOAT, NULL, 0);

        RT* pDestData = NULL;
        RT* pDestDataCopy = NULL;

        API_CHECK_WEIGHTS_RETVAL(D1Blob, interleavedData);
        API_CHECK_WEIGHTS_RETVAL(D2Blob, interleavedData);

        IT* pD1Data = reinterpret_cast<IT*>(const_cast<void*>(D1Blob.values));
        IT* pD2Data = reinterpret_cast<IT*>(const_cast<void*>(D2Blob.values));

        ASSERT( D1Blob.count == D2Blob.count );

        pDestData = reinterpret_cast<RT*>(engine_ast::MemoryCollector::getInstance()->allocateMemory(D1Blob.count * 2 * sizeof(RT)));

        memset(pDestData, 0, D1Blob.count * 2 * sizeof(RT));
        pDestDataCopy = pDestData;

        for (NvU64 c = 0 ; c < static_cast<NvU64>(D1Blob.count); ++c)
        {
            IT d1Val = pD1Data[c];
            IT d2Val = pD2Data[c];

            *pDestData = RT(d1Val);
            ++pDestData;
            *pDestData = RT(d2Val);
            ++pDestData;
            trnsCount++;
        }

        interleavedData.type   = getEnumFromType<RT>();
        interleavedData.values = pDestDataCopy;
        interleavedData.count  = trnsCount;

        fail:
        return interleavedData;
    }

    template <typename IT, typename RT>
    static Weights translatePESclDataToFD
        (
            WeightDims dims,
            Weights srcData,
            int channelsPerGroup = 0
        )
    {
        return translateCRSToFeatureData<IT,RT>(dims, srcData, channelsPerGroup);
    }

    //!<  translate raw caffe bias-data to that suitable for SDP Bias op
    template <typename IT, typename RT>
    static Weights translateDataForBias
        (
            engine_ast::SDPMode  biasMode,      //!<  per-layer/channel/elementwise
            WeightDims           biasDims,      //!<  dims of orig caffe bias-data blob
            Weights&             srcBias,        //!<  ptr to orig caffe mean blob
            int channelsPerGroup = 0
        )
    {
        NvDlaError e = NvDlaSuccess;
        NVDLA_UNUSED(e);
        NvS64 trnsCount = 0;
        Weights Bias_tr_data = Weights(nvdla::DataType::FLOAT, NULL, 0);

        API_CHECK_WEIGHTS_RETVAL(srcBias, Bias_tr_data);

        IT* pSrcBias = reinterpret_cast<IT*>(const_cast<void*>(srcBias.values));

        RT* pDestBias =
            reinterpret_cast<RT*>(engine_ast::MemoryCollector::getInstance()->allocateMemory(biasDims.wtSize * sizeof(RT)));
        memset(pDestBias, 0, biasDims.wtSize * sizeof(RT));
        RT* pDestBiasCopy = pDestBias;

        /* Bias:-> y = x + bias*/

        // Bias data can be of 3 types: per-layer/per-channel/per-element
        // Per-Layer:
        if (biasMode.v() == engine_ast::SDPModeEnum::SDP_MODE_PER_LAYER)
        {
            *pDestBias = pSrcBias[0];
            trnsCount++;
        }
            // Per-Channel:
        else if (biasMode.v() == engine_ast::SDPModeEnum::SDP_MODE_PER_CHANNEL)
        {
            int c = 0;
            for ( ; c < biasDims.numChannels; ++c)
            {
                IT srcBias = pSrcBias[c];
                *pDestBias = RT(srcBias);
                ++pDestBias;
                trnsCount++;
            }
        }
            // Per-Element:
        else if (biasMode.v() == engine_ast::SDPModeEnum::SDP_MODE_PER_ELEMENT)
        {
            /* Translate to feature format */
            Bias_tr_data = translatePEBiasToFD<IT,RT>(biasDims, srcBias, channelsPerGroup);
            goto fail;
        }
        else
        {
            ORIGINATE_ERROR_FAIL(NvDlaError_NotSupported, "Don't support data translation for "
                                                          "bias op with sdp mode: %s", biasMode.c_str());
        }

        Bias_tr_data.type   = getEnumFromType<RT>();
        Bias_tr_data.values = pDestBiasCopy;
        Bias_tr_data.count  = trnsCount;

        fail:
        return Bias_tr_data;
    }

    //!<  translate raw caffe mean/variance to that suitable for SDP BatchNorm
    template <typename IT, typename RT>
    static Weights translateDataForBatchNorm
        (
            engine_ast::SDPMode  bnMode,         //!<  per-channel/layer
            WeightDims           bnDims,         //!<  dims of orig caffe mean/variance blobs
            Weights&             srcMean,        //!<  ptr to orig caffe mean blob
            Weights&             srcVar          //!<  ptr to orig caffe variance blob
        )
    {
        NvDlaError e = NvDlaSuccess;

        NVDLA_UNUSED(e);
        NVDLA_UNUSED(bnDims);

        Weights BNTrnsData = Weights(nvdla::DataType::FLOAT, NULL, 0);

        /* Batch-Norm:-> y = (x - mean) / sqrt(variance+eps)
         * SDP can do ADD and MUL, not SUB and DIV
         */

        // Batch norm data can be only of 2 types: per-layer/per-channel
        if ( bnMode.v() == engine_ast::SDPModeEnum::SDP_MODE_PER_LAYER ||
             bnMode.v() == engine_ast::SDPModeEnum::SDP_MODE_PER_CHANNEL)
        {
            BNTrnsData = interlayDataBlobs<IT, RT>(srcMean, srcVar);
        }
        else if (bnMode.v() == engine_ast::SDPModeEnum::SDP_MODE_PER_ELEMENT)
        {
            ORIGINATE_ERROR_FAIL(NvDlaError_NotSupported, "BN per-elemet translation is not yet supported\n");
        }
        else
        {
            ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Unknown BN mode\n");
        }

        fail:
        return BNTrnsData;
    }

    /**
     * Creates a unit(or identity) scale Weight based on mode and precision.
     * This is useful when rescaling is handled by separate scale node.
     **/
    template <typename MP, typename CP>
    static Weights createUnitScaleData
        (
            engine_ast::SDPMode scaleMode,
            Dims4               scaleDims
        )
    {
        NvDlaError e = NvDlaSuccess;
        NVDLA_UNUSED(e);
        Weights unitScaleData = Weights(nvdla::DataType::FLOAT, NULL, 0);

        NvS32 unitScaleCount = scaleDims.c * scaleDims.h * scaleDims.w;
        NvS32 trnsCnt = 0;

        MP* unitScaleValues =
            (MP*)engine_ast::MemoryCollector::getInstance()->allocateMemory(unitScaleCount * sizeof(MP));
        memset(unitScaleValues, 0, unitScaleCount * sizeof(MP));

        if (scaleMode.v() == engine_ast::SDPModeEnum::SDP_MODE_PER_LAYER)
        {
            unitScaleValues[0] = 1;
            trnsCnt++;
        }
        else if (scaleMode.v() == engine_ast::SDPModeEnum::SDP_MODE_PER_CHANNEL)
        {
            int c = 0;
            for ( ; c < scaleDims.c; ++c)
            {
                unitScaleValues[c] = 1;
                trnsCnt++;
            }
        }
        else if (scaleMode.v() == engine_ast::SDPModeEnum::SDP_MODE_PER_ELEMENT)
        {
            int i = 0;
            for ( ; i < (scaleDims.c * scaleDims.w * scaleDims.h); ++i)
            {
                unitScaleValues[i] = 1;
                trnsCnt++;
            }
        }
        else
        {
            ORIGINATE_ERROR_FAIL(NvDlaError_NotSupported, "Unsupported scale mode: %d\n", scaleMode.v());
        }

        ASSERT(trnsCnt == unitScaleCount);

        unitScaleData.type = getEnumFromType<MP>();
        unitScaleData.count = unitScaleCount;
        unitScaleData.values = unitScaleValues;

        fail:
        return unitScaleData;
    }


    //!<  Converts per-layer scale data to per-channel scale data
    // Repeats the same data #numChannels number of times. Data type is maintained after conversion
    template <typename IT, typename RT>
    static Weights translatePLScaleToPCScale
        (
            Weights&            srcScale,       //!< ptr to orig caffe mean blob
            NvU32               numChannels     //!< #channels
        )
    {
        Weights Scale_tr_data = Weights(nvdla::DataType::FLOAT, NULL, 0);
        NvU32 newCount = numChannels * srcScale.count;

        API_CHECK_WEIGHTS_RETVAL(srcScale, Scale_tr_data);

        IT* pSrcScale = reinterpret_cast<IT*>(const_cast<void*>(srcScale.values));
        IT* pDestScale =
            reinterpret_cast<IT*>(engine_ast::MemoryCollector::getInstance()->allocateMemory(newCount * sizeof(IT)));

        for (NvU32 ii = 0; ii < numChannels; ii++)
        {
            pDestScale[ii] = pSrcScale[0];
        }

        Scale_tr_data.type = srcScale.type;
        Scale_tr_data.count = newCount;
        Scale_tr_data.values = pDestScale;

        return Scale_tr_data;
    }


    //!<  translate raw caffe scale-data to that suitable for SDP Scale
    template <typename IT, typename RT>
    static Weights translateDataForScale
        (
            engine_ast::SDPMode  scaleMode,      //!<  per-layer/channel/elementwise
            WeightDims           scaleDims,      //!<  dims of orig caffe scale-data blob
            Weights&             srcScale,       //!<  ptr to orig caffe mean blob
            int channelsPerGroup = 0
        )
    {
        NvS64 trnsCount = 0;
        Weights Scale_tr_data = Weights(nvdla::DataType::FLOAT, NULL, 0);

        API_CHECK_WEIGHTS_RETVAL(srcScale, Scale_tr_data);

        IT* pSrcScale = reinterpret_cast<IT*>(const_cast<void*>(srcScale.values));

        RT* pDestScale =
            reinterpret_cast<RT*>(engine_ast::MemoryCollector::getInstance()->allocateMemory(scaleDims.wtSize * sizeof(RT)));
        memset(pDestScale, 0, scaleDims.wtSize * sizeof(RT));
        RT* pDestScaleCopy = pDestScale;

        /* Scale:-> y = x * scale_factor */

        // Scale data can be of 3 types: per-layer/per-channel/per-element
        // Per-Layer:
        if (scaleMode.v() == engine_ast::SDPModeEnum::SDP_MODE_PER_LAYER)
        {
            *pDestScale = RT(pSrcScale[0]);
            trnsCount++;
        }
            // Per-Channel:
        else if (scaleMode.v() == engine_ast::SDPModeEnum::SDP_MODE_PER_CHANNEL)
        {
            int c = 0;
            for ( ; c < scaleDims.numChannels; ++c)
            {
                IT srcScale = pSrcScale[c];
                *pDestScale = RT(srcScale);
                ++pDestScale;
                trnsCount++;
            }
        }
            // Per-Element
        else if (scaleMode.v() == engine_ast::SDPModeEnum::SDP_MODE_PER_ELEMENT)
        {
            /* Translate to feature format */
            Scale_tr_data = translatePESclDataToFD<IT,RT>(scaleDims, srcScale, channelsPerGroup);
            goto fail;
        }

        Scale_tr_data.type   = getEnumFromType<RT>();
        Scale_tr_data.values = pDestScaleCopy;
        Scale_tr_data.count  = trnsCount;

        fail:
        return Scale_tr_data;
    }

    /* Reciprocal of the data blob. */
    template <typename IT, typename RT>
    static Weights invertDataBlob
        (
            engine_ast::SDPMode mode,           // per-channel/layer
            WeightDims          dataDims,
            Weights&            rawData
        )
    {
        NvDlaError e = NvDlaSuccess;
        Weights invertData = Weights(nvdla::DataType::FLOAT, NULL, 0);
        NVDLA_UNUSED(e);

        IT* pRawBlob = reinterpret_cast<IT*>(const_cast<void*>(rawData.values));
        IT* pInvertBlob;

        API_CHECK_WEIGHTS_RETVAL(rawData, invertData);

        invertData.type   = rawData.type;
        invertData.count  = rawData.count;
        pInvertBlob       =
            reinterpret_cast<IT*>(engine_ast::MemoryCollector::getInstance()->allocateMemory(rawData.count * sizeof(IT)));
        invertData.values = NULL;

        if (mode.v() == engine_ast::SDPModeEnum::SDP_MODE_PER_LAYER)
        {
            IT invertVal = IT(1) / pRawBlob[0];
            if (!isWithinNumericalLimits<IT, RT>(invertVal))
            {
                engine_ast::MemoryCollector::getInstance()->freeMemory(pInvertBlob);
                goto fail;
            }
            pInvertBlob[0] = invertVal;
        }
        else if (mode.v() == engine_ast::SDPModeEnum::SDP_MODE_PER_CHANNEL)
        {
            for (int cc = 0; cc < dataDims.numChannels; ++cc)
            {
                IT invertVal = IT(1) / pRawBlob[cc];
                if (!isWithinNumericalLimits<IT, RT>(invertVal))
                {
                    engine_ast::MemoryCollector::getInstance()->freeMemory(pInvertBlob);
                    goto fail;
                }
                pInvertBlob[cc] = invertVal;
            }
        }
        else
        {
            ORIGINATE_ERROR_FAIL(NvDlaError_NotSupported, "Don't support invert data "
                                                          "for SDP mode %s", mode.c_str());
        }

        invertData.values = pInvertBlob;

        fail:
        return invertData;
    }

private:

    struct Oplimits {
        int startIndex;
        int limit;

        Oplimits(int start, int limit):
            startIndex(start), limit(limit) { };
    };

    struct AtomicWtOp {
        Oplimits kg;    //!<  the prevalent kernel group
        Oplimits cg;    //!<  num of channel groups in 1 atomic wt op
        Oplimits r;     //!<  kernel height to be covered in 1 atomic wt op
        Oplimits s;     //!<  kernel width to be covered in 1 atomic wt op
        Oplimits k;     //!<  num of kernels in 1 atomic wt op
        Oplimits c;     //!<  num of channels in 1 atomic wt op

        AtomicWtOp(Oplimits kg,
                   Oplimits cg,
                   Oplimits r,
                   Oplimits s,
                   Oplimits k,
                   Oplimits c):
            kg(kg), cg(cg), r(r), s(s), k(k), c(c) {}
    };

    struct IMGAtomicWtOp {
        Oplimits kg;    //!<  the prevalent kernel group
        Oplimits rg;    //!<  number of vertical lines in 1 atomic wt op
        Oplimits s;     //!<  kernel width to be covered in 1 atomic wt op
        Oplimits k;     //!<  number of kernels to be covered in 1 atomic wt op
        Oplimits r;     //!<  kernel height to be covered in 1 atomic wt op
        Oplimits c;     //!<  number of channels to be covered in 1 atomic wt op

        IMGAtomicWtOp(Oplimits kg,
                      Oplimits rg,
                      Oplimits s,
                      Oplimits k,
                      Oplimits r,
                      Oplimits c):
            kg(kg), rg(rg), s(s), k(k), r(r), c(c) {}
    };

    template <typename T>
    static nvdla::DataType getEnumFromType()
    {
        nvdla::DataType ret = nvdla::DataType::UNKNOWN;
        T max           = std::numeric_limits<T>::max();
        NvS8 nvs8_max   = std::numeric_limits<NvS8>::max();
        NvS16 nvs16_max = std::numeric_limits<NvS16>::max();
        NvF32 fp32_max  = std::numeric_limits<NvF32>::max();
        half_float::half half_max   = std::numeric_limits<half_float::half>::max();


        if (max == (T)nvs8_max)
        {
            ret = nvdla::DataType::INT8;
        }
        else if (max == (T)nvs16_max)
        {
            ret = nvdla::DataType::INT16;
        }
        else if (max == (T)half_max)
        {
            ret = nvdla::DataType::HALF;
        }
        else if (max == (T)fp32_max)
        {
            ret = nvdla::DataType::FLOAT;
        }
        return ret;
    }

    //!<  determine position in the wt blob from array indexes
    template <typename T>
    static T* getAddrOffset
        (
            int indK,             //!< kernel index in the blob
            int indC,             //!< chnl index
            int indR,             //!< row index
            int indS,             //!< column index
            WeightDims wDims,     //!< dims of the wt blob
            T* pWts               //!< ptr to the wt blob
        )
    {
        if (indK >= wDims.numKernels ||
            indC >= wDims.numChannels ||
            indR >= wDims.height ||
            indS >= wDims.width)
        {
            return NULL;
        }

        return &pWts[indS +
                     wDims.width*(indR +
                                  wDims.height*(indC +
                                                wDims.numChannels*(indK)))];
    }

    //!<  determine position in the wt blob from array indexes for group operation
    template <typename T>
    static T* getAddrOffsetForGroup
        (
            int indK,             //!< kernel index in the blob
            int indC,             //!< chnl index
            int indR,             //!< row index
            int indS,             //!< column index
            int indG,             //!< group index
            int numChannels,      //!< channel size of original weights
            WeightDims wDims,     //!< dims of the wt blob
            T* pWts               //!< ptr to the wt blob
        )
    {
        indC = indG * numChannels + indC;

        if (indK >= wDims.numKernels ||
            indC >= wDims.numChannels ||
            indR >= wDims.height ||
            indS >= wDims.width)
        {
            return NULL;
        }

        return &pWts[indS +
                     wDims.width*(indR +
                                  wDims.height*(indC +
                                                wDims.numChannels*(indK)))];
    }

    //!<  determine position in the raw caffe weights from array indexes suitable for IMD Pre Chnl Ext
    template <typename T>
    static T* getCaffeAddrForIMGPreChnlExt
        (
            int indK,                   //!<  kernel index in the raw caffe wt blob
            int indC,                   //!<  channel index
            int indR,                   //!<  row index
            int indS,                   //!<  channel index
            WeightDims caffeWDims,      //!<  dims of the wt blob
            T* pCaffeWts                //!<  pts to the wt blob
        )
    {
        int caffeS = caffeWDims.width;
        int caffeR = caffeWDims.height;
        int caffeC = caffeWDims.numChannels;
        NVDLA_UNUSED(indS); // indS is always 1 for pre Chnl Ext

        int ind_s = indC / caffeC;
        int ind_r = indR;
        int ind_c = indC % caffeC;

        if (ind_s < caffeS && ind_r < caffeR && ind_c < caffeC)
            return &pCaffeWts[ind_s +
                              caffeS*(ind_r +
                                      caffeR*(ind_c +
                                              caffeC*(indK)))];
        else
            return NULL;
    }

    //!<  determine position in the raw caffe weights from array indexes suitable for WG Chnl Ext
    template <typename T>
    static T* getCaffeAddrForWGChnlExt
        (
            int indK,                   //!<  kernel index in theraw caffe wt blob
            int indC,                   //!<  channel index
            int indR,                   //!<  row index
            int indS,                   //!<  channel index
            WeightDims caffeWDims,      //!<  dims of the wt blob
            T* pCaffeWts                //!<  pts to the wt blob
        )
    {
        int caffeStX = caffeWDims.strideX;
        int caffeStY = caffeWDims.strideY;
        int caffeR   = caffeWDims.height;
        int caffeS   = caffeWDims.width;
        int caffeC   = caffeWDims.numChannels;

        // change caffe_s only after 1 origC is finished
        int ind_s = (indS * caffeStX) + ((indC % (caffeC * caffeStX)) / caffeC);
        // change caffe_r only after 1 ceS row is finished
        int ind_r = (indR * caffeStY) + (indC / (caffeC * caffeStX));
        int ind_c = indC % caffeC;

        if (ind_r < caffeR && ind_s < caffeS)
            return &pCaffeWts[ind_s +
                              caffeS*(ind_r +
                                      caffeR*(ind_c +
                                              caffeC*(indK)))];
        else
            return NULL;
    }

    //!<  determine position in the raw caffe weights from array indexes suitable for DC Chnl Ext (which is de-featured; keeping as backup)
    template <typename T>
    static T* getCaffeAddrForDCChnlExt
        (
            int indK,                   //!<  kernel index in theraw caffe wt blob
            int indC,                   //!<  channel index
            int indR,                   //!<  row index
            int indS,                   //!<  channel index
            WeightDims caffeWDims,      //!<  dims of the wt blob
            T* pCaffeWts                //!<  pts to the wt blob
        )
    {
        return getCaffeAddrForWGChnlExt<T>(indK, indC, indR, indS, caffeWDims, pCaffeWts);
    }

    //!<  prepare weight translation sub-ops for IMG post channel extension
    template <typename RT>
    static void prepWtTrnsOpsForIMGPostCE
        (
            WeightDims                  wDims,      //!<  dimensions of the weights
            NvU32                       postExtFactor,
            std::vector<IMGAtomicWtOp>& vWtOps,      //!<  list of all operations to achieve the translation
            int atomicKSize,
            int atomicCSize
        )
    {
        int s   = wDims.width;
        int c   = wDims.numChannels;
        int rf  = postExtFactor;
        int kf  = (sizeof(RT) == 1 ? atomicKSize : (atomicKSize / 2));
        int rfg = wDims.height / rf;
        int kfg = wDims.numKernels / kf;
        int rp  = wDims.height % rf;
        int kp  = wDims.numKernels % kf;
        int rpg = 1;    //!<  partial row group will always be 1 in number if any
        int kpg = 1;    //!<  partial kernel group will always be 1 in number if any

        const int FULL_ROWS_PER_GROUP = rf;

        if (c > atomicCSize || rfg <= 1)
            return;

        bool isFullRowGroupsPoss  = rfg > 0 ? true : false;
        bool isFullKrnlGroupsPoss = kfg > 0 ? true : false;
        bool isPartRowGroupsPoss  = rp > 0 ? true : false;
        bool isPartKrnlGroupsPoss = kp > 0 ? true : false;

        //!<  Prepare atomic ops for full kernel groups
        if (isFullKrnlGroupsPoss)
        {
            for (int ind_kfg = 0; ind_kfg < kfg; ++ind_kfg)
            {
                if (isFullRowGroupsPoss)
                {
                    vWtOps.push_back(
                        IMGAtomicWtOp(Oplimits(ind_kfg, 1),
                                      Oplimits(0, rfg),
                                      Oplimits(0, s),
                                      Oplimits(0, kf),
                                      Oplimits(0, rf),
                                      Oplimits(0, c)));
                }
                if (isPartRowGroupsPoss)
                {
                    vWtOps.push_back(
                        IMGAtomicWtOp(Oplimits(ind_kfg, 1),
                                      Oplimits(0, rpg),
                                      Oplimits(0, s),
                                      Oplimits(0, kf),
                                      Oplimits(rfg * FULL_ROWS_PER_GROUP, rp),
                                      Oplimits(0, c)));
                }
            }
        }

        //!<  Prepare atomic ops for the partial kernel group
        if (isPartKrnlGroupsPoss)
        {
            if (isFullRowGroupsPoss)
            {
                vWtOps.push_back(
                    IMGAtomicWtOp(Oplimits(kfg, kpg),
                                  Oplimits(0, rfg),
                                  Oplimits(0, s),
                                  Oplimits(0, kp),
                                  Oplimits(0, rf),
                                  Oplimits(0, c)));
            }
            if (isPartRowGroupsPoss)
            {
                vWtOps.push_back(
                    IMGAtomicWtOp(Oplimits(kfg, kpg),
                                  Oplimits(0, rpg),
                                  Oplimits(0, s),
                                  Oplimits(0, kp),
                                  Oplimits(rfg * FULL_ROWS_PER_GROUP, rp),
                                  Oplimits(0, c)));
            }
        }
    }

    //!<  execute the wt translation sub-ops for IMG Post Chnl Ext
    template <typename IT, typename RT>
    static Weights execWtTrnsOpsForIMGPostCE
        (
            WeightDims                        wDims,       //!<  dims of the conv layer wts
            IT*                               pSrcWts,     //!<  surface ptr of caffe wts
            const std::vector<IMGAtomicWtOp>& vWtOps,       //!<  list of wt translation ops
            int atomicKSize,
            int atomicCSize,
            int cbufWidth
        )
    {
        NvS64 trnsCnt = 0;
        Weights IMG_Post_CE_Wts = Weights(nvdla::DataType::FLOAT, NULL, 0);

        RT* pIMGPostCEWts =
            reinterpret_cast<RT*>(engine_ast::MemoryCollector::getInstance()->allocateMemory(
                ROUNDUP_AND_ALIGN(wDims.wtSize * sizeof(RT), cbufWidth)));
        memset(pIMGPostCEWts, 0, ROUNDUP_AND_ALIGN(wDims.wtSize * sizeof(RT), cbufWidth));
        RT* pIMGPostCEWtsCopy = pIMGPostCEWts;

        int ind_rg = 0;
        int ind_s = 0;
        int ind_k = 0;
        int ind_r = 0;
        int ind_c = 0;
        int krnlPerGrp = (sizeof(RT) == 1 ? atomicKSize : (atomicKSize / 2));

        const int FULL_ROWS_PER_GROUP = atomicCSize / wDims.numChannels;

        std::vector<IMGAtomicWtOp>::const_iterator iterWtOp = vWtOps.begin();
        for ( ; iterWtOp != vWtOps.end(); ++iterWtOp)
        {
            for (ind_rg = iterWtOp->rg.startIndex;
                 ind_rg < iterWtOp->rg.limit;
                 ++ind_rg)
            {
                for (ind_s = iterWtOp->s.startIndex;
                     ind_s < iterWtOp->s.limit;
                     ++ind_s)
                {
                    for (ind_k = iterWtOp->kg.startIndex*krnlPerGrp;
                         ind_k < iterWtOp->kg.startIndex*krnlPerGrp +
                                 iterWtOp->k.limit;
                         ++ind_k)
                    {
                        for (ind_r = ind_rg*FULL_ROWS_PER_GROUP +
                                     iterWtOp->r.startIndex;
                             ind_r < (ind_rg*FULL_ROWS_PER_GROUP +
                                      iterWtOp->r.startIndex +
                                      iterWtOp->r.limit);
                             ++ind_r)
                        {
                            for (ind_c = iterWtOp->c.startIndex;
                                 ind_c < iterWtOp->c.limit;
                                 ++ind_c)
                            {
                                // We only support:
                                //  - int8 -> int8 translation
                                //  - int16-> int16 translation
                                //  - fp16 -> fp16 translation
                                //  - fp32 -> fp32 translation (will deprecate soon)
                                //  - fp32 -> fp16 translation
                                IT value    = pSrcWts[ind_s +
                                                      wDims.width*(ind_r +
                                                                   wDims.height*(ind_c +
                                                                                 wDims.numChannels*(ind_k)))];
                                *pIMGPostCEWts = RT(value);
                                ++pIMGPostCEWts;
                                ++trnsCnt;
                            }
                        }
                    }
                }
            }
        }

        IMG_Post_CE_Wts.type   = getEnumFromType<RT>();
        IMG_Post_CE_Wts.values = pIMGPostCEWtsCopy;
        IMG_Post_CE_Wts.count  = trnsCnt;

        return IMG_Post_CE_Wts;
    }

    //!<  do a slow channel extension for direct convolution
    template <typename T>
    static T* doSlowCEForDC
        (
            WeightDims& origWDims,  //!<  dims of orig non chnl extnd wt blob
            T* pSrcWts              //!<  ptr to orig non chnl extnd wt blob
        )
    {
        T* pDCCEWts;
        T* pDCCEWtsCopy;
        WeightDims ceWDims;

        int origR = origWDims.height;
        int origS = origWDims.width;
        int origC = origWDims.numChannels;
        int origK = origWDims.numKernels;
        int origStX = origWDims.strideX;
        int origStY = origWDims.strideY;

        if (origStX == 1 && origStY == 1)
        {
            return pSrcWts;
        }

        int ceR  = (origR + origStY - 1)/origStY;
        int ceS  = (origS + origStX - 1)/origStX;
        int ceC  = origC * origStX * origStY;
        int ceWtSize = origK * ceC * ceR * ceS;

        ceWDims.height  = ceR;
        ceWDims.width   = ceS;
        ceWDims.numChannels = ceC;
        ceWDims.numKernels  = origK;
        ceWDims.strideX = 1;
        ceWDims.strideY = 1;
        ceWDims.wtSize  = ceWtSize;

        pDCCEWts = reinterpret_cast<T*>(engine_ast::MemoryCollector::getInstance()->allocateMemory(ceWtSize * sizeof(T)));
        memset(pDCCEWts, 0, ceWtSize * sizeof(T));
        pDCCEWtsCopy = pDCCEWts;

        for (int ind_k = 0; ind_k < origK; ++ind_k)
            for (int ind_c = 0; ind_c < ceC; ++ind_c)
                for (int ind_r = 0; ind_r < ceR; ++ind_r)
                    for (int ind_s = 0; ind_s < ceS; ++ind_s)
                    {
                        T* dest = getAddrOffset<T>(ind_k, ind_c, ind_r, ind_s,
                                                   ceWDims, pDCCEWts);
                        T* src  = getCaffeAddrForDCChnlExt<T>(ind_k, ind_c, ind_r,
                                                              ind_s, origWDims, pSrcWts);
                        if (src != NULL)
                            *dest = *src;
                        else
                            *dest = 0.0f;
                    }

        origWDims = ceWDims;
        return pDCCEWtsCopy;
    }

    //!<  do chnl extension for direct convolution
    template <typename T>
    static T* doCEForDC
        (
            WeightDims& wDims,    //!<  dims of orig non chnl extnd wt blob
            T*          pSrcWts   //!<  ptr to orig non chnl extnd wt blob
        )
    {
        T* pDCCEWts;
        T* pDCCEWtsCopy;

        int origR = wDims.height;
        int origS = wDims.width;
        int origC = wDims.numChannels;
        int origK = wDims.numKernels;
        int origStX = wDims.strideX;
        int origStY = wDims.strideY;

        if (origStX == 1 && origStY == 1)
        {
            return pSrcWts;
        }

        int ceR  = (origR + origStY - 1)/origStY;
        int ceS  = (origS + origStX - 1)/origStX;
        int ceC  = origC * origStX * origStY;
        int ceWtSize = origK * ceC * ceR * ceS;

        pDCCEWts = reinterpret_cast<T*>(engine_ast::MemoryCollector::getInstance()->allocateMemory(ceWtSize * sizeof(T)));
        memset(pDCCEWts, 0, ceWtSize * sizeof(T));
        pDCCEWtsCopy = pDCCEWts;

        for (int ind_k = 0; ind_k < origK; ++ind_k)
            for (int ind_c = 0; ind_c < ceC; ++ind_c)
                for (int ind_r = 0; ind_r < ceR; ++ind_r)
                    for (int ind_s = 0; ind_s < ceS; ++ind_s)
                    {
                        // change caffe_s only after 1 origC is finished
                        int caffe_s = (ind_s * origStX) +
                                      ((ind_c % (origC * origStX)) / origC);
                        // change caffe_r only after 1 ceS row is finished
                        int caffe_r = (ind_r * origStY) +
                                      (ind_c / (origC * origStX));
                        int caffe_c = ind_c % origC;
                        if (caffe_r < origR && caffe_s < origS)
                            *pDCCEWts = pSrcWts[caffe_s +
                                                origS*(caffe_r +
                                                       origR*(caffe_c +
                                                              origC*(ind_k)))];
                        else
                            *pDCCEWts = 0;
                        ++pDCCEWts;
                    }
        wDims.height = ceR;
        wDims.width = ceS;
        wDims.numChannels = ceC;
        wDims.wtSize = ceWtSize;
        wDims.strideX = 1;
        wDims.strideY = 1;

        return pDCCEWtsCopy;
    }

    //!<  prepare weight translation sub-ops for direct convolution
    template <typename RT>
    static void prepWtTrnsOpsForDC
        (
            WeightDims               wDims,      //!<  dimensions of the weights
            std::vector<AtomicWtOp>& vWtOps,      //!<  list of all operations to achieve the translation
            int atomicKSize,
            int atomicCSize
        )
    {
        int r   = wDims.height;
        int s   = wDims.width;
        int cf  = atomicCSize;
        int cp  = wDims.numChannels % atomicCSize;
        int cfg = wDims.numChannels / atomicCSize;
        int cpg = 1;    //!<  Partial channel group will always be 1 in number if any
        int kf  = (sizeof(RT) == 1 ? atomicKSize : (atomicKSize / 2));
        int kp  = wDims.numKernels % kf;
        int kfg = wDims.numKernels / kf;
        int kpg = 1;    //!<  Partial kernel group will always be 1 in number if any

        bool isFullChnlGroupsPoss = cfg > 0 ? true : false;
        bool isFullKrnlGroupsPoss = kfg > 0 ? true : false;
        bool isPartChnlGroupsPoss = cp > 0 ? true : false;
        bool isPartKrnlGroupsPoss = kp > 0 ? true : false;

        //!<  Prepare atomic ops for full kernel groups
        if (isFullKrnlGroupsPoss)
        {
            for (int ind_kfg = 0; ind_kfg < kfg; ++ind_kfg)
            {
                if (isFullChnlGroupsPoss)
                {
                    vWtOps.push_back(
                        AtomicWtOp(Oplimits(ind_kfg, 1),
                                   Oplimits(0, cfg),
                                   Oplimits(0, r),
                                   Oplimits(0, s),
                                   Oplimits(0, kf),
                                   Oplimits(0, cf)));
                }
                if (isPartChnlGroupsPoss)
                {
                    vWtOps.push_back(
                        AtomicWtOp(Oplimits(ind_kfg, 1),
                                   Oplimits(0, cpg),
                                   Oplimits(0, r),
                                   Oplimits(0, s),
                                   Oplimits(0, kf),
                                   Oplimits(cfg*atomicCSize, cp)));
                }
            }
        }

        //!<  Prepare atomic ops for the partial kernel group
        if (isPartKrnlGroupsPoss)
        {
            if (isFullChnlGroupsPoss)
            {
                vWtOps.push_back(
                    AtomicWtOp(Oplimits(kfg, kpg),
                               Oplimits(0, cfg),
                               Oplimits(0, r),
                               Oplimits(0, s),
                               Oplimits(0, kp),
                               Oplimits(0, cf)));
            }
            if (isPartChnlGroupsPoss)
            {
                vWtOps.push_back(
                    AtomicWtOp(Oplimits(kfg, kpg),
                               Oplimits(0, cpg),
                               Oplimits(0, r),
                               Oplimits(0, s),
                               Oplimits(0, kp),
                               Oplimits(cfg*atomicCSize, cp)));
            }
        }
    }

    //!<  prepare weight translation sub-ops for deconvolution (same as direct convolution)
    template <typename RT>
    static void prepWtTrnsOpsForDeconv
        (
            WeightDims               wDims,      //!<  dimensions of the weights
            std::vector<AtomicWtOp>& vWtOps,      //!<  list of all operations to achieve the translation
            int atomicKSize,
            int atomicCSize
        )
    {
        prepWtTrnsOpsForDC<RT>(wDims, vWtOps, atomicKSize, atomicCSize);
    }

    //!<  execute the wt translation sub-ops for DC
    template <typename IT, typename RT>
    static Weights execWtTrnsOpsForDC
        (
            WeightDims wDims,                       //!<  dims of the conv layer wts
            IT* pSrcWts,                            //!<  surface ptr of caffe wts
            const std::vector<AtomicWtOp>& vWtOps,   //!<  list of wt translation ops
            int atomicKSize,
            int atomicCSize,
            int cbufWidth
        )
    {
        NvS64 trns_size = 0;
        int ind_cg = 0;
        int ind_r = 0;
        int ind_s = 0;
        int ind_k = 0;
        int ind_c = 0;
        Weights DC_tr_wts;
        std::vector<AtomicWtOp>::const_iterator iterWtOp = vWtOps.begin();
        int krnlPerGrp = (sizeof(RT) == 1 ? atomicKSize : (atomicKSize / 2));

        RT* pDCDestWts =
            reinterpret_cast<RT*>(engine_ast::MemoryCollector::getInstance()->allocateMemory(
                ROUNDUP_AND_ALIGN(wDims.wtSize * sizeof(RT), cbufWidth)));
        memset(pDCDestWts, 0, ROUNDUP_AND_ALIGN(wDims.wtSize * sizeof(RT), cbufWidth));
        RT* pDCDestWtsCopy = pDCDestWts;

        for ( ; iterWtOp != vWtOps.end(); ++iterWtOp)
        {
            for (ind_cg = iterWtOp->cg.startIndex;
                 ind_cg < iterWtOp->cg.limit;
                 ++ind_cg)
            {
                for (ind_r = iterWtOp->r.startIndex;
                     ind_r < iterWtOp->r.limit;
                     ++ind_r)
                {
                    for (ind_s = iterWtOp->s.startIndex;
                         ind_s < iterWtOp->s.limit;
                         ++ind_s)
                    {
                        for (ind_k = iterWtOp->kg.startIndex*krnlPerGrp;
                             ind_k < iterWtOp->kg.startIndex*krnlPerGrp +
                                     iterWtOp->k.limit;
                             ++ind_k)
                        {
                            for (ind_c = ind_cg*atomicCSize +
                                         iterWtOp->c.startIndex;
                                 ind_c < (ind_cg*atomicCSize +
                                          iterWtOp->c.startIndex +
                                          iterWtOp->c.limit);
                                 ++ind_c)
                            {
                                // We only support:
                                //  - int8 -> int8 translation
                                //  - int16-> int16 translation
                                //  - fp16 -> fp16 translation
                                //  - fp32 -> fp32 translation (will deprecate soon)
                                //  - fp32 -> fp16 translation
                                IT value    = pSrcWts[ind_s +
                                                      iterWtOp->s.limit*(ind_r +
                                                                         iterWtOp->r.limit*(ind_c +
                                                                                            wDims.numChannels*(ind_k)))];
                                *pDCDestWts = RT(value);
                                ++pDCDestWts;
                                ++trns_size;
                            }
                        }
                    }
                }
            }
        }

        DC_tr_wts.type = getEnumFromType<RT>();
        DC_tr_wts.values = pDCDestWtsCopy;
        DC_tr_wts.count  = trns_size;

        return DC_tr_wts;
    }

    //!<  execute the wt translation sub-ops for Deconvoltion
    //!<  (same as direct convolution, except that the kernel bytes are laid out in the reverse raster scan order
    //!<  for eg: (Sn-1,Rn-1) -> (Sn-2,Rn-1) ------> (S1,R0) -> (S0,R0)
    template <typename IT, typename RT>
    static Weights execWtTrnsOpsForDeconv
        (
            WeightDims wDims,                       //!<  dims of the conv layer wts
            IT* pSrcWts,                            //!<  surface ptr of caffe wts
            const std::vector<AtomicWtOp>& vWtOps,   //!<  list of wt translation ops
            int atomicKSize,
            int atomicCSize,
            int cbufWidth
        )
    {
        NvS64 trns_size = 0;
        int ind_cg = 0;
        int ind_r = 0;
        int ind_s = 0;
        int ind_k = 0;
        int ind_c = 0;
        Weights DC_tr_wts;
        std::vector<AtomicWtOp>::const_iterator iterWtOp = vWtOps.begin();
        int krnlPerGrp = (sizeof(RT) == 1 ? atomicKSize : (atomicKSize / 2));

        RT* pDCDestWts =
            reinterpret_cast<RT*>(engine_ast::MemoryCollector::getInstance()->allocateMemory(
                ROUNDUP_AND_ALIGN(wDims.wtSize * sizeof(RT), cbufWidth)));
        memset(pDCDestWts, 0, ROUNDUP_AND_ALIGN(wDims.wtSize * sizeof(RT), cbufWidth));
        RT* pDCDestWtsCopy = pDCDestWts;

        for ( ; iterWtOp != vWtOps.end(); ++iterWtOp)
        {
            for (ind_cg = iterWtOp->cg.startIndex;
                 ind_cg < iterWtOp->cg.limit;
                 ++ind_cg)
            {
                for (ind_r = iterWtOp->r.limit - 1;
                     ind_r >= iterWtOp->r.startIndex;
                     --ind_r)
                {
                    for (ind_s = iterWtOp->s.limit - 1;
                         ind_s >= iterWtOp->s.startIndex;
                         --ind_s)
                    {
                        for (ind_k = iterWtOp->kg.startIndex*krnlPerGrp;
                             ind_k < iterWtOp->kg.startIndex*krnlPerGrp +
                                     iterWtOp->k.limit;
                             ++ind_k)
                        {
                            for (ind_c = ind_cg*atomicCSize +
                                         iterWtOp->c.startIndex;
                                 ind_c < (ind_cg*atomicCSize +
                                          iterWtOp->c.startIndex +
                                          iterWtOp->c.limit);
                                 ++ind_c)
                            {
                                // We only support:
                                //  - int8 -> int8 translation
                                //  - int16-> int16 translation
                                //  - fp16 -> fp16 translation
                                //  - fp32 -> fp32 translation (will deprecate soon)
                                //  - fp32 -> fp16 translation
                                IT value    = pSrcWts[ind_s +
                                                      iterWtOp->s.limit*(ind_r +
                                                                         iterWtOp->r.limit*(ind_c +
                                                                                            wDims.numChannels*(ind_k)))];
                                *pDCDestWts = RT(value);
                                ++pDCDestWts;
                                ++trns_size;
                            }
                        }
                    }
                }
            }
        }

        DC_tr_wts.type = getEnumFromType<RT>();
        DC_tr_wts.values = pDCDestWtsCopy;
        DC_tr_wts.count  = trns_size;

        return DC_tr_wts;
    }

    //!<  add zero padding to the wt blob for Winograd convolution
    template <typename T>
    static T* padZerosForWG
        (
            WeightDims&     wDims,         //!<  dims of the raw wt blob
            T*              pSrcWts,        //!<  original wt blob
            int             numPadChnls,    //!<  numChnls to be zero padded after existing chnls
            std::map<std::string, T>& mCaffeHash      //!<  hash of the entire wt blob
        )
    {
        bool isSanityOn = mCaffeHash.size() > 0 ? true : false;
        std::string key;
        T* pWGZPWts;
        T* pWGZPWtsCopy;

        NvU64 zeroPadWtSize = wDims.numKernels *
                              wDims.height *
                              wDims.width  *
                              (wDims.numChannels + numPadChnls);
        pWGZPWts = reinterpret_cast<T*>(engine_ast::MemoryCollector::getInstance()->allocateMemory(zeroPadWtSize * sizeof(T)));
        pWGZPWtsCopy = pWGZPWts;

        for (int ind_k = 0; ind_k < wDims.numKernels; ++ind_k)
        {
            //!<  copy existing channels as it is upto origC
            for (int ind_c = 0; ind_c < wDims.numChannels; ++ind_c)
                for (int ind_r = 0; ind_r < wDims.height; ++ind_r)
                    for (int ind_s = 0; ind_s < wDims.width; ++ind_s)
                    {
                        *pWGZPWts = pSrcWts[ind_s +
                                            wDims.width*(ind_r +
                                                         wDims.height*(ind_c +
                                                                       wDims.numChannels*(ind_k)))];
                        key = toString(ind_k) + "-" +
                              toString(ind_c) + "-" +
                              toString(ind_r) + "-" +
                              toString(ind_s);

                        if (isSanityOn)
                            mCaffeHash.insert(std::pair<std::string, T>(key, *pWGZPWts));

                        ++pWGZPWts;
                    }
            //!<  add zero padded channels upto numPadChnls
            for (int ind_czp = wDims.numChannels;
                 ind_czp < (wDims.numChannels + numPadChnls); ++ind_czp)
                for (int ind_r = 0; ind_r < wDims.height; ++ind_r)
                    for (int ind_s = 0; ind_s < wDims.width; ++ind_s)
                    {
                        *pWGZPWts = 0;
                        key = toString(ind_k) + "-" +
                              toString(ind_czp) + "-" +
                              toString(ind_r) + "-" +
                              toString(ind_s);

                        if (isSanityOn)
                            mCaffeHash.insert(std::pair<std::string, T>(key, *pWGZPWts));

                        ++pWGZPWts;
                    }
        }

        wDims.numChannels += numPadChnls;
        wDims.wtSize = zeroPadWtSize;

        return pWGZPWtsCopy;
    }


    //!<  do chnl extension for WG
    template <typename T>
    static T* doCEForWG
        (
            WeightDims       &origWDims,         //!<  dims of orig non chnl extnd wt blob
            T*               pSrcWts,            //!<  ptr to orig non chnl extnd wt blob
            std::map<std::string, T>&  mCaffeHash          //!<  hash of the entire wt blob
        )
    {
        bool isSanityOn = mCaffeHash.size() > 0 ? true : false;
        std::string key;
        T* pWGCEWts;
        T* pWGCEWtsCopy;
        WeightDims ceWDims;

        int origR = origWDims.height;
        int origS = origWDims.width;
        int origC = origWDims.numChannels;
        int origK = origWDims.numKernels;
        int origStX = origWDims.strideX;
        int origStY = origWDims.strideY;

        int ceR = (origR + origStY - 1)/origStY;
        int ceS  = (origS + origStX - 1)/origStX;

        if (ceR != 3 || ceS != 3)
        {
            return pSrcWts;
        }

        int ceC  = origC * origStX * origStY;
        int ceWtSize = origK * ceC * ceR * ceS;

        ceWDims.height = ceR;
        ceWDims.width = ceS;
        ceWDims.numChannels = ceC;
        ceWDims.numKernels  = origK;
        ceWDims.strideX = 1;
        ceWDims.strideY = 1;
        ceWDims.wtSize = ceWtSize;

        pWGCEWts = reinterpret_cast<T*>(engine_ast::MemoryCollector::getInstance()->allocateMemory(ceWtSize * sizeof(T)));
        pWGCEWtsCopy = pWGCEWts;

        for (int ind_k = 0; ind_k < origK; ++ind_k)
            for (int ind_c = 0; ind_c < ceC; ++ind_c)
                for (int ind_r = 0; ind_r < ceR; ++ind_r)
                    for (int ind_s = 0; ind_s < ceS; ++ind_s)
                    {
                        T* dest = getAddrOffset<T>(ind_k, ind_c, ind_r, ind_s,
                                                   ceWDims, pWGCEWts);
                        T* src  = getCaffeAddrForWGChnlExt<T>(ind_k, ind_c, ind_r,
                                                              ind_s, origWDims, pSrcWts);
                        if (src != NULL)
                            *dest = *src;
                        else
                            *dest = 0.0f;
                        key = toString(ind_k) + "-" +
                              toString(ind_c) + "-" +
                              toString(ind_r) + "-" +
                              toString(ind_s);

                        if (isSanityOn)
                            mCaffeHash.insert(std::pair<std::string, T>(key, *dest));
                    }


        origWDims = ceWDims;
        return pWGCEWtsCopy;
    }

    //!<  convert a 3x3 matrix to 4x4 for Winograd
    template <typename T>
    static void WGMatrixUtil
        (
            int indKrnl,
            int indChnl,
            int numChnls,
            T* pSrcWts,
            T* pWGMatTrWts
        )
    {
        const NvF32 G[4][3] = {
            {1, 0, 0},
            {0.5, 0.5, 0.5},
            {0.5, -0.5, 0.5},
            {0, 0, 1},
        };
        const NvF32 Gt[3][4] = {
            {1, 0.5, 0.5, 0},
            {0, 0.5, -0.5, 0},
            {0, 0.5, 0.5, 1}
        };
        T g[3][3] = {{T(0)}};
        NvF32 temp[4][3] = {{0.0f}};
        NvF32 trnsFP32Wts[4][4] = {{0.0f}};

        for (int ind_r = 0; ind_r < 3; ind_r++)
        {
            for (int ind_s = 0; ind_s < 3; ind_s++)
            {
                g[ind_r][ind_s] = pSrcWts[ind_s +
                                          3*(ind_r +
                                             3*(indChnl +
                                                numChnls*(indKrnl)))];
            }
        }

        //Matrix multiply (Gxg)xGt
        for (int ind_m = 0; ind_m < 4; ++ind_m)
        {
            for (int ind_p = 0; ind_p < 3; ++ind_p)
            {
                temp[ind_m][ind_p] = 0.0f;
                for (int ind_n = 0; ind_n < 3; ++ind_n)
                    temp[ind_m][ind_p] += G[ind_m][ind_n] * g[ind_n][ind_p];
            }
        }

        //Matrix multiply Gx(gxGt)
        for (int ind_r = 0; ind_r < 4; ++ind_r)
        {
            for (int ind_s = 0; ind_s < 4; ++ind_s)
            {
                for (int ind_n = 0; ind_n < 3; ++ind_n)
                {
                    trnsFP32Wts[ind_r][ind_s] += temp[ind_r][ind_n] * Gt[ind_n][ind_s];
                }
                NvU32 target_ind = ind_s + 4*(ind_r + 4*(indChnl + numChnls*(indKrnl)));
                pWGMatTrWts[target_ind] = T(trnsFP32Wts[ind_r][ind_s]);
            }
        }
    }

    //!<  convert 3x3xC matrix to 4x4xC
    template <typename T>
    static T* WGMatrixTrns
        (
            WeightDims&     wDims,
            T*              pSrcWts
        )
    {
        T* pWGMatTrWts;
        T* pWGMatTrWtsCopy;

        //Here the kernels would be resized to 4x4xC each
        NvU64 mtWtSize = wDims.numKernels * wDims.numChannels * 4 * 4;

        pWGMatTrWts = reinterpret_cast<T*>(engine_ast::MemoryCollector::getInstance()->allocateMemory(mtWtSize * sizeof(T)));
        pWGMatTrWtsCopy = pWGMatTrWts;

        for (int ind_k = 0; ind_k < wDims.numKernels; ++ind_k)
        {
            for (int ind_c = 0; ind_c < wDims.numChannels; ++ind_c)
            {
                //Convert the weights from 3x3xC to 4x4xC format1 chnl at a time
                WGMatrixUtil<T>(ind_k, ind_c, wDims.numChannels,
                                pSrcWts, pWGMatTrWts);
            }
        }


        wDims.width = 4;
        wDims.height = 4;
        wDims.wtSize = mtWtSize;

        return pWGMatTrWtsCopy;
    }

    //!<  prepare wt translation ops for WG
    template <typename RT>
    static void prepWtTrnsOpsForWG
        (
            WeightDims          wDims,      //!<  dimensions of the weights
            std::vector<AtomicWtOp>& vWtOps      //!<  list of all operations to achieve the translation
        )
    {
        int r = wDims.height;
        int s = wDims.width;
        int cf = WG_FULL_CHANNELS_PER_ATOM;
        int cfg = wDims.numChannels / WG_FULL_CHANNELS_PER_ATOM;
        int kf = sizeof(RT) == 1 ? 32 : 16;
        int kp = wDims.numKernels % kf;
        int kfg = wDims.numKernels / kf;
        int kpg = 1;  //!<  Partial kernel group will always be 1 in number

        bool isFullKrnlGroupsPoss = kfg > 0 ? true : false;
        bool isFullChnlGroupsPoss = cfg > 0 ? true : false;
        bool isPartKrnlGroupsPoss = kp > 0 ? true : false;
        bool isPartChnlGroupsPoss = false; //!<  In WG, #chanls are rounded upto nearest multiple of 32
        NVDLA_UNUSED(isPartChnlGroupsPoss);

        //!<  Prepare atomic ops for full kernel groups
        if (isFullKrnlGroupsPoss)
        {
            for (int ind_kfg = 0; ind_kfg < kfg; ++ind_kfg)
            {
                if (isFullChnlGroupsPoss)
                {
                    vWtOps.push_back(
                        AtomicWtOp(Oplimits(ind_kfg, 1),
                                   Oplimits(0, cfg),
                                   Oplimits(0, r),
                                   Oplimits(0, s),
                                   Oplimits(0, kf),
                                   Oplimits(0, cf)));
                }
            }
        }

        //!<  Prepare atomic ops for the partial kernel group
        if (isPartKrnlGroupsPoss)
        {
            if (isFullChnlGroupsPoss)
            {
                vWtOps.push_back(
                    AtomicWtOp(Oplimits(kfg, kpg),
                               Oplimits(0, cfg),
                               Oplimits(0, r),
                               Oplimits(0, s),
                               Oplimits(0, kp),
                               Oplimits(0, cf)));
            }
        }
    }


    //!<  exec wt translation ops for WG
    template <typename IT, typename RT>
    static Weights execWtTrnsOpsForWG
        (
            WeightDims                wDims,        //!<  dims of the output wt blob
            IT*                       pSrcWts,      //!<  ptr to the processed caffe wt blob
            const std::vector<AtomicWtOp>& vWtOps        //!<  list of ops to trns the caffe blob
        )
    {
        int krnlPerGrp = sizeof(RT) == 1 ? 32 : 16;
        RT* pWGDestWts;
        RT* pWGDestWtsCopy;
        NvS64 trns_size = 0;
        Weights WG_tr_wts;

        std::vector<AtomicWtOp>::const_iterator iterWtOp = vWtOps.begin();

        pWGDestWts =
            reinterpret_cast<RT*>(engine_ast::MemoryCollector::getInstance()->allocateMemory(
                ROUNDUP_AND_ALIGN(wDims.wtSize * sizeof(RT), WG_WTS_SIZE_ALIGN)));
        memset(pWGDestWts, 0, ROUNDUP_AND_ALIGN(wDims.wtSize * sizeof(RT), WG_WTS_SIZE_ALIGN));
        pWGDestWtsCopy = pWGDestWts;

        for (int loop = 0; iterWtOp != vWtOps.end(); ++iterWtOp, ++loop)
        {
            for (int ind_cg = iterWtOp->cg.startIndex;
                 ind_cg < iterWtOp->cg.limit;
                 ++ind_cg)
            {
                for (int ind_k = iterWtOp->kg.startIndex*krnlPerGrp;
                     ind_k < iterWtOp->kg.startIndex*krnlPerGrp +
                             iterWtOp->k.limit;
                     ++ind_k)
                {
                    for (int ind_r = iterWtOp->r.startIndex;
                         ind_r < iterWtOp->r.limit;
                         ++ind_r)
                    {
                        for (int ind_s = iterWtOp->s.startIndex;
                             ind_s < iterWtOp->s.limit;
                             ++ind_s)
                        {
                            for (int ind_c = ind_cg*WG_FULL_CHANNELS_PER_ATOM;
                                 ind_c < (ind_cg*WG_FULL_CHANNELS_PER_ATOM +
                                          iterWtOp->c.limit);
                                 ++ind_c)
                            {
                                IT value = pSrcWts[ind_s +
                                                   iterWtOp->s.limit*(ind_r +
                                                                      iterWtOp->r.limit*(ind_c +
                                                                                         wDims.numChannels*(ind_k)))];
                                // We only support:
                                //  - int8 -> int8 translation
                                //  - int16-> int16 translation
                                //  - fp16 -> fp16 translation
                                //  - fp32 -> fp32 translation (will deprecate soon)
                                //  - fp32 -> fp16 translation
                                *pWGDestWts = RT(value);
                                ++pWGDestWts;
                                ++trns_size;
                            }
                        }
                    }
                }
            }
        }

        WG_tr_wts.type   = getEnumFromType<RT>();
        WG_tr_wts.values = pWGDestWtsCopy;
        WG_tr_wts.count  = trns_size;

        return WG_tr_wts;
    }


    //!<  get original 3x3 matrix from 4x4 by doing inverse WG translation
    template <typename T>
    static void getOrigWGMat
        (
            T trnsMat[4][4],
            T (&origMat)[3][3]
        )
    {
        T temp[3][4] = {{T(0)}};
        const NvS8 GInv[3][4] = {
            {1, 0, 0, 0},
            {0, 1, -1, 0},
            {0, 0, 0, 1},
        };

        const NvS8 GtInv[4][3] = {
            {1, 0, 0},
            {0, 1, 0},
            {0, -1, 0},
            {0, 0, 1},
        };

        //Matrix multiply [G^(-1)xGxgxGt]xGt^(-1)
        for(int ind_m = 0; ind_m < 3; ++ind_m)
        {
            for (int ind_p = 0; ind_p < 4; ++ind_p)
            {
                temp[ind_m][ind_p] = 0.0f;
                for (int ind_n = 0; ind_n < 4; ++ind_n)
                    temp[ind_m][ind_p] += GInv[ind_m][ind_n] *
                                          trnsMat[ind_n][ind_p];
            }
        }

        //Matrix multiply G^(-1)x[GxgxGtxGt^(-1)]
        for (int ind_r = 0; ind_r < 3; ++ind_r)
        {
            for (int ind_s = 0; ind_s < 3; ++ind_s)
            {
                origMat[ind_r][ind_s] = 0;
                for (int ind_n = 0; ind_n < 4; ++ind_n) {
                    origMat[ind_r][ind_s] += temp[ind_r][ind_n] *
                                             GtInv[ind_n][ind_s];
                }
            }
        }
    }

    //!<  compare 2 3x3 matrices
    template <typename T>
    static int compare3By3Matrices
        (
            T mat1[][3],
            T mat2[][3]
        )
    {
        for (int ind_m = 0; ind_m < 3; ++ind_m)
            for (int ind_n = 0; ind_n < 3; ++ind_n)
                if (round(1000 * mat1[ind_m][ind_n]) !=
                    round(1000 * mat2[ind_m][ind_n]))
                {
                    // try a bigger hammer one more time b4 failing
                    if (round(100 * mat1[ind_m][ind_n]) !=
                        round(100 * mat2[ind_m][ind_n]))
                        return -1;
                    else
                        continue;
                }
        return 0;
    }


    //!<  get original 3x3x3 cube from 4x4x4
    template <typename T>
    static void getOrigWGCube
        (
            T trnsCube[4][4][4],
            T (&origCube)[3][3][3]
        )
    {
        T trnsMat[4][4] = {{T(0)}};
        T origMat[3][3] = {{T(0)}};
        for (int ind_c = 0; ind_c < 4; ++ind_c)
        {
            memset(trnsMat, 0, sizeof(trnsMat[0][0]) * 4 * 4);
            for (int ind_r = 0; ind_r < 4; ++ind_r)
            {
                for (int ind_s = 0; ind_s < 4; ++ind_s)
                {
                    trnsMat[ind_r][ind_s] = trnsCube[ind_r][ind_s][ind_c];
                }
            }

            memset(origMat, 0, sizeof(origMat[0][0]) * 3 * 3);
            getOrigWGMat<T>(trnsMat, origMat);
            for (int ind_r = 0; ind_r < 3; ++ind_r)
            {
                for (int ind_s = 0; ind_s < 3; ++ind_s)
                {
                    origCube[ind_r][ind_s][ind_c] = origMat[ind_r][ind_s];
                }
            }
        }
    }

    //!<  run sanity after doing CE for DC
    template <typename T>
    static int runSanityForDCWtChnlExt
        (
            T* pSrcWts,             //!<  orig wt blob
            WeightDims srcWDims,    //!<  dims of orig wt blob
            T* pDCCEWts,            //!<  chnl extnd wt blob
            WeightDims ceWDims      //!<  dims of chnl extnd wt blob
        )
    {
        T ceWt;
        T caffeWt;
        for (int ind_k = 0; ind_k < ceWDims.numKernels; ++ind_k)
            for (int ind_c = 0; ind_c < ceWDims.numChannels; ++ind_c)
                for (int ind_r = 0; ind_r < ceWDims.height; ++ind_r)
                    for (int ind_s = 0; ind_s < ceWDims.width; ++ind_s)
                    {
                        ceWt = pDCCEWts[ind_s +
                                        ceWDims.width*(ind_r +
                                                       ceWDims.height*(ind_c +
                                                                       ceWDims.numChannels*(ind_k)))];
                        int orig_c = ind_c % srcWDims.numChannels;
                        int orig_r = (ind_r * srcWDims.strideY) +
                                     ((ind_c / srcWDims.numChannels) /
                                      srcWDims.strideY);
                        int orig_s = (ind_s * srcWDims.strideX) +
                                     ((ind_c / srcWDims.numChannels) %
                                      srcWDims.strideX);
                        if (orig_r < srcWDims.height &&
                            orig_s < srcWDims.width)
                            caffeWt = pSrcWts[orig_s +
                                              srcWDims.width*(orig_r +
                                                              srcWDims.height*(orig_c +
                                                                               srcWDims.numChannels*(ind_k)))];
                        else
                            caffeWt = 0;
                        if (ceWt != caffeWt)
                        {
                            return -1;
                        }
                    }
        return 0;
    }

    //!<  run sanity after complete wt translation for DC
    template <typename IT, typename RT>
    static int runSanityForDCWtTrns
        (
            RT*                 pDCTrWts,   //!<  ptr to translated wt blob for DC
            WeightDims          srcWDims,   //!<  dims of translated wt blob for DC
            std::vector<AtomicWtOp>  vWtOps,     //!<  list of all ops to achieve wt translation for DC
            std::map<std::string, IT>&    mCaffeHash,  //!<  hash of entire raw caffe wt blob
            int atomicKSize,
            int atomicCSize
        )
    {
        int err = 0;
        std::map<std::string, RT> mDCHash;
        mDCHash.clear();

        int krnlPerGrp = (sizeof(RT) == 1 ? atomicKSize : (atomicKSize / 2));

        std::vector<AtomicWtOp>::const_iterator iterWtOp = vWtOps.begin();

        for ( ; iterWtOp != vWtOps.end(); ++iterWtOp)
        {
            for (int ind_cg = iterWtOp->cg.startIndex;
                 ind_cg < iterWtOp->cg.limit;
                 ++ind_cg)
            {
                for (int ind_r = iterWtOp->r.startIndex;
                     ind_r < iterWtOp->r.limit;
                     ++ind_r)
                {
                    for (int ind_s = iterWtOp->s.startIndex;
                         ind_s < iterWtOp->s.limit;
                         ++ind_s)
                    {
                        for (int ind_k = iterWtOp->kg.startIndex*krnlPerGrp;
                             ind_k < iterWtOp->kg.startIndex*krnlPerGrp +
                                     iterWtOp->k.limit;
                             ++ind_k)
                        {
                            for (int ind_c = ind_cg*atomicCSize +
                                             iterWtOp->c.startIndex;
                                 ind_c < (ind_cg*atomicCSize +
                                          iterWtOp->c.startIndex +
                                          iterWtOp->c.limit);
                                 ++ind_c)
                            {
                                int orig_c = ind_c % srcWDims.numChannels;
                                int orig_r = (ind_r * srcWDims.strideY) +
                                             ((ind_c / srcWDims.numChannels) /
                                              srcWDims.strideY);
                                int orig_s = (ind_s * srcWDims.strideX) +
                                             ((ind_c / srcWDims.numChannels) %
                                              srcWDims.strideX);
                                std::string key;
                                if (orig_r < srcWDims.height &&
                                    orig_s < srcWDims.width)
                                {
                                    key = toString(ind_k) + "-" +
                                          toString(orig_c) + "-" +
                                          toString(orig_r) + "-" +
                                          toString(orig_s);
                                    mDCHash.insert(std::pair<std::string, RT>(key, *pDCTrWts));
                                }
                                ++pDCTrWts;
                            }
                        }
                    }
                }
            }
        }

        typename std::map<std::string, IT>::iterator iterCaffe = mCaffeHash.begin();
        typename std::map<std::string, RT>::iterator iterDC = mDCHash.begin();

        if (mCaffeHash.size() != mDCHash.size())
        {
            err = -1;
            goto exit;
        }

        // FIXME: have stricter check here
        for ( ; iterCaffe != mCaffeHash.end(); ++iterCaffe)
        {
            iterDC = mDCHash.find(iterCaffe->first);
            if (iterDC == mDCHash.end())
            {
                err = -1;
                goto exit;
            }
        }

        exit:
        mDCHash.clear();
        return err;
    }


    //!<  run sanity after complete wt translation for Deconvolution (same as direct convolution - UNVERIFIED)
    template <typename IT, typename RT>
    static int runSanityForDeconvWtTrns
        (
            RT*                 pDCTrWts,   //!<  ptr to translated wt blob for DC
            WeightDims          srcWDims,   //!<  dims of translated wt blob for DC
            std::vector<AtomicWtOp>  vWtOps,     //!<  list of all ops to achieve wt translation for DC
            std::map<std::string, IT>&    mCaffeHash,  //!<  hash of entire raw caffe wt blob
            int atomicKSize,
            int atomicCSize
        )
    {
        return runSanityForDCWtTrns<IT, RT>(pDCTrWts, srcWDims, vWtOps, mCaffeHash, atomicKSize, atomicCSize);
    }

    //!<  run sanity after zero padding wt blob for WG
    template <typename T>
    static int runSanityForWGWtZeroPadded
        (
            T* pSrcWts,         //!<  ptr to orig non zero padded wt blob
            WeightDims srcWDims,//!<  dims of orig non zero padded wt blob
            T* pWGZPWts,        //!<  ptr to zero padded wt blob
            WeightDims zpWDims  //!<  dims of zero padded wt blob
        )
    {
        int ind_zp = 0;
        int ind_src = 0;
        for (; ind_zp < zpWDims.wtSize; ++ind_zp)
        {
            if (!pWGZPWts[ind_zp])
            {
                if (ind_src < srcWDims.wtSize &&
                    !pSrcWts[ind_src])
                    ind_src++;
                continue;
            }

            if (pWGZPWts[ind_zp] != pSrcWts[ind_src])
            {
                return -1;
            }
            else
                ind_src++;
        }
        return 0;
    }


    //!<  run sanity after doing CE for WG
    template <typename T>
    static int runSanityForWGWtChnlExt
        (
            T* pSrcWts,         //!<  ptr to orig non chnl extnd wt blob
            WeightDims srcWDims,//!<  dims of orig non chnl extnd wt blob
            T* pWGCEWts,        //!<  ptr to chnl extnd wt blob
            WeightDims ceWDims  //!<  dims of chnl extnd wt blob
        )
    {
        T ceWt = T(0);
        T caffeWt = T(0);
        for (int ind_k = 0; ind_k < ceWDims.numKernels; ++ind_k)
            for (int ind_c = 0; ind_c < ceWDims.numChannels; ++ind_c)
                for (int ind_r = 0; ind_r < ceWDims.height; ++ind_r)
                    for (int ind_s = 0; ind_s < ceWDims.width; ++ind_s)
                    {
                        ceWt = pWGCEWts[ind_s +
                                        ceWDims.width*(ind_r +
                                                       ceWDims.height*(ind_c +
                                                                       ceWDims.numChannels*(ind_k)))];
                        int orig_c = ind_c % srcWDims.numChannels;
                        int orig_r = (ind_r * srcWDims.strideY) +
                                     ((ind_c / srcWDims.numChannels) /
                                      srcWDims.strideY);
                        int orig_s = (ind_s * srcWDims.strideX) +
                                     ((ind_c / srcWDims.numChannels) %
                                      srcWDims.strideX);
                        if (orig_r < srcWDims.height &&
                            orig_s < srcWDims.width)
                            caffeWt = pSrcWts[orig_s +
                                              srcWDims.width*(orig_r +
                                                              srcWDims.height*(orig_c +
                                                                               srcWDims.numChannels*(ind_k)))];
                        else
                            caffeWt = 0;
                        if (ceWt != caffeWt)
                        {
                            return -1;
                        }
                    }
        return 0;
    }

    //!<  run sanity after doing matrix translation on the wt blob for WG
    template <typename T>
    static int runSanityForWGWtMatrixTrns
        (
            T*              pSrcWts,    //!<  ptr to orig non matrix translated wt blob
            WeightDims      srcWDims,   //!<  dims of orig non matrix translated wt blob
            T*              pWGMatTrWts,//!<  ptr to matrix translated wt blob
            WeightDims      mtWDims     //!<  dims of matrix translated wt blob
        )
    {
        T trnsMat[4][4] = {{T(0)}};
        T origMat[3][3] = {{T(0)}};
        T retrievedMat[3][3] = {{T(0)}};

        for (int ind_k = 0; ind_k < mtWDims.numKernels; ++ind_k)
        {
            for (int ind_c = 0; ind_c < mtWDims.numChannels; ++ind_c)
            {
                memset(trnsMat, 0.0f, sizeof(trnsMat[0][0]) * 4 * 4);
                for (int ind_rt = 0; ind_rt < mtWDims.height; ++ind_rt)
                {
                    for (int ind_st = 0; ind_st < mtWDims.width; ++ind_st)
                    {
                        trnsMat[ind_rt][ind_st] = pWGMatTrWts[ind_st +
                                                              mtWDims.width*(ind_rt +
                                                                             mtWDims.height*(ind_c +
                                                                                             mtWDims.numChannels*(ind_k)))];
                    }
                }

                memset(origMat, 0.0f, sizeof(origMat[0][0]) * 3 * 3);
                for (int ind_ro = 0; ind_ro < srcWDims.height; ++ind_ro)
                {
                    for (int ind_so = 0; ind_so < srcWDims.width; ++ind_so)
                    {
                        origMat[ind_ro][ind_so] = pSrcWts[ind_so +
                                                          srcWDims.width*(ind_ro +
                                                                          srcWDims.height*(ind_c +
                                                                                           srcWDims.numChannels*(ind_k)))];
                    }
                }

                memset(retrievedMat, 0.0f, sizeof(retrievedMat[0][0]) * 3 * 3);
                getOrigWGMat<T>(trnsMat, retrievedMat);
                if (compare3By3Matrices<T>(origMat, retrievedMat))
                {
                    return -1;
                }
            }
        }
        return 0;
    }


    //!<  run sanity after complete wt translation for WG
    template <typename IT, typename RT>
    static int runSanityForWGWtTrns
        (
            RT*                pWGTrWts,  //!<  ptr to translated wt blob for WG
            std::vector<AtomicWtOp> vWtOps,    //!<  list of all ops to achieve wt translation
            std::map<std::string, IT>&   mCaffeHash //!<  hash of the entire raw caffe wt blob
        )
    {
        int err = 0;
        RT trnsWGCube[4][4][4] = {{{RT(0)}}};
        RT origWGCube[3][3][3] = {{{RT(0)}}};
        std::map<std::string, RT> mWGHash;
        mWGHash.clear();

        int krnlPerGrp = (sizeof(RT) == 1 ? 32 : 16);

        std::vector<AtomicWtOp>::const_iterator iterWtOp = vWtOps.begin();

        for (; iterWtOp != vWtOps.end(); ++iterWtOp)
        {
            for (int ind_cg = iterWtOp->cg.startIndex;
                 ind_cg < iterWtOp->cg.limit;
                 ++ind_cg)
            {
                for (int ind_k = iterWtOp->kg.startIndex*krnlPerGrp;
                     ind_k < iterWtOp->kg.startIndex*krnlPerGrp +
                             iterWtOp->k.limit;
                     ++ind_k)
                {
                    memset(trnsWGCube, 0, sizeof(trnsWGCube[0][0][0]) * 4 * 4 * 4);
                    for (int ind_rt = 0; ind_rt < 4; ++ind_rt)
                    {
                        for (int ind_st = 0; ind_st < 4; ++ind_st)
                        {
                            for (int ind_c = 0; ind_c < WG_FULL_CHANNELS_PER_ATOM; ++ind_c)
                            {
                                trnsWGCube[ind_rt][ind_st][ind_c] = *pWGTrWts;
                                ++pWGTrWts;
                            }
                        }
                    }

                    memset(origWGCube, 0, sizeof(origWGCube[0][0][0]) * 3 * 3 * 3);
                    getOrigWGCube<RT>(trnsWGCube, origWGCube);

                    for (int ind_ro = 0; ind_ro < 3; ++ind_ro)
                    {
                        for (int ind_so = 0; ind_so < 3; ++ind_so)
                        {
                            for (int ind_c = 0; ind_c < WG_FULL_CHANNELS_PER_ATOM; ++ind_c)
                            {
                                int orig_c = ind_cg * WG_FULL_CHANNELS_PER_ATOM + ind_c;
                                std::string key = toString(ind_k) + "-" +
                                                  toString(orig_c) + "-" +
                                                  toString(ind_ro) + "-" +
                                                  toString(ind_so);
                                mWGHash.insert(std::pair<std::string, RT>(key,
                                                                          origWGCube[ind_ro][ind_so][ind_c]));
                            }
                        }
                    }
                }
            }
        }

        typename std::map<std::string, IT>::iterator iterCaffe = mCaffeHash.begin();
        typename std::map<std::string, RT>::iterator iterWG = mWGHash.begin();

        if (mCaffeHash.size() != mWGHash.size())
        {
            err = -1;
            goto exit;
        }

        for ( ; iterCaffe != mCaffeHash.end(); ++iterCaffe)
        {
            iterWG = mWGHash.find(iterCaffe->first);
            if (iterWG == mWGHash.end())
            {
                err = -1;
                goto exit;
            }
        }

        exit:
        mWGHash.clear();
        return err;
    }
};


} // nvdla::priv
} // nvdla

#endif /* NVDLA_PRIV_WEIGHT_TRNS_UNIT_H */
