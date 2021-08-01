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

#ifndef NVDLA_PRIV_ENGINE_AST_H
#define NVDLA_PRIV_ENGINE_AST_H

#include <vector>
#include <unordered_set>
#include <map>
#include <algorithm>
#include <queue>
#include <array>

#include "AST.h"
#include "Tensor.h"
#include "Type.h" // for misc

#include "CanonicalAST.h"
#include "MultiBatch.h"
#include "Surface.h"
#include "TargetConfig.h"
#include "priv/Loadable.h"

#include "DLAInterface.h"
#include "DLAResourceManager.h"
#include "EngineASTEnums.h"
#include "LutManager.h"
#include "priv/EMUInterface.h"

#ifdef NVDLA_COMPILER_OUTPUT_FOR_PROTOTEST
#include "priv/DlaPrototestInterface.pb.h"
#endif

#define NUM_MAX_BDMA_OPS            1
#define SDP_ADDER_DATA_INDEX        0
#define SDP_MULTIPLIER_DATA_INDEX   1
#define SDP_LEFT_SHIFT_MAX_PLACES   16
#define SDP_RIGHTT_SHIFT_MAX_PLACES 32

#define FOR_EACH(SEQ, ITR, FUNC)                   \
    for (ITR i = SEQ.begin(); i != SEQ.end(); ++i) \
    {                                              \
        PROPAGATE_ERROR_FAIL((*i)->FUNC());        \
    }

//
// provide explicit hash fn specialization for std::pair<engine_ast::Node *, engine_ast::Edge *> (Elem)
//
namespace nvdla {
namespace priv {
namespace engine_ast {
class Node;
class Edge;
} // namespace engine_ast
} // namespace priv
} // namespace nvdla
namespace std {
template<>
struct hash<std::pair<nvdla::priv::engine_ast::Node*, nvdla::priv::engine_ast::Edge*> >
{
    std::size_t operator()(const std::pair<nvdla::priv::engine_ast::Node*, nvdla::priv::engine_ast::Edge*>& v) const noexcept
    {
        std::size_t h;
        // one side or the other will be non-zero.
        if (v.first)
        {
            h = std::hash<nvdla::priv::engine_ast::Node*>{}(v.first);
        }
        else
        {
            h = std::hash<nvdla::priv::engine_ast::Edge*>{}(v.second);
        }
        return h;
    }
};
} // namespace std

namespace nvdla {

namespace priv {

static surface::SurfaceFormatEnum IMG_FORMATS[] = {
    surface::SurfaceFormatEnum::NVDLA_IMG_R8,
    surface::SurfaceFormatEnum::NVDLA_IMG_R10,
    surface::SurfaceFormatEnum::NVDLA_IMG_R12,
    surface::SurfaceFormatEnum::NVDLA_IMG_R16,
    surface::SurfaceFormatEnum::NVDLA_IMG_R16_I,
    surface::SurfaceFormatEnum::NVDLA_IMG_R16_F,
    surface::SurfaceFormatEnum::NVDLA_IMG_A16B16G16R16,
    surface::SurfaceFormatEnum::NVDLA_IMG_X16B16G16R16,
    surface::SurfaceFormatEnum::NVDLA_IMG_A16B16G16R16_F,
    surface::SurfaceFormatEnum::NVDLA_IMG_A16Y16U16V16,
    surface::SurfaceFormatEnum::NVDLA_IMG_V16U16Y16A16,
    surface::SurfaceFormatEnum::NVDLA_IMG_A16Y16U16V16_F,
    surface::SurfaceFormatEnum::NVDLA_IMG_A8B8G8R8,
    surface::SurfaceFormatEnum::NVDLA_IMG_A8R8G8B8,
    surface::SurfaceFormatEnum::NVDLA_IMG_B8G8R8A8,
    surface::SurfaceFormatEnum::NVDLA_IMG_R8G8B8A8,
    surface::SurfaceFormatEnum::NVDLA_IMG_X8B8G8R8,
    surface::SurfaceFormatEnum::NVDLA_IMG_X8R8G8B8,
    surface::SurfaceFormatEnum::NVDLA_IMG_B8G8R8X8,
    surface::SurfaceFormatEnum::NVDLA_IMG_R8G8B8X8,
    surface::SurfaceFormatEnum::NVDLA_IMG_A2B10G10R10,
    surface::SurfaceFormatEnum::NVDLA_IMG_A2R10G10B10,
    surface::SurfaceFormatEnum::NVDLA_IMG_B10G10R10A2,
    surface::SurfaceFormatEnum::NVDLA_IMG_R10G10B10A2,
    surface::SurfaceFormatEnum::NVDLA_IMG_A2Y10U10V10,
    surface::SurfaceFormatEnum::NVDLA_IMG_V10U10Y10A2,
    surface::SurfaceFormatEnum::NVDLA_IMG_A8Y8U8V8,
    surface::SurfaceFormatEnum::NVDLA_IMG_V8U8Y8A8,
    surface::SurfaceFormatEnum::NVDLA_IMG_Y8___U8V8_N444,
    surface::SurfaceFormatEnum::NVDLA_IMG_Y8___V8U8_N444,
    surface::SurfaceFormatEnum::NVDLA_IMG_Y10___U10V10_N444,
    surface::SurfaceFormatEnum::NVDLA_IMG_Y10___V10U10_N444,
    surface::SurfaceFormatEnum::NVDLA_IMG_Y12___U12V12_N444,
    surface::SurfaceFormatEnum::NVDLA_IMG_Y12___V12U12_N444,
    surface::SurfaceFormatEnum::NVDLA_IMG_Y16___U16V16_N444,
    surface::SurfaceFormatEnum::NVDLA_IMG_Y16___V16U16_N444};

class Loadable;

namespace engine_ast {

//
// EngineParams hold all the details needed to program the HW engine
// Some of the engine params are directly inherited from the canonical AST
// equivalent operations, whereas some others are computed over the course
// of engine AST compilation.
//
// In short, only those params which are directly needed for HW engine programming
// should be held in EngineParams. Any dynamic state for assisting compilation
// should be held in the OpParams of respective engine nodes.
//

enum ConcatAxisEnum
{
    ENGINE_AST_CONCAT_AXIS_ENUMS(GEN_ENUM)
};
typedef SequenceEnum<ConcatAxisEnum, NvU8> ConcatAxis;

enum SplitAxisEnum
{
    ENGINE_AST_SPLIT_AXIS_ENUMS(GEN_ENUM)
};
typedef SequenceEnum<SplitAxisEnum, NvU8> SplitAxis;

enum ConvolutionModeEnum
{
    ENGINE_AST_CONVOLUTION_MODE_ENUMS(GEN_ENUM)
};
typedef SequenceEnum<ConvolutionModeEnum, NvU8> ConvolutionMode;

enum SDPModeEnum
{
    ENGINE_AST_SDP_MODE_ENUMS(GEN_ENUM)
};
typedef SequenceEnum<SDPModeEnum, NvU8> SDPMode;

enum SDPALUTypeEnum
{
    ENGINE_AST_SDP_ALU_TYPE_ENUMS(GEN_ENUM)
};
typedef SequenceEnum<SDPALUTypeEnum, NvU8> SDPALUType;

enum SDPOpTypeEnum
{
    ENGINE_AST_SDP_OP_TYPE_ENUMS(GEN_ENUM)
};
typedef SequenceEnum<SDPOpTypeEnum, NvU8> SDPOpType;

enum SDPActTypeEnum
{
    ENGINE_AST_SDP_ACT_TYPE_ENUMS(GEN_ENUM)
};
typedef SequenceEnum<SDPActTypeEnum, NvU8> SDPActType;

enum SDPSubEngineTypeEnum
{
    ENGINE_AST_SDP_SUBENGINE_TYPE_ENUMS(GEN_ENUM)
};
typedef SequenceEnum<SDPSubEngineTypeEnum, NvU8> SDPSubEngineType;

enum RubikModeEnum
{
    ENGINE_AST_RUBIK_MODE_ENUMS(GEN_ENUM)
};
typedef SequenceEnum<RubikModeEnum, NvU8> RubikMode;

class EngineParams
{
public:
    virtual ~EngineParams()
    {
    }
};

class PrecisionCVTParams
{
public:
    PrecisionCVTParams()
        : m_truncate(0),
          m_scale(1),
          m_offset(0),
          m_enable(false)
    {
    }
    virtual ~PrecisionCVTParams()
    {
    }

    /* clone */
    PrecisionCVTParams(const PrecisionCVTParams& other)
        : m_truncate(other.m_truncate),
          m_scale(other.m_scale),
          m_offset(other.m_offset),
          m_enable(other.m_enable)
    {
    }

    NvS16 scale() const
    {
        return m_scale;
    }
    void setScale(NvS16 scale)
    {
        m_scale = scale;
    }

    NvU8 truncate() const
    {
        return m_truncate;
    }
    void setTruncate(NvU8 truncate)
    {
        m_truncate = truncate;
    }

    NvU8 offset() const
    {
        return m_offset;
    }
    void setOffset(NvU8 offset)
    {
        m_offset = offset;
    }

    bool isEnable() const
    {
        return m_enable;
    }
    void setEnable(bool enable)
    {
        m_enable = enable;
    }

protected:
    NvU8 m_truncate; // u6 right shifter
    NvS16 m_scale;   // i16 scalar
    NvS32 m_offset;
    bool m_enable;
};

class ConvCoreCVTParams
{
public:
    ConvCoreCVTParams()
        : m_pra_truncate(0),
          m_out_truncate(0)
    {
    }
    virtual ~ConvCoreCVTParams()
    {
    }

    /* clone */
    ConvCoreCVTParams(const ConvCoreCVTParams& other)
        : m_pra_truncate(other.m_pra_truncate),
          m_out_truncate(other.m_out_truncate),
          m_input_cvt(other.m_input_cvt)
    {
    }

    NvU8 outTruncate() const
    {
        return m_out_truncate;
    }
    void setOutTruncate(NvU8 outTruncate)
    {
        m_out_truncate = outTruncate;
    }

    NvU8 praTruncate() const
    {
        return m_pra_truncate;
    }
    void setPraTruncate(NvU8 praTruncate)
    {
        m_pra_truncate = praTruncate;
    }

    PrecisionCVTParams& inputCVT()
    {
        return m_input_cvt;
    }
    void setInputCVT(const PrecisionCVTParams& inputCvt)
    {
        m_input_cvt = inputCvt;
    }

protected:
    NvU8 m_pra_truncate;            // u2 right shifter
    NvU8 m_out_truncate;            // u6 right shifter
    PrecisionCVTParams m_input_cvt; // input side CVT
};

static EngineParams EngineParamsNULL;

class ConvCoreEngineParams : public EngineParams
{
public:
    ConvCoreEngineParams()
        : EngineParams(),
          m_has_bias_term(0),
          m_padding_value(0),
          m_raw_weights(),
          m_dla_weights(),
          m_conv_mode(ConvolutionModeEnum::CONV_MODE_UNKNOWN),
          m_reuse_data(false),
          m_reuse_weights(false),
          m_release_data(true),
          m_release_weights(true),
          m_retain_slices(0),
          m_data_banks_allotted(0),
          m_weight_banks_allotted(0),
          m_num_groups(1)
    {
    }
    virtual ~ConvCoreEngineParams()
    {
    }

    /* clone */
    ConvCoreEngineParams(const ConvCoreEngineParams& other)
        : m_has_bias_term(other.m_has_bias_term),
          m_padding_value(other.m_padding_value),
          m_raw_weights(other.m_raw_weights),
          m_dla_weights(other.m_dla_weights),
          m_conv_mode(other.m_conv_mode),
          m_wg_params(other.m_wg_params),
          m_reuse_data(other.m_reuse_data),
          m_reuse_weights(other.m_reuse_weights),
          m_release_data(other.m_release_data),
          m_release_weights(other.m_release_weights),
          m_retain_slices(other.m_retain_slices),
          m_data_banks_allotted(other.m_data_banks_allotted),
          m_weight_banks_allotted(other.m_weight_banks_allotted),
          m_num_groups(other.m_num_groups)
    {
    }

    // hold all winograd specific params here
    struct WinogradParams
    {
        Dims4 inDims;
        Dims4 outDims;
        Dims4 auxDims;
    };

    Dims2 dilation() const
    {
        return m_dilation;
    }
    void setDilation(Dims2 dilation)
    {
        m_dilation = dilation;
    }

    NvU8 hasBiasTerm() const
    {
        return m_has_bias_term;
    }
    void setHasBiasTerm(NvU8 hasBias)
    {
        m_has_bias_term = hasBias;
    }

    int paddingValue() const
    {
        return m_padding_value;
    }
    void setPaddingValue(int p)
    {
        m_padding_value = p;
    }

    Dims2 bottomRightPadding() const
    {
        return m_BR_padding;
    }
    void setBottomRightPadding(Dims2 br)
    {
        m_BR_padding = br;
    }

    Dims2 topLeftPadding() const
    {
        return m_TL_padding;
    }
    void setTopLeftPadding(Dims2 tl)
    {
        m_TL_padding = tl;
    }

    Dims2 stride() const
    {
        return m_stride;
    }
    void setStride(Dims2 stride)
    {
        m_stride = stride;
    }

    Weights rawWeights() const
    {
        return m_raw_weights;
    }
    void setRawWeights(Weights raw)
    {
        m_raw_weights = raw;
    }

    Weights DLAWeights() const
    {
        return m_dla_weights;
    }
    void setDLAWeights(Weights trns)
    {
        m_dla_weights = trns;
    }

    const Dims4 weightDims() const
    {
        return m_weight_dims;
    }
    void setWeightDims(const Dims4 wd)
    {
        m_weight_dims = wd;
    }

    ConvolutionMode convMode() const
    {
        return m_conv_mode;
    }
    void setConvMode(ConvolutionMode convMode)
    {
        m_conv_mode = convMode;
    }

    WinogradParams& winogradParams()
    {
        return m_wg_params;
    }
    void setWinogradParams(const WinogradParams& wgp)
    {
        m_wg_params = wgp;
    }
    void clearWinogradParams()
    {
        m_wg_params.inDims = Dims4(-1, -1, -1, -1);
        m_wg_params.outDims = Dims4(-1, -1, -1, -1);
        m_wg_params.auxDims = Dims4(-1, -1, -1, -1);
    }

    ConvCoreCVTParams& convCoreCVT()
    {
        return m_cc_cvt;
    }
    void setConvCoreCVT(const ConvCoreCVTParams& ccCvt)
    {
        m_cc_cvt = ccCvt;
    }

    NvU8 dataBanksAllotted() const
    {
        return m_data_banks_allotted;
    }
    void setAllottedDataBanks(NvU8 db)
    {
        m_data_banks_allotted = db;
    }

    NvU8 weightBanksAllotted() const
    {
        return m_weight_banks_allotted;
    }
    void setAllottedWeightBanks(NvU8 wb)
    {
        m_weight_banks_allotted = wb;
    }

    bool isReuseData() const
    {
        return m_reuse_data;
    }
    void setReuseData(bool dataReuse)
    {
        m_reuse_data = dataReuse;
    }

    bool isReuseWeights() const
    {
        return m_reuse_weights;
    }
    void setReuseWeights(bool weightsReuse)
    {
        m_reuse_weights = weightsReuse;
    }

    bool isReleaseData() const
    {
        return m_release_data;
    }
    void setReleaseData(bool dataRelease)
    {
        m_release_data = dataRelease;
    }

    bool isReleaseWeights() const
    {
        return m_release_weights;
    }
    void setReleaseWeights(bool weightsRelease)
    {
        m_release_weights = weightsRelease;
    }

    NvU16 retainSlices() const
    {
        return m_retain_slices;
    }
    void setRetainSlices(NvU16 releaseSlices)
    {
        m_retain_slices = releaseSlices;
    }

    NvU32 numGroups() const
    {
        return m_num_groups;
    }
    void setNumGroups(NvU16 numGroups)
    {
        m_num_groups = numGroups;
    }

    std::vector<NvF32>& filterScales()
    {
        return m_filter_scales;
    }
    void setFilterScales(const std::vector<NvF32>& filterScales)
    {
        m_filter_scales = filterScales;
    }

protected:
    NvU8 m_has_bias_term;
    NvU32 m_padding_value;
    Dims2 m_stride;
    Dims2 m_dilation;
    Dims2 m_TL_padding;
    Dims2 m_BR_padding;
    Dims4 m_weight_dims;
    Weights m_raw_weights;
    Weights m_dla_weights;

    ConvolutionMode m_conv_mode;
    WinogradParams m_wg_params;
    ConvCoreCVTParams m_cc_cvt;

    bool m_reuse_data;
    bool m_reuse_weights;
    bool m_release_data;
    bool m_release_weights;
    NvU16 m_retain_slices;
    NvU8 m_data_banks_allotted;
    NvU8 m_weight_banks_allotted;
    NvU16 m_num_groups;
    std::vector<NvF32> m_filter_scales;
};

class SDPSubEngineParams
{
public:
    SDPSubEngineParams()
        : m_enabled(false),
          m_mode(SDPModeEnum::SDP_MODE_UNKNOWN),
          m_alu_type(SDPALUTypeEnum::SDP_ALU_TYPE_UNKNOWN),
          m_op_type(SDPOpTypeEnum::SDP_OP_TYPE_NONE),
          m_act_type(SDPActTypeEnum::SDP_ACT_TYPE_NONE),
          m_truncate(0),
          m_shift_value(0),
          m_alu_operand(0),
          m_mul_operand(0),
          m_precision(surface::SurfacePrecisionEnum::NVDLA_PRECISION_UNKNOWN),
          m_isINT8Rescaling(false)
    {
    }
    virtual ~SDPSubEngineParams()
    {
    }

    /* clone */
    SDPSubEngineParams(const SDPSubEngineParams& other)
        : m_enabled(other.m_enabled),
          m_mode(other.m_mode),
          m_alu_type(other.m_alu_type),
          m_op_type(other.m_op_type),
          m_act_type(other.m_act_type),
          m_truncate(other.m_truncate),
          m_shift_value(other.m_shift_value),
          m_alu_operand(other.m_alu_operand),
          m_mul_operand(other.m_mul_operand),
          m_precision(other.m_precision),
          m_isINT8Rescaling(other.m_isINT8Rescaling)
    {
    }

    bool enabled() const
    {
        return m_enabled;
    }
    void setEnabled(bool enabled)
    {
        m_enabled = enabled;
    }

    SDPMode mode() const
    {
        return m_mode;
    }
    void setMode(SDPMode mode)
    {
        m_mode = mode;
    }

    SDPActType actType() const
    {
        return m_act_type;
    }
    void setActType(SDPActType actType)
    {
        m_act_type = actType;
    }

    SDPOpType opType() const
    {
        return m_op_type;
    }
    void setOpType(SDPOpType opType)
    {
        m_op_type = opType;
    }

    SDPALUType aluType() const
    {
        return m_alu_type;
    }
    void setAluType(SDPALUType aluType)
    {
        m_alu_type = aluType;
    }

    NvU8 truncate() const
    {
        return m_truncate;
    }
    void setTruncate(NvU8 truncate)
    {
        m_truncate = truncate;
    }

    NvU8 shiftValue() const
    {
        return m_shift_value;
    }
    void setShiftValue(NvU8 shiftValue)
    {
        m_shift_value = shiftValue;
    }

    NvS16 aluOperand() const
    {
        return m_alu_operand;
    }
    void setAluOperand(NvS16 aluOperand)
    {
        m_alu_operand = aluOperand;
    }

    NvS16 mulOperand() const
    {
        return m_mul_operand;
    }
    void setMulOperand(NvS16 mulOperand)
    {
        m_mul_operand = mulOperand;
    }

    surface::SurfacePrecision precision() const
    {
        return m_precision;
    }
    void setPrecision(surface::SurfacePrecision precision)
    {
        m_precision = precision;
    }

    PrecisionCVTParams& mulCVT()
    {
        return m_mul_cvt;
    }
    void setMULCVT(const PrecisionCVTParams& mulCvt)
    {
        m_mul_cvt = mulCvt;
    }

    PrecisionCVTParams& aluCVT()
    {
        return m_alu_cvt;
    }
    void setALUCVT(const PrecisionCVTParams& aluCvt)
    {
        m_alu_cvt = aluCvt;
    }

    bool isINT8Rescaling()
    {
        return m_isINT8Rescaling;
    }
    void setINT8Rescaling(bool enb)
    {
        m_isINT8Rescaling = enb;
    }

protected:
    bool m_enabled;
    SDPMode m_mode;
    SDPALUType m_alu_type;
    SDPOpType m_op_type;
    SDPActType m_act_type;
    NvU8 m_truncate;
    NvU8 m_shift_value;
    NvS16 m_alu_operand;
    NvS16 m_mul_operand;
    surface::SurfacePrecision m_precision;
    PrecisionCVTParams m_alu_cvt;
    PrecisionCVTParams m_mul_cvt;
    bool m_isINT8Rescaling;
};

class SDPEngineParams : public EngineParams
{
public:
    SDPEngineParams()
        : EngineParams(),
          m_x1_params(),
          m_x2_params(),
          m_y_params(),
          m_conv_mode(ConvolutionModeEnum::CONV_DIRECT),
          m_wg_params(),
          m_num_groups(1)
    {
    }
    virtual ~SDPEngineParams()
    {
    }

    /* clone */
    SDPEngineParams(const SDPEngineParams& other)
        : m_x1_params(other.m_x1_params),
          m_x2_params(other.m_x2_params),
          m_y_params(other.m_y_params),
          m_conv_mode(other.m_conv_mode),
          m_wg_params(other.m_wg_params),
          m_num_groups(other.m_num_groups),
          m_out_cvt(other.m_out_cvt)
    {
    }

    // hold all winograd specific params here
    struct WinogradParams
    {
        Dims4 ioDims;
        Dims4 auxDims;
    };

    SDPSubEngineParams& x1Params()
    {
        return m_x1_params;
    }
    void setX1Params(const SDPSubEngineParams& x1Params)
    {
        m_x1_params = x1Params;
    }

    SDPSubEngineParams& x2Params()
    {
        return m_x2_params;
    }
    void setX2Params(const SDPSubEngineParams& x2Params)
    {
        m_x2_params = x2Params;
    }

    SDPSubEngineParams& yParams()
    {
        return m_y_params;
    }
    void setYParams(const SDPSubEngineParams& yParams)
    {
        m_y_params = yParams;
    }

    ConvolutionMode convMode() const
    {
        return m_conv_mode;
    }
    void setConvMode(ConvolutionMode convMode)
    {
        m_conv_mode = convMode;
    }

    WinogradParams& winogradParams()
    {
        return m_wg_params;
    }
    void setWinogradParams(const WinogradParams& wgp)
    {
        m_wg_params = wgp;
    }

    NvU16 numGroups() const
    {
        return m_num_groups;
    }
    void setNumGroups(NvU16 groups)
    {
        m_num_groups = groups;
    }

    PrecisionCVTParams& outCVT()
    {
        return m_out_cvt;
    }
    void setOutCVT(const PrecisionCVTParams& outCvt)
    {
        m_out_cvt = outCvt;
    }

protected:
    SDPSubEngineParams m_x1_params; // for x1,x2,y
    SDPSubEngineParams m_x2_params;
    SDPSubEngineParams m_y_params;
    ConvolutionMode m_conv_mode;
    WinogradParams m_wg_params;
    NvU16 m_num_groups;
    PrecisionCVTParams m_out_cvt;
};

class CPUParams : public EngineParams
{
public:
    CPUParams()
        : EngineParams()
    {
    }
    virtual ~CPUParams()
    {
    }
};

class RubikEngineParams : public EngineParams
{
public:
    RubikEngineParams()
        : m_mode(RubikModeEnum::RUBIK_MODE_UNKNOWN)
    {
    }
    virtual ~RubikEngineParams()
    {
    }

    RubikMode mode() const
    {
        return m_mode;
    }
    void setMode(RubikMode mode)
    {
        m_mode = mode;
    }

    Dims2 deconvStride() const
    {
        return m_deconv_stride;
    }
    void setDeconvStride(Dims2 stride)
    {
        m_deconv_stride = stride;
    }

    // hold all contract op related params here
    struct ContractOpParams
    {
        Dims4 inDims;
        Dims4 outDims;
    };

    ContractOpParams& contractOpParams()
    {
        return m_contract_op_params;
    }
    void setContractOpParams(const ContractOpParams& cp)
    {
        m_contract_op_params = cp;
    }

protected:
    RubikMode m_mode;
    Dims2 m_deconv_stride;
    ContractOpParams m_contract_op_params;
};

class PDPEngineParams : public EngineParams
{
public:
    PDPEngineParams()
    {
    }
    virtual ~PDPEngineParams()
    {
    }

    struct hwSplitWidthInfo
    {
        NvS32 numSplits;
        NvS32 rightPadding;
        NvS32 bottomPadding;
        // input
        NvS32 firstInWidth;
        NvS32 midInWidth;
        NvS32 lastInWidth;
        NvS32 numOverlapStripes;
        // output
        NvS32 firstOutWidth;
        NvS32 midOutWidth;
        NvS32 lastOutWidth;

        hwSplitWidthInfo()
            : numSplits(0),
              rightPadding(0),
              bottomPadding(0),
              firstInWidth(0),
              midInWidth(0),
              lastInWidth(0),
              numOverlapStripes(0),
              firstOutWidth(0),
              midOutWidth(0),
              lastOutWidth(0)
        {
        }
    };

    nvdla::PoolingType poolingType() const
    {
        return m_pooling_type;
    }
    void setPoolingType(nvdla::PoolingType poolingType)
    {
        m_pooling_type = poolingType;
    }

    virtual int paddingValue() const
    {
        return m_padding_value;
    }
    virtual void setPaddingValue(int p)
    {
        m_padding_value = p;
    }

    virtual Dims2 bottomRightPadding() const
    {
        return m_BR_padding;
    }
    virtual void setBottomRightPadding(Dims2 pad)
    {
        m_BR_padding = pad;
    }

    virtual Dims2 topLeftPadding() const
    {
        return m_TL_padding;
    }
    virtual void setTopLeftPadding(Dims2 pad)
    {
        m_TL_padding = pad;
    }

    virtual Dims2 stride() const
    {
        return m_stride;
    }
    virtual void setStride(Dims2 stride)
    {
        m_stride = stride;
    }

    virtual Dims2 poolingWindow() const
    {
        return m_pooling_window;
    }
    virtual void setPoolingWindow(Dims2 window)
    {
        m_pooling_window = window;
    }

    hwSplitWidthInfo getHwSplitWidthInfo() const
    {
        return m_hwSplitWidthInfo;
    }
    void setHwSplitWidthInfo(hwSplitWidthInfo sinfo)
    {
        m_hwSplitWidthInfo = sinfo;
    }

protected:
    nvdla::PoolingType m_pooling_type;
    Dims2 m_TL_padding;
    Dims2 m_BR_padding;
    NvU32 m_padding_value;
    Dims2 m_stride;
    Dims2 m_pooling_window;
    hwSplitWidthInfo m_hwSplitWidthInfo;
};

class CDPEngineParams : public EngineParams
{
public:
    CDPEngineParams()
    {
    }
    virtual ~CDPEngineParams()
    {
    }

    virtual NvU32 localSize() const
    {
        return m_local_size;
    }
    virtual void setLocalSize(NvU32 localSize)
    {
        m_local_size = localSize;
    }

    virtual NvF32 alpha() const
    {
        return m_alpha;
    }
    virtual void setAlpha(NvF32 alpha)
    {
        m_alpha = alpha;
    }

    virtual NvF32 beta() const
    {
        return m_beta;
    }
    virtual void setBeta(NvF32 beta)
    {
        m_beta = beta;
    }

    virtual NvF32 k() const
    {
        return m_k;
    }
    virtual void setK(NvF32 k)
    {
        m_k = k;
    }

protected:
    NvU32 m_local_size;
    NvF32 m_alpha;
    NvF32 m_beta;
    NvF32 m_k;
};

class BDMAEngineParams : public EngineParams
{
public:
    BDMAEngineParams()
    {
    }
    virtual ~BDMAEngineParams()
    {
    }

    NvU32 destLine() const
    {
        return m_dest_line;
    }
    void setDestLine(NvU32 destLine)
    {
        m_dest_line = destLine;
    }

    NvU32 destSurface() const
    {
        return m_dest_surface;
    }
    void setDestSurface(NvU32 destSurface)
    {
        m_dest_surface = destSurface;
    }

    NvU32 lineRepeat() const
    {
        return m_line_repeat;
    }
    void setLineRepeat(NvU32 lineRepeat)
    {
        m_line_repeat = lineRepeat;
    }

    NvU32 lineSize() const
    {
        return m_line_size;
    }
    void setLineSize(NvU32 lineSize)
    {
        m_line_size = lineSize;
    }

    NvU32 srcLine() const
    {
        return m_src_line;
    }
    void setSrcLine(NvU32 srcLine)
    {
        m_src_line = srcLine;
    }

    NvU32 srcSurface() const
    {
        return m_src_surface;
    }
    void setSrcSurface(NvU32 srcSurface)
    {
        m_src_surface = srcSurface;
    }

    NvU32 surfaceRepeat() const
    {
        return m_surface_repeat;
    }
    void setSurfaceRepeat(NvU32 surfaceRepeat)
    {
        m_surface_repeat = surfaceRepeat;
    }

    NvU32 numTransfers() const
    {
        return m_num_transfers;
    }
    void setNumTransfers(NvU32 numTransfers)
    {
        m_num_transfers = numTransfers;
    }

private:
    NvU32 m_line_size;
    NvU32 m_line_repeat;
    NvU32 m_src_line;
    NvU32 m_dest_line;
    NvU32 m_surface_repeat;
    NvU32 m_src_surface;
    NvU32 m_dest_surface;
    NvU32 m_num_transfers;
};

class SplitParams : public EngineParams
{
public:
    SplitParams()
    {
    }
    virtual ~SplitParams()
    {
    }
};

}; // namespace engine_ast

}; // namespace priv

}; // namespace nvdla

namespace nvdla {

class INetwork;
class ILayer;
class ITensor;

namespace priv {

class Layer;
class Network;
class Tensor;
class Wisdom;
class WisdomContainerEntry;
class Profile;
class TargetConfig;
class LutManager;

namespace memory {
class MemoryResolver;
};

namespace engine_ast {
//
// types related to business handled int the edge and node info
// any of these which should be visible in consumers of the graph
// need to stay public.
//
enum EdgeTypeEnum
{
    ENGINE_AST_EDGE_TYPE_ENUMS(GEN_ENUM)
};
typedef SequenceEnum<EdgeTypeEnum, NvU16> EdgeType;

enum EngineTypeEnum
{
    ENGINE_AST_ENGINE_TYPE_ENUMS(GEN_ENUM)
};
typedef SequenceEnum<EngineTypeEnum, NvU16> EngineType;

enum EngineOpTypeEnum
{
    ENGINE_AST_ENGINE_OP_TYPE_ENUMS(GEN_ENUM)
};
typedef SequenceEnum<EngineOpTypeEnum, NvU16> EngineOpType;

enum IODirectionEnum
{
    ENGINE_AST_IO_DIRECTION_ENUMS(GEN_ENUM)
};
typedef SequenceEnum<IODirectionEnum, NvU8> IODirection;

enum OperationEventTypeEnum
{
    ENGINE_AST_OPERATION_EVENT_TYPE_ENUMS(GEN_ENUM)
};
typedef SequenceEnum<OperationEventTypeEnum, NvU8> OperationEventType;

static std::vector<engine_ast::EdgeType> viaData = {{engine_ast::EdgeTypeEnum::DATA}};
static std::vector<engine_ast::EdgeType> viaCompute = {{engine_ast::EdgeTypeEnum::COMPUTE}};
static std::vector<engine_ast::EdgeType> viaHazard = {{engine_ast::EdgeTypeEnum::HAZARD}};
static std::vector<engine_ast::EdgeType> viaComputeData = {{engine_ast::EdgeTypeEnum::COMPUTE},
                                                           {engine_ast::EdgeTypeEnum::DATA}};
static std::vector<engine_ast::EdgeType> viaComputeHazard = {{engine_ast::EdgeTypeEnum::COMPUTE},
                                                             {engine_ast::EdgeTypeEnum::HAZARD}};
static std::vector<engine_ast::EdgeType> allowData = {{engine_ast::EdgeTypeEnum::DATA}};
static std::vector<engine_ast::EdgeType> allowDataCompute = {{engine_ast::EdgeTypeEnum::DATA},
                                                             {engine_ast::EdgeTypeEnum::COMPUTE}};
static std::vector<engine_ast::EdgeType> allowAll;

class ASTToEMUInterface
{
public:
    static NvU16 getDataFormat(EMUInterface*, surface::SurfaceFormat, NvU32 mem_atomic_size);
};

class ASTToDLAInterface
{
public:
    static NvS8 getEngineType(DLAInterface*, EngineType);
    static NvU8 getOperationEventType(DLAInterface*, OperationEventType);
    static NvU8 getDataFormat(DLAInterface*, surface::SurfaceFormat);
    static NvU8 getConvCorePrecision(DLAInterface*, surface::SurfacePrecision);
    static NvU8 getSDPEnable(DLAInterface*, bool enabled);
    static NvU8 getSDPPrecision(DLAInterface*, surface::SurfacePrecision);
    static NvU8 getSDPActType(DLAInterface* dla_if, engine_ast::SDPActType sat);
    static NvU8 getSDPALUType(DLAInterface* dla_if, engine_ast::SDPALUType sat);
    static NvU8 getSDPMode(DLAInterface* dla_if, engine_ast::SDPMode smode);
    static NvU8 getSDPOpType(DLAInterface* dla_if, engine_ast::SDPOpType sat);
    static NvU8 getPDPPrecision(DLAInterface*, surface::SurfacePrecision);
    static NvU8 getCDPPrecision(DLAInterface*, surface::SurfacePrecision);
    static NvU8 getPDPMode(DLAInterface*, nvdla::PoolingType);
    static NvU8 getRubikPrecision(DLAInterface*, surface::SurfacePrecision);
    static NvU8 getRubikMode(DLAInterface*, engine_ast::RubikMode);
};

class MemoryCollector;
class Graph;
class Node;
class Edge;
class ScoredDependencyOrdering;
class DependencyOrdering;

class MemoryCollector
{
public:
    static MemoryCollector* getInstance()
    {
        if (collector == NULL)
        {
            collector = new MemoryCollector();
        }
        return collector;
    }

    MemoryCollector()
    {
    }
    virtual ~MemoryCollector()
    {
    }

    void* allocateMemory(size_t size)
    {
        // Allocate memory and register it internally.
        void* memory = malloc(size);
        if (memory != NULL)
        {
            registerMemory(memory);
        }

        return memory;
    }

    void freeMemory(void* memory)
    {
        // unregister memory and free it.
        if (memory != NULL)
        {
            unregisterMemory(memory);
            free(memory);
        }
    }

    void freeRemainingMemories()
    {
        // Free all remaining unfreed memories.
        for (auto it = m_registered_memories.begin(); it != m_registered_memories.end(); ++it)
        {
            free(*it);
        }
        m_registered_memories.clear();

        delete collector;
        collector = NULL;
    }

private:
    void registerMemory(void* memory)
    {
        m_registered_memories.insert(memory);
    }
    void unregisterMemory(void* memory)
    {
        m_registered_memories.erase(memory);
    }

    // singleton
    static MemoryCollector* collector;
    std::unordered_set<void*> m_registered_memories;
};

class Graph : public ast::Graph<engine_ast::Node, engine_ast::Edge>
{
public:
    Graph(Profile* profile, TargetConfig* target_config);

    virtual ~Graph();
    Graph* clone()
    {
        return new Graph(*this);
    }

    virtual std::string name()
    {
        return std::string("OUTER_GRAPH");
    }
    virtual void setScoredOrdering(ScoredDependencyOrdering* o)
    {
        m_scored_ordering = o;
    }
    virtual void setOrdering(DependencyOrdering* o)
    {
        m_ordering = o;
    }
    virtual DependencyOrdering* ordering()
    {
        return m_ordering;
    }
    virtual ScoredDependencyOrdering* scoredOrdering()
    {
        return m_scored_ordering;
    }

    /* Getters and Setters */
    virtual std::string nextNodeId()
    {
        return std::string("n-") + toString(m_next_node_id++);
    }
    virtual std::string nextEdgeId()
    {
        return std::string("e-") + toString(m_next_edge_id++);
    }

    // downstreamDataEdges := downstreamEdges where type == DATA
    std::vector<Edge*> downstreamDataEdges(Node* node);
    std::vector<Edge*> downstreamComputeEdges(Node* node);
    std::vector<Edge*> downstreamHazardEdges(Node* node);
    std::vector<Node*> downstreamDataNodes(Node* node);
    std::vector<Node*> downstreamComputeNodes(Node* node);
    std::vector<Node*> downstreamHazardNodes(Node* node);

    std::vector<Edge*> upstreamDataEdges(Node* node);
    std::vector<Edge*> upstreamHazardEdges(Node* node);

    Edge* connectingDataEdge(Node* fromNode, Node* toNode, ast::EdgeSide fromDir);

    std::vector<Edge*> upstreamAuxEdges(Node* node);
    Edge* getUpstreamAuxEdge(Node* node, NvU8 id = 0);

    std::vector<Node*> upstreamDataNodes(Node* node);

    std::vector<Edge*> siblingDataEdges(Edge* edge);

    // the profile using which this graph and its constituents are compiled
    virtual Profile* profile() const
    {
        return m_profile;
    }
    virtual TargetConfig* target_config() const
    {
        return m_targetconfig;
    }
    virtual memory::DLAResourceManager* resourceMgr()
    {
        return &m_resource_mgr;
    }
    virtual LutManager* lutManager()
    {
        return m_lutManager;
    }

    NvDlaError initGraphResources();

    // the caffe parser doesnt encapsulate the auxiliary inputs like
    // kernel and bias weights into tensors. but we need them in the graph
    std::string newAuxTensorName();
    Tensor* addAuxTensor(const std::string& s, const Dims4 dims, TensorType tt);

    surface::TensorSurfaceDesc* nodeInputTensorSurface(const Node*, size_t i, const std::vector<surface::SurfaceCategory>&);
    surface::TensorSurfaceDesc* nodeOutputTensorSurface(const Node*, size_t i, const std::vector<surface::SurfaceCategory>&);

    inline bool debugGraphDump() const
    {
        return true;
    }
    inline bool debugClone() const
    {
        return true;
    }
    inline bool debugOps() const
    {
        return true;
    }
    inline bool debugGroupOps() const
    {
        return true;
    }
    inline bool debugMathOptz() const
    {
        return true;
    }
    inline bool debugWeights() const
    {
        return true;
    }
    inline bool debugQuantization() const
    {
        return true;
    }
    inline bool debugFuseSubEngineOps() const
    {
        return true;
    }
    inline bool debugSurfaces() const
    {
        return true;
    }
    inline bool debugBuffers() const
    {
        return true;
    }
    inline bool debugCopyOutDebug() const
    {
        return true;
    }
    inline bool debugMemoryLayout() const
    {
        return true;
    }
    inline bool debugBinding() const
    {
        return true;
    }
    inline bool debugDepGraph() const
    {
        return true;
    }
    inline bool debugMemHazards() const
    {
        return true;
    }
    inline bool debugRelocs() const
    {
        return true;
    }

    class Graphlet
    {
    public:
        Graphlet()
            : m_node_list(NodeSequence())
        {
            m_engine_op_heads = NodeSequence(EngineType::num_elements(), NULL);
        }
        virtual ~Graphlet()
        {
        }
        virtual Graphlet* clone()
        {
            return new Graphlet(*this);
        }
        NodeSequence& nodeList()
        {
            return m_node_list;
        }
        void setNodeList(const NodeSequence& nl)
        {
            m_node_list = nl;
        }
        NodeSequence& opHeads()
        {
            return m_engine_op_heads;
        }

    protected:
        NodeSequence m_node_list;
        NodeSequence m_engine_op_heads; // FIXME: clone this suitably
    };

    inline std::vector<Graphlet*>& graphlets()
    {
        return m_graphlets;
    }

    ScoredDependencyOrdering* m_scored_ordering;
    DependencyOrdering* m_ordering;

    virtual void checkDirty();

    virtual const NodeSequence& orderedNodes();
    virtual const EdgeSequence& orderedEdges();
    virtual const ElemSequence& orderedElems();

    const EdgeSequence& orderedComputeEdges();
    const EdgeSequence& orderedDataEdges();

    NvDlaError determineNwSurfaceFormat(TensorType);
    surface::SurfaceFormat suggestNwSurfaceFormat(TensorType);

    /* Compiler operations on graph */
    NvDlaError registerAllSurfaces();
    NvDlaError registerAllBuffers();
    NvDlaError reserveAllBuffers();
    NvDlaError preProcessAuxData();
    NvDlaError mergeUnitScaleOperations();
    NvDlaError mergeActivationOperations();
    NvDlaError updateScalingFactors();
    NvDlaError quantizeAuxData();
    NvDlaError handleLowPrecisionConversions();
    NvDlaError translateAuxData();
    NvDlaError fuseOnTheFlyNodes();
    NvDlaError groupAtomicOperations();
    NvDlaError splitNodes();
    NvDlaError handleMultiBatch();
    NvDlaError fuseSDPSubEngineOps();
    NvDlaError flattenGraph();
    NvDlaError topologicalSort(NodeSequence& topological_order);
    NvDlaError resolveDataDependencies(const NodeSequence& allNodes);
    NvDlaError resolveComputeDependencies(const NodeSequence& allNodes);
    NvDlaError resolveSoftwareDependencies();
    NvDlaError resolveMultiBatchDependencies();
    NvDlaError regroupAtomicOperations();
    NvDlaError determineTaskBoundaries(const NodeSequence& allNodes);
    NvDlaError annotateNodes(NvS16& lastUsedAnnId);
    NvDlaError resolveMemory(const NodeSequence& topological_order);

    /* verification/sanity operations on graph */
    NvDlaError verifyAllSurfaces();
    NvDlaError verifyAllBuffers();
    NvDlaError verifyDependencyGraph();

    /* refresh graph state */
    NvDlaError refreshGraphState();

    /* Code emission operations on graph */
    NvDlaError prepareMemoryListEntries(Loadable* l);
    NvDlaError createTensorDescListEntry(surface::TensorSurfaceDesc*, ILoadable::TensorDescListEntry&, NvU32 memAtomsize);

    /* graph utils */
    bool connectedComputeNodes(Node* upStream, Node* downStream);
    bool connectedDataNodes(Node* upStream, Node* downStream);
    void replaceEdgeNodes(Edge* edge, ast::EdgeSide dir, Node* oldNode, Node* newNode);
    void replaceNodeEdges(Node* node, ast::EdgeSide dir, Edge* oldEdge, Edge* newEdge);
    virtual bool connectNodesWithEdge(Edge* newEdge, Node* fromNode, Node* toNode);
    NvDlaError removeNodeFromAST(Node* killNode, IODirection iod);
    NvDlaError substituteNodeInAST(Node* origNode, NodeSequence subNodes);
    NvDlaError substituteEdgeInAST(Edge* origEdge, Edge* subEdge);
    Edge* addComputeEdge(Node* fromNode, Node* toNode);
    Edge* addDataEdge(canonical_ast::Edge* canEdge, Node* fromNode, Node* toNode, Tensor* tensor);
    Edge* addDataEdge(engine_ast::Edge* cloneEdge, Node* fromNode, Node* toNode, Tensor* tensor);
    Edge* addHazardEdge(Node* fromNode, Node* toNode);

    void resetRelocEntries();
    void insertRelocEntry(ILoadable::RelocEntry);
    NvDlaError gatherRelocEntries(NvS16 opsId, NvU8* opsBase,
                                  NvS16 surfsId, NvU8* surfsBase,
                                  NvS16 depsId, NvU8* depsBase);
    std::vector<ILoadable::RelocEntry>& getRelocEntries()
    {
        return m_relocEntries;
    }

    static void printGraph(engine_ast::Graph* g, bool nested, std::string graphName = "");

protected:
    Graph(const Graph& other);
    surface::TensorSurfaceDesc* nodeTensorSurface(const Node*, size_t i,
                                                  const std::vector<surface::SurfaceCategory>& types,
                                                  ast::EdgeSideEnum dir);
    int m_next_node_id;
    int m_next_edge_id;
    EdgeSequence m_graph_input_edges;
    EdgeSequence m_graph_output_edges;
    memory::DLAResourceManager m_resource_mgr;

    std::vector<Graphlet*> m_graphlets;
    std::vector<Tensor*> m_stream_tensors;
    std::vector<Tensor*> m_aux_tensors;
    std::vector<Tensor*> m_io_tensors;
    std::vector<Tensor*> m_debug_tensors;
    std::map<memory::TensorBufferDesc*, size_t> m_tbd_memory_list_id;
    Profile* m_profile;
    TargetConfig* m_targetconfig;

    // helper class used to implement resolveMemory()
    memory::MemoryResolver* m_memoryResolver;

    LutManager* m_lutManager;

    std::vector<ILoadable::RelocEntry> m_relocEntries;
};

typedef Graph::NodeSequence NodeSequence;

class DependencyParams
{
public:
    typedef engine_ast::Graph::NodeSequence NodeSequence;

    class BindNode
    {
    public:
        BindNode()
            : m_bind_node(NULL),
              m_bind_node_ann_id(-1),
              m_op_event(OperationEventTypeEnum::OP_COMPLETED)
        {
        }
        virtual ~BindNode()
        {
        }

        OperationEventType opEvent()
        {
            return m_op_event;
        }
        void setOpEvent(const OperationEventType opEvent)
        {
            m_op_event = opEvent;
        }

        Node* node()
        {
            return m_bind_node;
        }
        void setNode(Node* node)
        {
            m_bind_node = node;
        }

        NvS16 nodeAnnId() const
        {
            return m_bind_node_ann_id;
        }
        void setNodeAnnId(NvS16 nodeAnnId)
        {
            m_bind_node_ann_id = nodeAnnId;
        }

    protected:
        Node* m_bind_node;
        NvS16 m_bind_node_ann_id;
        OperationEventType m_op_event;
    };

    DependencyParams()
        : m_annotation_id(-1), m_fused_nodes(NodeSequence())
    {
        // init null initialized fused node array of fixed size
        m_fused_nodes = NodeSequence(IODirection::num_elements(), NULL);

        // init null initialized producer and consumer arrays of fixed sizes
        m_consumers = std::vector<BindNode>(EngineType::num_elements(), BindNode());
        m_producers = std::vector<BindNode>(EngineType::num_elements(), BindNode());
    }
    virtual ~DependencyParams()
    {
    }
    virtual DependencyParams* clone()
    {
        return new DependencyParams(*this);
    }

    BindNode& consumer(NvU8 index)
    {
        return m_consumers[index];
    }
    std::vector<BindNode>& consumers()
    {
        return m_consumers;
    }

    BindNode& producer(NvU8 index)
    {
        return m_producers[index];
    }
    std::vector<BindNode>& producers()
    {
        return m_producers;
    }

    Node* fusedNode(IODirection dir) const
    {
        return m_fused_nodes[dir.v()];
    }
    void setFusedNode(IODirection dir, Node* node)
    {
        m_fused_nodes[dir.v()] = node;
    }

    void setAnnotationId(NvS16 id)
    {
        m_annotation_id = id;
    }
    NvS16 annotationId() const
    {
        return m_annotation_id;
    }

    NvU16 getDependencyCount();

    void clear()
    {
        for (size_t ii = 0; ii < EngineType::num_elements(); ++ii)
        {
            m_producers[ii].setNode(NULL);
            m_producers[ii].setOpEvent(OperationEventTypeEnum::OP_COMPLETED);
            m_consumers[ii].setNode(NULL);
            m_consumers[ii].setOpEvent(OperationEventTypeEnum::OP_COMPLETED);
        }
        setAnnotationId(-1);
        setFusedNode(IODirectionEnum::INPUT, NULL);
        setFusedNode(IODirectionEnum::OUTPUT, NULL);
    }

    /* clone */
    DependencyParams(const DependencyParams& other)
        : m_annotation_id(other.m_annotation_id),
          m_fused_nodes(other.m_fused_nodes)
    {
    }

protected:
    NvS16 m_annotation_id;
    NodeSequence m_fused_nodes; // [IODirection::num_elements()];
    std::vector<BindNode> m_consumers;
    std::vector<BindNode> m_producers;
};

class SDPNode; //fwd decl

class Node
{
public:
    virtual std::string className() const
    {
        return std::string("Node");
    }
    virtual std::string pretty() const
    {
        return "<pretty-node>";
    } // override in specific nodes to add. careful...

    static std::string prettyId(const Node* n, NvU8 flags = ast::PrettyId_Default)
    {
        if (!n) { return std::string("(null)"); }
        std::stringstream r;
        std::string sep("");
        if (flags & ast::PrettyId_Id)
        {
            r << sep << n->id();
            sep = " ";
        }
        if (flags & ast::PrettyId_ClassName)
        {
            r << sep << n->className();
            sep = " ";
        }
        if (flags & ast::PrettyId_Name)
        {
            r << sep << n->name();
            sep = " ";
        }
        if (flags & (ast::PrettyId_Verbose))
        {
            r << sep << n->pretty();
            sep = " ";
        }
        return r.str();
    }

    typedef engine_ast::Graph::NodeSequence NodeSequence;
    typedef engine_ast::Graph::EdgeSequence EdgeSequence;
    typedef engine_ast::Graph::NodeSequenceIterator NodeSequenceIterator;
    typedef engine_ast::Graph::EdgeSequenceIterator EdgeSequenceIterator;

    Node(NvU16 numBatches = 1)
        : m_containing_graph(0), m_taskId(-1)
    {
        m_unique_id = m_next_id++;
        m_mb_dependency_params = new MultiBatchState<DependencyParams>(numBatches);
        m_sup_in_surf_formats = std::vector<surface::SurfaceFormat>();
        m_sup_out_surf_formats = std::vector<surface::SurfaceFormat>();
        m_sup_aux_surf_formats = std::vector<surface::SurfaceFormat>();
    }
    virtual ~Node()
    {
    }
    virtual Node* clone()
    {
        return new Node(*this);
    }

    const std::string id() const
    {
        return m_id;
    }
    void setId(const std::string id)
    {
        m_id = id;
    }

    NvU32 uniqueId() const
    {
        return m_unique_id;
    }

    const std::string name() const
    {
        return m_name;
    }
    void setName(const std::string name)
    {
        m_name = name;
    }

    inline Graph* graph() const
    {
        return m_containing_graph;
    }
    void setGraph(Graph* g)
    {
        m_containing_graph = g;
    }

    EngineType engineType() const
    {
        return m_engine_type;
    }
    void setEngineType(EngineType et)
    {
        m_engine_type = et;
    }

    EngineOpType engineOpType() const
    {
        return m_engine_op_type;
    }
    void setEngineOpType(EngineOpType eot)
    {
        m_engine_op_type = eot;
    }

    bool isEMUEngineType() const
    {
        return m_engine_type.e() == CPU;
    }
    bool isDLAEngineType() const
    {
        return !(m_engine_type.e() == CPU || m_engine_type.e() == SPLIT || m_engine_type.e() == CONCATENATION);
    }
    bool isSoftwareNode() const
    {
        return (m_engine_type.e() == SPLIT || m_engine_type.e() == CONCATENATION);
    }
    virtual bool isEngineType(EngineType et)
    {
        return m_engine_type == et;
    }

    virtual EngineParams& params(NvU16)
    {
        return EngineParamsNULL;
    }
    virtual void inheritParams(Node*)
    {
        return;
    }
    virtual const void* getAuxData(Edge* auxEdge)
    {
        return NULL;
    }
    virtual Dims4 getAuxDims()
    {
        return Dims4(-1, -1, -1, -1);
    }

    virtual canonical_ast::Node* canonicalNode() const
    {
        return NULL;
    }

    inline const std::vector<surface::SurfaceFormat>& supportedInSurfFormats() const
    {
        return m_sup_in_surf_formats;
    }
    inline const std::vector<surface::SurfaceFormat>& supportedOutSurfFormats() const
    {
        return m_sup_out_surf_formats;
    }
    inline const std::vector<surface::SurfaceFormat>& supportedAuxSurfFormats() const
    {
        return m_sup_aux_surf_formats;
    }

    const std::vector<surface::SurfaceCategory> supportedInSurfCategories() const;
    const std::vector<surface::SurfaceCategory> supportedOutSurfCategories() const;
    const std::vector<surface::SurfaceCategory> supportedAuxSurfCategories() const;

    bool dependsOn(Node* on, const std::vector<EdgeType>& requireVia, const std::vector<EdgeType>& allowVia) const;

    std::vector<surface::TensorSurfaceDesc*> auxSurfaces() const;
    std::vector<surface::TensorSurfaceDesc*> inputSurfaces() const;
    std::vector<surface::TensorSurfaceDesc*> outputSurfaces() const;

    void emitDependencyParams(DLAInterface* target_dla, DLACommonOpDescAccessor dep, NvU32 batchId);
    void setDataCubeAccessor(DLADataCubeAccessor acc, surface::TensorSurfaceDesc* tsd, IODirection iod, NvU32 batchId);

    virtual std::vector<surface::SurfaceFormat> suggestAuxSurfaceFormats(Edge* auxEdge);
    std::vector<surface::SurfaceFormat> suggestOutputSurfaceFormats();
    std::vector<surface::SurfaceFormat> suggestInputSurfaceFormats();
    NvDlaError supportsSurfaceFormat(surface::SurfaceFormat, std::vector<surface::SurfaceFormat>);

    virtual Dims4 suggestSurfaceDims(surface::TensorSurfaceDesc* tsd);
    virtual NvU32 suggestLineStride(surface::TensorSurfaceDesc* tsd);
    virtual NvU32 suggestSurfaceStride(surface::TensorSurfaceDesc* tsd);
    virtual NvU64 suggestSurfaceSize(surface::TensorSurfaceDesc* tsd);
    virtual NvU64 suggestSurfaceOffsetInBuffer(surface::TensorSurfaceDesc* tsd);
    virtual memory::TensorBufferDesc* suggestBuffer(surface::TensorSurfaceDesc* tsd);

    virtual NvDlaError preProcessAuxData()
    {
        return NvDlaSuccess;
    }
    virtual Node* mergeWithSDPOp(SDPNode* other_op)
    {
        return NULL;
    }
    virtual NvDlaError updateScalingFactors()
    {
        return NvDlaSuccess;
    }
    virtual NvDlaError quantizeAuxData()
    {
        return NvDlaSuccess;
    }
    virtual NvDlaError fuseOnTheFlyNodes()
    {
        return NvDlaSuccess;
    }
    virtual NvDlaError handleLowPrecisionConversions()
    {
        return NvDlaSuccess;
    }
    virtual NvDlaError translateAuxData()
    {
        return NvDlaSuccess;
    }
    virtual NvDlaError splitNodes()
    {
        return NvDlaSuccess;
    }
    virtual NvDlaError handleMultiBatch()
    {
        return NvDlaSuccess;
    }
    virtual NvDlaError resolveDataDependencies(Node* next);
    virtual NvDlaError resolveComputeDependencies(const NodeSequence& ordered_nodes);
    NvDlaError resolveSoftwareDependencies();
    virtual NvDlaError resolveMultiBatchDependencies();
    virtual NvDlaError selfAnnotate(NvS16& lastUsedAnnId);

    virtual NvDlaError verifyEdgePorts()
    {
        return NvDlaSuccess;
    }
    virtual NvDlaError verifySurfaceDims(surface::TensorSurfaceDesc*)
    {
        return NvDlaSuccess;
    }
    NvDlaError verifySurfaces();
    NvDlaError verifyDependencyParams();

    const EdgeSequence& inputEdges() const
    {
        return m_input_edges;
    }
    void markInputEdge(Edge* input)
    {
        if (std::find(m_input_edges.begin(), m_input_edges.end(), input) == m_input_edges.end())
            m_input_edges.push_back(input);
    }

    const EdgeSequence& outputEdges() const
    {
        return m_output_edges;
    }
    void markOutputEdge(Edge* output)
    {
        if (std::find(m_output_edges.begin(), m_output_edges.end(), output) == m_output_edges.end())
            m_output_edges.push_back(output);
    }

    const EdgeSequence& auxEdges() const
    {
        return m_aux_edges;
    }
    void markAuxEdge(Edge* aux)
    {
        if (std::find(m_aux_edges.begin(), m_aux_edges.end(), aux) == m_aux_edges.end())
            m_aux_edges.push_back(aux);
    }

    // cache reference to input/aux/output edges of each node into the respective edge ports
    virtual NvDlaError populateEdgePorts();
    void unpopulateEdgePorts()
    {
        m_input_edges.clear();
        m_aux_edges.clear();
        m_output_edges.clear();
    }
    NvDlaError repopulateEdgePorts()
    {
        unpopulateEdgePorts();
        return populateEdgePorts();
    }

    // get the data edge which matches any of the supplied tensor surface categories
    NvDlaError nodeDataEdge(const std::vector<surface::SurfaceCategory>& types, ast::EdgeSideEnum dir, engine_ast::Edge** retEdge);
    // get the data edge which matches the supplied raw tensor type (used when SD is not yet defined for the edge)
    NvDlaError nodeDataEdge(TensorType raw_tt, ast::EdgeSideEnum dir, engine_ast::Edge** retEdge);

    // get the aux edge associated with the node
    virtual NvDlaError nodeAuxEdge(engine_ast::Edge** ret_edge)
    {
        *ret_edge = NULL;
        return NvDlaSuccess;
    }

    virtual void captureCanonicalParams()
    {
        return;
    }

    DependencyParams& dependencyParams()
    {
        return m_mb_dependency_params->batch(0);
    }
    DependencyParams& dependencyParams(NvU16 batchId)
    {
        return m_mb_dependency_params->batch(batchId);
    }

    /* Code Emission and DLA interface APIs */
    void setTaskId(NvS16 id)
    {
        m_taskId = id;
    }
    NvS16 taskId() const
    {
        return m_taskId;
    }

    inline bool debugWinograd() const
    {
        return true;
    }
    inline bool debugSplits() const
    {
        return true;
    }
    inline bool debugFusion() const
    {
        return true;
    }
    inline bool debugResolveDependencies() const
    {
        return true;
    }

    virtual NvDlaError emitOp(Graph*, DLAInterface*, NvU32 op_slot, NvU32 batch_id,
                              DLACommonOpDescAccessor,
                              DLAOperationContainerAccessor,
                              DLASurfaceContainerAccessor);

#ifdef NVDLA_COMPILER_OUTPUT_FOR_PROTOTEST
    virtual NvDlaError emitOp(NvU32 /*op_slot*/, NvU32 /*batch_id*/,
                              DLAInterface*,
                              DLACommonOpDescAccessor&,
                              DLAOperationContainerAccessor&,
                              DLASurfaceContainerAccessor&,
                              nvdla_prototest_interface::Layer*)
    {
        return NvDlaSuccess;
    }
#endif

    virtual NvDlaError emitOp(Graph*, EMUInterface*, NvU32 op_slot, NvU32 batch_id,
                              EMUOperationContainerAccessor,
                              EMUOperationBufferContainerAccessor);

    NvDlaError clearNodeTSDStateMapping()
    {
        m_nodeTSDSurfaceOffsetInBuffer.clear();
        m_nodeTSDLineStride.clear();
        m_nodeTSDSurfaceStride.clear();
        m_nodeTSDSurfaceSize.clear();

        return NvDlaSuccess;
    }

protected:
    /* for clone */
    Node(const Node& other)
        : m_id(other.m_id),
          m_unique_id(m_next_id++),
          m_name(other.m_name),
          m_containing_graph(0),
          m_engine_type(other.m_engine_type),
          m_engine_op_type(other.m_engine_op_type),
          m_aux_edges(other.m_aux_edges),
          m_input_edges(other.m_input_edges),
          m_output_edges(other.m_output_edges),
          m_mb_dependency_params(other.m_mb_dependency_params),
          m_sup_in_surf_formats(other.m_sup_in_surf_formats),
          m_sup_aux_surf_formats(other.m_sup_aux_surf_formats),
          m_sup_out_surf_formats(other.m_sup_out_surf_formats),
          m_taskId(-1),
          m_nodeTSDSurfaceOffsetInBuffer(),
          m_nodeTSDLineStride(),
          m_nodeTSDSurfaceStride(),
          m_nodeTSDSurfaceSize()
    {
        // of all the attributes, you don't want the cloned node to inherit
        // the containing_graph from orig node. The cloned node should belong
        // to the new graph under construction. (see Graph(const Graph& other))
    }

    friend class Graph;
    std::string m_id;  // unique within the graph
    NvU32 m_unique_id; // id for graph ordering. u32 instead of string.
    static NvU32 m_next_id;
    std::string m_name;
    Graph* m_containing_graph;
    EngineType m_engine_type;
    EngineOpType m_engine_op_type;
    EdgeSequence m_aux_edges;    // definitive aux edge ports
    EdgeSequence m_input_edges;  // definitive input edge ports
    EdgeSequence m_output_edges; // definitive output edge ports
    MultiBatchState<DependencyParams>* m_mb_dependency_params;
    std::vector<surface::SurfaceFormat> m_sup_in_surf_formats;
    std::vector<surface::SurfaceFormat> m_sup_aux_surf_formats;
    std::vector<surface::SurfaceFormat> m_sup_out_surf_formats;
    NvS16 m_taskId;
    std::map<surface::TensorSurfaceDesc*, NvU64> m_nodeTSDSurfaceOffsetInBuffer;
    std::map<surface::TensorSurfaceDesc*, NvU32> m_nodeTSDLineStride;
    std::map<surface::TensorSurfaceDesc*, NvU32> m_nodeTSDSurfaceStride;
    std::map<surface::TensorSurfaceDesc*, NvU64> m_nodeTSDSurfaceSize;
};

class NestedGraph : public Graph
{
public:
    NestedGraph()
        : Graph(0, 0)
    {
        m_ng_scored_ordering = 0;
        m_ng_ordering = 0;
        m_containing_super_node = 0;
    }
    virtual ~NestedGraph()
    {
    }
    NestedGraph* clone()
    {
        return new NestedGraph(*this);
    }

    virtual std::string name()
    {
        return std::string("INNER_GRAPH");
    }

    virtual bool insertEdge(Edge*);
    virtual bool insertNode(Node*);
    virtual bool removeEdge(Edge*);
    virtual bool removeNode(Node*);
    virtual bool removeEdgeFromNode(Edge*, ast::EdgeSide, Node*);
    virtual bool removeNodeFromEdge(Edge*, ast::EdgeSide, Node*);
    virtual bool appendNodeToEdge(Edge*, ast::EdgeSide, Node*);

    bool connectNodesWithEdge(Edge* newEdge, Node* fromNode, Node* toNode);

    virtual void setScoredOrdering(ScoredDependencyOrdering* o)
    {
        m_ng_scored_ordering = o;
    }
    virtual void setOrdering(DependencyOrdering* o)
    {
        m_ng_ordering = o;
    }
    virtual DependencyOrdering* ordering()
    {
        return m_ng_ordering;
    }
    virtual ScoredDependencyOrdering* scoredOrdering()
    {
        return m_ng_scored_ordering;
    }

    virtual std::string nextNodeId()
    {
        return std::string("ng-n-") + toString(m_next_node_id++);
    }
    virtual std::string nextEdgeId()
    {
        return std::string("ng-e-") + toString(m_next_edge_id++);
    }

    virtual void checkDirty();
    virtual const NodeSequence& orderedNodes();
    virtual const EdgeSequence& orderedEdges();
    virtual const ElemSequence& orderedElems();

    virtual Profile* profile() const
    {
        return m_containing_super_node->graph()->profile();
    }
    virtual TargetConfig* target_config() const
    {
        return m_containing_super_node->graph()->target_config();
    }
    virtual memory::DLAResourceManager* resourceMgr()
    {
        return m_containing_super_node->graph()->resourceMgr();
    }
    virtual LutManager* lutManager()
    {
        return m_containing_super_node->graph()->lutManager();
    }

    NvDlaError populateNestedGraph(NodeSequence& groupedOps);

    inline Node* containingSuperNode() const
    {
        return m_containing_super_node;
    }
    void setContainingSuperNode(Node* sn)
    {
        m_containing_super_node = sn;
    }

    NodeSequence topNodes();
    NodeSequence bottomNodes();

    inline bool debugNestedGraph()
    {
        return true;
    }

protected:
    ScoredDependencyOrdering* m_ng_scored_ordering;
    DependencyOrdering* m_ng_ordering;
    Node* m_containing_super_node;
};

// collection of engine op nodes that can be executed atomically as a group
class MultiOpsNode : public Node
{
public:
    virtual std::string className() const
    {
        return std::string("MultiOpsNode");
    }

    MultiOpsNode(NvU16 numBatches = 1)
        : Node(numBatches)
    {
        m_engine_type = MULTI_OPS;
        m_nested_graph = 0;
        m_is_online = false;
        m_sup_in_surf_formats.assign(IMG_FORMATS, IMG_FORMATS + (sizeof(IMG_FORMATS) / sizeof(IMG_FORMATS[0])));
        m_sup_in_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_FEATURE_DATA_INT8);
        m_sup_in_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_FEATURE_DATA_FP16);
        m_sup_out_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_FEATURE_DATA_INT8);
        m_sup_out_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_FEATURE_DATA_FP16);
    }
    virtual ~MultiOpsNode()
    {
    }

    /*
     * Return true if the super node or any of its inner nodes have the same engine type
     * as the one provided
     */
    virtual bool isEngineType(EngineType et);

    virtual NvDlaError populateEdgePorts();
    virtual NvDlaError repopulateEdgePorts();

    virtual NvDlaError splitNodes();
    virtual NvDlaError handleMultiBatch();
    virtual NvDlaError selfAnnotate(NvS16& lastUsedAnnId);
    virtual NvDlaError resolveMultiBatchDependencies();

    inline NestedGraph* nestedGraph()
    {
        return m_nested_graph;
    }
    void setNestedGraph(NestedGraph* ng)
    {
        m_nested_graph = ng;
    }

    bool isOnline() const
    {
        return m_is_online;
    }
    void setIsOnline(bool isOnline)
    {
        m_is_online = isOnline;
    }

    /*
     * Some compiler operations like to see flattened graph (resolving dependencies, etc)
     * these api's allow to replace the multi-op node with its contents in place
     * and vice-versa
     */
    NvDlaError plugNestedGraph();
    NvDlaError unplugNestedGraph();

    Edge* outerGraphIsomorphEdgeOf(Edge*);

    const std::unordered_map<Edge*, Edge*>& isomorphicEdgeMap()
    {
        return m_isomorphic_edge_map;
    }
    void setIsomorphicEdgeMap(const std::unordered_map<Edge*, Edge*>& iemap)
    {
        m_isomorphic_edge_map = iemap;
    }

    virtual NvDlaError verifyEdgePorts();

protected:
    /* state of the multi-ops node as connected/disconnected from
     * outer graph
     */
    bool m_is_online;
    NestedGraph* m_nested_graph;

    /*
     * The super node should maintain a cache of isomorphic sets of edges:
     * 1 set represents the input-output edges to the super node when it is a part of the outer graph
     * and the 2nd set represents the graph input-output edges of the nested graph within
     */
    std::unordered_map<Edge*, Edge*> m_isomorphic_edge_map;
};

// scale (when power non-trivial)
// softmax
class CPUNode : public Node
{
public:
    virtual std::string className() const
    {
        return std::string("CPUNode");
    }

    CPUNode(NvU16 numBatches)
        : Node(numBatches)
    {
        m_engine_type = CPU;
    }
    virtual ~CPUNode()
    {
    }

    virtual CPUNode* clone()
    {
        return new CPUNode(*this);
    }
    virtual canonical_ast::Node* canonicalNode() const
    {
        return NULL;
    }
    virtual void captureCanonicalParams()
    {
        return;
    }
};

// scale op in CPU engine :- y = (scale_factor * x)
class CPUScaleOpNode : public CPUNode
{
public:
    virtual std::string className() const
    {
        return std::string("CPUScaleOpNode");
    }

    class OpParams : public CPUParams
    {
    public:
        OpParams()
        {
        }
        virtual ~OpParams()
        {
        }

        const Weights& power() const
        {
            return m_power;
        }
        void setPower(const Weights& power)
        {
            m_power = power;
        }

        const Weights& scale() const
        {
            return m_scale;
        }
        void setScale(const Weights& scale)
        {
            m_scale = scale;
        }

        const Weights& shift() const
        {
            return m_shift;
        }
        void setShift(const Weights& shift)
        {
            m_shift = shift;
        }

    protected:
        // add everything relevant to CPU based scale op
        Weights m_scale;
        Weights m_power;
        Weights m_shift;
    };

    CPUScaleOpNode(canonical_ast::ScaleNode* can_scale_node, NvU16 numBatches = 1)
        : CPUNode(numBatches)
    {
        m_engine_type = CPU;
        m_engine_op_type = CPU_SCALE;
        m_can_scale_node = can_scale_node;
        m_mb_op_params = new MultiBatchState<OpParams>(numBatches);
        m_sup_in_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_FEATURE_DATA_INT8);
        m_sup_in_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_FEATURE_DATA_FP16);
        m_sup_out_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_FEATURE_DATA_INT8);
        m_sup_out_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_FEATURE_DATA_FP16);
    }
    virtual ~CPUScaleOpNode()
    {
    }

    OpParams& params(NvU16 batchId = 0)
    {
        return m_mb_op_params->batch(batchId);
    }
    virtual void captureCanonicalParams();
    virtual CPUScaleOpNode* clone()
    {
        return new CPUScaleOpNode(*this);
    }
    virtual canonical_ast::ScaleNode* canonicalNode() const
    {
        return m_can_scale_node;
    }
    virtual NvDlaError emitOp(Graph*, EMUInterface*, NvU32 op_slot, NvU32 batch_id,
                              EMUOperationContainerAccessor,
                              EMUOperationBufferContainerAccessor);

protected:
    canonical_ast::ScaleNode* m_can_scale_node;
    MultiBatchState<OpParams>* m_mb_op_params;
};

// softmax op in CPU node
class CPUSoftMaxOpNode : public CPUNode
{
public:
    virtual std::string className() const
    {
        return std::string("CPUSoftMaxOpNode");
    }

    class OpParams : public CPUParams
    {
    public:
        OpParams()
        {
        }
        virtual ~OpParams()
        {
        }

    protected:
        // add everything relevant to CPU based softmax op
    };

    CPUSoftMaxOpNode(canonical_ast::SoftMaxNode* can_sm_node, NvU16 numBatches = 1)
        : CPUNode(numBatches)
    {
        m_engine_type = CPU;
        m_engine_op_type = CPU_SOFTMAX;
        m_can_sm_node = can_sm_node;
        m_mb_op_params = new MultiBatchState<OpParams>(numBatches);
        m_sup_in_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_FEATURE_DATA_INT8);
        m_sup_in_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_FEATURE_DATA_FP16);
        m_sup_out_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_FEATURE_DATA_INT8);
        m_sup_out_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_FEATURE_DATA_FP16);
    }
    virtual ~CPUSoftMaxOpNode()
    {
    }

    virtual void captureCanonicalParams();
    virtual CPUSoftMaxOpNode* clone()
    {
        return new CPUSoftMaxOpNode(*this);
    }
    virtual canonical_ast::SoftMaxNode* canonicalNode() const
    {
        return m_can_sm_node;
    }
    virtual NvDlaError emitOp(Graph*, EMUInterface*, NvU32 op_slot, NvU32 batch_id,
                              EMUOperationContainerAccessor,
                              EMUOperationBufferContainerAccessor);

protected:
    canonical_ast::SoftMaxNode* m_can_sm_node;
    MultiBatchState<OpParams>* m_mb_op_params;
};

// concat
class ConcatenationNode : public Node
{
public:
    virtual std::string className() const
    {
        return std::string("ConcatenationNode");
    }

    class OpParams : public EngineParams
    {
    public:
        OpParams()
            : m_concat_axis(ConcatAxisEnum::CONCAT_AXIS_UNKNOWN)
        {
        }
        virtual ~OpParams()
        {
        }

        ConcatAxis concatAxis() const
        {
            return m_concat_axis;
        }
        void setConcatAxis(ConcatAxis axis)
        {
            m_concat_axis = axis;
        }

    protected:
        ConcatAxis m_concat_axis;
    };

    ConcatenationNode(canonical_ast::ConcatenationNode* concat, NvU16 numBatches = 1)
        : Node(numBatches)
    {
        m_engine_type = CONCATENATION;
        m_engine_op_type = CONCATENATION_CONCAT;
        m_can_concat_node = concat;
        m_mb_op_params = new MultiBatchState<OpParams>(numBatches);
        m_sup_in_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_FEATURE_DATA_INT8);
        m_sup_in_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_FEATURE_DATA_FP16);
        m_sup_out_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_FEATURE_DATA_INT8);
        m_sup_out_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_FEATURE_DATA_FP16);
    }
    virtual ~ConcatenationNode()
    {
    }

    inline bool debugConcat()
    {
        return true;
    }
    virtual void captureCanonicalParams();
    OpParams& params(NvU16 batchId = 0)
    {
        return m_mb_op_params->batch(batchId);
    }

    NvDlaError verifySurfaceIsPartOfConcat(surface::TensorSurfaceDesc* src, surface::TensorSurfaceDesc* dst, ConcatAxis axis);
    NvU64 suggestOffsetInConcatChain(surface::TensorSurfaceDesc* inTSD);
    virtual Dims4 suggestSurfaceDims(surface::TensorSurfaceDesc* tsd);
    virtual NvU32 suggestLineStride(surface::TensorSurfaceDesc* tsd);
    virtual NvU32 suggestSurfaceStride(surface::TensorSurfaceDesc* tsd);
    virtual NvU64 suggestSurfaceSize(surface::TensorSurfaceDesc* tsd);
    virtual NvU64 suggestSurfaceOffsetInBuffer(surface::TensorSurfaceDesc* tsd);
    virtual memory::TensorBufferDesc* suggestBuffer(surface::TensorSurfaceDesc* tsd);

    // cache reference to input/output edges in their order of appearance
    virtual NvDlaError populateEdgePorts();
    virtual NvDlaError resolveMultiBatchDependencies()
    {
        return NvDlaSuccess;
    }
    virtual ConcatenationNode* clone()
    {
        return new ConcatenationNode(*this);
    }
    virtual canonical_ast::ConcatenationNode* canonicalNode() const
    {
        return m_can_concat_node;
    }

    virtual NvDlaError verifyEdgePorts()
    {
        if (outputEdges().size() != 1)
            return NvDlaError_BadValue;
        else
            return NvDlaSuccess;
    }
    virtual NvDlaError verifySurfaceDims(surface::TensorSurfaceDesc*);

protected:
    canonical_ast::ConcatenationNode* m_can_concat_node;
    MultiBatchState<OpParams>* m_mb_op_params;
};

// sw split
class SplitNode : public Node
{
public:
    virtual std::string className() const
    {
        return std::string("SplitNode");
    }

    class OpParams : public EngineParams
    {
    public:
        OpParams()
            : m_split_axis(SplitAxisEnum::SPLIT_AXIS_UNKNOWN)
        {
        }
        virtual ~OpParams()
        {
        }

        SplitAxis splitAxis() const
        {
            return m_split_axis;
        }
        void setSplitAxis(SplitAxis axis)
        {
            m_split_axis = axis;
        }

    protected:
        SplitAxis m_split_axis;
    };

    SplitNode(canonical_ast::SplitNode* split, NvU16 numBatches = 1)
        : Node(numBatches)
    {
        m_engine_type = SPLIT;
        m_engine_op_type = SPLIT_SOFTWARE;
        m_can_split_node = split;
        m_mb_op_params = new MultiBatchState<OpParams>(numBatches);
        m_sup_in_surf_formats.assign(IMG_FORMATS, IMG_FORMATS + (sizeof(IMG_FORMATS) / sizeof(IMG_FORMATS[0])));
        m_sup_in_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_FEATURE_DATA_INT8);
        m_sup_in_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_FEATURE_DATA_FP16);
        m_sup_out_surf_formats.assign(IMG_FORMATS, IMG_FORMATS + (sizeof(IMG_FORMATS) / sizeof(IMG_FORMATS[0])));
        m_sup_out_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_FEATURE_DATA_INT8);
        m_sup_out_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_FEATURE_DATA_FP16);
    }
    virtual ~SplitNode()
    {
    }

    inline bool debugSplit()
    {
        return true;
    }
    virtual void captureCanonicalParams();
    OpParams& params(NvU16 batchId = 0)
    {
        return m_mb_op_params->batch(batchId);
    }

    NvDlaError verifySurfaceIsPartOfSplit(surface::TensorSurfaceDesc* src, surface::TensorSurfaceDesc* dst, SplitAxis axis);
    NvU64 suggestOffsetInSplitChain(surface::TensorSurfaceDesc* outTSD);
    virtual Dims4 suggestSurfaceDims(surface::TensorSurfaceDesc* tsd);
    virtual NvU32 suggestLineStride(surface::TensorSurfaceDesc* tsd);
    virtual NvU32 suggestSurfaceStride(surface::TensorSurfaceDesc* tsd);
    virtual NvU64 suggestSurfaceSize(surface::TensorSurfaceDesc* tsd);
    virtual NvU64 suggestSurfaceOffsetInBuffer(surface::TensorSurfaceDesc* tsd);
    virtual memory::TensorBufferDesc* suggestBuffer(surface::TensorSurfaceDesc* tsd);

    // cache reference to input/output edges in their order of appearance
    virtual NvDlaError populateEdgePorts();
    virtual NvDlaError resolveMultiBatchDependencies()
    {
        return NvDlaSuccess;
    }
    virtual SplitNode* clone()
    {
        return new SplitNode(*this);
    }
    virtual canonical_ast::SplitNode* canonicalNode() const
    {
        return m_can_split_node;
    }

    virtual NvDlaError verifyEdgePorts()
    {
        if (inputEdges().size() != 1)
            return NvDlaError_BadValue;
        else
            return NvDlaSuccess;
    }
    virtual NvDlaError verifySurfaceDims(surface::TensorSurfaceDesc*);

protected:
    canonical_ast::SplitNode* m_can_split_node;
    MultiBatchState<OpParams>* m_mb_op_params;
};

class SDPNode; //fwd decl

// convolution layer
// fully connected
// deconvolve 1st
class ConvCoreNode : public Node
{
public:
    virtual std::string className() const
    {
        return std::string("ConvCoreNode");
    }

    ConvCoreNode(NvU16 numBatches)
        : Node(numBatches)
    {
        m_engine_type = CONVOLUTION;
        m_mb_engine_params = new MultiBatchState<ConvCoreEngineParams>(numBatches);
    }
    virtual ~ConvCoreNode()
    {
    }

    // data-split can happen for conv/fc/deconv
    struct SplitDataInfo
    {
        // for split-h
        NvS32 topSliceID;
        NvS32 bottomSliceID;
        NvS32 topPadding;
        NvS32 bottomPadding;
        // for split-w
        NvS32 leftSliceID;
        NvS32 rightSliceID;
        NvS32 leftPadding;
        NvS32 rightPadding;
        // common
        Dims4 inDims;
        Dims4 outDims;
        NvS32 numConvs;
        NvS32 numOverlapSlices;
        NvS32 numRetainSlices;

        NvU64 inputBufferOffset;
        NvU64 outputBufferOffset;

        NvU16 wtBanks;
        NvU16 dataBanks;

        SplitDataInfo()
            : topSliceID(-1), bottomSliceID(-1), topPadding(-1), bottomPadding(-1), leftSliceID(-1), rightSliceID(-1), leftPadding(-1), rightPadding(-1), numConvs(-1), numOverlapSlices(-1), numRetainSlices(-1), inputBufferOffset(0), outputBufferOffset(0), wtBanks(0), dataBanks(0)
        {
            inDims = Dims4(0, 0, 0, 0);
            outDims = Dims4(0, 0, 0, 0);
        }
    };

    NvDlaError captureCanonicalWeights();

    bool debugFactorization()
    {
        return false;
    }

    virtual Dims4 suggestSurfaceDims(surface::TensorSurfaceDesc* tsd);
    virtual NvU32 suggestLineStride(surface::TensorSurfaceDesc* tsd);
    virtual NvU32 suggestSurfaceStride(surface::TensorSurfaceDesc* tsd);
    virtual NvU64 suggestSurfaceSize(surface::TensorSurfaceDesc* tsd);
    virtual NvU64 suggestSurfaceOffsetInBuffer(surface::TensorSurfaceDesc* tsd);
    virtual NvDlaError preProcessAuxData();
    virtual Node* mergeWithSDPOp(SDPNode* other_op);
    virtual NvDlaError quantizeAuxData();
    virtual NvDlaError handleLowPrecisionConversions();
    virtual NvDlaError fuseOnTheFlyNodes();
    virtual NvDlaError translateAuxData();
    virtual NvDlaError splitNodes();
    virtual NvDlaError handleMultiBatch();
    virtual NvDlaError resolveDataDependencies(Node* next);

    virtual ConvCoreEngineParams& params(NvU16 batchId = 0)
    {
        return m_mb_engine_params->batch(batchId);
    }
    virtual ConvCoreNode* clone()
    {
        return new ConvCoreNode(*this);
    }
    virtual void captureCanonicalParams()
    {
        return;
    }

    virtual NvDlaError nodeAuxEdge(engine_ast::Edge** ret_edge);

    // Conv Core doesn't have a write port, so the FD o/p from conv core
    // has to pass through SDP to be written into memory
    SDPNode* addSDPJointOpNode(canonical_ast::Node* canConv);
    SDPNode* addSDPJointOpNode(SDPNode* copyFromSDP);

    virtual std::vector<surface::SurfaceFormat> suggestAuxSurfaceFormats(Edge* auxEdge = NULL);

    NvDlaError determineWinogradParams();

    SplitDataInfo& splitDataInfo()
    {
        return m_split_data_info;
    }
    void setSplitDataInfo(const SplitDataInfo& sdi)
    {
        m_split_data_info = sdi;
    }

    NvDlaError processWtsForIMG();
    NvDlaError mandatoryChnlExtForIMG();
    NvDlaError optionalChnlExtForIMG();
    Node* tryToMergeWithScaleOp(SDPNode* sdp_scl_op);
    Node* tryToMergeWithBatchNormOp(SDPNode* sdp_bn_op);
    NvDlaError squashWeightGroups();
    NvDlaError splitNodesInternal();
    NvDlaError splitData(NvU16 avlbDataBanks);
    NvDlaError splitWeightsAndData(NvU16 avlbWtBanks, NvU16 avlbDataBanks);
    NvDlaError determineSplitDataRatios(NvU16& avlbDataBanks, std::vector<SplitDataInfo>& splitChunks);

    virtual NvDlaError verifyEdgePorts()
    {
        if (inputEdges().size() != 1 || auxEdges().size() != 1 || outputEdges().size() != 1)
            return NvDlaError_BadValue;
        else
            return NvDlaSuccess;
    }
    virtual NvDlaError verifySurfaceDims(surface::TensorSurfaceDesc*);

protected:
    NvS16 calculateEPS(surface::TensorSurfaceDesc*);
    NvU16 calculateTotalBanksForData(surface::TensorSurfaceDesc*);
    NvU16 calculateMinBanksForWeight(surface::TensorSurfaceDesc*);
    NvU16 calculateTotalBanksForWeight(surface::TensorSurfaceDesc*);
    NvDlaError verifyPartialHInfo(const std::vector<SplitDataInfo>&, bool);
    MultiBatchState<ConvCoreEngineParams>* m_mb_engine_params;
    SplitDataInfo m_split_data_info;
};

// convolution op in conv engine
class ConvCoreConvolutionOpNode : public ConvCoreNode
{
public:
    virtual std::string className() const
    {
        return std::string("ConvolutionOpNode");
    }

    class OpParams : public ConvCoreEngineParams
    {
    public:
        OpParams()
            : ConvCoreEngineParams()
        {
            m_post_extension = 0;
        }
        virtual ~OpParams()
        {
        }

        NvU8 postExtension() const
        {
            return m_post_extension;
        }
        void setPostExtension(NvU8 ext)
        {
            m_post_extension = ext;
        }

    protected:
        NvU8 m_post_extension;
    };

    ConvCoreConvolutionOpNode(canonical_ast::ConvolutionNode* can_conv_node, NvU16 numBatches = 1)
        : ConvCoreNode(numBatches)
    {
        m_engine_op_type = CONVOLUTION_CONV;
        m_can_conv_node = can_conv_node;
        m_mb_op_params = new MultiBatchState<OpParams>(numBatches);
        m_sup_in_surf_formats.assign(IMG_FORMATS, IMG_FORMATS + (sizeof(IMG_FORMATS) / sizeof(IMG_FORMATS[0])));
        m_sup_in_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_FEATURE_DATA_INT8);
        m_sup_in_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_FEATURE_DATA_FP16);
        m_sup_out_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_FEATURE_DATA_INT8);
        m_sup_out_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_FEATURE_DATA_FP16);
        m_sup_aux_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_WEIGHT_DC_INT8);
        m_sup_aux_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_WEIGHT_DC_FP16);
        m_sup_aux_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_WEIGHT_IMG_INT8);
        m_sup_aux_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_WEIGHT_IMG_FP16);
        m_sup_aux_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_WEIGHT_WG_INT8);
        m_sup_aux_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_WEIGHT_WG_FP16);
    }
    virtual ~ConvCoreConvolutionOpNode()
    {
    }

    NvDlaError fallbackWGConvToDC();

    virtual OpParams& params(NvU16 batchId = 0)
    {
        return m_mb_op_params->batch(batchId);
    }
    virtual ConvCoreConvolutionOpNode* clone()
    {
        return new ConvCoreConvolutionOpNode(*this);
    }
    virtual canonical_ast::ConvolutionNode* canonicalNode() const
    {
        return m_can_conv_node;
    }
    virtual void captureCanonicalParams();
    virtual const void* getAuxData(Edge* auxEdge)
    {
        return params().DLAWeights().values;
    }
    virtual Dims4 getAuxDims()
    {
        return params().weightDims();
    }
    virtual void inheritParams(Node*);

    virtual NvDlaError emitOp(Graph*, DLAInterface*, NvU32 op_slot, NvU32 batch_id,
                              DLACommonOpDescAccessor,
                              DLAOperationContainerAccessor,
                              DLASurfaceContainerAccessor);

#ifdef NVDLA_COMPILER_OUTPUT_FOR_PROTOTEST
    virtual NvDlaError emitOp(NvU32 op_slot, NvU32 batch_id,
                              DLAInterface*,
                              DLACommonOpDescAccessor&,
                              DLAOperationContainerAccessor&,
                              DLASurfaceContainerAccessor&,
                              nvdla_prototest_interface::Layer*);
#endif

protected:
    canonical_ast::ConvolutionNode* m_can_conv_node;
    MultiBatchState<OpParams>* m_mb_op_params;
};

// fc op in conv engine
class ConvCoreFullyConnectedOpNode : public ConvCoreNode
{
public:
    virtual std::string className() const
    {
        return std::string("FullyConnectedOpNode");
    }

    class OpParams : public ConvCoreEngineParams
    {
        // add that is relevant to FC
    };

    ConvCoreFullyConnectedOpNode(canonical_ast::FullyConnectedNode* can_fc_node, NvU16 numBatches = 1)
        : ConvCoreNode(numBatches)
    {
        m_engine_op_type = CONVOLUTION_FC;
        m_can_fc_node = can_fc_node;
        m_mb_op_params = new MultiBatchState<OpParams>(numBatches);
        m_sup_in_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_IMG_R16_F);
        m_sup_in_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_FEATURE_DATA_INT8);
        m_sup_in_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_FEATURE_DATA_FP16);
        m_sup_out_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_FEATURE_DATA_INT8);
        m_sup_out_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_FEATURE_DATA_FP16);
        m_sup_aux_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_WEIGHT_DC_INT8);
        m_sup_aux_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_WEIGHT_DC_FP16);
    }
    virtual ~ConvCoreFullyConnectedOpNode()
    {
    }

    virtual OpParams& params(NvU16 batchId = 0)
    {
        return m_mb_op_params->batch(batchId);
    }
    virtual ConvCoreFullyConnectedOpNode* clone()
    {
        return new ConvCoreFullyConnectedOpNode(*this);
    }
    virtual canonical_ast::FullyConnectedNode* canonicalNode() const
    {
        return m_can_fc_node;
    }
    virtual void captureCanonicalParams();
    virtual const void* getAuxData(Edge* auxEdge)
    {
        return params().DLAWeights().values;
    }
    virtual Dims4 getAuxDims()
    {
        return params().weightDims();
    }

    virtual NvDlaError emitOp(Graph*, DLAInterface*, NvU32 op_slot, NvU32 batch_id,
                              DLACommonOpDescAccessor,
                              DLAOperationContainerAccessor,
                              DLASurfaceContainerAccessor);

#ifdef NVDLA_COMPILER_OUTPUT_FOR_PROTOTEST
    virtual NvDlaError emitOp(NvU32 op_slot, NvU32 batch_id,
                              DLAInterface*,
                              DLACommonOpDescAccessor&,
                              DLAOperationContainerAccessor&,
                              DLASurfaceContainerAccessor&,
                              nvdla_prototest_interface::Layer*);
#endif

protected:
    canonical_ast::FullyConnectedNode* m_can_fc_node;
    MultiBatchState<OpParams>* m_mb_op_params;
};

// deconv op in conv engine
class ConvCoreDeconvolutionOpNode : public ConvCoreNode
{
public:
    virtual std::string className() const
    {
        return std::string("DeconvolutionOpNode");
    }

    class OpParams : public ConvCoreEngineParams
    {
        // add all that is relevant to deconv
    };
    ConvCoreDeconvolutionOpNode(canonical_ast::DeconvolutionNode* can_deconv_node, NvU16 numBatches = 1)
        : ConvCoreNode(numBatches)
    {
        m_engine_op_type = CONVOLUTION_DECONV;
        m_can_deconv_node = can_deconv_node;
        m_mb_op_params = new MultiBatchState<OpParams>(numBatches);
        m_sup_in_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_FEATURE_DATA_INT8);
        m_sup_in_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_FEATURE_DATA_FP16);
        m_sup_out_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_FEATURE_DATA_INT8);
        m_sup_out_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_FEATURE_DATA_FP16);
        m_sup_aux_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_WEIGHT_DC_INT8);
        m_sup_aux_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_WEIGHT_DC_FP16);
    }
    virtual ~ConvCoreDeconvolutionOpNode()
    {
    }

    virtual OpParams& params(NvU16 batchId = 0)
    {
        return m_mb_op_params->batch(batchId);
    }
    virtual ConvCoreDeconvolutionOpNode* clone()
    {
        return new ConvCoreDeconvolutionOpNode(*this);
    }
    virtual canonical_ast::DeconvolutionNode* canonicalNode() const
    {
        return m_can_deconv_node;
    }
    virtual void captureCanonicalParams();
    virtual void inheritParams(Node*);
    virtual const void* getAuxData(Edge* auxEdge)
    {
        return params().DLAWeights().values;
    }
    virtual Dims4 getAuxDims()
    {
        return params().weightDims();
    }

    virtual Dims4 suggestSurfaceDims(surface::TensorSurfaceDesc* tsd);
    virtual NvDlaError preProcessAuxData();
    virtual NvDlaError quantizeAuxData();
    virtual NvDlaError translateAuxData();

    virtual NvDlaError emitOp(Graph*, DLAInterface*, NvU32 op_slot, NvU32 batch_id,
                              DLACommonOpDescAccessor,
                              DLAOperationContainerAccessor,
                              DLASurfaceContainerAccessor);

protected:
    canonical_ast::DeconvolutionNode* m_can_deconv_node;
    MultiBatchState<OpParams>* m_mb_op_params;
};

// activation
// scale (sometimes, unless power != 0? then cpu?)
// elementwise
// bias reduction
// SDP super op node (x1 + x2 + y ops)
class SDPSuperOpNode; //fwd declaration

class SDPNode : public Node
{
public:
    virtual std::string className() const
    {
        return std::string("SDPNode");
    }

    SDPNode(NvU16 numBatches)
        : Node(numBatches)
    {
        m_engine_type = SDP;
        m_mb_engine_params = new MultiBatchState<SDPEngineParams>(numBatches);
        m_isUnitScale = false;
    }
    virtual ~SDPNode()
    {
    }

    bool debugFactorization()
    {
        return false;
    }

    virtual std::vector<surface::SurfaceFormat> suggestAuxSurfaceFormats(Edge* auxEdge = NULL);
    virtual Dims4 suggestSurfaceDims(surface::TensorSurfaceDesc* tsd);
    virtual NvU32 suggestLineStride(surface::TensorSurfaceDesc* tsd);
    virtual NvU32 suggestSurfaceStride(surface::TensorSurfaceDesc* tsd);
    virtual NvU64 suggestSurfaceSize(surface::TensorSurfaceDesc* tsd);
    virtual NvU64 suggestSurfaceOffsetInBuffer(surface::TensorSurfaceDesc* tsd);

    Node* tryToMergeWithActOp(SDPNode* sdp_act_op);

    virtual Node* mergeWithSDPOp(SDPNode* /*other_op*/);
    virtual NvDlaError fuseOnTheFlyNodes();
    virtual NvDlaError handleMultiBatch();
    virtual NvDlaError resolveDataDependencies(Node* next);
    virtual Node* fuseSDPSubEngineOp(SDPNode* nextSDP);
    virtual Node* mergeUnitScaleOp(SDPNode* nextSDP);
    bool isFeasibleToFuseSDPSubEngineOp(SDPNode* nextSDP);
    bool isFeasibleToFuseSDPEltwiseOp(SDPNode* nextSDP);
    virtual NvDlaError configureSDPSuperOpSubEngine(SDPSuperOpNode* sdpSuperOp, SDPSubEngineType xN)
    {
        return NvDlaError_BadParameter;
    }

    NvDlaError determineWinogradParams(ConvCoreNode* wgConvNode);

    virtual SDPNode* clone()
    {
        return new SDPNode(*this);
    }
    virtual canonical_ast::Node* canonicalNode() const
    {
        return NULL;
    }
    virtual void captureCanonicalParams()
    {
        return;
    }
    virtual NvDlaError nodeAuxEdge(engine_ast::Edge** ret_edge);

    virtual SDPEngineParams& params(NvU16 batchId = 0)
    {
        return m_mb_engine_params->batch(batchId);
    }

    virtual NvDlaError verifySurfaceDims(surface::TensorSurfaceDesc*);

    bool isUnitScale()
    {
        return m_isUnitScale;
    }
    void setUnitScale(bool enb)
    {
        m_isUnitScale = enb;
    }

    const Weights& rescaleData() const
    {
        return m_rescaleData;
    }
    void setRescaleData(const Weights& rd)
    {
        m_rescaleData = rd;
    }

protected:
    MultiBatchState<SDPEngineParams>* m_mb_engine_params;
    bool m_isUnitScale;
    Weights m_rescaleData;
};

// activation op in SDP engine
class SDPActivationOpNode : public SDPNode
{
public:
    virtual std::string className() const
    {
        return std::string("SDPActivationNode");
    }

    class OpParams : public SDPEngineParams
    {
    public:
        OpParams()
            : SDPEngineParams()
        {
            SDPSubEngineParams x1params;
            SDPSubEngineParams x2params;
            SDPSubEngineParams yparams;

            x1params.setEnabled(false);
            x2params.setEnabled(false);
            yparams.setEnabled(false);

            setX1Params(x1params);
            setX2Params(x2params);
            setYParams(yparams);
        }
        virtual ~OpParams()
        {
        }

        // add anything relevant to Activation here
    };

    SDPActivationOpNode(canonical_ast::ActivationNode* can_act_node, NvU16 numBatches = 1)
        : SDPNode(numBatches)
    {
        m_engine_op_type = SDP_ACTIVATION;
        m_can_act_node = can_act_node;
        m_mb_op_params = new MultiBatchState<OpParams>(numBatches);
        m_hLut = -1;
        m_sup_in_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_FEATURE_DATA_INT8);
        m_sup_in_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_FEATURE_DATA_FP16);
        m_sup_out_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_FEATURE_DATA_INT8);
        m_sup_out_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_FEATURE_DATA_FP16);
    }
    virtual ~SDPActivationOpNode()
    {
    }

    virtual void inheritParams(Node* inheritFrom);
    virtual OpParams& params(NvU16 batchId = 0)
    {
        return m_mb_op_params->batch(batchId);
    }
    virtual SDPActivationOpNode* clone()
    {
        return new SDPActivationOpNode(*this);
    }
    virtual canonical_ast::ActivationNode* canonicalNode() const
    {
        return m_can_act_node;
    }
    virtual void captureCanonicalParams();

    virtual NvDlaError verifyEdgePorts()
    {
        if (inputEdges().size() != 1 || auxEdges().size() != 0 || outputEdges().size() != 1)
            return NvDlaError_BadValue;
        else
            return NvDlaSuccess;
    }

    virtual NvDlaError emitOp(Graph* g,
                              DLAInterface* target_dla,
                              NvU32 op_slot, NvU32 batch_id,
                              DLACommonOpDescAccessor dep,
                              DLAOperationContainerAccessor op,
                              DLASurfaceContainerAccessor surf);

#ifdef NVDLA_COMPILER_OUTPUT_FOR_PROTOTEST
    virtual NvDlaError emitOp(NvU32 op_slot, NvU32 batch_id,
                              DLAInterface*,
                              DLACommonOpDescAccessor&,
                              DLAOperationContainerAccessor&,
                              DLASurfaceContainerAccessor&,
                              nvdla_prototest_interface::Layer*);
#endif

    NvDlaError emitReLU(DLASDPOpDescAccessor sdp_op);
    NvDlaError emitLut(DLASDPOpDescAccessor sdp_op);

protected:
    canonical_ast::ActivationNode* m_can_act_node;
    MultiBatchState<OpParams>* m_mb_op_params;
    LutManager::LutHandle m_hLut;
};

// scale op in SDP engine :-  y = (scale_factor * x)
class SDPScaleOpNode : public SDPNode
{
public:
    virtual std::string className() const
    {
        return std::string("SDPScaleNode");
    }

    class OpParams : public SDPEngineParams
    {
    public:
        OpParams()
            : SDPEngineParams()
        {
            SDPSubEngineParams x1params;
            SDPSubEngineParams x2params;
            SDPSubEngineParams yparams;

            x1params.setEnabled(true);
            x2params.setEnabled(false);
            yparams.setEnabled(false);

            x1params.setAluType(SDPALUTypeEnum::SDP_ALU_TYPE_SUM);
            x1params.setOpType(SDPOpTypeEnum::SDP_OP_TYPE_MUL);

            setX1Params(x1params);
            setX2Params(x2params);
            setYParams(yparams);
        }
        virtual ~OpParams()
        {
        }

        const Dims4& scaleDims() const
        {
            return m_scale_dims;
        }
        void setScaleDims(const Dims4& sd)
        {
            m_scale_dims = sd;
        }

        Weights rawScaleData() const
        {
            return m_raw_scale_data;
        }
        void setRawScaleData(Weights sc)
        {
            m_raw_scale_data = sc;
        }

        const Weights& DLAScaleData() const
        {
            return m_dla_scale_data;
        }
        void setDLAScaleData(const Weights& dsd)
        {
            m_dla_scale_data = dsd;
        }

    protected:
        Dims4 m_scale_dims;
        Weights m_raw_scale_data;
        Weights m_dla_scale_data;
    };

    SDPScaleOpNode(canonical_ast::ScaleNode* can_scale_node, NvU16 numBatches = 1)
        : SDPNode(numBatches)
    {
        m_engine_op_type = SDP_SCALE;
        m_can_scale_node = can_scale_node;
        m_mb_op_params = new MultiBatchState<OpParams>(numBatches);
        m_sup_in_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_FEATURE_DATA_INT8);
        m_sup_in_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_FEATURE_DATA_FP16);
        m_sup_out_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_FEATURE_DATA_INT8);
        m_sup_out_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_FEATURE_DATA_FP16);
        m_sup_aux_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_SCALE_DATA_INT8);
        m_sup_aux_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_SCALE_DATA_INT16);
        m_sup_aux_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_SCALE_DATA_FP16);
    }
    virtual ~SDPScaleOpNode()
    {
    }

    template<typename MP, typename CP>
    Weights inverseScaleData(
        engine_ast::SDPMode scaleMode, // per-layer/channel/elementwise
        Dims4 scaleDims,               // dims of orig caffe scale-data blob
        Weights& srcScaleData          // ptr to orig caffe scale blob
    );

    NvDlaError populateWithUnitScaleParams(
        engine_ast::SDPMode scaleMode,
        Dims4 scaleDims);

    virtual void inheritParams(Node* inheritFrom);
    virtual OpParams& params(NvU16 batchId = 0)
    {
        return m_mb_op_params->batch(batchId);
    }
    virtual void captureCanonicalParams();
    virtual SDPScaleOpNode* clone()
    {
        return new SDPScaleOpNode(*this);
    }
    virtual canonical_ast::ScaleNode* canonicalNode() const
    {
        return m_can_scale_node;
    }

    virtual std::vector<surface::SurfaceFormat> suggestAuxSurfaceFormats(Edge* auxEdge = NULL);

    // Some scale operations have optional bias
    SDPNode* addSDPBiasOpNode(canonical_ast::Node* can_node);
    virtual Node* mergeWithSDPOp(SDPNode* other_op);
    virtual NvDlaError handleLowPrecisionConversions();
    virtual NvDlaError configureSDPSuperOpSubEngine(SDPSuperOpNode* sdpSuperOp, SDPSubEngineType xN) override;
    virtual NvDlaError translateAuxData();

    // Utility functions for handling sdp scales INT8
    NvDlaError getFp32ScaleData(const Weights data, std::vector<NvF32>& trnsFp32Scale);
    NvDlaError scaleDataToInt16();
    NvDlaError rescaleScaleDataForNoFusedConv();
    NvDlaError rescaleScaleDataForPerFilter();
    NvDlaError rescaleScaleDataForPerKernel();

    NvDlaError captureCanonicalScaleData();
    virtual const void* getAuxData(Edge* auxEdge)
    {
        return params().DLAScaleData().values;
    }
    virtual Dims4 getAuxDims()
    {
        return params().scaleDims();
    }

    virtual NvDlaError verifyEdgePorts()
    {
        if (inputEdges().size() != 1 || auxEdges().size() != 1 || outputEdges().size() != 1)
            return NvDlaError_BadValue;
        else
            return NvDlaSuccess;
    }

    virtual NvDlaError emitOp(Graph*, DLAInterface*, NvU32 op_slot, NvU32 batch_id,
                              DLACommonOpDescAccessor,
                              DLAOperationContainerAccessor,
                              DLASurfaceContainerAccessor);

#ifdef NVDLA_COMPILER_OUTPUT_FOR_PROTOTEST
    virtual NvDlaError emitOp(NvU32 op_slot, NvU32 batch_id,
                              DLAInterface*,
                              DLACommonOpDescAccessor&,
                              DLAOperationContainerAccessor&,
                              DLASurfaceContainerAccessor&,
                              nvdla_prototest_interface::Layer*);
#endif

protected:
    canonical_ast::ScaleNode* m_can_scale_node;
    MultiBatchState<OpParams>* m_mb_op_params;
};

// batch norm :- y = (x - mean)/var
class SDPBatchNormOpNode : public SDPNode
{
public:
    virtual std::string className() const
    {
        return std::string("SDPBatchNormOpNode");
    }

    class OpParams : public SDPEngineParams
    {
    public:
        OpParams()
            : SDPEngineParams(),
              m_epsilon(0.0f)
        {
            SDPSubEngineParams x1params;
            SDPSubEngineParams x2params;
            SDPSubEngineParams yparams;

            x1params.setEnabled(true);
            x2params.setEnabled(false);
            yparams.setEnabled(false);

            x1params.setAluType(SDPALUTypeEnum::SDP_ALU_TYPE_SUM);
            x1params.setOpType(SDPOpTypeEnum::SDP_OP_TYPE_BOTH);

            setX1Params(x1params);
            setX2Params(x2params);
            setYParams(yparams);
        }
        virtual ~OpParams()
        {
        }

        float epsilon() const
        {
            return m_epsilon;
        }
        void setEpsilon(const float e)
        {
            m_epsilon = e;
        }

        const Dims4& meanDims() const
        {
            return m_mean_dims;
        }
        void setMeanDims(const Dims4& md)
        {
            m_mean_dims = md;
        }

        const Dims4& varianceDims() const
        {
            return m_var_dims;
        }
        void setVarianceDims(const Dims4& vd)
        {
            m_var_dims = vd;
        }

        const Dims4& batchNormDims() const
        {
            return m_bn_dims;
        }
        void setBatchNormDims(const Dims4& dims)
        {
            m_bn_dims = dims;
        }

        const Weights& rawMeanData() const
        {
            return m_raw_mean_data;
        }
        void setRawMeanData(const Weights& rmd)
        {
            m_raw_mean_data = rmd;
        }

        const Weights& rawVarianceData() const
        {
            return m_raw_var_data;
        }
        void setRawVarianceData(const Weights& rvd)
        {
            m_raw_var_data = rvd;
        }

        const Weights& DLABatchNormData() const
        {
            return m_dla_bn_data;
        }
        void setDLABatchNormData(const Weights& bn)
        {
            m_dla_bn_data = bn;
        }

    protected:
        float m_epsilon;
        Dims4 m_mean_dims;
        Dims4 m_var_dims;
        Dims4 m_bn_dims;
        Weights m_raw_mean_data;
        Weights m_raw_var_data;
        Weights m_dla_bn_data;
    };

    SDPBatchNormOpNode(canonical_ast::BatchNormNode* can_bn_node, NvU16 numBatches)
        : SDPNode(numBatches)
    {
        m_engine_op_type = SDP_BATCH_NORM;
        m_can_bn_node = can_bn_node;
        m_mb_op_params = new MultiBatchState<OpParams>(numBatches);
        m_sup_in_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_FEATURE_DATA_INT8);
        m_sup_in_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_FEATURE_DATA_FP16);
        m_sup_out_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_FEATURE_DATA_INT8);
        m_sup_out_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_FEATURE_DATA_FP16);
        m_sup_aux_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_BATCH_NORM_DATA_INT8);
        m_sup_aux_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_BATCH_NORM_DATA_INT16);
        m_sup_aux_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_BATCH_NORM_DATA_FP16);
    }
    virtual ~SDPBatchNormOpNode()
    {
    }

    enum AuxDataType
    {
        MEAN_DATA = 0,
        VARIANCE_DATA = 1
    };

    virtual std::vector<surface::SurfaceFormat> suggestAuxSurfaceFormats(Edge* auxEdge = NULL);

    template<typename MP, typename CP>
    NvU32 factorsPerEntity(
        MP rawValue,
        AuxDataType auxType,
        MP* factorizedValue = NULL);

    template<typename MP, typename CP>
    NvDlaError maxFactorsPerEntity(
        Weights& auxData,
        AuxDataType auxType,
        NvU32* numFactors);

    template<typename MP, typename CP>
    NvDlaError processMeanAndVar(
        Weights& rawMeanData,
        Weights& processedMeanData,
        Weights& rawVarData,
        std::vector<Weights>& processedVarData);

    NvDlaError scaleMeanToInt16(ConvCoreNode* fusedConv, std::vector<NvF32>& filterScales, std::vector<NvF32>& inTensorScales);

    Node* tryToMergeWithScaleOp(SDPNode* sdp_scl_op);
    Node* tryToMergeWithBiasOp(SDPNode* sdp_bias_op);
    Node* tryToMergeWithBiasOpInplace(SDPNode* sdp_bias_op);
    virtual Node* mergeWithSDPOp(SDPNode* other_op);
    virtual NvDlaError preProcessAuxData();
    virtual NvDlaError quantizeAuxData();
    virtual NvDlaError handleLowPrecisionConversions();

    template<typename MP, typename CP>
    NvDlaError performPerLayerRescaling(
        ConvCoreNode* fusedConv,
        std::vector<NvF32>& filterScales,
        Weights& rawVarData,
        std::vector<NvF32>& inTensorScales,
        std::vector<NvF32>& outTensorScales);

    template<typename MP, typename CP>
    NvDlaError performPerChannelRescaling(
        ConvCoreNode* fusedConv,
        std::vector<NvF32>& filterScales,
        Weights& rawVarData,
        std::vector<NvF32>& inTensorScales,
        std::vector<NvF32>& outTensorScales);

    virtual NvDlaError translateAuxData();
    virtual NvDlaError configureSDPSuperOpSubEngine(SDPSuperOpNode* sdpSuperOp, SDPSubEngineType xN) override;

    bool debugFactorization()
    {
        return false;
    }
    virtual void inheritParams(Node* inheritFrom);
    virtual OpParams& params(NvU16 batchId = 0)
    {
        return m_mb_op_params->batch(batchId);
    }
    NvDlaError captureCanonicalBatchNormData();
    virtual void captureCanonicalParams();
    virtual SDPBatchNormOpNode* clone()
    {
        return new SDPBatchNormOpNode(*this);
    }
    virtual canonical_ast::BatchNormNode* canonicalNode() const
    {
        return m_can_bn_node;
    }
    virtual const void* getAuxData(Edge* auxEdge)
    {
        return params().DLABatchNormData().values;
    }
    virtual Dims4 getAuxDims()
    {
        return params().batchNormDims();
    }

    virtual NvDlaError verifyEdgePorts()
    {
        if (inputEdges().size() != 1 || auxEdges().size() != 1 || outputEdges().size() != 1)
            return NvDlaError_BadValue;
        else
            return NvDlaSuccess;
    }

    virtual NvDlaError emitOp(Graph*, DLAInterface*, NvU32 op_slot, NvU32 batch_id,
                              DLACommonOpDescAccessor,
                              DLAOperationContainerAccessor,
                              DLASurfaceContainerAccessor);

#ifdef NVDLA_COMPILER_OUTPUT_FOR_PROTOTEST
    virtual NvDlaError emitOp(NvU32 op_slot, NvU32 batch_id,
                              DLAInterface*,
                              DLACommonOpDescAccessor&,
                              DLAOperationContainerAccessor&,
                              DLASurfaceContainerAccessor&,
                              nvdla_prototest_interface::Layer*);
#endif

protected:
    canonical_ast::BatchNormNode* m_can_bn_node;
    MultiBatchState<OpParams>* m_mb_op_params;
};

// elementwise op in SDP engine
class SDPElementWiseOpNode : public SDPNode
{
public:
    virtual std::string className() const
    {
        return std::string("SDPElementWiseOpNode");
    }

    class OpParams : public SDPEngineParams
    {
    public:
        OpParams()
            : SDPEngineParams()
        {
            SDPSubEngineParams x1params;
            SDPSubEngineParams x2params;
            SDPSubEngineParams yparams;

            x1params.setEnabled(true);
            x2params.setEnabled(false);
            yparams.setEnabled(false);

            setX1Params(x1params);
            setX2Params(x2params);
            setYParams(yparams);
        }
        virtual ~OpParams()
        {
        }

        // add anything relevant to ElementWise here
    };

    SDPElementWiseOpNode(canonical_ast::ElementWiseNode* can_ew_node, NvU16 numBatches = 1)
        : SDPNode(numBatches)
    {
        m_engine_op_type = SDP_ELEMENTWISE;
        m_can_ew_node = can_ew_node;
        m_mb_op_params = new MultiBatchState<OpParams>(numBatches);
        m_sup_in_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_FEATURE_DATA_INT8);
        m_sup_in_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_FEATURE_DATA_FP16);
        m_sup_out_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_FEATURE_DATA_INT8);
        m_sup_out_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_FEATURE_DATA_FP16);
    }
    virtual ~SDPElementWiseOpNode()
    {
    }

    virtual NvDlaError handleLowPrecisionConversions();
    NvDlaError performPerTensorRescaling(
        std::vector<NvF32>& inTensorScales,
        std::vector<NvF32>& outTensorScales,
        PrecisionCVTParams& outCvt);

    virtual void inheritParams(Node* inheritFrom);
    virtual OpParams& params(NvU16 batchId = 0)
    {
        return m_mb_op_params->batch(batchId);
    }
    virtual void captureCanonicalParams();
    virtual SDPElementWiseOpNode* clone()
    {
        return new SDPElementWiseOpNode(*this);
    }
    virtual canonical_ast::ElementWiseNode* canonicalNode() const
    {
        return m_can_ew_node;
    }
    virtual NvDlaError configureSDPSuperOpSubEngine(SDPSuperOpNode* sdpSuperOp, SDPSubEngineType xN) override;
    Node* getPeerSource(SDPNode* currentSource);
    virtual Node* mergeWithSDPOp(SDPNode* other_op);
    virtual NvDlaError populateEdgePorts();
    virtual NvDlaError verifyEdgePorts()
    {
        if (inputEdges().size() != 2 || auxEdges().size() != 0 || outputEdges().size() != 1)
            return NvDlaError_BadValue;
        else
            return NvDlaSuccess;
    }

    virtual NvDlaError emitOp(Graph*, DLAInterface*, NvU32 op_slot, NvU32 batch_id,
                              DLACommonOpDescAccessor,
                              DLAOperationContainerAccessor,
                              DLASurfaceContainerAccessor);

#ifdef NVDLA_COMPILER_OUTPUT_FOR_PROTOTEST
    virtual NvDlaError emitOp(NvU32 op_slot, NvU32 batch_id,
                              DLAInterface*,
                              DLACommonOpDescAccessor&,
                              DLAOperationContainerAccessor&,
                              DLASurfaceContainerAccessor&,
                              nvdla_prototest_interface::Layer*);
#endif

protected:
    canonical_ast::ElementWiseNode* m_can_ew_node;
    MultiBatchState<OpParams>* m_mb_op_params;
};

// sdp bias to club with conv w/ bias term (bias-reduction)
// sdp bias to club with sdp scale w/ shift term (bias-addition)
class SDPBiasOpNode : public SDPNode
{
public:
    virtual std::string className() const
    {
        return std::string("SDPBiasOpNode");
    }

    class OpParams : public SDPEngineParams
    {
    public:
        OpParams()
            : SDPEngineParams(),
              m_HasBiasReduction(false),
              m_axis(0)
        {
            SDPSubEngineParams x1params;
            SDPSubEngineParams x2params;
            SDPSubEngineParams yparams;

            x1params.setEnabled(true);
            x2params.setEnabled(false);
            yparams.setEnabled(false);

            x1params.setAluType(SDPALUTypeEnum::SDP_ALU_TYPE_SUM);
            x1params.setOpType(SDPOpTypeEnum::SDP_OP_TYPE_ADD);

            setX1Params(x1params);
            setX2Params(x2params);
            setYParams(yparams);
        }
        virtual ~OpParams()
        {
        }

        bool hasBiasReduction() const
        {
            return m_HasBiasReduction;
        }
        void setHasBiasReduction(bool br)
        {
            m_HasBiasReduction = br;
        }

        NvS32 axis() const
        {
            return m_axis;
        }
        void setAxis(NvS32 a)
        {
            m_axis = a;
        }

        virtual Weights rawBiasData() const
        {
            return m_raw_bias_data;
        }
        virtual void setRawBiasData(Weights raw)
        {
            m_raw_bias_data = raw;
        }

        virtual Weights DLABiasData() const
        {
            return m_dla_bias_data;
        }
        virtual void setDLABiasData(Weights trns)
        {
            m_dla_bias_data = trns;
        }

        const Dims4& biasDims() const
        {
            return m_bias_dims;
        }
        void setBiasDims(const Dims4& bd)
        {
            m_bias_dims = bd;
        }

    protected:
        bool m_HasBiasReduction;
        NvS32 m_axis;
        Dims4 m_bias_dims;
        Weights m_raw_bias_data;
        Weights m_dla_bias_data;
    };

    SDPBiasOpNode(canonical_ast::Node* can_node, NvU16 numBatches = 1)
        : SDPNode(numBatches)
    {
        m_engine_op_type = SDP_BIAS;
        m_can_node = can_node;
        m_mb_op_params = new MultiBatchState<OpParams>(numBatches);
        m_sup_in_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_FEATURE_DATA_INT8);
        m_sup_in_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_FEATURE_DATA_FP16);
        m_sup_out_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_FEATURE_DATA_INT8);
        m_sup_out_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_FEATURE_DATA_FP16);
        m_sup_aux_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_BIAS_DATA_INT8);
        m_sup_aux_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_BIAS_DATA_INT16);
        m_sup_aux_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_BIAS_DATA_FP16);
    }
    virtual ~SDPBiasOpNode()
    {
    }

    virtual std::vector<surface::SurfaceFormat> suggestAuxSurfaceFormats(Edge* auxEdge = NULL);

    NvDlaError quantizeBiasToInt8(ConvCoreNode* fusedConv, std::vector<NvF32>& filterScales, std::vector<NvF32>& inTensorScales);
    NvDlaError scaleBiasToInt16(ConvCoreNode* fusedConv, std::vector<NvF32>& filterScales, std::vector<NvF32>& inTensorScales);

    virtual NvDlaError quantizeAuxData();
    virtual NvDlaError handleLowPrecisionConversions();

    NvDlaError performPerKernelRescaling(
        ConvCoreNode* fusedConv,
        std::vector<NvF32>& filterScales,
        std::vector<NvF32>& inTensorScales,
        std::vector<NvF32>& outTensorScales,
        PrecisionCVTParams& outCvt);
    NvDlaError performPerChannelRescaling(
        ConvCoreNode* fusedConv,
        std::vector<NvF32>& filterScales,
        std::vector<NvF32>& inTensorScales,
        std::vector<NvF32>& outTensorScales,
        PrecisionCVTParams& outCvt);

    Node* tryToMergeWithBiasOp(SDPNode* sdp_bias_op);
    Node* tryToMergeWithScaleOp(SDPNode* sdp_scl_op);
    Node* tryToMergeWithBatchNormOp(SDPNode* sdp_bn_op);
    virtual NvDlaError configureSDPSuperOpSubEngine(SDPSuperOpNode* sdpSuperOp, SDPSubEngineType xN) override;
    virtual Node* mergeWithSDPOp(SDPNode* /*other_op*/);
    virtual NvDlaError translateAuxData();
    NvDlaError captureCanonicalBiasData();

    virtual void inheritParams(Node* inheritFrom);
    virtual OpParams& params(NvU16 batchId = 0)
    {
        return m_mb_op_params->batch(batchId);
    }
    virtual void captureCanonicalParams();
    virtual SDPBiasOpNode* clone()
    {
        return new SDPBiasOpNode(*this);
    }
    virtual canonical_ast::Node* canonicalNode() const
    {
        return m_can_node;
    }
    virtual const void* getAuxData(Edge* auxEdge)
    {
        return params().DLABiasData().values;
    }
    virtual Dims4 getAuxDims()
    {
        return params().biasDims();
    }

    virtual NvDlaError verifyEdgePorts()
    {
        if (inputEdges().size() != 1 || auxEdges().size() != 1 || outputEdges().size() != 1)
            return NvDlaError_BadValue;
        else
            return NvDlaSuccess;
    }

    virtual NvDlaError emitOp(Graph*, DLAInterface*, NvU32 op_slot, NvU32 batch_id,
                              DLACommonOpDescAccessor,
                              DLAOperationContainerAccessor,
                              DLASurfaceContainerAccessor);

#ifdef NVDLA_COMPILER_OUTPUT_FOR_PROTOTEST
    virtual NvDlaError emitOp(NvU32 op_slot, NvU32 batch_id,
                              DLAInterface*,
                              DLACommonOpDescAccessor&,
                              DLAOperationContainerAccessor&,
                              DLASurfaceContainerAccessor&,
                              nvdla_prototest_interface::Layer*);
#endif

protected:
    canonical_ast::Node* m_can_node;
    MultiBatchState<OpParams>* m_mb_op_params;
};

// sdp nop to club with conv w/o bias term
class SDPNOPNode : public SDPNode
{
public:
    virtual std::string className() const
    {
        return std::string("SDPNOPNode");
    }

    class OpParams : public SDPEngineParams
    {
    public:
        OpParams()
            : SDPEngineParams()
        {
            SDPSubEngineParams x1params;
            SDPSubEngineParams x2params;
            SDPSubEngineParams yparams;

            x1params.setEnabled(false);
            x2params.setEnabled(false);
            yparams.setEnabled(false);

            setX1Params(x1params);
            setX2Params(x2params);
            setYParams(yparams);
        }
        virtual ~OpParams()
        {
        }

        // add anything relevant to NOP here
    };

    SDPNOPNode(canonical_ast::Node* can_node, NvU16 numBatches = 1)
        : SDPNode(numBatches)
    {
        m_engine_op_type = SDP_NOP;
        m_can_node = can_node;
        m_mb_op_params = new MultiBatchState<OpParams>(numBatches);
        m_sup_in_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_FEATURE_DATA_INT8);
        m_sup_in_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_FEATURE_DATA_FP16);
        m_sup_out_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_FEATURE_DATA_INT8);
        m_sup_out_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_FEATURE_DATA_FP16);
    }
    virtual ~SDPNOPNode()
    {
    }

    virtual void inheritParams(Node* inheritFrom);
    virtual OpParams& params(NvU16 batchId = 0)
    {
        return m_mb_op_params->batch(batchId);
    }
    virtual void captureCanonicalParams();
    virtual SDPNOPNode* clone()
    {
        return new SDPNOPNode(*this);
    }
    virtual canonical_ast::Node* canonicalNode() const
    {
        return m_can_node;
    }

    virtual NvDlaError verifyEdgePorts()
    {
        if (inputEdges().size() != 1 || auxEdges().size() != 0 || outputEdges().size() != 1)
            return NvDlaError_BadValue;
        else
            return NvDlaSuccess;
    }

    virtual NvDlaError emitOp(Graph*, DLAInterface*, NvU32 op_slot, NvU32 batch_id,
                              DLACommonOpDescAccessor,
                              DLAOperationContainerAccessor,
                              DLASurfaceContainerAccessor);

#ifdef NVDLA_COMPILER_OUTPUT_FOR_PROTOTEST
    virtual NvDlaError emitOp(NvU32 op_slot, NvU32 batch_id,
                              DLAInterface*,
                              DLACommonOpDescAccessor&,
                              DLAOperationContainerAccessor&,
                              DLASurfaceContainerAccessor&,
                              nvdla_prototest_interface::Layer*);
#endif

protected:
    canonical_ast::Node* m_can_node;
    MultiBatchState<OpParams>* m_mb_op_params;
};

// SDP Super Op node
class SDPSuperOpNode : public SDPNode
{
public:
    virtual std::string className() const
    {
        return std::string("SDPSuperOpNode");
    }

    class OpParams : public SDPEngineParams
    {
    public:
        OpParams()
            : SDPEngineParams()
        {
            SDPSubEngineParams x1params;
            SDPSubEngineParams x2params;
            SDPSubEngineParams yparams;

            x1params.setEnabled(false);
            x2params.setEnabled(false);
            yparams.setEnabled(false);

            setX1Params(x1params);
            setX2Params(x2params);
            setYParams(yparams);
        }
        virtual ~OpParams()
        {
        }

        const Dims4& multiplierDims(SDPSubEngineType xN) const
        {
            return m_DataParams[xN.e()].multiplier_dims;
        }
        void setMultiplierDims(SDPSubEngineType xN, const Dims4& dims)
        {
            m_DataParams[xN.e()].multiplier_dims = dims;
        }

        const Dims4& adderDims(SDPSubEngineType xN) const
        {
            return m_DataParams[xN.e()].adder_dims;
        }
        void setAdderDims(SDPSubEngineType xN, const Dims4& dims)
        {
            m_DataParams[xN.e()].adder_dims = dims;
        }

        const Dims4& dlaDataDims(SDPSubEngineType xN) const
        {
            return m_DataParams[xN.e()].dla_data_dims;
        }
        void setDLADataDims(SDPSubEngineType xN, const Dims4& dims)
        {
            m_DataParams[xN.e()].dla_data_dims = dims;
        }

        const Weights& rawMultiplierData(SDPSubEngineType xN) const
        {
            return m_DataParams[xN.e()].multiplier_data;
        }
        void setRawMultiplierData(SDPSubEngineType xN, const Weights& data)
        {
            m_DataParams[xN.e()].multiplier_data = data;
        }

        const Weights& rawAdderData(SDPSubEngineType xN) const
        {
            return m_DataParams[xN.e()].adder_data;
        }
        void setRawAdderData(SDPSubEngineType xN, const Weights& data)
        {
            m_DataParams[xN.e()].adder_data = data;
        }

        const Weights& dlaData(SDPSubEngineType xN) const
        {
            return m_DataParams[xN.e()].dla_data;
        }
        void setDLAData(SDPSubEngineType xN, const Weights& data)
        {
            m_DataParams[xN.e()].dla_data = data;
        }

        TensorType auxDataType(SDPSubEngineType xN) const
        {
            return m_DataParams[xN.e()].data_type;
        }
        void setAuxDataType(SDPSubEngineType xN, TensorType dt)
        {
            m_DataParams[xN.e()].data_type = dt;
        }

        const std::vector<surface::SurfaceFormat> auxSurfaceFormats(SDPSubEngineType xN) const
        {
            return m_DataParams[xN.e()].aux_surface_formats;
        }
        void setAuxSurfaceFormats(SDPSubEngineType xN, const std::vector<surface::SurfaceFormat> data)
        {
            m_DataParams[xN.e()].aux_surface_formats = data;
        }

    protected:
        class DataParams
        {
        public:
            TensorType data_type;

            Weights multiplier_data;
            Dims4 multiplier_dims;

            Weights adder_data;
            Dims4 adder_dims;

            Weights dla_data;
            Dims4 dla_data_dims;

            std::vector<surface::SurfaceFormat> aux_surface_formats;
        };
        // X1, X2 and Y engine params
        std::array<DataParams, 3> m_DataParams;
    };

protected:
    canonical_ast::Node* m_can_node;
    MultiBatchState<OpParams>* m_mb_op_params;
    std::map<NvU8, engine_ast::Edge*> m_sdpXengineToAuxEdgeMap;
    typedef std::map<NvU8, engine_ast::Edge*>::iterator SdpXengineToEdgeMapIterator;
    engine_ast::Edge* m_InputEdge;

    NvDlaError translateAuxDataInternal(SDPSubEngineType xN, SDPSubEngineParams& xParams);
    SdpXengineToEdgeMapIterator findSdpAuxEdge(Edge* edge);
    void printSdpXEdgeMap();
    SDPSubEngineParams* subEngineParams(SDPSubEngineType xN);

public:
    SDPSuperOpNode(canonical_ast::Node* can_node, NvU16 numBatches)
        : SDPNode(numBatches)
    {
        m_engine_op_type = SDP_SUPER;
        m_can_node = can_node;
        m_mb_op_params = new MultiBatchState<OpParams>(numBatches);
        m_sup_in_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_FEATURE_DATA_INT8);
        m_sup_in_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_FEATURE_DATA_FP16);
        m_sup_out_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_FEATURE_DATA_INT8);
        m_sup_out_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_FEATURE_DATA_FP16);
        m_sup_aux_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_BIAS_DATA_INT8);
        m_sup_aux_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_BIAS_DATA_INT16);
        m_sup_aux_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_BIAS_DATA_FP16);
        m_sup_aux_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_SCALE_DATA_INT8);
        m_sup_aux_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_SCALE_DATA_INT16);
        m_sup_aux_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_SCALE_DATA_FP16);
        m_sup_aux_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_BATCH_NORM_DATA_INT8);
        m_sup_aux_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_BATCH_NORM_DATA_INT16);
        m_sup_aux_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_BATCH_NORM_DATA_FP16);
        m_InputEdge = NULL;
    }
    virtual ~SDPSuperOpNode()
    {
    }

    virtual NvDlaError preProcessAuxData();
    virtual NvDlaError translateAuxData();

    virtual std::vector<surface::SurfaceFormat> suggestAuxSurfaceFormats(Edge* auxEdge = NULL);

    bool debugFactorization()
    {
        return false;
    }
    virtual void inheritParams(Node* inheritFrom);
    void inheritParamsForSubEngine(SDPSuperOpNode* otherSuperOp, SDPSubEngineType xN);
    virtual OpParams& params(NvU16 batchId = 0)
    {
        return m_mb_op_params->batch(batchId);
    }

    virtual NvDlaError captureCanonicalData(SDPSubEngineType xN, TensorType tN);
    //virtual void captureCanonicalParams();
    virtual SDPSuperOpNode* clone()
    {
        return new SDPSuperOpNode(*this);
    }
    virtual canonical_ast::Node* canonicalNode() const
    {
        return m_can_node;
    }
    virtual const void* getAuxData(Edge* auxEdge);

    void markSdpAuxEdge(SDPSubEngineType xN, Edge* edge)
    {
        m_sdpXengineToAuxEdgeMap[xN.e()] = edge;
    }
    NvDlaError auxEdgeBySubEngine(SDPSubEngineType xN, Edge** ret_edge);

    virtual NvDlaError populateEdgePorts();
    virtual NvDlaError verifyEdgePorts();
    virtual NvDlaError emitOp(Graph*, DLAInterface*, NvU32 op_slot, NvU32 batch_id,
                              DLACommonOpDescAccessor,
                              DLAOperationContainerAccessor,
                              DLASurfaceContainerAccessor);
    virtual NvU64 suggestSurfaceSize(surface::TensorSurfaceDesc* tsd);

#ifdef NVDLA_COMPILER_OUTPUT_FOR_PROTOTEST
    virtual NvDlaError emitOp(NvU32 op_slot, NvU32 batch_id,
                              DLAInterface*,
                              DLACommonOpDescAccessor&,
                              DLAOperationContainerAccessor&,
                              DLASurfaceContainerAccessor&,
                              nvdla_prototest_interface::Layer*);
#endif
};

// pooling
class PDPNode : public Node
{
public:
    virtual std::string className() const
    {
        return std::string("PDPNode");
    }

    class OpParams : public PDPEngineParams
    {
    public:
        OpParams()
            : PDPEngineParams()
        {
            m_pdpOnFlying = false;
        }
        virtual ~OpParams()
        {
        }

        bool isPDPOnFlying() const
        {
            return m_pdpOnFlying;
        }
        void setPDPFlyingMode(bool isOnFly)
        {
            m_pdpOnFlying = isOnFly;
        }

    protected:
        bool m_pdpOnFlying;
    };

    PDPNode(canonical_ast::PoolingNode* pool, NvU16 numBatches = 1)
        : Node(numBatches)
    {
        m_engine_type = PDP;
        m_engine_op_type = PDP_POOL;
        m_can_pool_node = pool;
        m_mb_op_params = new MultiBatchState<OpParams>(numBatches);
        m_sup_in_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_FEATURE_DATA_INT8);
        m_sup_in_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_FEATURE_DATA_FP16);
        m_sup_out_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_FEATURE_DATA_INT8);
        m_sup_out_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_FEATURE_DATA_FP16);
    }
    virtual ~PDPNode()
    {
    }

    virtual NvDlaError fuseOnTheFlyNodes();
    virtual NvDlaError splitNodes();
    virtual NvDlaError handleMultiBatch();
    void adjustBRPadding();
    NvDlaError pdpSWSplit();
    NvDlaError pdpHWSplitWidth();
    NvDlaError verifySplitWInfo(engine_ast::PDPEngineParams::hwSplitWidthInfo);

    OpParams& params(NvU16 batchId = 0)
    {
        return m_mb_op_params->batch(batchId);
    }
    virtual PDPNode* clone()
    {
        return new PDPNode(*this);
    }
    virtual canonical_ast::PoolingNode* canonicalNode() const
    {
        return m_can_pool_node;
    }
    virtual void captureCanonicalParams();

    virtual NvDlaError verifyEdgePorts()
    {
        if (inputEdges().size() != 1 || auxEdges().size() != 0 || outputEdges().size() != 1)
            return NvDlaError_BadValue;
        else
            return NvDlaSuccess;
    }
    virtual NvDlaError verifySurfaceDims(surface::TensorSurfaceDesc*);

    virtual NvDlaError emitOp(Graph*, DLAInterface*, NvU32 op_slot, NvU32 batch_id,
                              DLACommonOpDescAccessor,
                              DLAOperationContainerAccessor,
                              DLASurfaceContainerAccessor);

#ifdef NVDLA_COMPILER_OUTPUT_FOR_PROTOTEST
    virtual NvDlaError emitOp(NvU32 op_slot, NvU32 batch_id,
                              DLAInterface*,
                              DLACommonOpDescAccessor&,
                              DLAOperationContainerAccessor&,
                              DLASurfaceContainerAccessor&,
                              nvdla_prototest_interface::Layer*);
#endif

protected:
    NvU16 calculateMaxWidth(surface::SurfacePrecision, NvU16, NvU16);

    canonical_ast::PoolingNode* m_can_pool_node;
    MultiBatchState<OpParams>* m_mb_op_params;
};

// lrn
class CDPNode : public Node
{
public:
    virtual std::string className() const
    {
        return std::string("CDPNode");
    }

    class OpParams : public CDPEngineParams
    {
    public:
        OpParams()
            : CDPEngineParams()
        {
        }
        virtual ~OpParams()
        {
        }
    };

    CDPNode(NvU16 numBatches = 1)
        : Node(numBatches)
    {
        m_engine_type = CDP;
    }
    virtual ~CDPNode()
    {
    }

    virtual void captureCanonicalParams();
    virtual CDPNode* clone()
    {
        return new CDPNode(*this);
    }

    virtual NvDlaError handleMultiBatch();

    virtual NvDlaError verifyEdgePorts()
    {
        if (inputEdges().size() != 1 || auxEdges().size() != 0 || outputEdges().size() != 1)
            return NvDlaError_BadValue;
        else
            return NvDlaSuccess;
    }
    virtual NvDlaError verifySurfaceDims(surface::TensorSurfaceDesc*);
};

// lrn
class CDPLRNOpNode : public CDPNode
{
public:
    virtual std::string className() const
    {
        return std::string("CDPLRNOpNode");
    }

    class OpParams : public CDPEngineParams
    {
    public:
        OpParams()
            : CDPEngineParams()
        {
        }
        virtual ~OpParams()
        {
        }
    };

    CDPLRNOpNode(canonical_ast::LRNNode* can_lrn_node, NvU16 numBatches = 1)
        : CDPNode(numBatches)
    {
        m_engine_op_type = CDP_LRN;
        m_can_lrn_node = can_lrn_node;
        m_mb_op_params = new MultiBatchState<OpParams>(numBatches);
        m_hLut = -1;
        m_sup_in_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_FEATURE_DATA_FP16);
        m_sup_out_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_FEATURE_DATA_FP16);
    }
    virtual ~CDPLRNOpNode()
    {
    }

    OpParams& params(NvU16 batchId = 0)
    {
        return m_mb_op_params->batch(batchId);
    }
    virtual void captureCanonicalParams();
    virtual CDPLRNOpNode* clone()
    {
        return new CDPLRNOpNode(*this);
    }
    virtual canonical_ast::LRNNode* canonicalNode() const
    {
        return m_can_lrn_node;
    }

    virtual NvDlaError verifyEdgePorts()
    {
        if (inputEdges().size() != 1 || auxEdges().size() != 0 || outputEdges().size() != 1)
            return NvDlaError_BadValue;
        else
            return NvDlaSuccess;
    }

    virtual NvDlaError emitOp(Graph*, DLAInterface*, NvU32 op_slot, NvU32 batch_id,
                              DLACommonOpDescAccessor,
                              DLAOperationContainerAccessor,
                              DLASurfaceContainerAccessor);

#ifdef NVDLA_COMPILER_OUTPUT_FOR_PROTOTEST
    virtual NvDlaError emitOp(NvU32 op_slot, NvU32 batch_id,
                              DLAInterface*,
                              DLACommonOpDescAccessor&,
                              DLAOperationContainerAccessor&,
                              DLASurfaceContainerAccessor&,
                              nvdla_prototest_interface::Layer*);
#endif

protected:
    canonical_ast::LRNNode* m_can_lrn_node;
    MultiBatchState<OpParams>* m_mb_op_params;
    LutManager::LutHandle m_hLut;
};

// deconvolve 2nd
class RubikNode : public Node
{
public:
    virtual std::string className() const
    {
        return std::string("RubikNode");
    }

    class OpParams : public RubikEngineParams
    {
    public:
        OpParams()
            : RubikEngineParams()
        {
        }
        virtual ~OpParams()
        {
        }
        // add anything relevant to rubik here
    };

    RubikNode(canonical_ast::DeconvolutionNode* deconv, NvU16 numBatches = 1)
        : Node(numBatches)
    {
        m_engine_type = RUBIK;
        m_engine_op_type = RUBIK_DECONV;
        m_can_deconv_node = deconv;
        m_mb_op_params = new MultiBatchState<OpParams>(numBatches);
        m_sup_in_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_FEATURE_DATA_INT8);
        m_sup_in_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_FEATURE_DATA_FP16);
        m_sup_out_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_FEATURE_DATA_INT8);
        m_sup_out_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_FEATURE_DATA_FP16);
    }
    virtual ~RubikNode()
    {
    }

    virtual Dims4 suggestSurfaceDims(surface::TensorSurfaceDesc* tsd);
    virtual NvU32 suggestLineStride(surface::TensorSurfaceDesc* tsd);
    virtual NvU32 suggestSurfaceStride(surface::TensorSurfaceDesc* tsd);
    virtual NvU64 suggestSurfaceSize(surface::TensorSurfaceDesc* tsd);
    virtual NvDlaError handleMultiBatch();

    NvDlaError determineContractOpParams();

    inline bool debugRubik() const
    {
        return true;
    }

    OpParams& params(NvU16 batchId = 0)
    {
        return m_mb_op_params->batch(batchId);
    }
    virtual void captureCanonicalParams();
    virtual RubikNode* clone()
    {
        return new RubikNode(*this);
    }
    virtual canonical_ast::DeconvolutionNode* canonicalNode() const
    {
        return m_can_deconv_node;
    }

    virtual NvDlaError emitOp(Graph*, DLAInterface*, NvU32 op_slot, NvU32 batch_id,
                              DLACommonOpDescAccessor,
                              DLAOperationContainerAccessor,
                              DLASurfaceContainerAccessor);

protected:
    canonical_ast::DeconvolutionNode* m_can_deconv_node;
    MultiBatchState<OpParams>* m_mb_op_params;
};

// bdma - single transfer
// bdma - group transfer
class BDMANode : public Node
{
public:
    virtual std::string className() const
    {
        return std::string("BDMANode");
    }

    BDMANode()
        : Node()
    {
        m_engine_type = BDMA;
        m_sup_in_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_FEATURE_DATA_FP16);
        m_sup_out_surf_formats.push_back(surface::SurfaceFormatEnum::NVDLA_FEATURE_DATA_FP16);
    }
    virtual ~BDMANode()
    {
    }

    virtual BDMANode* clone()
    {
        return new BDMANode(*this);
    }
    virtual canonical_ast::Node* canonicalNode() const
    {
        return NULL;
    }

    BDMAEngineParams calcTransferParams(surface::TensorSurfaceDesc*, surface::TensorSurfaceDesc*);
};

class BDMASingleDMAOpNode : public BDMANode
{
public:
    virtual std::string className() const
    {
        return std::string("BDMASingleDMAOpNode");
    }

    class OpParams : public BDMAEngineParams
    {
    public:
        OpParams()
            : BDMAEngineParams()
        {
        }
        virtual ~OpParams()
        {
        }

        // anything relevant to single bdma op
    };

    BDMASingleDMAOpNode()
        : BDMANode()
    {
        m_engine_op_type = BDMA_SINGLE_DMA;
    }

    OpParams& getTransferParams()
    {
        return m_params;
    }
    void setTransferParams(OpParams trns_params)
    {
        m_params = trns_params;
    }

    virtual NvDlaError emitOp(Graph*, DLAInterface*, NvU32, NvU32,
                              DLACommonOpDescAccessor,
                              DLAOperationContainerAccessor,
                              DLASurfaceContainerAccessor);

protected:
    OpParams m_params; // note: bdma doesnt need multi batch support
};

class BDMAGroupDMAOpNode : public BDMANode
{
public:
    virtual std::string className() const
    {
        return std::string("BDMAGroupDMAOpNode");
    }

    class OpParams : public BDMAEngineParams
    {
    public:
        OpParams()
            : BDMAEngineParams()
        {
        }
        virtual ~OpParams()
        {
        }

        // anything relevant to group BDMA goes here
    };

    BDMAGroupDMAOpNode()
        : BDMANode()
    {
        m_engine_op_type = BDMA_GROUP_DMA;
    }

    std::vector<OpParams>& getTransferParams()
    {
        return m_params;
    }
    void setTransferParams(std::vector<OpParams> trns_params)
    {
        m_params = trns_params;
    }

protected:
    std::vector<OpParams> m_params; // note: bdma doesnt need multi batch support
};

class Edge
{
public:
    virtual std::string className() const
    {
        return std::string("Edge");
    }

    Edge(canonical_ast::Edge* can_edge)
        : m_containing_graph(NULL),
          m_can_edge(can_edge),
          m_original_tensor(NULL),
          m_tensor_surface_desc(NULL),
          m_bindId(-1),         // non-bindable
          m_bindDomain(IOD_Max) // n/a
    {
        m_unique_id = m_next_id++;
    }
    virtual ~Edge()
    {
    }

    static inline bool debugBinding()
    {
        return true;
    }

    const std::string id() const
    {
        return m_id;
    }
    void setId(const std::string id)
    {
        m_id = id;
    }

    NvU32 uniqueId() const
    {
        return m_unique_id;
    }

    Graph* graph()
    {
        return m_containing_graph;
    }
    void setGraph(Graph* g)
    {
        m_containing_graph = g;
    }

    canonical_ast::Edge* canonicalEdge() const
    {
        return m_can_edge;
    }

    inline EdgeType edgeType() const
    {
        return m_edge_type;
    }

    inline void setEdgeType(EdgeType et)
    {
        m_edge_type = et;
    }

    inline bool isComputeEdge() const
    {
        return m_edge_type.v() == EdgeTypeEnum::COMPUTE;
    }
    inline void setComputeEdge()
    {
        setEdgeType(engine_ast::EdgeTypeEnum::COMPUTE);
    }

    inline bool isDataEdge() const
    {
        return m_edge_type.v() == EdgeTypeEnum::DATA;
    }
    inline void setDataEdge()
    {
        setEdgeType(engine_ast::EdgeTypeEnum::DATA);
    }

    inline bool isHazardEdge() const
    {
        return m_edge_type.v() == EdgeTypeEnum::HAZARD;
    }
    inline void setHazardEdge()
    {
        setEdgeType(engine_ast::EdgeTypeEnum::HAZARD);
    }

    inline bool isAuxEdge() const
    {
        bool retval = false;
        if (isDataEdge() && originalTensor())
        {
            nvdla::TensorType tt = originalTensor()->getTensorType();
            retval = (tt == nvdla::TensorType::kBATCH_NORM || tt == nvdla::TensorType::kBIAS || tt == nvdla::TensorType::kSCALE || tt == nvdla::TensorType::kWEIGHT);
        }
        return retval;
    }

    Tensor* originalTensor() const
    {
        return m_original_tensor;
    }
    void setOriginalTensor(Tensor* tensor)
    {
        m_original_tensor = tensor;
    }

    NvDlaError registerSurface();
    NvDlaError determineSurfaceClients();
    NvDlaError determineSurfaceFormat();
    NvDlaError determineSurfaceStrides();
    NvDlaError determineSurfaceSize();
    NvDlaError determineSurfaceOffsetInBuffer();
    NvDlaError registerBuffer();
    NvDlaError reserveBuffer();
    NvDlaError handleMultiBatch();

    NvDlaError verifySurfaceClients();
    NvDlaError verifySurfaceFormat();
    NvDlaError verifySurfaceDims();
    NvDlaError verifySurfaceStrides();
    NvDlaError verifySurfaceSize();
    NvDlaError verifySurfaceOffsetInBuffer();
    NvDlaError verifySurfaceTensorScales();
    NvDlaError verifySurface();
    NvDlaError verifyBuffer();

    virtual Edge* clone()
    {
        return new Edge(*this);
    }

    static std::string prettyId(const Edge* e, NvU8 flags = ast::PrettyId_All)
    {
        if (!e) { return std::string("(null)"); }
        std::stringstream r;
        std::string sep("");
        if (flags & ast::PrettyId_Id)
        {
            r << sep << e->id();
            sep = " ";
        }
        if (flags & ast::PrettyId_ClassName)
        {
            r << sep << e->className();
            sep = " ";
        }
        if (flags & ast::PrettyId_Type)
        {
            r << sep << e->edgeType().c_str();
            sep = " ";
        }
        return r.str();
    }

    bool bindable() const
    {
        bool isBindable = m_bindId >= 0;
        if (debugBinding())
        {
            gLogInfo << "::Edge edge=" << m_id << " bindable=" << isBindable << std::endl;
        }
        return isBindable;
    }
    NvS16 bindId() const
    {
        if (debugBinding())
        {
            gLogInfo << "::Edge edge=" << m_id << " bindid=" << m_bindId << std::endl;
        }
        return m_bindId;
    }

    IOD bindDomain() const
    {
        return m_bindDomain;
    }

    NvS16 bindId(enum IOD& bindDomain) const
    {
        if (debugBinding())
        {
            gLogInfo << "::Edge edge=" << m_id << " domain=" << (int)m_bindDomain << " bind_id=" << m_bindId << std::endl;
        }
        bindDomain = m_bindDomain;
        return m_bindId;
    }
    void setBindId(NvS16 id, enum IOD bindDomain)
    {
        if (debugBinding())
        {
            gLogInfo << "::Edge setBindId edge=" << m_id << " domain=" << (int)bindDomain << " id=" << id << std::endl;
        }
        m_bindDomain = bindDomain;
        m_bindId = id;
    }

    surface::TensorSurfaceDesc* tensorSurfaceDesc() const
    {
        return m_tensor_surface_desc;
    }
    void setTensorSurfaceDesc(surface::TensorSurfaceDesc* t)
    {
        m_tensor_surface_desc = t;
    }
    memory::TensorBufferDesc* tensorBufferDesc() const
    {
        return m_tensor_surface_desc ? m_tensor_surface_desc->tensorBufferDesc() : 0;
    }

    /* for clone */
    Edge(const Edge& other)
        : m_id(other.m_id + std::string("-cloned")),
          m_unique_id(m_next_id++),
          m_containing_graph(0),
          m_can_edge(other.m_can_edge),
          m_edge_type(other.m_edge_type),
          m_original_tensor(other.m_original_tensor),
          m_tensor_surface_desc(other.m_tensor_surface_desc),
          m_bindId(other.m_bindId),
          m_bindDomain(other.m_bindDomain)
    {
        // of all the attributes, you don't want the cloned edge to inherit
        // the containing_graph from orig edge. The cloned edge should belong
        // to the new graph under construction. (see AST.h:Graph(const Graph& other)
    }

protected:
    std::string m_id;  // unique within the graph
    NvU32 m_unique_id; // id for graph ordering. u32 instead of string.
    static NvU32 m_next_id;
    Graph* m_containing_graph;
    canonical_ast::Edge* m_can_edge;
    EdgeType m_edge_type; // compute or data
    Tensor* m_original_tensor;
    surface::TensorSurfaceDesc* m_tensor_surface_desc;
    NvS16 m_bindId;
    enum IOD m_bindDomain;
};

class IsSameCanonicalEdge
{
public:
    IsSameCanonicalEdge(canonical_ast::Edge* can_edge)
        : m_can_edge(can_edge)
    {
    }
    bool operator()(engine_ast::Edge* engEdge)
    {
        if (engEdge->canonicalEdge() == m_can_edge)
        {
            return true;
        }
        return false;
    }

protected:
    canonical_ast::Edge* m_can_edge;
};

class NodeWithSameEngineType
{
public:
    NodeWithSameEngineType(EngineType et)
        : m_engine_type(et)
    {
    }
    bool operator()(engine_ast::Node* test_node)
    {
        if (m_engine_type == test_node->engineType())
        {
            return true;
        }
        return false;
    }

protected:
    EngineType m_engine_type;
};

Graph* generateGraph(Profile*, TargetConfig*, canonical_ast::Graph*);

// certain can nodes get transformed into more than one engine nodes because
// of h/w limitation. eg: canonical_conv = eng_conv + eng_sdp-nop (if no bias term)
NvDlaError transformCanNode(
    Graph* engGraph,
    canonical_ast::Node* canNode,
    Graph::EdgeSequence engSrcEdges,
    Graph::EdgeSequence engSinkEdges,
    Graph::NodeSequence& transformedEngNodes);

// for sending to debug tools
std::ostream& outputJson(Graph*, std::ostream&);
std::ostream& outputJson(Graph*, Edge*, std::ostream&);
std::ostream& outputJson(Graph*, Node*, std::ostream&);

bool serializeTo(WisdomContainerEntry*);
bool deserializeFrom(WisdomContainerEntry*);
//    virtual bool assignSymbols(Wisdom *);

canonical_ast::Graph* mirrorEdges(Graph*, canonical_ast::Graph*, BiMap<canonical_ast::Edge*, Edge*>& edge_correlation);
void produceNodes(Graph*, canonical_ast::Graph*);

class NodeFactory
{
public:
    static void clearMaps(void);
    static ConvCoreConvolutionOpNode* newConvCoreConvolutionOpNode(canonical_ast::ConvolutionNode*, Graph*);
    static ConvCoreFullyConnectedOpNode* newConvCoreFullyConnectedOpNode(canonical_ast::FullyConnectedNode*, Graph*);
    static ConvCoreDeconvolutionOpNode* newConvCoreDeconvolutionOpNode(canonical_ast::DeconvolutionNode*, Graph*);
    static SDPScaleOpNode* newSDPScaleOpNode(canonical_ast::ScaleNode*, Graph*);
    static SDPBatchNormOpNode* newSDPBatchNormOpNode(canonical_ast::BatchNormNode*, Graph*);
    static SDPActivationOpNode* newSDPActivationOpNode(canonical_ast::ActivationNode*, Graph*);
    static SDPElementWiseOpNode* newSDPElementWiseOpNode(canonical_ast::ElementWiseNode*, Graph*);
    static SDPBiasOpNode* newSDPBiasOpNode(canonical_ast::Node*, Graph*);
    static SDPNOPNode* newSDPNOPNode(canonical_ast::Node*, Graph*);
    static SDPSuperOpNode* newSDPSuperOpNode(canonical_ast::Node*, Graph*);
    static PDPNode* newPDPNode(canonical_ast::PoolingNode*, Graph*);
    static CDPLRNOpNode* newCDPLRNOpNode(canonical_ast::LRNNode*, Graph*);
    static CPUScaleOpNode* newCPUScaleOpNode(canonical_ast::ScaleNode*, Graph*);
    static CPUSoftMaxOpNode* newCPUSoftMaxOpNode(canonical_ast::SoftMaxNode*, Graph*);
    static RubikNode* newRubikNode(canonical_ast::DeconvolutionNode*, Graph*);
    static ConcatenationNode* newConcatNode(canonical_ast::ConcatenationNode*, Graph*);
    static SplitNode* newSplitNode(canonical_ast::SplitNode*, Graph*);
    static BDMASingleDMAOpNode* newSingleBDMANode(Graph*);
    static BDMAGroupDMAOpNode* newGroupBDMANode(Graph*);
    static MultiOpsNode* newMultiOpsNode(Graph::NodeSequence&, Graph*);

    template<typename T>
    static T nodeCast(Node*);

protected:
    static std::map<Node*, ConvCoreConvolutionOpNode*> s_conv_conv_priv;
    static std::map<Node*, ConvCoreFullyConnectedOpNode*> s_conv_fc_priv;
    static std::map<Node*, ConvCoreDeconvolutionOpNode*> s_conv_deconv_priv;
    static std::map<Node*, SDPScaleOpNode*> s_sdp_scale_priv;
    static std::map<Node*, SDPBatchNormOpNode*> s_sdp_bn_priv;
    static std::map<Node*, SDPActivationOpNode*> s_sdp_act_priv;
    static std::map<Node*, SDPElementWiseOpNode*> s_sdp_ew_priv;
    static std::map<Node*, SDPBiasOpNode*> s_sdp_bias_priv;
    static std::map<Node*, SDPNOPNode*> s_sdp_nop_priv;
    static std::map<Node*, SDPSuperOpNode*> s_sdp_super_priv;
    static std::map<Node*, PDPNode*> s_pdp_priv;
    static std::map<Node*, CDPLRNOpNode*> s_cdp_lrn_priv;
    static std::map<Node*, CPUScaleOpNode*> s_cpu_scale_priv;
    static std::map<Node*, CPUSoftMaxOpNode*> s_cpu_sm_priv;
    static std::map<Node*, RubikNode*> s_rubik_priv;
    static std::map<Node*, ConcatenationNode*> s_concat_priv;
    static std::map<Node*, SplitNode*> s_split_priv;
    static std::map<Node*, BDMASingleDMAOpNode*> s_single_bdma_priv;
    static std::map<Node*, BDMAGroupDMAOpNode*> s_group_bdma_priv;
    static std::map<Node*, MultiOpsNode*> s_multi_ops_priv;
};

// the default generate() produces a depth first dependency ordering.
class ScoredDependencyOrdering : public ast::ScoredGraphOrdering<engine_ast::Graph>
{
public:
    ScoredDependencyOrdering(engine_ast::Graph* g)
        : ast::ScoredGraphOrdering<engine_ast::Graph>(g)
    {
    }
    virtual ~ScoredDependencyOrdering()
    {
    }

protected:
};

// this class flattens the scores out, but retains the original order given by the scored ordering.
// we treat this as the front-end in that it pulls through from the scored order.
class DependencyOrdering : public ast::GraphOrdering<engine_ast::Graph>
{
public:
    DependencyOrdering(ScoredDependencyOrdering* sdo)
        : ast::GraphOrdering<engine_ast::Graph>(sdo->graph()), m_sdo(sdo)
    {
    }
    virtual ~DependencyOrdering()
    {
    }
    virtual NvDlaError generate();
    virtual void clear();
    const EdgeSequence& dataEdgeOrder() const
    {
        return m_data_edge_order;
    }
    const EdgeSequence& computeEdgeOrder() const
    {
        return m_compute_edge_order;
    }
    const EdgeSequence& hazardEdgeOrder() const
    {
        return m_hazard_edge_order;
    }

    ScoredDependencyOrdering* scoredOrdering()
    {
        return m_sdo;
    }

protected:
    ScoredDependencyOrdering* m_sdo;
    EdgeSequence m_data_edge_order;
    EdgeSequence m_compute_edge_order;
    EdgeSequence m_hazard_edge_order;
};

class AddCopyOutDebugBDMA : public ast::GraphVisitor<engine_ast::Graph>
{
public:
    AddCopyOutDebugBDMA()
        : ast::GraphVisitor<engine_ast::Graph>()
    {
    }
    virtual ~AddCopyOutDebugBDMA()
    {
    }

    // only visiting nodes for this...
    virtual NvDlaError visitBegin(engine_ast::Graph*);
    virtual NvDlaError visitElem(engine_ast::Graph::Elem)
    {
        return NvDlaError_InvalidState;
    }
    virtual NvDlaError visitEdge(engine_ast::Edge*)
    {
        return NvDlaError_InvalidState;
    }
    virtual NvDlaError visitNode(engine_ast::Node*);
    virtual NvDlaError visitEnd(engine_ast::Graph*, NvDlaError ve)
    {
        if (ve == NvDlaSuccess)
        {
            m_graph->ordering()->generate();
        }
        return ve;
    }

protected:
    engine_ast::Graph* m_graph;
    NvS16 m_debugBindId;

    inline bool debugCopyOutDebug()
    {
        return true;
    }
};

}; // namespace engine_ast

namespace memory {

class MemoryResolver : public ast::GraphVisitor<engine_ast::Graph>
{
public:
    MemoryResolver(); // engine_ast::Graph *g);
    virtual ~MemoryResolver()
    {
    }

    // only visiting nodes for this...
    virtual NvDlaError visitBegin(engine_ast::Graph*);
    virtual NvDlaError visitElem(engine_ast::Graph::Elem)
    {
        return NvDlaError_InvalidState;
    }
    virtual NvDlaError visitEdge(engine_ast::Edge*)
    {
        return NvDlaError_InvalidState;
    }
    virtual NvDlaError visitNode(engine_ast::Node*);
    virtual NvDlaError visitEnd(engine_ast::Graph*, NvDlaError ve);

    // records earliest annotationId a deallocation becomes possible for a surface.
    typedef std::pair<int, surface::TensorSurfaceDesc*> DeallocableSurface;
    typedef std::pair<surface::TensorSurfaceDesc*, surface::TensorSurfaceDesc*> SurfacePair;

protected:
    NvDlaError findReuseHazards(engine_ast::Node*, surface::TensorSurfaceDesc*, std::vector<SurfacePair>& collisions);
    NvDlaError resolveHazards(engine_ast::Node*, std::vector<SurfacePair>& collisions);

    NvDlaError tryAllocInsidePool(engine_ast::Node* node,
                                  surface::TensorSurfaceDesc* tsd,
                                  memory::TensorBufferDesc* tbd,
                                  std::vector<memory::Pool*>& tryPools,
                                  bool isAUX, bool& retry);
    NvDlaError tryAllocOutsidePool(engine_ast::Node* node,
                                   surface::TensorSurfaceDesc* tsd,
                                   memory::TensorBufferDesc* tbd,
                                   bool isAUX);
    NvDlaError resolveSurfacePlacement(engine_ast::Node* node,
                                       surface::TensorSurfaceDesc* tsd,
                                       memory::TensorBufferDesc* tbd,
                                       bool isAUX, bool& retry,
                                       bool allowFallback);

    NvDlaError placeSurfaceContent(engine_ast::Node* node, surface::TensorSurfaceDesc* tsd);

    //note, these are snapshotted at visitBegin(...)
    bool m_useMemPool;
    bool m_useReusePooledMemory;
    bool m_useGreedyEviction;
    bool m_useCVSRAM;

    std::vector<memory::Pool>* m_pools;
    memory::Pool* m_localCVSRAM;
    memory::Pool* m_localSDRAM;
    memory::Pool* m_globalSDRAM;
    bool m_debug;

    std::list<DeallocableSurface> m_deallocated;
    std::set<surface::TensorSurfaceDesc*> m_inLocalPool;
};

}; // namespace memory

}; // namespace priv

}; // namespace nvdla

#endif // NVDLA_PRIV_ENGINE_AST_H
