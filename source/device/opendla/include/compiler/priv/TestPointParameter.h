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

#ifndef NVDLA_PRIV_TEST_POINT_PARAMETER_H
#define NVDLA_PRIV_TEST_POINT_PARAMETER_H

#include <vector>

#include "dlatypes.h"

#define GEN_ENUM(X, N) X = N,
#define GEN_STR(X, N)  #X,

#include "TestPointParameterEnum.h"

namespace nvdla {
namespace priv {

//
// A "setting" is a specific value as applied to a given "parameter".

//
// Wrapper class to represent parameters made up of enumerations.
// Here the enumerations are required to be unsigned contiguous
// sequences beginning at 0.  E.g. 0, 1, 2... not flags (0, 1, 2, 4)
// or booleans (flags)
//
template<typename EnumClass, typename UnderlyingType = NvU8>
class TestPointEnumParameter
{
public:
    typedef UnderlyingType underlying_type;

protected:
    underlying_type m_e;
    TestPointEnumParameter(underlying_type v)
        : m_e(v)
    {
    }

    static const char* s_c_str;          // class name str
    static const char* const s_c_strs[]; // class enum strs
    static const size_t s_num_elements;

public:
    static const char* parameter_name_c_str()
    {
        return s_c_str;
    }
    const char* c_str()
    {
        return s_c_strs[m_e];
    }

    underlying_type v()
    {
        return m_e;
    }
    EnumClass e()
    {
        return EnumClass(m_e);
    }
    bool valid()
    {
        return m_e < s_num_elements;
    }

    TestPointEnumParameter(EnumClass p)
        : m_e(p)
    {
    }
};

class BatchMode
{
public:
    enum Enum
    {
        BATCH_MODE_ENUMS(GEN_ENUM)
    };
};
typedef TestPointEnumParameter<BatchMode::Enum> BatchModeParameter;

class CVSRamSize
{
public:
    enum Enum
    {
        CVSRAM_SIZE_ENUMS(GEN_ENUM)
    };
};
typedef TestPointEnumParameter<CVSRamSize::Enum> CVSRamSizeParameter;

class HWLayerTuning
{
public:
    enum Enum
    {
        HW_LAYER_TUNING_ENUMS(GEN_ENUM)
    };
};

class MappingWeights
{
public:
    enum Enum
    {
        MAPPING_WEIGHTS_ENUMS(GEN_ENUM)
    };
};

class Padding
{
public:
    enum Enum
    {
        PADDING_ENUMS(GEN_ENUM)
    };
};

class OutputSequence
{
public:
    enum Enum
    {
        OUTPUT_SEQUENCE_ENUMS(GEN_ENUM)
    };
};

class Dilation
{
public:
    enum Enum
    {
        DILATION_ENUMS(GEN_ENUM)
    };
};

class WeightDensity
{
public:
    enum Enum
    {
        WEIGHT_DENSITY_ENUMS(GEN_ENUM)
    };
};

class FeatureDensity
{
public:
    enum Enum
    {
        FEATURE_DENSITY_ENUMS(GEN_ENUM)
    };
};

class ChannelExtension
{
public:
    enum Enum
    {
        CHANNEL_EXTENSION_ENUMS(GEN_ENUM)
    };
};

class ConvMACRedundancy
{
public:
    enum Enum
    {
        CONV_MAC_REDUNDANCY_ENUMS(GEN_ENUM)
    };
};

class ConvBufBankMgmt
{
public:
    enum Enum
    {
        CONV_BUF_BANK_MGMT_ENUMS(GEN_ENUM)
    };
};

class PDPOpMode
{
public:
    enum Enum
    {
        PDP_OP_MODE_ENUMS(GEN_ENUM)
    };
};

class OffFlyingOpMode
{
public:
    enum Enum
    {
        OFF_FLYING_OP_MODE_ENUMS(GEN_ENUM)
    };
};

class SDPOpMode
{
public:
    enum Enum
    {
        SDP_OP_MODE_ENUMS(GEN_ENUM)
    };
};

class AXIFSched
{
public:
    enum Enum
    {
        AXIF_SCHED_ENUMS(GEN_ENUM)
    };
};

class PixelDataFormat
{
public:
    enum Enum
    {
        PIXEL_DATA_FORMAT_ENUMS(GEN_ENUM)
    };
};

class NetworkForks
{
public:
    enum Enum
    {
        NETWORK_FORKS_ENUMS(GEN_ENUM)
    };
};

typedef TestPointEnumParameter<HWLayerTuning::Enum> HWLayerTuningParameter;
typedef TestPointEnumParameter<MappingWeights::Enum> MappingWeightsParameter;
typedef TestPointEnumParameter<Padding::Enum> PaddingParameter;
typedef TestPointEnumParameter<OutputSequence::Enum> OutputSequenceParameter;
typedef TestPointEnumParameter<Dilation::Enum> DilationParameter;
typedef TestPointEnumParameter<WeightDensity::Enum> WeightDensityParameter;
typedef TestPointEnumParameter<FeatureDensity::Enum> FeatureDensityParameter;
typedef TestPointEnumParameter<ChannelExtension::Enum> ChannelExtensionParameter;
typedef TestPointEnumParameter<ConvMACRedundancy::Enum> ConvMACRedundancyParameter;
typedef TestPointEnumParameter<ConvBufBankMgmt::Enum> ConvBufBankMgmtParameter;
typedef TestPointEnumParameter<PDPOpMode::Enum> PDPOpModeParameter;
typedef TestPointEnumParameter<OffFlyingOpMode::Enum> OffFlyingOpModeParameter;
typedef TestPointEnumParameter<SDPOpMode::Enum> SDPOpModeParameter;
typedef TestPointEnumParameter<AXIFSched::Enum> AXIFSchedParameter;
typedef TestPointEnumParameter<PixelDataFormat::Enum> PixelDataFormatParameter;
typedef TestPointEnumParameter<NetworkForks::Enum> NetworkForksParameter;

// global (system) parameters/constraints
//  e.g.:
//    dla0 - use cvsram
//    dla1 - no cvsram
//

class GlobalParameters
{
public:
    CVSRamSizeParameter m_cvsram_size[2];

    // ...
};

// these show up in the description of the network somewhere
class LayerSettings
{ // i.e. move these to the network layer?
public:
    //    PaddingParameter           m_padding;
    DilationParameter m_dilation;
    PixelDataFormatParameter m_pixel_data_format;
};

class LocalParameters
{
public:
    BatchModeParameter m_batch_mode;
    PaddingParameter m_padding;        // packed vs. padded stride or alignment choices instead?
    CVSRamSizeParameter m_cvsram_size; // discrete set of choices
    ChannelExtensionParameter m_channel_extension;
    HWLayerTuningParameter m_hw_layer_tuning;
    MappingWeightsParameter m_mapping_weights;
    OutputSequenceParameter m_output_sequence;
    WeightDensityParameter m_weight_density;
    FeatureDensityParameter m_feature_density;
    ConvMACRedundancyParameter m_conv_mac_redundancy;
    ConvBufBankMgmtParameter m_conv_buf_bank_mgmt;
    PDPOpModeParameter m_pdp_op_mode;
    OffFlyingOpModeParameter m_off_flying_op_mode;
    SDPOpModeParameter m_sdp_op_mode;
    AXIFSchedParameter m_axif_sched;
    NetworkForksParameter m_network_forks;
};

} // namespace priv
} // namespace nvdla

#endif // NVDLA_PRIV_TEST_POINT_PARAMETER_H
