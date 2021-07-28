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

#ifndef NVDLA_PRIV_TEST_POINT_PARAMETER_ENUM_H
#define NVDLA_PRIV_TEST_POINT_PARAMETER_ENUM_H

// These show up here instead of "next to" the enum class wrapper
// because the MISRA rules disallow defining macros at anything other
// than global scope.  And, the enum class wrappers are being used

// for class BatchModeParameter::SERIAL, ...
#define BATCH_MODE_ENUMS(op)                    \
    op(SERIAL, 0U)                              \
    op(MULTI,  1U)

// for class CVSRamSizeParameter::ZERO_MB, ...
#define CVSRAM_SIZE_ENUMS(op)                   \
    op(ZERO_MB, 0U)                             \
    op(TWO_MB, 1U)                              \
    op(FOUR_MB, 2U)

// for class HWLayerTuningParameter::DC, ...
#define HW_LAYER_TUNING_ENUMS(op)               \
    op(DC,       0U)                            \
    op(WINOGRAD, 1U)

// for class MappingWeightsParameter::COMPRESSED, ...
#define MAPPING_WEIGHTS_ENUMS(op)               \
    op(COMPRESSED,   0U)                        \
    op(UNCOMPRESSED, 1U)

// for class PaddingParameter::NO_PADDING, ...
#define PADDING_ENUMS(op)                       \
    op(NO_PADDING, 0U)                          \
    op(PADDED,     1U)

// for class OutputSequenceParameter::PARTIAL_HEIGHT, ...
#define OUTPUT_SEQUENCE_ENUMS(op)               \
    op(PARTIAL_HEIGHT,      0U)                 \
    op(PARTIAL_HEIGHT_LAST, 1U)

// for class DilationParameter::DISABLED, ...
#define DILATION_ENUMS(op)                      \
    op(DISABLED, 0U)                            \
    op(ENABLED,  1U)

// for class WeightDensityParameter::FULL, ...
#define WEIGHT_DENSITY_ENUMS(op)                \
    op(FULL,    0U)                             \
    op(PARTIAL, 1U)

// for class FeatureDensityParameter::FULL, ...
#define FEATURE_DENSITY_ENUMS(op)               \
    op(FULL,    0U)                             \
    op(PARTIAL, 1U)

// for class ChannelExtensionParameter::DISABLED, ...
#define CHANNEL_EXTENSION_ENUMS(op)             \
    op(DISABLED, 0U)                            \
    op(ENABLED, 1U)

// for class ConvMACRedundancyParameter::DISABLED, ...
#define CONV_MAC_REDUNDANCY_ENUMS(op)           \
    op(DISABLED, 0U)                            \
    op(ENABLED,  1U)

// for class ConvBufBankMgmtParameter::DISABLED, ...
#define CONV_BUF_BANK_MGMT_ENUMS(op)            \
    op(DISABLED, 0U)                            \
    op(ENABLED,  1U)

// for class PDPOpModeParameter::ON_FLYING, ...
#define PDP_OP_MODE_ENUMS(op)                   \
    op(ON_FLYING,  0U)                          \
    op(OFF_FLYING, 1U)

// for class OffFlyingOpModeParameter::NO_REFETCH, ...
#define OFF_FLYING_OP_MODE_ENUMS(op)            \
    op(NO_REFETCH, 0U)                          \
    op(REFETCH,    1U)

// for class SDPOpModeParameter::ON_FLYING, ...
#define SDP_OP_MODE_ENUMS(op)                   \
    op(ON_FLYING,  0U)                          \
    op(OFF_FLYING, 1U)

// for class AXIFSchedParameter::DISABLED, ...
#define AXIF_SCHED_ENUMS(op)                    \
    op(DISABLED, 0U)                            \
    op(ENABLED,  1U)

// for class PixelDataFormatParameter::PITCH_LINEAR, ...
#define PIXEL_DATA_FORMAT_ENUMS(op)             \
    op(PITCH_LINEAR, 0U)

// for class NetworkForksParameter::DISABLED, ...
#define NETWORK_FORKS_ENUMS(op)                 \
    op(DISABLED, 0U)                            \
    op(ENABLED,  1U)

#endif // NVDLA_PRIV_TEST_POINT_PARAMETER_ENUM
