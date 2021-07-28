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

#ifndef NVDLA_PRIV_ENGINE_AST_ENUMS_H
#define NVDLA_PRIV_ENGINE_AST_ENUMS_H

// EngineAST::EdgeType
#define ENGINE_AST_EDGE_TYPE_ENUMS(op)   \
    op(DATA, 0U)                         \
    op(COMPUTE, 1U)                      \
    op(HAZARD, 2U)

// EngineAST::EngineType
#define ENGINE_AST_ENGINE_TYPE_ENUMS(op) \
    op(CPU, 0U)                          \
    op(CONVOLUTION, 1U)                  \
    op(RUBIK, 2U)                        \
    op(SDP, 3U)                          \
    op(PDP, 4U)                          \
    op(CDP, 5U)                          \
    op(BDMA, 6U)                         \
    op(CONCATENATION, 7U)                \
    op(SPLIT, 8U)                        \
    op(MULTI_OPS, 9U)

// EngineAST::EngineOpType
#define ENGINE_AST_ENGINE_OP_TYPE_ENUMS(op) \
    op(CPU_SCALE, 0U)                       \
    op(CPU_SOFTMAX, 1U)                     \
    op(CONVOLUTION_CONV, 2U)                \
    op(CONVOLUTION_FC, 3U)                  \
    op(CONVOLUTION_DECONV, 4U)              \
    op(RUBIK_DECONV, 5U)                    \
    op(SDP_ACTIVATION, 6U)                  \
    op(SDP_SCALE, 7U)                       \
    op(SDP_BATCH_NORM, 8U)                  \
    op(SDP_ELEMENTWISE, 9U)                 \
    op(SDP_BIAS, 10U)                       \
    op(SDP_NOP,  11U)                       \
    op(PDP_POOL, 12U)                       \
    op(CDP_LRN, 13U)                        \
    op(BDMA_SINGLE_DMA, 14U)                \
    op(BDMA_GROUP_DMA, 15U)                 \
    op(CONCATENATION_CONCAT, 16U)           \
    op(SPLIT_SOFTWARE, 17U)                 \
    op(SDP_SUPER, 18U)


// EngineAST::IODirection
#define ENGINE_AST_IO_DIRECTION_ENUMS(op) \
    op(UNKNOWN, 0U)                       \
    op(INPUT,   1U)                       \
    op(OUTPUT,  2U)

// EngineAST::OperationEventType
#define ENGINE_AST_OPERATION_EVENT_TYPE_ENUMS(op) \
    op(OP_COMPLETED,        0U)                   \
    op(OP_PROGRAMMED,       1U)                   \
    op(OP_ENABLED,          2U)                   \
    op(OP_CDMA_WEIGHT_DONE, 3U)                   \
    op(OP_CDMA_DATA_DONE,   4U)

// EngineAST::ConcatAxis
#define ENGINE_AST_CONCAT_AXIS_ENUMS(op) \
    op(CONCAT_AXIS_UNKNOWN, 0U)          \
    op(CONCAT_ALONG_C,      1U)          \
    op(CONCAT_ALONG_H,      2U)          \
    op(CONCAT_ALONG_W,      3U)

// EngineAST::SplitAxis
#define ENGINE_AST_SPLIT_AXIS_ENUMS(op) \
    op(SPLIT_AXIS_UNKNOWN, 0U)          \
    op(SPLIT_ALONG_C,      1U)          \
    op(SPLIT_ALONG_H,      2U)          \
    op(SPLIT_ALONG_W,      3U)          \
    op(SPLIT_ALONG_NONE,   4U)

// EngineAST::ConvCoreEngineParams::ConvolutionMode
#define ENGINE_AST_CONVOLUTION_MODE_ENUMS(op)   \
    op(CONV_MODE_UNKNOWN,   0U)                 \
    op(CONV_DIRECT,         1U)                 \
    op(CONV_WINOGRAD,       2U)

// EngineAST::SDPSubEngineParams::SDPMode
#define ENGINE_AST_SDP_MODE_ENUMS(op)       \
    op(SDP_MODE_UNKNOWN,     0U)            \
    op(SDP_MODE_PER_LAYER,   1U)            \
    op(SDP_MODE_PER_CHANNEL, 2U)            \
    op(SDP_MODE_PER_ELEMENT, 3U)

// EngineAST::SDPSubEngineParams::SDPALUType
#define ENGINE_AST_SDP_ALU_TYPE_ENUMS(op)   \
    op(SDP_ALU_TYPE_UNKNOWN, 0U)            \
    op(SDP_ALU_TYPE_MAX,     1U)            \
    op(SDP_ALU_TYPE_MIN,     2U)            \
    op(SDP_ALU_TYPE_SUM,     3U)            \
    op(SDP_ALU_TYPE_EQL,     4U)

// EngineAST::SDPSubEngineParams::SDPOpType
#define ENGINE_AST_SDP_OP_TYPE_ENUMS(op)    \
    op(SDP_OP_TYPE_UNKNOWN, 0U)             \
    op(SDP_OP_TYPE_NONE,    1U)             \
    op(SDP_OP_TYPE_MUL,     2U)             \
    op(SDP_OP_TYPE_ADD,     3U)             \
    op(SDP_OP_TYPE_BOTH,    4U)

// EngineAST::SDPSubEngineParams::SDPActType
#define ENGINE_AST_SDP_ACT_TYPE_ENUMS(op)   \
    op(SDP_ACT_TYPE_UNKNOWN,    0U)         \
    op(SDP_ACT_TYPE_NONE,       1U)         \
    op(SDP_ACT_TYPE_RELU,       2U)         \
    op(SDP_ACT_TYPE_SIGMOID,    3U)         \
    op(SDP_ACT_TYPE_TANH,       4U)

// EngineAST::SDPSubEngineParams::SDPSubEngineType
#define ENGINE_AST_SDP_SUBENGINE_TYPE_ENUMS(op)   \
    op(SDP_ENGINE_X1,           0U)         \
    op(SDP_ENGINE_X2,           1U)         \
    op(SDP_ENGINE_Y,            2U)

// EngineAST::RubikEngineParams::RubikMode
#define ENGINE_AST_RUBIK_MODE_ENUMS(op) \
    op(RUBIK_MODE_UNKNOWN,      0U)     \
    op(RUBIK_MODE_CONTRACT,     1U)     \
    op(RUBIK_MODE_SPLIT,        2U)     \
    op(RUBIK_MODE_MERGE,        3U)

#endif // NVDLA_PRIV_ENGINE_AST_ENUMS_H

