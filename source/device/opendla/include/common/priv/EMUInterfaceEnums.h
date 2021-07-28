/*
 * Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#ifndef NVDLA_PRIV_EMU_INTERFACE_ENUMS_H
#define NVDLA_PRIV_EMU_INTERFACE_ENUMS_H

// These show up here instead of "next to" the enum class wrapper
// because the MISRA rules disallow defining macros at anything other
// than global scope.  And, the enum class wrappers are being used

// for class EMUBufferType::DLA_FEATURE_INT8_FORMAT, ...
#define EMU_BUFFER_TYPE_ENUMS(op)               \
    op(DLA_FEATURE_INT8_FORMAT, 0U)             \
    op(DLA_FEATURE_INT16_FORMAT, 1U)            \
    op(DLA_FEATURE_FP16_FORMAT, 2U)

// for class EMUOpType::SOFTMAX, ...
#define EMU_OP_TYPE_ENUMS(op)               \
    op(POWER, 0U)                           \
    op(SOFTMAX, 1U)

#endif // NVDLA_PRIV_EMU_INTERFACE_ENUMS_H
