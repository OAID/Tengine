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

#ifndef NVDLA_PRIV_EMU_EMU1_A_EMU_INTERFACE_H
#define NVDLA_PRIV_EMU_EMU1_A_EMU_INTERFACE_H

#define NVDLA_EMU_MAX_BUFFERS_PER_TASK (6144)

/**
 * @name Op Type
 * Network is formed using a list of these operations
 * @{
 */
#define NVDLA_EMU_OP_POWER    0
#define NVDLA_EMU_OP_SOFTMAX  1
/** @} */

/**
 * Address
 */
struct emu_address
{
    void *hMem;
    NvU32 offset;
};

/**
 * Task Descriptor
 */
struct emu_task_desc
{
    NvU32 num_addresses;
    emu_address address_list[NVDLA_EMU_MAX_BUFFERS_PER_TASK];
} __attribute__ ((packed, aligned(256)));

/**
 * Network Descriptor
 *
 * Contains all information to execute a network
 *
 * @num_operations: Number of operations in the lists
 */
struct emu_network_desc
{
    NvS16 operation_desc_index;
    NvS16 operation_buffer_desc_index;
    NvU16 num_operations;
} __attribute__ ((packed, aligned(256)));

struct emu_common_op_desc
{
    NvU8 op_type;
    NvF32 input_scale_factor;
    NvF32 output_scale_factor;
};

struct emu_power_op_desc
{
    emu_common_op_desc common;
    NvF32 power;
    NvF32 scale;
    NvF32 shift;
} __attribute__ ((packed, aligned(4)));

struct emu_softmax_op_desc
{
    emu_common_op_desc common;
    NvU8 axis;
} __attribute__ ((packed, aligned(4)));

union emu_operation_container
{
    struct emu_power_op_desc power_op;
    struct emu_softmax_op_desc softmax_op;
};

struct emu_buffer_desc
{
    /* offset to the actual IOVA in task.address_list */
    NvS16 addressIndex;
    NvU32 addressIndexOffset;
    NvU32 size;

    /* surface format */
    NvU16 format;

    /* cube dimensions */
    NvU16 width;
    NvU16 height;
    NvU16 channel;

    /* stride information */
    NvU32 line_stride;
    NvU32 surf_stride;
} __attribute__ ((packed, aligned(256)));

struct emu_power_buffer_descs
{
    /* Buffer Descriptors */
    struct emu_buffer_desc src_data;
    struct emu_buffer_desc dst_data;
} __attribute__ ((packed, aligned(4)));

struct emu_softmax_buffer_descs
{
    /* Buffer Descriptors */
    struct emu_buffer_desc src_data;
    struct emu_buffer_desc dst_data;
} __attribute__ ((packed, aligned(4)));

union emu_operation_buffer_container
{
    struct emu_power_buffer_descs power_buffers;
    struct emu_softmax_buffer_descs softmax_buffers;
};


#endif // NVDLA_PRIV_EMU_EMU1_A_EMU_INTERFACE_H
