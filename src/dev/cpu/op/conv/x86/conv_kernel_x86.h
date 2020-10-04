/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * License); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*
 * Copyright (c) 2020, OPEN AI LAB
 * Author: qtang@openailab.com
 */

#ifndef _CONV_KERNEL_X86_H_
#define _CONV_KERNEL_X86_H_

#include "tengine_ir.h"
#include "convolution_param.h"

struct conv_priv_info
{
    void* interleave_buffer;    // kernel transform buffer
    void* interleave_buffer_pack4;    // kernel pack4
    void* im2col_buffer;    // input data transform buffer
    void* im2col_buffer_pack4;    // input data transform buffer pack4
    void* input_pad;
    void* dot_block;
    void* transform_input;
    void* output_bordered;
    int im2col_buffer_size;    // kernel transform buffer size
    int im2col_buffer_pack4_size;    // kernel transform buffer size
    int interleave_buffer_size;    // input data transform buffer size
    int interleave_buffer_pack4_size;
    int external_im2col_mem;    // flag
    int external_im2col_pack4_mem;    // flag
    int external_interleave_mem;    // flag
    int external_interleave_pack4_mem;    // flag
    int cpu_type;
    int winograd;

    /* hybrid int8 params */
    void* p_input_max;
    void* p_kernel_max;
};

/* float32 */
int conv_hcl_prerun(struct ir_tensor* input_tensor, struct ir_tensor* filter_tensor, struct ir_tensor* output_tensor,
                    struct conv_priv_info* info, struct conv_param* param) __attribute__((weak));

int conv_hcl_postrun(struct conv_priv_info* info) __attribute__((weak));

int conv_hcl_run(struct ir_tensor* input_tensor, struct ir_tensor* filter_tensor, struct ir_tensor* bias_tensor,
                 struct ir_tensor* output_tensor, struct conv_priv_info* conv_info, struct conv_param* param,
                 int num_thread, int cpu_affinity) __attribute__((weak));

int conv_hcl_get_shared_mem_size(struct ir_tensor* input_tensor, struct ir_tensor* output_tensor,
                                 struct conv_param* param) __attribute__((weak));
int conv_hcl_get_shared_pack4_mem_size(struct ir_tensor* input_tensor, struct ir_tensor* output_tensor,
                                       struct conv_param* param) __attribute__((weak));

int conv_hcl_set_shared_mem(struct conv_priv_info* priv_info, void* mem, int mem_size) __attribute__((weak));

int conv_hcl_set_shared_pack4_mem(struct conv_priv_info* priv_info, void* mem, int mem_size) __attribute__((weak));

#endif
