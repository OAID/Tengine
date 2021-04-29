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
 * Copyright (c) 2021, OPEN AI LAB
 * Author: haitao@openailab.com
 */

#ifndef __CONV_PARAM_H__
#define __CONV_PARAM_H__

struct conv_param
{
    int kernel_h;
    int kernel_w;
    int stride_h;
    int stride_w;
    int pad_h0;
    int pad_h1;
    int pad_w0;
    int pad_w1;
    int dilation_h;
    int dilation_w;
    int input_channel;
    int output_channel;
    int group;
    int activation;
    int wino_off;
};

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
    int wino_off;

    /* int8 params */
    int* q_shift;
    int* multi;
    int activation_min;
    int activation_max;
};
#endif
