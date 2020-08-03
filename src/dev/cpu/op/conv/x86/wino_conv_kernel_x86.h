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
 * Author: haoluo@openailab.com
 */
#ifndef __WINO_CONV_KERNEL_X86_H_
#define __WINO_CONV_KERNEL_X86_H_

#include "tengine_ir.h"
#include "convolution_param.h"
#include "../conv_hcl_kernel.h"

#if __SSE2__
#include <emmintrin.h>
#endif
#if __AVX__
#include <immintrin.h>
#endif

int wino_conv_hcl_prerun(struct ir_tensor* input_tensor, struct ir_tensor* filter_tensor,
                         struct ir_tensor* output_tensor, struct conv_priv_info* info, struct conv_param* param)
    __attribute__((weak));

int wino_conv_hcl_postrun(struct conv_priv_info* info) __attribute__((weak));

int wino_conv_hcl_run(struct ir_tensor* input_tensor, struct ir_tensor* filter_tensor, struct ir_tensor* bias_tensor,
                      struct ir_tensor* output_tensor, struct conv_priv_info* conv_info, struct conv_param* param,
                      int num_thread, int affinity) __attribute__((weak));

#endif
