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
 * Author: qwang@openailab.com
 */

#ifndef __CONV_DW_KERNEL_INT8_ARM_H_
#define __CONV_DW_KERNEL_INT8_ARM_H_

#include "tengine_ir.h"
#include "convolution_param.h"
#include "conv_dw_kernel_arm.h"

int conv_dw_int8_prerun(struct ir_tensor* input_tensor, struct ir_tensor* filter_tensor, struct ir_tensor* output_tensor,
                    struct conv_priv_info* priv_info, struct conv_param* param) __attribute__((weak));
int conv_dw_int8_postrun(struct conv_priv_info* priv_info) __attribute__((weak));
int conv_dw_int8_run(struct ir_tensor* input_tensor, struct ir_tensor* filter_tensor, struct ir_tensor* bias_tensor,
                 struct ir_tensor* output_tensor, struct conv_priv_info* priv_info, struct conv_param* param,
                 int num_thread, int cpu_affinity) __attribute__((weak));

#endif