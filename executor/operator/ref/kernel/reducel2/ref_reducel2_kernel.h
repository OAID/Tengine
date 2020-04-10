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
 * Copyright (c) 2019, Open AI Lab
 * Author: zpluo@openailab.com
 */

#ifndef __REF_REDUCEL2_KERNEL_H__
#define __REF_REDUCEL2_KERNEL_H__

#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "compiler_fp16.h"

#ifdef __cplusplus
extern "C" {
#endif

struct reducel2_param
{
    int axis;
    int dims[4];

};

typedef int (*ref_reducel2_t)(void* in_data, void* out_data,reducel2_param* param);

#ifdef CONFIG_KERNEL_FP32
#include "ref_reducel2_fp32.c"
#endif

#ifdef __cplusplus
}
#endif

#endif
