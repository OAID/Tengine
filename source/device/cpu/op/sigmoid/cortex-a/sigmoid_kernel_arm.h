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

#ifndef _SIGMOID_KERNEL_ARM_H_
#define _SIGMOID_KERNEL_ARM_H_

#include "graph/tensor.h"

#include <arm_neon.h>

struct tab
{
    float32x4_t a0;
    float32x4_t a1;
    float32x4_t a2;
    float32x4_t a3;
    float32x4_t a4;
    float32x4_t a5;
    float32x4_t a6;
    float32x4_t a7;
};

int sigmoid_run(struct tensor* output_tensor, struct tensor* input_tensor, int num_thread);

#endif
