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
 * Author:
 */

#ifndef __BATCHNORM_KERNEL_REF_H__
#define __BATCHNORM_KERNEL_REF_H__


#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"

#include <stdbool.h>
#include <math.h>

struct ref_batchnorm_param
{
    int input_n;
    int input_h;
    int input_w;
    int input_c;
    int layout;
    bool iscaffe;
    float* scale_mean;
    float* scale_var_inv;
    float* gamma;
    float* beta;
    float in_scale;
    int in_zero;
    float out_scale;
    int out_zero;
};

int ref_batchnorm_fp32(float* input, float* output, const struct ref_batchnorm_param* param);

int ref_batchnorm_uint8(struct tensor* input_tensor, struct tensor* output_tensor, const struct ref_batchnorm_param* param);

#endif
