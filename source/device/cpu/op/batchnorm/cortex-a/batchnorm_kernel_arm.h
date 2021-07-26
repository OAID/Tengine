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

#ifndef _BATCHNORM_KERNEL_ARM_H_
#define _BATCHNORM_KERNEL_ARM_H_

#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"

struct hcl_batchnorm_param
{
    float* scale_mean;
    float* scale_var_inv;
};

int batchnorm_run(struct tensor* output_tensor, struct tensor* input_tensor, float* scale_mean, float* scale_var_inv, int num_thread);

#endif
