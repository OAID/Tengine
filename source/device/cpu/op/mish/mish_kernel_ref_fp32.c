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

#include "mish_kernel_ref.h"

#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "module/module.h"
#include "operator/op.h"
#include "utility/float.h"
#include "utility/sys_port.h"
#include "utility/log.h"
#include "device/cpu/cpu_node.h"
#include "device/cpu/cpu_graph.h"
#include "device/cpu/cpu_module.h"

#include <math.h>

int ref_mish_fp32(struct tensor* input_tensor, struct tensor* output_tensor, int num_thread)
{
    int w = input_tensor->dims[3];
    int h = output_tensor->dims[2];
    int channels = input_tensor->dims[1];
    int size = h * w;
    int c_step = h * w;

    float* input_data = (float*)input_tensor->data;
    float* out_data = (float*)output_tensor->data;

#pragma omp parallel for num_threads(num_thread)
    for (int q = 0; q < channels; q++)
    {
        float* src = input_data + c_step * q;
        float* dst = out_data + c_step * q;

        for (int i = 0; i < size; i++)
        {
            dst[i] = src[i] * tanhf(log(1 + exp(src[i])));
        }
    }

    return 0;
}
