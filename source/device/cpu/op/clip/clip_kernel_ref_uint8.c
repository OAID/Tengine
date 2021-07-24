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

#include "clip_kernel_ref.h"

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


int ref_clip_uint8(struct tensor* input_tensor, struct tensor* output_tensor, float max, float min)
{
    int total_size = input_tensor->elem_num;
    uint8_t* input_uint8 = ( uint8_t* )input_tensor->data;
    uint8_t* output_uint8 = ( uint8_t* )output_tensor->data;

    float input_scale = input_tensor->scale;
    float output_scale = output_tensor->scale;
    int input_zero = input_tensor->zero_point;
    int output_zero = output_tensor->zero_point;

    /* input dequant */
    float* input_fp32 = ( float* )sys_malloc(total_size * sizeof(float));
    float* output_fp32 = ( float* )sys_malloc(total_size * sizeof(float));

    for (uint32_t i = 0; i < input_tensor->elem_num; i++)
        input_fp32[i] = ((float )input_uint8[i] - (float )input_zero) * input_scale;

    for (int i = 0; i < total_size; i++)
    {
        output_fp32[i] = input_fp32[i];

        if (output_fp32[i] > max)
            output_fp32[i] = max;
        if (output_fp32[i] < min)
            output_fp32[i] = min;
    }

    /* output quant */
    for (int i = 0; i < total_size; i++)
    {
        int output_data = (int)roundf(output_fp32[i] / output_scale) + output_zero;
        output_uint8[i] = output_data > 255 ? 255 : output_data;
    }

    sys_free(input_fp32);
    sys_free(output_fp32); 

    return 0;
}
