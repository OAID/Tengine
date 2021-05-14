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

#include "unary_kernel_ref.h"

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


int ref_unary_uint8(struct tensor* input_tensor, struct tensor* output_tensor, struct unary_param* param)
{
    /* dequant */
    uint8_t* input_uint8 = input_tensor->data;
    uint8_t* output_uint8 = output_tensor->data;
    float input_scale = input_tensor->scale;
    float output_scale = output_tensor->scale;
    int32_t input_zero = input_tensor->zero_point;
    int32_t output_zero = output_tensor->zero_point;
    int input_size = input_tensor->elem_num;
    int output_size = output_tensor->elem_num;

    float* in_data = ( float* )sys_malloc(input_size * sizeof(float));
    float* out_data = ( float* )sys_malloc(output_size * sizeof(float));

    for (int i = 0; i < input_size; i++)
    {
        in_data[i] = (( float )input_uint8[i] - ( float )input_zero) * input_scale;
    }

    int size = input_tensor->elem_num;

    int type = param->type;

    switch (type)
    {
        case 0:
            for (int i = 0; i < size; i++)
            {
                out_data[i] = fabs(in_data[i]);
            }
            break;
        case 1:
            for (int i = 0; i < size; i++)
            {
                out_data[i] = -(in_data[i]);
            }
            break;
        case 2:
            for (int i = 0; i < size; i++)
            {
                out_data[i] = floor(in_data[i]);
            }
            break;
        case 3:
            for (int i = 0; i < size; i++)
            {
                out_data[i] = ceil(in_data[i]);
            }
            break;
        case 4:
            for (int i = 0; i < size; i++)
            {
                out_data[i] = in_data[i] * in_data[i];
            }
            break;
        case 5:
            for (int i = 0; i < size; i++)
            {
                out_data[i] = sqrt(in_data[i]);
            }
            break;
        case 6:
            for (int i = 0; i < size; i++)
            {
                out_data[i] = 1.f / sqrt(in_data[i]);
            }
            break;
        case 7:
            for (int i = 0; i < size; i++)
            {
                out_data[i] = exp(in_data[i]);
            }
            break;
        case 8:
            for (int i = 0; i < size; i++)
            {
                out_data[i] = log(in_data[i]);
            }
            break;
        case 9:
            for (int i = 0; i < size; i++)
            {
                out_data[i] = sin(in_data[i]);
            }
            break;
        case 10:
            for (int i = 0; i < size; i++)
            {
                out_data[i] = cos(in_data[i]);
            }
            break;
        case 11:
            for (int i = 0; i < size; i++)
            {
                out_data[i] = tan(in_data[i]);
            }
            break;
        case 12:
            for (int i = 0; i < size; i++)
            {
                out_data[i] = asin(in_data[i]);
            }
            break;
        case 13:
            for (int i = 0; i < size; i++)
            {
                out_data[i] = acos(in_data[i]);
            }
            break;
        case 14:
            for (int i = 0; i < size; i++)
            {
                out_data[i] = atan(in_data[i]);
            }
            break;
        case 15:
            for (int i = 0; i < size; i++)
            {
                out_data[i] = 1.f / (in_data[i]);
            }
            break;
        case 16:
            for (int i = 0; i < size; i++)
            {
                out_data[i] = tanh(in_data[i]);
            }
            break;
        default:
            break;
    }

    /* quant */
    for (int i = 0; i < output_size; i++)
    {
        int udata = round(out_data[i] / output_scale + output_zero);
        if (udata > 255)
            udata = 255;
        else if (udata < 0)
            udata = 0;
        output_uint8[i] = udata;
    }

    sys_free(in_data);
    sys_free(out_data);

    return 0;
}
