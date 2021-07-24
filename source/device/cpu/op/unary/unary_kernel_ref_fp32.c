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


int ref_unary_fp32(struct tensor* input_tensor, struct tensor* output_tensor, struct unary_param* param)
{
    float* in_data = input_tensor->data;
    float* out_data = output_tensor->data;

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

    return 0;
}
