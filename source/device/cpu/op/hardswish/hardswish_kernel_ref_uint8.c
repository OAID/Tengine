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

#include "hardswish_kernel_ref.h"

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

int ref_hardswish_uint8(struct tensor* input_tensor, struct tensor* output_tensor)
{
    int size = input_tensor->elem_num;

    // dequant
    uint8_t* input_uint8 = (uint8_t*)input_tensor->data;
    uint8_t* output_uint8 = (uint8_t*)output_tensor->data;
    float input_scale = input_tensor->scale;
    float output_scale = output_tensor->scale;
    int32_t input_zero = input_tensor->zero_point;
    int32_t output_zero = output_tensor->zero_point;

    float* data_fp32 = (float*)sys_malloc(size * sizeof(float));

    for (int i = 0; i < size; i++)
        data_fp32[i] = ((float)input_uint8[i] - (float)input_zero) * input_scale;

    for (int i = 0; i < size; i++)
    {
        float tmp = data_fp32[i] + 3.f;

        if (tmp < 0.f)
            tmp = 0.f;
        if (tmp > 6.f)
            tmp = 6.f;

        data_fp32[i] = data_fp32[i] * (tmp / 6.f);
    }

    // quant
    for (int i = 0; i < size; i++)
    {
        int udata = round(data_fp32[i] / output_scale + output_zero);
        if (udata > 255)
            udata = 255;
        else if (udata < 0)
            udata = 0;
        output_uint8[i] = udata;
    }
    sys_free(data_fp32);

    return 0;
}
