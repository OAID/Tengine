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

#include "softmax_kernel_ref.h"

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

int ref_softmax_uint8(struct tensor* input_tensor, struct tensor* output_tensor, int axis)
{
    int element_size = input_tensor->elem_size;
    int type = input_tensor->data_type;

    int* dims = (int*)sys_malloc(input_tensor->dim_num * sizeof(int));
    for (int i = 0; i < input_tensor->dim_num; i++)
    {
        dims[i] = input_tensor->dims[i];
    }

    int out_size, in_size, on_size;

    out_size = 1;
    for (int i = 0; i < axis; i++)
    {
        out_size *= dims[i];
    }

    in_size = 1;
    for (size_t i = axis + 1; i < input_tensor->dim_num; i++)
    {
        in_size *= dims[i];
    }
    on_size = dims[axis];

    float* max_array = (float*)sys_malloc(in_size * sizeof(float));
    float* sum_array = (float*)sys_malloc(in_size * sizeof(float));

    int on_in_size = on_size * in_size;
    int totol_size = on_in_size * out_size;

    uint8_t* input = (uint8_t*)input_tensor->data;
    uint8_t* output = (uint8_t*)output_tensor->data;
    float* input_f = (float*)sys_malloc(totol_size * 4);
    float* output_f = (float*)sys_malloc(totol_size * 4);

    float input_scale = input_tensor->scale;
    float output_scale = output_tensor->scale;
    uint8_t input_zero = input_tensor->zero_point;
    uint8_t output_zero = output_tensor->zero_point;

    /* dequant to fp32 */
    for (int i = 0; i < out_size; i++)
        for (int j = 0; j < on_in_size; j++)
            input_f[i * on_in_size + j] = ((float)input[i * on_in_size + j] - (float)input_zero) * input_scale;

    /* fp32 softmax */
    for (int i = 0; i < out_size; i++)
    {
        /* get max */
        int img_base = i * in_size * on_size;
        GetMaxArray(input_f + img_base, max_array, in_size, on_size);
        GetOutResult(input_f + img_base, output_f + img_base, max_array, sum_array, in_size, on_size);
    }

    /* quant to uint8 */
    for (int i = 0; i < out_size; i++)
    {
        for (int j = 0; j < on_in_size; j++)
        {
            int udata = (int)(round(output_f[i * on_in_size + j] / output_scale) + output_zero);
            if (udata > 255)
                udata = 255;
            else if (udata < 0)
                udata = 0;
            output[i * on_in_size + j] = udata;
        }
    }

    sys_free(input_f);
    sys_free(output_f);

    sys_free(max_array);
    sys_free(sum_array);

    sys_free(dims);
    return 0;
}
