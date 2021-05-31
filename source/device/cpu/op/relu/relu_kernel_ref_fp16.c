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

#include "relu_kernel_ref.h"

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


#if MACOS
#else
int ref_relu_fp16(struct tensor* input_tensor, struct tensor* output_tensor, float negative_slope)
{
    int total_size = input_tensor->elem_num;
    float* input_data = input_tensor->data;
    float* output_data = output_tensor->data;

    /* cost fp16 to fp32 */
    fp16_t* input_fp16 = input_tensor->data;
    fp16_t* output_fp16 = output_tensor->data;
    float* input_fp32 = (float*)sys_malloc(total_size * sizeof(float));

    for(int i=0; i< total_size; i++)
    {
        input_fp32[i] = fp16_to_fp32(input_fp16[i]);
    }

    /* process */
    if (negative_slope == 0)
    {
        for (int i = 0; i < total_size; i++)
        {
            if (input_fp32[i] < 0)
                input_fp32[i] = 0;
            else
                input_fp32[i] = input_fp32[i];
        }
    }
    else
    {
        for (int i = 0; i < total_size; i++)
        {
            if (input_fp32[i] < 0)
                input_fp32[i] = input_fp32[i] * negative_slope;
            else
                input_fp32[i] = input_fp32[i];
        }
    }

    /* cost fp32 to fp16 */
    for(int i=0; i<total_size; i++)
    {
        output_fp16[i] = fp32_to_fp16(input_fp32[i]);
    }

    sys_free(input_fp32);

    return 0;
}
#endif
