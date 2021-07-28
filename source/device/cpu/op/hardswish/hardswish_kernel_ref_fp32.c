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

int ref_hardswish_fp32(struct tensor* input_tensor, struct tensor* output_tensor)
{
    float* input_data = (float*)input_tensor->data;
    float* output_data = (float*)output_tensor->data;
    int size = input_tensor->elem_num;

    for (int i = 0; i < size; i++)
    {
        float tmp = input_data[i] + 3.f;

        if (tmp < 0.f)
            tmp = 0.f;
        if (tmp > 6.f)
            tmp = 6.f;

        output_data[i] = input_data[i] * (tmp / 6.f);
    }

    return 0;
}
