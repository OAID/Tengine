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
 * Copyright (c) 2019, Open AI Lab
 * Author: haitao@openailab.com
 */

#include <string.h>

#include <math.h>
#include <stdlib.h>

int ref_softmax_kernel_fp16(__fp16* input, __fp16* output, float* max_array, float* sum_array, op_data* op_param)
{
    int out_size = op_param->out_size;
    int in_size = op_param->in_size;
    int on_size = op_param->on_size;
    int on_in_size = in_size * on_size;

    float* input_f = ( float* )malloc(out_size * on_in_size * sizeof(float));
    float* output_f = ( float* )malloc(out_size * on_in_size * sizeof(float));

    for(int i = 0; i < out_size; i++)
        for(int j = 0; j < on_in_size; j++)
            input_f[i * on_in_size + j] = fp16_to_fp32(input[i * on_in_size + j]);

    for(int i = 0; i < out_size; i++)
    {
        /* get max */
        int img_base = i * in_size * on_size;
        GetMaxArray(input_f + img_base, max_array, in_size, on_size);
        GetOutResult(input_f + img_base, output_f + img_base, max_array, sum_array, in_size, on_size);
    }

    for(int i = 0; i < out_size; i++)
        for(int j = 0; j < on_in_size; j++)
            output[i * on_in_size + j] = fp32_to_fp16(output_f[i * on_in_size + j]);

    free(input_f);
    free(output_f);

    return 0;
}
