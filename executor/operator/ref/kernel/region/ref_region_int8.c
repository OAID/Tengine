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
 * Author: jingyou@openailab.com
 */

static int ref_region_int8(const int8_t* in_data, int8_t* out_data, ref_region_param* param)
{
    int input_size = param->dims[0] * param->dims[1] * param->dims[2] * param->dims[3];
    float* input_f = ( float* )malloc(input_size * sizeof(float));
    float* output_f = ( float* )malloc(input_size * sizeof(float));
    for(int i = 0; i < input_size; i++)
        input_f[i] = in_data[i] * param->scale[0];

    ref_region_common(input_f, output_f, param);

    float max_val = 0.0f;
    for(int i = 0; i < input_size; i++)
    {
        if(max_val < fabs(output_f[i]))
            max_val = fabs(output_f[i]);
    }
    float out_scale = max_val / 127;
    for(int i = 0; i < input_size; i++)
    {
        out_data[i] = (int8_t)(round(output_f[i] / out_scale));
    }
    param->scale[1] = out_scale;

    free(input_f);
    free(output_f);
    return 0;
}
