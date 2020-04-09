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
 * Author: jjzeng@openailab.com
 */

static int ref_concat_uint8(const uint8_t** in_data, uint8_t* out_data, const struct concat_param* param)
{
    int axis = param->axis;
    int concat_dim = 0;
    for(int ii = 0; ii < param->input_counts; ++ii)
    {
        concat_dim += param->input_shape[ii].dim[axis];
    }

    if(concat_dim != param->output_shape.dim[axis])
    {
        printf("concat dimensions is not same output: ( %d -- %d )\n", concat_dim, param->output_shape.dim[axis]);
        return -1;
    }

    int outer_size, in_size;
    outer_size = 1;
    for(int ii = 0; ii < axis; ++ii)
    {
        outer_size *= param->output_shape.dim[ii];
    }
    in_size = 1;
    for(int ii = axis + 1; ii < param->output_dim; ++ii)
    {
        in_size *= param->output_shape.dim[ii];
    }

    int output_size = 1;
    for(int ii = 0; ii < param->output_dim; ++ii)
    {
        output_size *= param->output_shape.dim[ii];
    }

    uint8_t* output_ptr = out_data;
    float out_scale = param->output_shape.scale;
    uint8_t out_zero = param->output_shape.zero;
    for(int k = 0; k < outer_size; ++k)
    {
        for(int j = 0; j < param->input_counts; ++j)
        {
            int cp_size = param->input_shape[j].dim[axis] * in_size;
            float scale = param->input_shape[j].scale;
            uint8_t input_zero = param->input_shape[j].zero;

            const uint8_t* input_ptr = ( const uint8_t* )(in_data[j] + k * cp_size);

            if(scale == out_scale && input_zero == out_zero)
            {
                memcpy(output_ptr, input_ptr, cp_size);
            }
            else
            {
                float t_scale = scale / out_scale;
                for(int ii = 0; ii < cp_size; ++ii)
                {
                    output_ptr[ii] = round((input_ptr[ii] - input_zero)*t_scale) + out_zero;
                }
            }
            output_ptr += cp_size;
        }
    }

    return 0;
}
