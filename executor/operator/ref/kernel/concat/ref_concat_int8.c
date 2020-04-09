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

static int ref_concat_int8(const int8_t** in_data, int8_t* out_data, const struct concat_param* param)
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

    float* output_tmp = ( float* )malloc(output_size * 4);
    if(NULL == output_tmp)
    {
        return -1;
    }

    float* output_ptr = output_tmp;
    float max_scale = 0.0f;
    for(int k = 0; k < outer_size; ++k)
    {
        for(int j = 0; j < param->input_counts; ++j)
        {
            int cp_size = param->input_shape[j].dim[axis] * in_size;
            float scale = param->input_shape[j].scale;
            const int8_t* input_ptr = in_data[j] + k * cp_size;

            for(int ii = 0; ii < cp_size; ++ii)
            {
                float val = (input_ptr[ii]) * scale;
                output_ptr[ii] = val;
            }
            output_ptr += cp_size;

            if(max_scale < scale)
                max_scale = scale;
        }
    }

    int8_t* last_output_ptr = out_data;
    for(int ii = 0; ii < output_size; ++ii)
    {
        last_output_ptr[ii] = round(output_tmp[ii] / max_scale);
    }

    concat_param* out_param = const_cast<concat_param*>(param);
    out_param->out_scale = max_scale;

    free(output_tmp);
    return 0;
}
