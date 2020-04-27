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
 * Author: zpluo@openailab.com
 */

static int ref_split_fp32(const float* in_data, float** out_data, struct split_param* param)
{
    int slice_axis = param->axis;
    int num_slices = 1;
    int slice_size = 1;
    for(int i = 0; i < slice_axis; i++)
    {
        num_slices = num_slices * param->input_shape.dim[i];
    }
    for(int i = slice_axis + 1; i < param->input_dim; i++)
    {
        slice_size = slice_size * param->input_shape.dim[i];
    }
    int in_slice = param->input_shape.dim[slice_axis];
    int slice_index = 0;
    unsigned int out_num = param->output_counts;
    for(unsigned int i = 0; i < out_num; i++)
    {
        float* output = ( float* )out_data[i];
        if(param->is_caffe){
            int size = 1;
            for(int i = 0; i < 4; i++){
                size *= param->input_shape.dim[i];
            }
            memcpy(output, in_data, sizeof(float)*size);
        } else {
            int out_slice = 0;
            // if(param->squeeze_dim == 1)
            // {
            //     out_slice = 1;
            // }
            // else
            {
                out_slice = param->output_shape[i].dim[slice_axis];
            }
            for(int n = 0; n < num_slices; n++)
            {
                int in_offset = (n * in_slice + slice_index) * slice_size;
                int out_offset = n * out_slice * slice_size;
                memcpy(output + out_offset, in_data + in_offset, slice_size * out_slice * sizeof(float));
            }
            slice_index += out_slice;
        }

    }
    return 0;
}
