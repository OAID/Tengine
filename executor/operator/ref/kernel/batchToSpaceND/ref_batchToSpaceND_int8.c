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
 * Author: bingzhang@openailab.com
 */

static int ref_batchToSpaceND_int8(const int8_t* in_data, int8_t* out_data, struct batchToSpaceND_param* param)
{
    int o_size = param->out_dims[0]*param->out_dims[1]*param->out_dims[2]*param->out_dims[3];

    float* output_tmp = (float*)malloc(o_size * 4);
    if(output_tmp == NULL){
        return -1;
    }
    float* output_ptr = output_tmp;  

    for(int in_batch = 0; in_batch < param->in_dims[0]; ++in_batch){
        const int out_batch = in_batch % param->out_dims[0];
        const int spatial_offset = in_batch / param->out_dims[0];
        float scale = param->in_scale;        
        for(int in_h = 0; in_h < param->in_dims[1]; ++in_h){
            const int out_h = in_h * param->dilation_y + spatial_offset / param->dilation_x - param->crop_top;
            if(out_h < 0 ||  out_h >= param->out_dims[1]){
                continue;
            }
            for(int in_w = 0; in_w < param->in_dims[2]; ++in_w){
                const int out_w = in_w * param->dilation_x + spatial_offset % param->dilation_x - param->crop_left;
                if(out_w < 0 || out_w >= param->out_dims[2]){
                    continue;
                }
                int outOffset = out_batch*param->out_dims[1]*param->out_dims[2]*param->out_dims[3] + out_h * param->out_dims[2]*param->out_dims[3] + out_w *param->in_dims[3];
                output_ptr +=  outOffset;
                int inOffset = in_batch*param->in_dims[1]*param->in_dims[2]*param->in_dims[3] + in_h *param->in_dims[2] *param->in_dims[3] + in_w *param->in_dims[3];
                const int8_t* in = in_data + inOffset;
                for(int i = 0; i < param->out_dims[3]; ++i){
                    output_ptr[i] = (in[i]) * scale;
                }
                output_ptr += param->out_dims[3];                
            }
        }
    }
    float maxScale = 0.0f;
    for(int i = 0; i < o_size; i++){
        if(maxScale < output_ptr[i])
            maxScale = output_ptr[i];
    }
    float o_scale = maxScale / 127;
    param->out_scale = o_scale;

    for(int i = 0; i < o_size; i++){
        out_data[i] = round(output_ptr[i] / param->out_scale);
    }
    free(output_ptr);

    return 0;
}
