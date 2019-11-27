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

static int ref_spaceToBatchND_int8(const int8_t* in_data, int8_t* out_data, struct spaceToBatchND_param* param)
{
    

    int o_size = param->out_dims[0]*param->out_dims[1]*param->out_dims[2]*param->out_dims[3];

    float* output_tmp = (float*)malloc(o_size * 4);
    if(output_tmp == NULL){
        return -1;
    }
    float* output_ptr = output_tmp;    
    for(int out_b = 0; out_b < param->out_dims[0]; ++out_b){
        int input_batch = out_b % param->in_dims[0];
        int shift_w = (out_b / param->in_dims[0]) % param->dilation_x;
        int shift_h = (out_b / param->in_dims[0]) / param->dilation_x;
        for(int out_h = 0; out_h < param->out_dims[1]; ++out_h){
            for(int out_w = 0; out_w < param->out_dims[2]; ++out_w){
                int index = out_b * param->out_dims[1] * param->out_dims[2] * param->out_dims[3] + out_h * param->out_dims[2] * param->out_dims[3] + out_w * param->out_dims[3];
                output_ptr += index;
                float scale = param->in_scale;
                
                if( (out_h * param->dilation_y + shift_h < param->pad_top) || (out_h * param->dilation_y + shift_h >= param->pad_top + param->in_dims[1])
                    || (out_w * param->dilation_x + shift_w < param->pad_left) || (out_w *param->dilation_x + shift_w  >= param->pad_left + param->in_dims[2]) )
                {
                    //memset(out, pad_value, param->out_dims[3]*sizeof(int8_t));
                    for(int i = 0; i < param->out_dims[3]; ++i){
                        output_ptr[i] = 0 ;
                    }
                    output_ptr += param->out_dims[3];
                }
                else
                {
                    int index_h = out_h * param->dilation_y + shift_h - param->pad_top;
                    int index_w = out_w * param->dilation_x + shift_w - param->pad_left;
                    int index = input_batch * param->in_dims[1] * param->in_dims[2] * param->in_dims[3] + index_h * param->in_dims[2] * param->in_dims[3] + index_w * param->in_dims[3];
                    const int8_t* in = in_data + index;
                    //memcpy(out, in, param->out_dims[3]*sizeof(int8_t));
                    for(int i = 0; i < param->out_dims[3]; ++i){
                        output_ptr[i] = (in[i]) * scale;
                    }
                    output_ptr += param->out_dims[3];
                }
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
