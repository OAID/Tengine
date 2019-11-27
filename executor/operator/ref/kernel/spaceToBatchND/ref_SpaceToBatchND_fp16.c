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

static int ref_spaceToBatchND_fp16(const __fp16* in_data, __fp16* out_data, struct spaceToBatchND_param* param)
{
    /*
    int pad_value = 0;

    for(int out_b = 0; out_b < param->out_dims[0]; ++out_b){
        int input_batch = out_b % param->in_dims[0];
        int shift_w = (out_b / param->in_dims[0]) % param->dilation_x;
        int shift_h = (out_b / param->in_dims[0]) / param->dilation_x;
        for(int out_h = 0; out_h < param->out_dims[1]; ++out_h){
            for(int out_w = 0; out_w < param->out_dims[2]; ++out_w){
                int index = out_b * param->out_dims[1] * param->out_dims[2] * param->out_dims[3] + out_h * param->out_dims[2] * param->out_dims[3] + out_w * param->out_dims[3] + param->out_dims[3];
                __fp16* out = out_data + index;
                if( (out_h * param->dilation_y + shift_h < param->pad_top) || (out_h * param->dilation_y + shift_h >= param->pad_top + param->in_dims[1])
                    || (out_w * param->dilation_x + shift_w < param->pad_left) || (out_w *param->dilation_x + shift_w  >= param->pad_left + param->in_dims[2]) )
                {
                    memset(out, pad_value, param->out_dims[3]*sizeof(__fp16));
                }
                else
                {
                    int index_h = out_h * param->dilation_y + shift_h - param->pad_top;
                    int index_w = out_w * param->dilation_x + shift_w - param->pad_left;
                    int index = input_batch * param->in_dims[1] * param->in_dims[2] * param->in_dims[3] + index_h * param->in_dims[2] * param->in_dims[3] + index_w * param->in_dims[3] + param->in_dims[3];
                    const __fp16* in = in_data + index;
                    memcpy(out, in, param->out_dims[3]*sizeof(__fp16));
                }
            }
        }
    }
    */

        __fp16* output_ptr = out_data;
        int output_batch_size = param->out_dims[0];
        int input_batch_size = param->in_dims[0];

        int block_shape_width = param->dilation_x;
        int block_shape_height = param->dilation_y;
        int output_height = param->out_dims[1];
        int output_width = param->out_dims[2];
        int padding_top = param->pad_top;
        //int padding_right = param->pad_right;
        //int padding_bottom = param->pad_bottom;
        int padding_left = param->pad_left;
        int input_height = param->in_dims[1];
        int input_width = param->in_dims[2];
        int depth = param->in_dims[3];

        int out_stride_depth = param->out_dims[3];
        int out_stride_width = out_stride_depth * param->out_dims[2];
        int out_stride_height = out_stride_width * param->out_dims[1];

        int in_stride_depth = param->in_dims[3];
        int in_stride_width = in_stride_depth * param->in_dims[2];
        int in_stride_height = in_stride_width * param->in_dims[1];
        printf("%d %d %d %d %d %d \n",out_stride_depth,out_stride_width,out_stride_height,in_stride_depth,in_stride_width,in_stride_height );
        for (int out_b = 0; out_b < output_batch_size; ++out_b) {
            int input_batch = out_b % input_batch_size;
            int shift_w = (out_b / input_batch_size) % block_shape_width;
            int shift_h = (out_b / input_batch_size) / block_shape_width;
            for (int out_h = 0; out_h < output_height; ++out_h) {
                for (int out_w = 0; out_w < output_width; ++out_w) {
                    __fp16 * out = output_ptr + out_b*out_stride_height + out_h*out_stride_width + out_w*out_stride_depth;
                    if (out_h * block_shape_height + shift_h < padding_top ||
                        out_h * block_shape_height + shift_h >=
                        padding_top + input_height ||
                        out_w * block_shape_width + shift_w < padding_left ||
                        out_w * block_shape_width + shift_w >= padding_left + input_width) {
                        // This may not execute correctly when pad_value != 0 and T != uint8.
                        memset(out,0, depth * sizeof(__fp16));
                    } 
                    else 
                    {
                        const __fp16 * in = in_data + input_batch*in_stride_height +
                            ((out_h * block_shape_height + shift_h) - padding_top)*in_stride_width +
                            ((out_w * block_shape_width + shift_w) - padding_left)*in_stride_depth;
                            memcpy(out, in, depth * sizeof(__fp16));
                    }
                }
            }
        }


    return 0;
}
