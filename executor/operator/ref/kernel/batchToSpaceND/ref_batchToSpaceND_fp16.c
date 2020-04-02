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

static int ref_batchToSpaceND_fp16(const __fp16* in_data, __fp16* out_data, struct batchToSpaceND_param* param)
{
    for(int in_batch = 0; in_batch < param->in_dims[0]; ++in_batch){
        const int out_batch = in_batch % param->out_dims[0];
        const int spatial_offset = in_batch / param->out_dims[0];
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
                __fp16* out = out_data + outOffset;
                int inOffset = in_batch*param->in_dims[1]*param->in_dims[2]*param->in_dims[3] + in_h *param->in_dims[2] *param->in_dims[3] + in_w *param->in_dims[3];
                const __fp16* in = in_data + inOffset;
                memcpy(out, in, param->in_dims[3] * sizeof(__fp16));
            }
        }
    }

    return 0;
}
