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

int ref_crop_int8(int8_t* input, int8_t* output,ref_crop_param* param)
{
    if(param->flag == 1){
        if(param->num_args == 1)
        {
            int offsetH = (param->iDataH - param->crop_h) / 2;
            int offsetW = (param->iDataW - param->crop_w) / 2;
            if((param->offset_h + param->oDataH <= param->iDataH) && (param->offset_w + param->oDataW <= param->iDataW))
            {
                for(int n = 0; n < param->oDataN; n++)
                {
                    for(int c = 0; c <param->oDataC; c++)
                    {
                        for(int h = 0; h < param->oDataH; h++)
                        {
                            int i_h = h + offsetH;
                            for(int w = 0; w < param->oDataW; w++)
                            {
                                int i_w = w + offsetW;
                                output[n * param->oDataC * param->oDataH * param->oDataW + c * param->oDataH * param->oDataW + h *param->oDataW + w] =
                                    input[n * param->iDataC * param->iDataH * param->iDataW + c * param->iDataH * param->iDataW + i_h * param->iDataW + i_w]*param->scale[0]/param->scale[1];
                            }
                        }
                    }
                }
            }
        }
        if(param->num_args == 2)
        {
            if((param->offset_h + param->oDataH <= param->iDataH) && (param->offset_w + param->oDataW <= param->iDataW))
            {
                for(int n = 0; n < param->oDataN; n++)
                {
                    for(int c = 0; c < param->oDataC; c++)
                    {
                        for(int h = 0; h < param->oDataH; h++)
                        {
                            int i_h = h + param->offset_h;
                            for(int w = 0; w < param->oDataW; w++)
                            {
                                int i_w = w + param->offset_w;
                                output[n * param->oDataC * param->oDataH * param->oDataW + c * param->oDataH * param->oDataW + h * param->oDataW + w] =
                                    input[n * param->iDataC * param->iDataH * param->iDataW + c * param->iDataH * param->iDataW + i_h * param->iDataW + i_w]*param->scale[0]/param->scale[1];
                            }
                        }
                    }
                }
            }
        }
    }
    if(param->flag == 0){
        if(param->axis == 1){
            for(int n = 0; n < param->oDataN; n++){
                for(int c = 0; c < param->oDataC; c++){
                    int i_c = param->offset_c + c;
                    for(int h = 0; h < param->oDataH; h++){
                        int i_h = param->offset_h + h;
                        for(int w = 0; w < param->oDataW; w++){
                            int i_w = param->offset_w + w;
                            output[n * param->oDataC * param->oDataH * param->oDataW + c * param->oDataH * param->oDataW + h * param->oDataW + w] =
                                input[n * param->iDataC * param->iDataH * param->iDataW + i_c * param->iDataH * param->iDataW + i_h * param->iDataW + i_w]*param->scale[0]/param->scale[1];
                        }
                    }
                }
            }
        }
        if(param->axis == 2){
            for(int n = 0; n < param->oDataN; n++){
                for(int c = 0; c < param->oDataC; c++){
                    for(int h = 0; h < param->oDataH; h++){
                        int i_h = param->offset_h + h;
                        for(int w = 0; w < param->oDataW; w++){
                            int i_w = param->offset_w + w;
                            output[n * param->oDataC * param->oDataH * param->oDataW + c * param->oDataH * param->oDataW + h * param->oDataW + w] =
                                input[n * param->iDataC * param->iDataH * param->iDataW + c * param->iDataH * param->iDataW + i_h * param->iDataW + i_w]*param->scale[0]/param->scale[1];
                        }
                    }
                }
            }
        }
    }
    return 0;
}
