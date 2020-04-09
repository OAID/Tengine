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

int ref_unary_fp16(__fp16* in_data, __fp16* out_data, int size,int type, float scale, int zero_point)
{
    // float* out_data = ( float* )data;
    switch(type){
        case 0:
            for(int i= 0; i < size; i++){
                out_data[i] = fp32_to_fp16(fabs(fp16_to_fp32(in_data[i])));
            }
            break;
        case 1:
            for(int i= 0; i < size; i++){
                out_data[i] =fp32_to_fp16( -(fp16_to_fp32(in_data[i])));
            }
            break;
        case 2:
            for(int i= 0; i < size; i++){
                out_data[i] = fp32_to_fp16(floor(fp16_to_fp32(in_data[i])));
            }
            break;
        case 3:
            for(int i= 0; i < size; i++){
                out_data[i] = fp32_to_fp16(ceil(fp16_to_fp32(in_data[i])));
            }
            break;
        case 4:
            for(int i= 0; i < size; i++){
                out_data[i] = fp32_to_fp16(fp16_to_fp32(in_data[i])*fp16_to_fp32(in_data[i]));
            }
            break;
        case 5:
            for(int i= 0; i < size; i++){
                out_data[i] = fp32_to_fp16(fabs(fp16_to_fp32(in_data[i])));
            }
            break;
        case 6:
            for(int i= 0; i < size; i++){
                out_data[i] = fp32_to_fp16(sqrt(fp16_to_fp32(in_data[i])));
            }
            break;
        case 7:
            for(int i= 0; i < size; i++){
                out_data[i] = fp32_to_fp16(1.f / sqrt(fp16_to_fp32(in_data[i])));
            }
            break;
        case 8:
            for(int i= 0; i < size; i++){
                out_data[i] = fp32_to_fp16(exp(fp16_to_fp32(in_data[i])));
            }
            break;
        case 9:
            for(int i= 0; i < size; i++){
                out_data[i] = fp32_to_fp16(log(fp16_to_fp32(in_data[i])));
            }
            break;
        case 10:
            for(int i= 0; i < size; i++){
                out_data[i] = fp32_to_fp16(sin(fp16_to_fp32(in_data[i])));
            }
            break;
        case 11:
            for(int i= 0; i < size; i++){
                out_data[i] = fp32_to_fp16(cos(fp16_to_fp32(in_data[i])));
            }
            break;
        case 12:
            for(int i= 0; i < size; i++){
                out_data[i] = fp32_to_fp16(asin(fp16_to_fp32(in_data[i])));
            }
            break;
        case 13:
            for(int i= 0; i < size; i++){
                out_data[i] = fp32_to_fp16(acos(fp16_to_fp32(in_data[i])));
            }
            break;
        case 14:
            for(int i= 0; i < size; i++){
                out_data[i] = fp32_to_fp16(atan(fp16_to_fp32(in_data[i])));
            }
            break;
        case 15:
            for(int i= 0; i < size; i++){
                out_data[i] = fp32_to_fp16(1.f / (fp16_to_fp32(in_data[i])));
            }
            break;
        case 16:
            for(int i= 0; i < size; i++){
                out_data[i] = fp32_to_fp16(tanh(fp16_to_fp32(in_data[i])));
            }
            break;
        default:
            break;
    }
    return 0;
}
