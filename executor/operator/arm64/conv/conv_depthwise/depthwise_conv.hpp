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
 * Author: haoluo@openailab.com
 */
 
#ifndef __DEPTHWISE_CONV_HPP
#define __DEPTHWISE_CONV_HPP


void depthwise_conv_k5s1(float* input_buf, float* weight_buf, float* bias, float* output_buf, int input_h, int input_w,
                int channel_num, int out_h, int out_w, int pad0, int pad1, int activation);
                
void depthwise_conv_k5s2(float* input_buf, float* weight_buf, float* bias, float* output_buf, int input_h, int input_w,
                int channel_num, int out_h, int out_w, int activation);
                
void depthwise_conv_k7s1(float* input_buf, float* weight_buf, float* bias, float* output_buf, int input_h, int input_w,
                int channel_num, int out_h, int out_w, int activation);
                
void depthwise_conv_k7s2(float* input_buf, float* weight_buf, float* bias, float* output_buf, int input_h, int input_w,
                int channel_num, int out_h, int out_w, int activation);


#endif

