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
 * Copyright (c) 2021, OPEN AI LAB
 * Author: haitao@openailab.com
 */

#ifndef __DECONV_PARAM_H__
#define __DECONV_PARAM_H__

struct deconv_param
{
    int num_output;
    int kernel_h;
    int kernel_w;
    int stride_h;
    int stride_w;
    int pad_h0;
    int pad_w0;
    int pad_h1;
    int pad_w1;
    int dilation_h;
    int dilation_w;
    int group;
    int activation;
    int output_pad_h0;
    int output_pad_w0;
};

#endif
