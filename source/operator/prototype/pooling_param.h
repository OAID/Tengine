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

#ifndef __POOLING_PARAM_H__
#define __POOLING_PARAM_H__

#define COUNT_INCLUDE_PAD_MSK 0x010

enum
{
    POOL_MAX = 0,
    POOL_AVG
};

struct pool_param
{
    int pool_method; // 0:max    1:avg
    int kernel_h;
    int kernel_w;
    int stride_h;
    int stride_w;
    int pad_h0;
    int pad_h1;
    int pad_w0;
    int pad_w1;
    int global; // 0:general    1:global
    int caffe_flavor;
    void* funct;

    /* to support dynamic shape, need to save the original pad values*/
    int pad_h0_org;
    int pad_h1_org;
    int pad_w0_org;
    int pad_w1_org;
    void* input_pad;
};

static int calc_output_size(int input, int kernel, int stride, int pad, int caffe)
{
    int output = 1;
    if (pad >= 0)
    {
        if (1 == caffe)
        {
            output = 2 + ((input - kernel + 2 * pad - 1) / stride);
            if (pad > 0 && ((output - 1) * stride >= input + pad))
                output--;
        }
        else if (2 == caffe)
        {
            output = 1 + (input - kernel + pad) / stride;
        }
        else
            output = 1 + (input - kernel + 2 * pad) / stride;
    }
    else
    {
        output = 1 + (input - 1) / stride;
    }
    return output;
}

static void calc_real_pads(int out, int in, int kernel, int stride, const int pad_org, int* pad0, int* pad1)
{
    int total = (out - 1) * stride + kernel;
    int pad_num = total - in;

    if (pad_num < 0)
        pad_num = 0;

    /* for same */
    if (pad_org < 0)
    {
        *pad0 = pad_num / 2;
        *pad1 = pad_num - pad_org;
    }
    else
    {
        *pad0 = pad_org;
        *pad1 = pad_num - pad_org;
    }
}

#endif
