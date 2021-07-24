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
 * Parts of the following code in this file refs to
 * https://github.com/Tencent/ncnn/tree/master/src/layer/vulkan/
 * Tencent is pleased to support the open source community by making ncnn
 * available.
 *
 * Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
 *
 * Licensed under the BSD 3-Clause License (the "License"); you may not use this
 * file except in compliance with the License. You may obtain a copy of the
 * License at
 *
 * https://opensource.org/licenses/BSD-3-Clause
 */

/*
 * Copyright (c) 2020, Open AI Lab
 * Author: ddzhao@openailab.com
 */

#include "vulkan_tensor.hpp"

namespace TEngine {

void convert_packing(tensor* src, Tensor& dst, int elempack, const Option& opt)
{
    const Tensor _src = Tensor(src);
    // printf("convert packing ir_tensor to Tensor : %d %d %d %d %d\n", _src.c, _src.h, _src.w, _src.elempack, _src.elemsize);
}

void convert_packing(const Tensor& src, Tensor& dst, int _elempack, const Option& opt)
{
    int elempack = src.elempack;
    int out_elempack = _elempack;

    if (elempack == out_elempack)
    {
        dst = src;
        return;
    }

    int w = src.w;
    int h = src.h;
    int channels = src.c;
    int dims = src.dims;
    size_t elemsize = src.elemsize;

    if (dims == 1)
    {
        if (out_elempack == 1)
        {
            dst = src;
            dst.w = w * elempack;
            dst.cstep = w * elempack;
            dst.elemsize = elemsize / elempack;
            dst.elempack = out_elempack;
            return;
        }

        int outw = (w * elempack + out_elempack - 1) / out_elempack;
        size_t out_elemsize = elemsize / elempack * out_elempack;

        dst.create(outw, out_elemsize, out_elempack, opt.blob_allocator);
        if (dst.empty())
            return;

        memcpy(dst.data, src.data, w * elemsize);

        return;
    }

    if (dims == 2)
    {
        int outh = (h * elempack + out_elempack - 1) / out_elempack;
        size_t out_elemsize = elemsize / elempack * out_elempack;
        size_t lane_size = out_elemsize / out_elempack;

        dst.create(w, outh, out_elemsize, out_elempack, opt.blob_allocator);
        if (dst.empty())
            return;

        #pragma omp parallel for
        for (int i = 0; i < outh; i++)
        {
            unsigned char* outptr = (unsigned char*)dst + i * w * out_elemsize;

            for (int j = 0; j < w; j++)
            {
                unsigned char* out_elem_ptr = outptr + j * out_elemsize;

                for (int k = 0; k < out_elempack; k++)
                {
                    int srcy = (i * out_elempack + k) / elempack;
                    if (srcy >= h)
                        break;

                    int srck = (i * out_elempack + k) % elempack;

                    const unsigned char* ptr = (const unsigned char*)src + srcy * w * elemsize;
                    const unsigned char* elem_ptr = ptr + j * elemsize;
                    memcpy(out_elem_ptr + k * lane_size, elem_ptr + srck * lane_size, lane_size);
                }
            }
        }

        return;
    }

    if (dims == 3)
    {
        int outc = (channels * elempack + out_elempack - 1) / out_elempack;
        size_t out_elemsize = elemsize / elempack * out_elempack;
        size_t lane_size = out_elemsize / out_elempack;

        dst.create(w, h, outc, out_elemsize, out_elempack, opt.blob_allocator);
        if (dst.empty())
            return;

        #pragma omp parallel for
        for (int q = 0; q < outc; q++)
        {
            Tensor out = dst.channel(q);

            for (int i = 0; i < h; i++)
            {
                unsigned char* outptr = (unsigned char*)out + i * w * out_elemsize;

                for (int j = 0; j < w; j++)
                {
                    unsigned char* out_elem_ptr = outptr + j * out_elemsize;

                    for (int k = 0; k < out_elempack; k++)
                    {
                        int srcq = (q * out_elempack + k) / elempack;
                        if (srcq >= channels)
                            break;

                        int srck = (q * out_elempack + k) % elempack;

                        const Tensor m = src.channel(srcq);
                        const unsigned char* ptr = (const unsigned char*)m + i * w * elemsize;
                        const unsigned char* elem_ptr = ptr + j * elemsize;
                        memcpy(out_elem_ptr + k * lane_size, elem_ptr + srck * lane_size, lane_size);
                    }
                }
            }
        }

        return;
    }
}

unsigned short float32_to_float16(float value)
{
    // 1 : 8 : 23
    union
    {
        unsigned int u;
        float f;
    } tmp;

    tmp.f = value;

    // 1 : 8 : 23
    unsigned short sign = (tmp.u & 0x80000000) >> 31;
    unsigned short exponent = (tmp.u & 0x7F800000) >> 23;
    unsigned int significand = tmp.u & 0x7FFFFF;

    //     TLOG_INFO("%d %d %d", sign, exponent, significand);

    // 1 : 5 : 10
    unsigned short fp16;
    if (exponent == 0)
    {
        // zero or denormal, always underflow
        fp16 = (sign << 15) | (0x00 << 10) | 0x00;
    }
    else if (exponent == 0xFF)
    {
        // infinity or NaN
        fp16 = (sign << 15) | (0x1F << 10) | (significand ? 0x200 : 0x00);
    }
    else
    {
        // normalized
        short newexp = exponent + (-127 + 15);
        if (newexp >= 31)
        {
            // overflow, return infinity
            fp16 = (sign << 15) | (0x1F << 10) | 0x00;
        }
        else if (newexp <= 0)
        {
            // underflow
            if (newexp >= -10)
            {
                // denormal half-precision
                unsigned short sig = (significand | 0x800000) >> (14 - newexp);
                fp16 = (sign << 15) | (0x00 << 10) | sig;
            }
            else
            {
                // underflow
                fp16 = (sign << 15) | (0x00 << 10) | 0x00;
            }
        }
        else
        {
            fp16 = (sign << 15) | (newexp << 10) | (significand >> 13);
        }
    }

    return fp16;
}

float float16_to_float32(unsigned short value)
{
    // 1 : 5 : 10
    unsigned short sign = (value & 0x8000) >> 15;
    unsigned short exponent = (value & 0x7c00) >> 10;
    unsigned short significand = value & 0x03FF;

    //     TLOG_INFO("%d %d %d", sign, exponent, significand);

    // 1 : 8 : 23
    union
    {
        unsigned int u;
        float f;
    } tmp;
    if (exponent == 0)
    {
        if (significand == 0)
        {
            // zero
            tmp.u = (sign << 31);
        }
        else
        {
            // denormal
            exponent = 0;
            // find non-zero bit
            while ((significand & 0x200) == 0)
            {
                significand <<= 1;
                exponent++;
            }
            significand <<= 1;
            significand &= 0x3FF;
            tmp.u = (sign << 31) | ((-exponent + (-15 + 127)) << 23) | (significand << 13);
        }
    }
    else if (exponent == 0x1F)
    {
        // infinity or NaN
        tmp.u = (sign << 31) | (0xFF << 23) | (significand << 13);
    }
    else
    {
        // normalized
        tmp.u = (sign << 31) | ((exponent + (-15 + 127)) << 23) | (significand << 13);
    }

    return tmp.f;
}

void cast_float32_to_float16(const Tensor& src, Tensor& dst, const Option& opt)
{
    // printf("function cast_float32_to_float16 not done, fix me\n!!!!!");

    int w = src.w;
    int h = src.h;
    int channels = src.c;
    int dims = src.dims;
    size_t elemsize = src.elemsize;
    int elempack = src.elempack;

    size_t out_elemsize = 2 * elempack;

    if (dims == 1)
    {
        dst.create(w, out_elemsize, elempack, opt.blob_allocator);
    }
    else if (dims == 2)
    {
        dst.create(w, h, out_elemsize, elempack, opt.blob_allocator);
    }
    else if (dims == 3)
    {
        dst.create(w, h, channels, out_elemsize, elempack, opt.blob_allocator);
    }
    if (dst.empty())
        return ;

    int size = w * h * elempack;

    #pragma omp parallel for 
    for (int q = 0; q < channels; q++)
    {
        const float* ptr = src.channel(q);
        unsigned short* outptr = dst.channel(q);

        for (int i = 0; i < size; i++)
        {
            outptr[i] = float32_to_float16(ptr[i]);
        }
    }

}

void cast_float16_to_float32(const Tensor& src, Tensor& dst, const Option& opt)
{
    // printf("function cast_float16_to_float32 not done, fix me\n!!!!!");

    int w = src.w;
    int h = src.h;
    int channels = src.c;
    int dims = src.dims;
    size_t elemsize = src.elemsize;
    int elempack = src.elempack;

    size_t out_elemsize = 4 * elempack;

    if (dims == 1)
    {
        dst.create(w, out_elemsize, elempack, opt.blob_allocator);
    }
    else if (dims == 2)
    {
        dst.create(w, h, out_elemsize, elempack, opt.blob_allocator);
    }
    else if (dims == 3)
    {
        dst.create(w, h, channels, out_elemsize, elempack, opt.blob_allocator);
    }
    if (dst.empty())
        return ;

    int size = w * h * elempack;

    #pragma omp parallel for
    for (int q = 0; q < channels; q++)
    {
        const unsigned short* ptr = src.channel(q);
        float* outptr = dst.channel(q);

        for (int i = 0; i < size; i++)
        {
            outptr[i] = float16_to_float32(ptr[i]);
        }
    }

}

}   // namespace TEngine
