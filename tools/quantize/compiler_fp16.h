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
 * Copyright (c) 2020, OPEN AI LAB
 * Author: haitao@openailab.com
 */

#ifndef __COMPILIER_FP16_H__
#define __COMPILIER_FP16_H__

#ifdef MACOS

#else
#ifdef __cplusplus
extern "C" {
#endif

#if defined __ARM_ARCH || defined __riscv

#define fp16_to_fp32(data) \
    ({                     \
        float f = data;    \
        f;                 \
    })

#define fp32_to_fp16(data) \
    ({                     \
        __fp16 f = data;   \
        f;                 \
    })

#else
#ifdef _MSC_VER
#pragma  pack (push,1)
struct fp16_pack
{
    unsigned short frac : 10;
    unsigned char exp : 5;
    unsigned char sign : 1;
};

struct fp32_pack
{
    unsigned int frac : 23;
    unsigned char exp : 8;
    unsigned char sign : 1;
};
#pragma pack(pop)
#else
struct fp16_pack
{
    unsigned short frac : 10;
    unsigned char exp : 5;
    unsigned char sign : 1;
} __attribute__((packed));

struct fp32_pack
{
    unsigned int frac : 23;
    unsigned char exp : 8;
    unsigned char sign : 1;
} __attribute__((packed));
#endif

typedef struct fp16_pack __fp16;

static inline float fp16_to_fp32(__fp16 data)
{
    float f;
    struct fp32_pack* fp32 = ( struct fp32_pack* )&f;
    struct fp16_pack* fp16 = &data;

    int exp = fp16->exp;

    if(exp == 31 && fp16->frac != 0)
    {
        // return __builtin_inf()-__builtin_inf();
        fp32->sign = fp16->sign;
        fp32->exp = 255;
        fp32->frac = 1;

        return f;
    }

    if(exp == 31)
        exp = 255;
    if(exp == 0)
        exp = 0;
    else
        exp = (exp - 15) + 127;

    fp32->exp = exp;
    fp32->sign = fp16->sign;
    fp32->frac = (( int )fp16->frac) << 13;

    return f;
}

static inline __fp16 fp32_to_fp16(float data)
{
    struct fp32_pack* fp32 = ( struct fp32_pack* )&data;
    struct fp16_pack fp16;

    int exp = fp32->exp;

    if(fp32->exp == 255 && fp32->frac != 0)
    {
        // NaN
        fp16.exp = 31;
        fp16.frac = 1;
        fp16.sign = fp32->sign;

        return fp16;
    }

    if((exp - 127) < -14)
        exp = 0;
    else if((exp - 127) > 15)
        exp = 31;
    else
        exp = exp - 127 + 15;

    fp16.exp = exp;
    fp16.frac = fp32->frac >> 13;
    fp16.sign = fp32->sign;

    return fp16;
}
#endif

#endif

#ifdef __cplusplus
}
#endif
#endif
