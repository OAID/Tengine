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
 * Author: lswang@openailab.com
 */

#pragma once

#include <stdint.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#ifdef __GNUC__
#define PACKAGE_MARK __attribute__((packed))
#else
#define PACKAGE_MARK
#endif

// IEEE 754
// ISO/IEC/IEEE FDIS 60559:2010


#define BF16_EXP_MAX  ( 256 - 1)   //  2^8 - 1
#define FP16_EXP_MAX  (  32 - 1)   //  2^5 - 1
#define FP32_EXP_MAX  ( 256 - 1)   //  2^8 - 1
#define FP64_EXP_MAX  (2048 - 1)   // 2^11 - 1

#define FP16_NAN      ((FP16_EXP_MAX << 10) + 1)
#define FP16_INF      ((FP16_EXP_MAX << 10) + 0)
#define BF16_NAN      ((BF16_EXP_MAX <<  7) + 1)
#define BF16_INF      ((BF16_EXP_MAX <<  7) + 0)
#define FP32_NAN      ((FP32_EXP_MAX << 23) + 1)
#define FP32_INF      ((FP32_EXP_MAX << 23) + 0)


#ifdef _MSC_VER
#pragma pack (push,1)
#endif
typedef union fp16_pack
{
    struct
    {
        uint16_t frac : 10;
        uint16_t exp  :  5;
        uint16_t sign :  1;
    } PACKAGE_MARK;
    uint16_t value;
} PACKAGE_MARK fp16_pack_t;

typedef union fp32_pack
{
    struct
    {
        uint32_t frac : 23;
        uint32_t exp  :  8;
        uint32_t sign :  1;
    } PACKAGE_MARK;
    float value;
} PACKAGE_MARK fp32_pack_t;
#ifdef _MSC_VER
#pragma pack(pop)
#endif


#ifdef __ARM_ARCH
typedef __fp16      fp16_t;
#else
typedef fp16_pack_t fp16_t;
#endif
typedef float       fp32_t;


#ifdef __ARM_ARCH
#define fp16_to_fp32(data) ({ float f = data; f; })
#define fp32_to_fp16(data) ({ __fp16 f = data; f; })
#else
/*!
 * @brief  Convert a number from float16 to float32.
 *
 * @param [in]  package: Input float16 precision number.
 *
 * @return  The converted float32 precision number.
 */
fp32_t fp16_to_fp32(fp16_t package);


/*!
 * @brief  Convert a number from float32 to float16.
 *
 * @param [in]  package: Input float32 precision number.
 *
 * @return  The converted float16 precision number.
 */
fp16_t fp32_to_fp16(fp32_t package);
#endif



#ifndef __ARM_ARCH
fp32_t fp16_to_fp32(fp16_t package)
{
    fp32_pack_t data;

    // means 0
    if (0 == package.exp && 0 == package.frac)
    {
        data.value = 0;
        data.sign = package.sign;

        return data.value;
    }

    // means normalized value
    if (FP16_EXP_MAX != package.exp && 0 != package.exp && 0 != package.frac)
    {
        data.frac = package.frac << 13;
        data.exp  = package.exp + (- 15 + 127);
        data.sign = package.sign;

        return data.value;
    }

    // means infinite
    if (FP16_EXP_MAX == package.exp && 0 == package.frac)
    {
        data.frac = 0;
        data.exp  = FP32_EXP_MAX;
        data.sign = package.sign;

        return data.value;
    }

    // means NaN
    if (FP16_EXP_MAX == package.exp && 0 != package.frac)
    {
        data.frac = 1;
        data.exp  = FP32_EXP_MAX;
        data.sign = package.sign;

        return data.value;
    }

    // means subnormal numbers
    if (0 == package.exp && 0 != package.frac)
    {
        uint16_t frac = package.frac;
        uint16_t exp  = 0;

        while (0 == (frac & (uint16_t)0x200))
        {
            frac <<= 1;
            exp++;
        }

        data.frac = (frac << 1) & (uint16_t)0x3FF;
        data.exp  = -exp + (-15 + 127);
        data.sign = package.sign;

        return data.value;
    }

    return data.value;
}


fp16_t fp32_to_fp16(fp32_t value)
{
    fp32_pack_t* package = (fp32_pack_t*)(&value);
    fp16_t data;

    // means 0 or subnormal numbers, and subnormal numbers means underflow
    if (0 == package->exp)
    {
        data.value = 0;
        data.sign  = package->sign;

        return data;
    }

    // means normalized value
    if (FP32_EXP_MAX != package->exp && 0 != package->exp && 0 != package->frac)
    {
        int16_t exp  = package->exp + (-15 + 127);

        // means overflow
        if (31 <= exp)
        {
            data.frac = 0;
            data.exp  = FP16_EXP_MAX;
            data.sign = package->sign;
        }
        else if (0 >= exp)
        {
            // means subnormal numbers
            if (-10 <= exp)
            {
                data.frac  = (package->frac | 0x800000) >> (14 - exp);
                data.exp   = 0;
                data.sign  = package->sign;
            }
            // means underflow
            else
            {
                data.value = 0;
                data.sign  = package->sign;
            }
        }
        else
        {
            data.frac = package->frac >> 13;
            data.exp  = exp;
            data.sign = package->sign;
        }

        return data;
    }

    // means infinite
    if (FP32_EXP_MAX == package->exp && 0 == package->frac)
    {
        data.frac = 0;
        data.exp  = FP16_EXP_MAX;
        data.sign = package->sign;

        return data;
    }

    // means NaN
    if (FP32_EXP_MAX == package->exp && 0 != package->frac)
    {
        data.frac = 1;
        data.exp  = FP16_EXP_MAX;
        data.sign = package->sign;

        return data;
    }

    data.value = 0;
    return data;
}
#endif
