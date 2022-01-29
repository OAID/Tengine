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

#include "utility/float.h"

#define BF16_EXP_MAX (256 - 1)  //  2^8 - 1
#define FP16_EXP_MAX (32 - 1)   //  2^5 - 1
#define FP32_EXP_MAX (256 - 1)  //  2^8 - 1
#define FP64_EXP_MAX (2048 - 1) // 2^11 - 1

#define FP16_NAN ((FP16_EXP_MAX << 10) + 1)
#define FP16_INF ((FP16_EXP_MAX << 10) + 0)
#define BF16_NAN ((BF16_EXP_MAX << 7) + 1)
#define BF16_INF ((BF16_EXP_MAX << 7) + 0)
#define FP32_NAN ((FP32_EXP_MAX << 23) + 1)
#define FP32_INF ((FP32_EXP_MAX << 23) + 0)

#if !defined(__ARM_ARCH) || (defined(__ARM_ARCH) && (0 == __ARM_FEATURE_FP16_VECTOR_ARITHMETIC))

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
        data.exp = package.exp + (-15 + 127);
        data.sign = package.sign;

        return data.value;
    }

    // means infinite
    if (FP16_EXP_MAX == package.exp && 0 == package.frac)
    {
        data.frac = 0;
        data.exp = FP32_EXP_MAX;
        data.sign = package.sign;

        return data.value;
    }

    // means NaN
    if (FP16_EXP_MAX == package.exp && 0 != package.frac)
    {
        data.frac = 1;
        data.exp = FP32_EXP_MAX;
        data.sign = package.sign;

        return data.value;
    }

    // means subnormal numbers
    if (0 == package.exp && 0 != package.frac)
    {
        uint16_t frac = package.frac;
        uint16_t exp = 0;

        while (0 == (frac & (uint16_t)0x200))
        {
            frac <<= 1;
            exp++;
        }

        data.frac = (frac << 1) & (uint16_t)0x3FF;
        data.exp = -exp + (-15 + 127);
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
        data.sign = package->sign;

        return data;
    }

    // means normalized value
    if (FP32_EXP_MAX != package->exp && 0 != package->exp && 0 != package->frac)
    {
        int16_t exp = package->exp + (-15 + 127);

        // means overflow
        if (31 <= exp)
        {
            data.frac = 0;
            data.exp = FP16_EXP_MAX;
            data.sign = package->sign;
        }
        else if (0 >= exp)
        {
            // means subnormal numbers
            if (-10 <= exp)
            {
                data.frac = (package->frac | 0x800000) >> (14 - exp);
                data.exp = 0;
                data.sign = package->sign;
            }
            // means underflow
            else
            {
                data.value = 0;
                data.sign = package->sign;
            }
        }
        else
        {
            data.frac = package->frac >> 13;
            data.exp = exp;
            data.sign = package->sign;
        }

        return data;
    }

    // means infinite
    if (FP32_EXP_MAX == package->exp && 0 == package->frac)
    {
        data.frac = 0;
        data.exp = FP16_EXP_MAX;
        data.sign = package->sign;

        return data;
    }

    // means NaN
    if (FP32_EXP_MAX == package->exp && 0 != package->frac)
    {
        data.frac = 1;
        data.exp = FP16_EXP_MAX;
        data.sign = package->sign;

        return data;
    }

    data.value = 0;
    return data;
}
#endif

fp32_t bf16_to_fp32(bf16_t package)
{
    fp32_pack_t data;
    data.value = (float)((uint32_t)(package.value) << 16);
    return data.value;
}

bf16_t fp32_to_bf16(fp32_t value)
{
    fp32_pack_t* package = (fp32_pack_t*)(&value);
    bf16_t data;
    data.value = (int32_t)(package->value) >> 16;
    return data;
}

#ifndef _MSC_VER
fp32_t pxr24_to_fp32(pxr24_pack_t package)
{
    fp32_pack_t data;

    uint32_t float_val = (*(uint32_t*)(&package) & (uint32_t)(0x00FFFFFF)) << 8;
    data.value = (float)(float_val);

    return data.value;
}

pxr24_pack_t fp32_to_pxr24(fp32_t value)
{
    fp32_pack_t* package = (fp32_pack_t*)(&value);
    pxr24_pack_t data;

    uint32_t pxr24_val = (uint32_t)(package->value) >> 8;
    pxr24_pack_t* ptr = (pxr24_pack_t*)((uint8_t*)(&pxr24_val));

    data.frac = ptr->frac;
    data.exp = ptr->exp;
    data.sign = ptr->sign;

    return data;
}
#endif
