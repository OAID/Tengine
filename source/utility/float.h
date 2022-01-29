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

#ifdef _MSC_VER
#pragma pack(push, 1)
#endif
typedef union fp16_pack
{
    struct
    {
        uint16_t frac : 10;
        uint16_t exp : 5;
        uint16_t sign : 1;
    } PACKAGE_MARK;
    uint16_t value;
} PACKAGE_MARK fp16_pack_t;

typedef union bf16_pack
{
    struct
    {
        uint16_t frac : 7;
        uint16_t exp : 8;
        uint16_t sign : 1;
    } PACKAGE_MARK;
    uint16_t value;
} PACKAGE_MARK bf16_pack_t;

#ifdef _MSC_VER
typedef struct afp24_pack
{
    uint16_t frac : 16;
    uint8_t exp : 7;
    uint8_t sign : 1;
} afp24_pack_t;

typedef struct pxr24_pack
{
    uint16_t frac : 15;
    uint16_t : 1;
    uint8_t : 7;
    uint8_t sign : 1;
} pxr24_pack_t;
#else
typedef struct afp24_pack
{
    uint32_t frac : 16;
    uint32_t exp : 7;
    uint32_t sign : 1;
} PACKAGE_MARK afp24_pack_t;

typedef struct pxr24_pack
{
    uint32_t frac : 15;
    uint32_t exp : 8;
    uint32_t sign : 1;
} PACKAGE_MARK pxr24_pack_t;
#endif

typedef union fp32_pack
{
    struct
    {
        uint32_t frac : 23;
        uint32_t exp : 8;
        uint32_t sign : 1;
    } PACKAGE_MARK;
    float value;
} PACKAGE_MARK fp32_pack_t;

typedef union fp64_pack
{
    struct
    {
        uint64_t frac : 52;
        uint64_t exp : 11;
        uint64_t sign : 1;
    } PACKAGE_MARK;
    double value;
} PACKAGE_MARK fp64_pack_t;
#ifdef _MSC_VER
#pragma pack(pop)
#endif

#ifdef __ARM_ARCH
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
typedef __fp16 fp16_t;
#else
typedef fp16_pack_t fp16_t;
#endif
#else
typedef fp16_pack_t fp16_t;
#endif
typedef bf16_pack_t bf16_t;
typedef float fp32_t;
typedef double fp64_t;

#ifndef __ARM_ARCH
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

/*!
 * @brief  Convert a number from float16 to float32.
 *
 * @param [in]  package: Input Brain Floating Point precision number.
 *
 * @return  The converted float32 precision number.
 */
fp32_t bf16_to_fp32(bf16_t package);

/*!
 * @brief  Convert a number from float32 to float16.
 *
 * @param [in]  package: Input float32 precision number.
 *
 * @return  The converted Brain Floating Point precision number.
 */
bf16_t fp32_to_bf16(fp32_t package);

#ifdef __ARM_ARCH

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
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

#endif
