/*
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef NVDLA_PRIV_SURFACE_ENUMS_H
#define NVDLA_PRIV_SURFACE_ENUMS_H

#define SURFACE_CATEGORY_ENUMS(op) \
    op(SURFACE_CATEGORY_UNKNOWN, 0U)    \
    op(IMG,              1U) /* input pixel data */                            \
    op(WEIGHT,           2U) /* Kernel weight data */                          \
    op(FEATURE_DATA,     3U) /* N(C/X)HWC - dla specific feature data format */\
    op(M_PLANAR,         4U) /* NCHW - format to talk with other engines */    \
    op(BIAS_DATA,        5U) /* Bias Data */                                   \
    op(BATCH_NORM_DATA,  6U) /* Batch Norm Data */                             \
    op(SCALE_DATA,       7U) /* Scale Data */
    /* other formats follow like mean data, PReLU data, etc */

#define SURFACE_PRECISION_ENUMS(op) \
    op(NVDLA_PRECISION_UNKNOWN, 0U) \
    op(NVDLA_PRECISION_INT8,    1U) \
    op(NVDLA_PRECISION_INT16,   2U) \
    op(NVDLA_PRECISION_FP16,    3U) \
    op(NVDLA_PRECISION_UINT8,   4U) \
    op(NVDLA_PRECISION_UINT16,  5U) \

#define BIAS_DATA_CATEGORY_ENUMS(op)    \
    op(BIAS_DATA_CATEGORY_UNKNOWN, 0U)  \
    op(PER_LAYER_BIAS_DATA,     1U)     \
    op(PER_CHANNEL_BIAS_DATA,   2U)     \
    op(PER_ELEMENT_BIAS_DATA,   3U)

#define BATCH_NORM_DATA_CATEGORY_ENUMS(op)      \
    op(BATCH_NORM_DATA_CATEGORY_UNKNOWN, 0U)    \
    op(PER_LAYER_BATCH_NORM_DATA,        1U)    \
    op(PER_CHANNEL_BATCH_NORM_DATA,      2U)

#define SCALE_DATA_CATEGORY_ENUMS(op)   \
    op(SCALE_DATA_CATEGORY_UNKNOWN, 0U) \
    op(PER_LAYER_SCALE_DATA,        1U) \
    op(PER_CHANNEL_SCALE_DATA,      2U) \
    op(PER_ELEMENT_SCALE_DATA,      3U)

#define PIXEL_MAPPING_ENUMS(op) \
    op(PIXEL_MAPPING_UNKNOWN, 0U)   \
    op(PITCH_LINEAR,          1U)

/* Enum_Name : Surface_Category : Precision : Bytes-Per-Element : Channels-Per-Atom : Enum_Ordinal */
#define SURFACE_FORMAT_ENUMS(op)   \
        op(NVDLA_IMG_R8,                  surface::SurfaceCategoryEnum::IMG,  surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT8,  1, 1, 0U) \
        op(NVDLA_IMG_R10,                 surface::SurfaceCategoryEnum::IMG,  surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT16, 2, 1, 1U) \
        op(NVDLA_IMG_R12,                 surface::SurfaceCategoryEnum::IMG,  surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT16, 2, 1, 2U) \
        op(NVDLA_IMG_R16,                 surface::SurfaceCategoryEnum::IMG,  surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT16, 2, 1, 3U) \
        op(NVDLA_IMG_R16_I,               surface::SurfaceCategoryEnum::IMG,  surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT16, 2, 1, 4U) \
        op(NVDLA_IMG_R16_F,               surface::SurfaceCategoryEnum::IMG,  surface::SurfacePrecisionEnum::NVDLA_PRECISION_FP16,  2, 1, 5U) \
        op(NVDLA_IMG_A16B16G16R16,        surface::SurfaceCategoryEnum::IMG,  surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT16, 2, 4, 6U) \
        op(NVDLA_IMG_X16B16G16R16,        surface::SurfaceCategoryEnum::IMG,  surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT16, 2, 4, 7U) \
        op(NVDLA_IMG_A16B16G16R16_F,      surface::SurfaceCategoryEnum::IMG,  surface::SurfacePrecisionEnum::NVDLA_PRECISION_FP16,  2, 4, 8U) \
        op(NVDLA_IMG_A16Y16U16V16,        surface::SurfaceCategoryEnum::IMG,  surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT16, 2, 4, 9U) \
        op(NVDLA_IMG_V16U16Y16A16,        surface::SurfaceCategoryEnum::IMG,  surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT16, 2, 4, 10U) \
        op(NVDLA_IMG_A16Y16U16V16_F,      surface::SurfaceCategoryEnum::IMG,  surface::SurfacePrecisionEnum::NVDLA_PRECISION_FP16,  2, 4, 11U) \
        op(NVDLA_IMG_A8B8G8R8,            surface::SurfaceCategoryEnum::IMG,  surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT8,  1, 4, 12U) \
        op(NVDLA_IMG_A8R8G8B8,            surface::SurfaceCategoryEnum::IMG,  surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT8,  1, 4, 13U) \
        op(NVDLA_IMG_B8G8R8A8,            surface::SurfaceCategoryEnum::IMG,  surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT8,  1, 4, 14U) \
        op(NVDLA_IMG_R8G8B8A8,            surface::SurfaceCategoryEnum::IMG,  surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT8,  1, 4, 15U) \
        op(NVDLA_IMG_X8B8G8R8,            surface::SurfaceCategoryEnum::IMG,  surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT8,  1, 4, 16U) \
        op(NVDLA_IMG_X8R8G8B8,            surface::SurfaceCategoryEnum::IMG,  surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT8,  1, 4, 17U) \
        op(NVDLA_IMG_B8G8R8X8,            surface::SurfaceCategoryEnum::IMG,  surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT8,  1, 4, 18U) \
        op(NVDLA_IMG_R8G8B8X8,            surface::SurfaceCategoryEnum::IMG,  surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT8,  1, 4, 19U) \
        op(NVDLA_IMG_A2B10G10R10,         surface::SurfaceCategoryEnum::IMG,  surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT16, 2, 4, 20U) \
        op(NVDLA_IMG_A2R10G10B10,         surface::SurfaceCategoryEnum::IMG,  surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT16, 2, 4, 21U) \
        op(NVDLA_IMG_B10G10R10A2,         surface::SurfaceCategoryEnum::IMG,  surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT16, 2, 4, 22U) \
        op(NVDLA_IMG_R10G10B10A2,         surface::SurfaceCategoryEnum::IMG,  surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT16, 2, 4, 23U) \
        op(NVDLA_IMG_A2Y10U10V10,         surface::SurfaceCategoryEnum::IMG,  surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT16, 2, 4, 24U) \
        op(NVDLA_IMG_V10U10Y10A2,         surface::SurfaceCategoryEnum::IMG,  surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT16, 2, 4, 25U) \
        op(NVDLA_IMG_A8Y8U8V8,            surface::SurfaceCategoryEnum::IMG,  surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT8,  1, 4, 26U) \
        op(NVDLA_IMG_V8U8Y8A8,            surface::SurfaceCategoryEnum::IMG,  surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT8,  1, 4, 27U) \
        op(NVDLA_IMG_Y8___U8V8_N444,      surface::SurfaceCategoryEnum::IMG,  surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT8,  1, 3, 28U) \
        op(NVDLA_IMG_Y8___V8U8_N444,      surface::SurfaceCategoryEnum::IMG,  surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT8,  1, 3, 29U) \
        op(NVDLA_IMG_Y10___U10V10_N444,   surface::SurfaceCategoryEnum::IMG,  surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT16, 2, 3, 30U) \
        op(NVDLA_IMG_Y10___V10U10_N444,   surface::SurfaceCategoryEnum::IMG,  surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT16, 2, 3, 31U) \
        op(NVDLA_IMG_Y12___U12V12_N444,   surface::SurfaceCategoryEnum::IMG,  surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT16, 2, 3, 32U) \
        op(NVDLA_IMG_Y12___V12U12_N444,   surface::SurfaceCategoryEnum::IMG,  surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT16, 2, 3, 33U) \
        op(NVDLA_IMG_Y16___U16V16_N444,   surface::SurfaceCategoryEnum::IMG,  surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT16, 2, 3, 34U) \
        op(NVDLA_IMG_Y16___V16U16_N444,   surface::SurfaceCategoryEnum::IMG,  surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT16, 2, 3, 35U) \
        op(NVDLA_WEIGHT_DC_INT8,                surface::SurfaceCategoryEnum::WEIGHT,  surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT8,  1, -1, 36U) \
        op(NVDLA_WEIGHT_DC_INT8_COMPRESSED,     surface::SurfaceCategoryEnum::WEIGHT,  surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT8,  1, -1, 37U) \
        op(NVDLA_WEIGHT_WG_INT8,                surface::SurfaceCategoryEnum::WEIGHT,  surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT8,  1, -1, 38U) \
        op(NVDLA_WEIGHT_WG_INT8_COMPRESSED,     surface::SurfaceCategoryEnum::WEIGHT,  surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT8,  1, -1, 39U) \
        op(NVDLA_WEIGHT_IMG_INT8,               surface::SurfaceCategoryEnum::WEIGHT,  surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT8,  1, -1, 40U) \
        op(NVDLA_WEIGHT_IMG_INT8_COMPRESSED,    surface::SurfaceCategoryEnum::WEIGHT,  surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT8,  1, -1, 41U) \
        op(NVDLA_WEIGHT_DECONV_INT8,            surface::SurfaceCategoryEnum::WEIGHT,  surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT8,  1, -1, 42U) \
        op(NVDLA_WEIGHT_DECONV_INT8_COMPRESSED, surface::SurfaceCategoryEnum::WEIGHT,  surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT8,  1, -1, 43U) \
        op(NVDLA_WEIGHT_DC_INT16,               surface::SurfaceCategoryEnum::WEIGHT,  surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT16, 2, -1, 44U) \
        op(NVDLA_WEIGHT_DC_INT16_COMPRESSED,    surface::SurfaceCategoryEnum::WEIGHT,  surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT16, 2, -1, 45U) \
        op(NVDLA_WEIGHT_WG_INT16,               surface::SurfaceCategoryEnum::WEIGHT,  surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT16, 2, -1, 46U) \
        op(NVDLA_WEIGHT_WG_INT16_COMPRESSED,    surface::SurfaceCategoryEnum::WEIGHT,  surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT16, 2, -1, 47U) \
        op(NVDLA_WEIGHT_IMG_INT16,              surface::SurfaceCategoryEnum::WEIGHT,  surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT16, 2, -1, 48U) \
        op(NVDLA_WEIGHT_IMG_INT16_COMPRESSED,   surface::SurfaceCategoryEnum::WEIGHT,  surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT16, 2, -1, 49U) \
        op(NVDLA_WEIGHT_DECONV_INT16,           surface::SurfaceCategoryEnum::WEIGHT,  surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT16, 2, -1, 50U) \
        op(NVDLA_WEIGHT_DECONV_INT16_COMPRESSED,surface::SurfaceCategoryEnum::WEIGHT,  surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT16, 2, -1, 51U) \
        op(NVDLA_WEIGHT_DC_FP16,                surface::SurfaceCategoryEnum::WEIGHT,  surface::SurfacePrecisionEnum::NVDLA_PRECISION_FP16,  2, -1, 52U) \
        op(NVDLA_WEIGHT_DC_FP16_COMPRESSED,     surface::SurfaceCategoryEnum::WEIGHT,  surface::SurfacePrecisionEnum::NVDLA_PRECISION_FP16,  2, -1, 53U) \
        op(NVDLA_WEIGHT_WG_FP16,                surface::SurfaceCategoryEnum::WEIGHT,  surface::SurfacePrecisionEnum::NVDLA_PRECISION_FP16,  2, -1, 54U) \
        op(NVDLA_WEIGHT_WG_FP16_COMPRESSED,     surface::SurfaceCategoryEnum::WEIGHT,  surface::SurfacePrecisionEnum::NVDLA_PRECISION_FP16,  2, -1, 55U) \
        op(NVDLA_WEIGHT_IMG_FP16,               surface::SurfaceCategoryEnum::WEIGHT,  surface::SurfacePrecisionEnum::NVDLA_PRECISION_FP16,  2, -1, 56U) \
        op(NVDLA_WEIGHT_IMG_FP16_COMPRESSED,    surface::SurfaceCategoryEnum::WEIGHT,  surface::SurfacePrecisionEnum::NVDLA_PRECISION_FP16,  2, -1, 57U) \
        op(NVDLA_WEIGHT_DECONV_FP16,            surface::SurfaceCategoryEnum::WEIGHT,  surface::SurfacePrecisionEnum::NVDLA_PRECISION_FP16,  2, -1, 58U) \
        op(NVDLA_WEIGHT_DECONV_FP16_COMPRESSED, surface::SurfaceCategoryEnum::WEIGHT,  surface::SurfacePrecisionEnum::NVDLA_PRECISION_FP16,  2, -1, 59U) \
        op(NVDLA_BIAS_DATA_INT8,                surface::SurfaceCategoryEnum::BIAS_DATA,  surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT8,  1, -1, 60U) \
        op(NVDLA_BIAS_DATA_INT16,               surface::SurfaceCategoryEnum::BIAS_DATA,  surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT16, 2, -1, 61U) \
        op(NVDLA_BIAS_DATA_FP16,                surface::SurfaceCategoryEnum::BIAS_DATA,  surface::SurfacePrecisionEnum::NVDLA_PRECISION_FP16,  2, -1, 62U) \
        op(NVDLA_FEATURE_DATA_INT8,       surface::SurfaceCategoryEnum::FEATURE_DATA,  surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT8,  1, -1, 63U) \
        op(NVDLA_FEATURE_DATA_INT16,      surface::SurfaceCategoryEnum::FEATURE_DATA,  surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT16, 2, -1, 64U) \
        op(NVDLA_FEATURE_DATA_FP16,       surface::SurfaceCategoryEnum::FEATURE_DATA,  surface::SurfacePrecisionEnum::NVDLA_PRECISION_FP16,  2, -1, 65U) \
        op(NVDLA_M_PLANAR_INT8,           surface::SurfaceCategoryEnum::M_PLANAR,      surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT8,  1, -1, 66U) \
        op(NVDLA_M_PLANAR_INT16,          surface::SurfaceCategoryEnum::M_PLANAR,      surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT16, 2, -1, 67U) \
        op(NVDLA_M_PLANAR_FP16,           surface::SurfaceCategoryEnum::M_PLANAR,      surface::SurfacePrecisionEnum::NVDLA_PRECISION_FP16,  2, -1, 68U) \
        op(NVDLA_BATCH_NORM_DATA_INT8,    surface::SurfaceCategoryEnum::BATCH_NORM_DATA,  surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT8,  1, -1, 69U) \
        op(NVDLA_BATCH_NORM_DATA_INT16,   surface::SurfaceCategoryEnum::BATCH_NORM_DATA,  surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT16, 2, -1, 70U) \
        op(NVDLA_BATCH_NORM_DATA_FP16,    surface::SurfaceCategoryEnum::BATCH_NORM_DATA,  surface::SurfacePrecisionEnum::NVDLA_PRECISION_FP16,  2, -1, 71U) \
        op(NVDLA_SCALE_DATA_INT8,         surface::SurfaceCategoryEnum::SCALE_DATA,  surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT8,  1, -1, 72U) \
        op(NVDLA_SCALE_DATA_INT16,        surface::SurfaceCategoryEnum::SCALE_DATA,  surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT16, 2, -1, 73U) \
        op(NVDLA_SCALE_DATA_FP16,         surface::SurfaceCategoryEnum::SCALE_DATA,  surface::SurfacePrecisionEnum::NVDLA_PRECISION_FP16,  2, -1, 74U) \
        op(NVDLA_UNKNOWN_FORMAT,          surface::SurfaceCategoryEnum::SURFACE_CATEGORY_UNKNOWN, surface::SurfacePrecisionEnum::NVDLA_PRECISION_UNKNOWN, 0, -1, 75U)

#endif
