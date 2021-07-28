/*
 * Copyright (c) 2017-2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef NVDLA_I_TYPE_H
#define NVDLA_I_TYPE_H

#include <stdexcept>
#include <string>
#include <map>

#include "dlaerror.h"

#include "nvdla/c/NvDlaType.h"

#define ENUM_CLASS_MEMBERS(ENUM_C)                                  \
public:                                                             \
 ENUM_C(UnderlyingType v) : m_v(v) { if ( m_v > max() ) throw std::out_of_range(#ENUM_C); } \
 ENUM_C(const ENUM_C &o) : m_v(o.m_v) { }                               \
 ENUM_C(Enum e) : m_v( UnderlyingType(e) ) { }                          \
 ENUM_C() : m_v( UnderlyingType(0) ) { } /*note default is zero*/       \
 bool operator ==(const ENUM_C &rhs) const { return m_v == rhs.m_v; }   \
 bool operator ==(const Enum &rhs) const { return m_v == UnderlyingType(rhs); } \
 operator UnderlyingType() const { return m_v; }                        \
 UnderlyingType v() const { return m_v; }                               \
protected:                                                              \
 UnderlyingType m_v;



namespace nvdla
{

template<typename T> inline int EnumMax(); // used by checkers


class DataFormat {
public:
    typedef NvU8 UnderlyingType;
    enum Enum {
        UNKNOWN = NVDLA_DATA_FORMAT_UNKNOWN,
        NCHW    = NVDLA_DATA_FORMAT_NCHW,
        NHWC    = NVDLA_DATA_FORMAT_NHWC,
        NCxHWx  = NVDLA_DATA_FORMAT_NCxHWx,
    };
    static inline UnderlyingType max() { return 3U; }
    const char* c_str() const {
        const char * names[4] = { "UNKNOWN", "NCHW", "NHWC", "NCxHWx" };
        return names[m_v];
    }

    ENUM_CLASS_MEMBERS(DataFormat);
};
template<> inline int EnumMax<DataFormat>() { return DataFormat::max() + 1; }


class DataType
{
public:
    typedef NvU8 UnderlyingType;
    enum Enum {
        UNKNOWN = NVDLA_DATA_TYPE_UNKNOWN,
        FLOAT   = NVDLA_DATA_TYPE_FLOAT,  //!< FP32 format
        HALF    = NVDLA_DATA_TYPE_HALF,   //!< FP16 format
        INT16   = NVDLA_DATA_TYPE_INT16,  //!< INT16 format
        INT8    = NVDLA_DATA_TYPE_INT8,   //!< INT8 format
        UINT8   = NVDLA_DATA_TYPE_UINT8,  //!< UINT8 format
        UINT16  = NVDLA_DATA_TYPE_UINT16, //!< UINT16 format
    };
    static inline UnderlyingType max() { return 6U; }
    const char* c_str() const {
        const char * names[7] = { "UNKNOWN", "FLOAT", "HALF", "INT16", "INT8", "UINT8", "UINT16" };
        return names[m_v];
    }
    static UnderlyingType getEnum(std::string find)
    {
        NvU8 i = 0;
        const char * names[7] = { "UNKNOWN", "FLOAT", "HALF", "INT16", "INT8", "UINT8", "UINT16" };
        NvU8 match = UNKNOWN;
        for (; i <= max(); ++i) {
            if (names[i] == find) {
                match = i;
                break;
            }
        }
        return match;
    }
    ENUM_CLASS_MEMBERS(DataType);
};
template<> inline int EnumMax<DataType>() { return DataType::max() + 1; } // used by checkers

class TensorScalingMode
{
public:
    typedef NvU8 UnderlyingType;
    enum Enum {
        NONE        = NVDLA_TENSOR_SCALING_MODE_NONE,
        PER_TENSOR  = NVDLA_TENSOR_SCALING_MODE_PER_TENSOR,
        PER_CHANNEL = NVDLA_TENSOR_SCALING_MODE_PER_CHANNEL,
    };
    static inline UnderlyingType max() { return 2U; }
    const char* c_str() const {
        const char * names[3] = { "NONE", "PER_TENSOR", "PER_CHANNEL"};
        return names[m_v];
    }
    static UnderlyingType getEnum(std::string find)
    {
        NvU8 i = 0;
        const char * names[3] = { "NONE", "PER_TENSOR", "PER_CHANNEL"};
        NvU8 match = NONE;
        for (; i <= max(); ++i) {
            if (names[i] == find) {
                match = i;
                break;
            }
        }
        return match;
    }
    ENUM_CLASS_MEMBERS(TensorScalingMode);
};
template<> inline int EnumMax<TensorScalingMode>() { return TensorScalingMode::max() + 1; } // used by checkers

class QuantizationMode
{
public:
    typedef NvU8 UnderlyingType;
    enum Enum {
        NONE        = NVDLA_TENSOR_QUANTIZATION_MODE_NONE,
        PER_KERNEL  = NVDLA_TENSOR_QUANTIZATION_MODE_PER_KERNEL,   // per KCRS
        PER_FILTER  = NVDLA_TENSOR_QUANTIZATION_MODE_PER_FILTER,   // per CRS
    };
    static inline UnderlyingType max() { return 2U; }
    const char* c_str() const {
        const char * names[3] = { "NONE", "PER_KERNEL", "PER_FILTER"};
        return names[m_v];
    }
    static UnderlyingType getEnum(std::string find)
    {
        NvU8 i = 0;
        const char * names[3] = { "NONE", "PER_KERNEL", "PER_FILTER"};
        NvU8 match = NONE;
        for (; i <= max(); ++i) {
            if (names[i] == find) {
                match = i;
                break;
            }
        }
        return match;
    }
    ENUM_CLASS_MEMBERS(QuantizationMode);
};
template<> inline int EnumMax<QuantizationMode>() { return QuantizationMode::max() + 1; } // used by checkers

class DataCategory
{
public:
    typedef NvU8 UnderlyingType;
    enum Enum {
        IMAGE   = NVDLA_DATA_CATEGORY_IMAGE,
        WEIGHT  = NVDLA_DATA_CATEGORY_WEIGHT,
        FEATURE = NVDLA_DATA_CATEGORY_FEATURE,
        PLANAR  = NVDLA_DATA_CATEGORY_PLANAR,
        BIAS    = NVDLA_DATA_CATEGORY_BIAS,
    };
    static inline UnderlyingType max() { return 4U; }
    const char* c_str() const {
        const char * names[5] = { "IMAGE", "WEIGHT", "FEATURE", "PLANAR", "BIAS" };
        return names[m_v];
    }
    ENUM_CLASS_MEMBERS(DataCategory);
};
template<> inline int EnumMax<DataCategory>() { return DataCategory::max() + 1; } // used by checkers

//
// the type of surface a raw tensor represents
// defined in nvdla::priv namespace
// strictly used in pre-dla context
//
enum TensorType
{
    kUNKNOWN   = 0,
    kNW_INPUT  = 1,   /* network input tensor */
    kNW_OUTPUT = 2,   /* network output tensor */
    kIO        = 3,   /* input-output tensor passed between operations */
    kWEIGHT    = 4,   /* conv weights */
    kBIAS      = 5,   /* bias data */
    kBATCH_NORM= 6,   /* mean/variance data */
    kSCALE     = 7,   /* scaling data */
    kSTREAM    = 8,   /* tensors on wire */
    kDEBUG     = 9,   /* debug output tensors */
};
template<> inline int EnumMax<TensorType>() {  return 10; }

class Dims2
{
public:
    Dims2() : h(0), w(0) {};
    Dims2(NvS32 h, NvS32 w) : h(h), w(w) {};
    NvS32 h;
    NvS32 w;
    inline bool operator==(const Dims2& other) const
    {
        return (h == other.h && w == other.w);
    }
    inline bool operator!=(const Dims2& other) const
    {
        return !(h == other.h && w == other.w);
    }
};

class Dims3
{
public:
    Dims3() : c(0), h(0), w(0) {};
    Dims3(NvS32 c, NvS32 h, NvS32 w) : c(c), h(h), w(w) {};
    NvS32 c; // channels
    NvS32 h;
    NvS32 w;
};

class Dims4
{
public:
    Dims4() : n(1), c(0), h(0), w(0) {};
    Dims4(NvS32 c, NvS32 h, NvS32 w) : n(1), c(c), h(h), w(w) {};
    Dims4(NvS32 n, NvS32 c, NvS32 h, NvS32 w) : n(n), c(c), h(h), w(w) {};
    NvS32 n;      //!< the number of images in the data or number of kernels in the weights (default = 1)
    NvS32 c;      //!< the number of channels in the data
    NvS32 h;      //!< the height of the data
    NvS32 w;      //!< the width of the data
    inline bool operator==(const Dims4& other) const
    {
        return (n == other.n && c == other.c && h == other.h && w == other.w);
    }
    inline bool operator!=(const Dims4& other) const
    {
        return !(n == other.n && c == other.c && h == other.h && w == other.w);
    }
};

class Weights
{
public:
    Weights() :
        type(DataType::UNKNOWN),
        values(NULL),
        count(0)
    {};
    Weights(DataType type, const void* values, NvS64 count) :
        type(type),
        values(values),
        count(count)
    {};
    Weights(const Weights& other) :
        type(other.type),
        values(other.values),
        count(other.count)
    {};

    DataType type;      //!< the type of the weights
    const void* values; //!< the weight values, in a contiguous array
    NvS64 count;        //!< the number of weights in the array
};

class LayerType
{
public:
    typedef NvU8 UnderlyingType;
    enum Enum {
        kCONVOLUTION = NVDLA_LAYER_TYPE_CONVOLUTION,            //!< Convolution layer
        kFULLY_CONNECTED = NVDLA_LAYER_TYPE_FULLY_CONNECTED,    //!< Fully connected layer
        kACTIVATION = NVDLA_LAYER_TYPE_ACTIVATION,              //!< Activation layer
        kPOOLING = NVDLA_LAYER_TYPE_POOLING,                    //!< Pooling layer
        kLRN = NVDLA_LAYER_TYPE_LRN,                            //!< LRN layer
        kSCALE = NVDLA_LAYER_TYPE_SCALE,                        //!< Scale Layer
        kBATCH_NORM = NVDLA_LAYER_TYPE_BATCH_NORM,              //!< Batch Norm Layer
        kSOFTMAX = NVDLA_LAYER_TYPE_SOFTMAX,                    //!< SoftMax layer
        kDECONVOLUTION = NVDLA_LAYER_TYPE_DECONVOLUTION,        //!< Deconvolution layer
        kCONCATENATION = NVDLA_LAYER_TYPE_CONCATENATION,        //!< Concatenation layer
        kELEMENTWISE = NVDLA_LAYER_TYPE_ELEMENTWISE,            //!< Elementwise layer
        kSLICE = NVDLA_LAYER_TYPE_SLICE,                        //!< Slice layer
        lt_kUNKNOWN = NVDLA_LAYER_TYPE_UNKNOWN,
    };
    static inline UnderlyingType max() { return 12U; }
    const char* c_str() const {
        const char * names[13] = { "CONVOLUTION", "FULLY_CONNECTED", "ACTIVATION",
                                    "POOLING", "LRN", "SCALE", "BATCHNORM", "SOFTMAX",
                                    "DECONVOLUTION", "CONCATENATION", "ELEMENTWISE", "SLICE", "UNKNOWN"};
        return names[m_v];
    }
    static UnderlyingType getEnum(std::string find)
    {
        NvU8 i = 0;
        const char * names[13] = { "CONVOLUTION", "FULLY_CONNECTED", "ACTIVATION",
                                    "POOLING", "LRN", "SCALE", "BATCHNORM", "SOFTMAX",
                                    "DECONVOLUTION", "CONCATENATION", "ELEMENTWISE", "SLICE", "UNKNOWN"};
        NvU8 match = lt_kUNKNOWN;
        for (; i <= max(); ++i) {
            if (names[i] == find) {
                match = i;
                break;
            }
        }
        return match;
    }

    ENUM_CLASS_MEMBERS(LayerType);
};
template<> inline int EnumMax<LayerType>() { return LayerType::max() + 1; } // used by checkers


enum ActivationType
{
    kRELU = NVDLA_ACTIVATION_TYPE_RELU,
    kSIGMOID = NVDLA_ACTIVATION_TYPE_SIGMOID,
    kTANH = NVDLA_ACTIVATION_TYPE_TANH,
};
template<> inline int EnumMax<ActivationType>() { return 3; } // used by checkers

class PoolingType
{
public:
    typedef NvU8 UnderlyingType;
    enum Enum {
        kMIN = NVDLA_POOLING_TYPE_MIN,    // Min over elements
        kMAX = NVDLA_POOLING_TYPE_MAX,    // Maximum over elements
        kAVERAGE = NVDLA_POOLING_TYPE_AVERAGE // Average over elements. If the tensor is padded, the count includes the padding
    };
    static inline UnderlyingType max() { return 2U; }
    const char* c_str() const {
        const char * names[3] = { "MIN", "MAX", "AVERAGE" };
        return names[m_v];
    }
    static UnderlyingType getEnum(std::string find)
    {
        NvU8 i = 0;
        const char * names[3] = { "MIN", "MAX", "AVERAGE" };
        NvU8 match = kAVERAGE;
        for (; i <= max(); ++i) {
            if (names[i] == find) {
                match = i;
                break;
            }
        }
        return match;
    }
    ENUM_CLASS_MEMBERS(PoolingType);
};
template<> inline int EnumMax<PoolingType>() { return PoolingType::max() + 1; } // used by checkers

enum BiasMode
{
    bNONE    = NVDLA_BIAS_MODE_NONE,          //!< no bias
    bUNIFORM = NVDLA_BIAS_MODE_UNIFORM,       //!< identical coefficients across all elements of the tensor
    bCHANNEL = NVDLA_BIAS_MODE_CHANNEL,       //!< per-channel coefficients
    bm_ELEMENTWISE = NVDLA_BIAS_MODE_ELEMENTWISE //!< elementwise coefficients
};
template<> inline int EnumMax<BiasMode>() { return 4; } // used by checkers

enum ScaleMode
{
    sUNKNOWN = NVDLA_SCALE_MODE_UNKNOWN,       //!< unknown scale mode
    sUNIFORM = NVDLA_SCALE_MODE_UNIFORM,       //!< identical coefficients across all elements of the tensor
    sCHANNEL = NVDLA_SCALE_MODE_CHANNEL,       //!< per-channel coefficients
    sm_ELEMENTWISE = NVDLA_SCALE_MODE_ELEMENTWISE //!< elementwise coefficients
};
template<> inline int EnumMax<ScaleMode>() { return 4; } // used by checkers

enum BatchNormMode
{
    bnUNIFORM = NVDLA_BATCH_NORM_MODE_UNIFORM,       //!<< identical coefficients across all elements of the tensor
    bnm_CHANNEL = NVDLA_BATCH_NORM_MODE_CHANNEL,    //!< per-channel coefficients
};
template<> inline int EnumMax<BatchNormMode>() { return 2; } // used by checkers

enum ElementWiseOperation
{
    kSUM = NVDLA_ELEMENTWISE_OPERATION_SUM,    //!< sum of the two elements
    kPROD = NVDLA_ELEMENTWISE_OPERATION_PROD,  //!< product of the two elements
    kMIN = NVDLA_ELEMENTWISE_OPERATION_MIN,    //!< minimum of the two elements
    ew_kMAX = NVDLA_ELEMENTWISE_OPERATION_MAX  //!< maximum of the two elements
};
template<> inline int EnumMax<ElementWiseOperation>() { return 3; } // used by checkers


class PixelMapping {
public:
    enum Enum {
        PITCH_LINEAR = NVDLA_PIXEL_MAPPING_PITCH_LINEAR,
    };
    typedef NvU8 UnderlyingType;

    static inline UnderlyingType max() { return 1U; }

    ENUM_CLASS_MEMBERS(PixelMapping)
};
template<> inline int EnumMax<PixelMapping>() { return PixelMapping::max() + 1; } // used by checkers


class PixelFormat {
public:
    enum Enum {
        R8 = NVDLA_PIXEL_FORMAT_R8,
        R10 = NVDLA_PIXEL_FORMAT_R10,
        R12 = NVDLA_PIXEL_FORMAT_R12,
        R16 = NVDLA_PIXEL_FORMAT_R16,
        R16_I = NVDLA_PIXEL_FORMAT_R16_I,
        R16_F = NVDLA_PIXEL_FORMAT_R16_F,
        A16B16G16R16 = NVDLA_PIXEL_FORMAT_A16B16G16R16,
        X16B16G16R16 = NVDLA_PIXEL_FORMAT_X16B16G16R16,
        A16B16G16R16_F = NVDLA_PIXEL_FORMAT_A16B16G16R16_F,
        A16Y16U16V16 = NVDLA_PIXEL_FORMAT_A16Y16U16V16,
        V16U16Y16A16 = NVDLA_PIXEL_FORMAT_V16U16Y16A16,
        A16Y16U16V16_F = NVDLA_PIXEL_FORMAT_A16Y16U16V16_F,
        A8B8G8R8 = NVDLA_PIXEL_FORMAT_A8B8G8R8,
        A8R8G8B8 = NVDLA_PIXEL_FORMAT_A8R8G8B8,
        B8G8R8A8 = NVDLA_PIXEL_FORMAT_B8G8R8A8,
        R8G8B8A8 = NVDLA_PIXEL_FORMAT_R8G8B8A8,
        X8B8G8R8 = NVDLA_PIXEL_FORMAT_X8B8G8R8,
        X8R8G8B8 = NVDLA_PIXEL_FORMAT_X8R8G8B8,
        B8G8R8X8 = NVDLA_PIXEL_FORMAT_B8G8R8X8,
        R8G8B8X8 = NVDLA_PIXEL_FORMAT_R8G8B8X8,
        A2B10G10R10 = NVDLA_PIXEL_FORMAT_A2B10G10R10,
        A2R10G10B10 = NVDLA_PIXEL_FORMAT_A2R10G10B10,
        B10G10R10A2 = NVDLA_PIXEL_FORMAT_B10G10R10A2,
        R10G10B10A2 = NVDLA_PIXEL_FORMAT_R10G10B10A2,
        A2Y10U10V10 = NVDLA_PIXEL_FORMAT_A2Y10U10V10,
        V10U10Y10A2 = NVDLA_PIXEL_FORMAT_V10U10Y10A2,
        A8Y8U8V8 = NVDLA_PIXEL_FORMAT_A8Y8U8V8,
        V8U8Y8A8 = NVDLA_PIXEL_FORMAT_V8U8Y8A8,
        Y8___U8V8_N444 = NVDLA_PIXEL_FORMAT_Y8___U8V8_N444,
        Y8___V8U8_N444 = NVDLA_PIXEL_FORMAT_Y8___V8U8_N444,
        Y10___U10V10_N444 = NVDLA_PIXEL_FORMAT_Y10___U10V10_N444,
        Y10___V10U10_N444 = NVDLA_PIXEL_FORMAT_Y10___V10U10_N444,
        Y12___U12V12_N444 = NVDLA_PIXEL_FORMAT_Y12___U12V12_N444,
        Y12___V12U12_N444 = NVDLA_PIXEL_FORMAT_Y12___V12U12_N444,
        Y16___U16V16_N444 = NVDLA_PIXEL_FORMAT_Y16___U16V16_N444,
        Y16___V16U16_N444 = NVDLA_PIXEL_FORMAT_Y16___V16U16_N444,
        FEATURE  = NVDLA_PIXEL_FORMAT_FEATURE,
        FEATURE_X8  = NVDLA_PIXEL_FORMAT_FEATURE_X8,
        UNKNOWN  = NVDLA_PIXEL_FORMAT_UNKNOWN,

        // Y16___U8V8_N444 = NVDLA_PIXEL_FORMAT_UNSUPPORTED,
        // Y16___V8U8_N444 = NVDLA_PIXEL_FORMAT_UNSUPPORTED-1,
        // Y8___U8_V8_N444 = NVDLA_PIXEL_FORMAT_UNSUPPORTED-2,
        // Y10___U10_V10_N444 = NVDLA_PIXEL_FORMAT_UNSUPPORTED-3,
        // Y12___U12_V12_N444 = NVDLA_PIXEL_FORMAT_UNSUPPORTED-4,
        // Y16___U16_V16_N444 = NVDLA_PIXEL_FORMAT_UNSUPPORTED-5,
        // Y16___U8_V8_N444 = NVDLA_PIXEL_FORMAT_UNSUPPORTED-6,

    };
    typedef NvU8 UnderlyingType;
    static inline UnderlyingType max() { return 38U; }
    const char* c_str() const {
        const char * names[39] = { "R8", "R10", "R12", "R16", "R16_I", "R16_F", "A16B16G16R16", "X16B16G16R16", "A16B16G16R16_F", "A16Y16U16V16", "V16U16Y16A16", "A16Y16U16V16_F",
                            "A8B8G8R8", "A8R8G8B8", "B8G8R8A8", "R8G8B8A8", "X8B8G8R8", "X8R8G8B8", "B8G8R8X8", "R8G8B8X8", "A2B10G10R10", "A2R10G10B10", "B10G10R10A2", "R10G10B10A2",
                            "A2Y10U10V10", "V10U10Y10A2", "A8Y8U8V8", "V8U8Y8A8", "Y8___U8V8_N444", "Y8___V8U8_N444", "Y10___U10V10_N444", "Y10___V10U10_N444", "Y12___U12V12_N444",
                            "Y12___V12U12_N444", "Y16___U16V16_N444", "Y16___V16U16_N444", "FEATURE", "FEATURE_X8", "UNKNOWN"};
        return names[m_v];
    }
    static UnderlyingType getEnum(std::string find)
    {
        NvU8 i = 0;
        const char * names[39] = { "R8", "R10", "R12", "R16", "R16_I", "R16_F", "A16B16G16R16", "X16B16G16R16", "A16B16G16R16_F", "A16Y16U16V16", "V16U16Y16A16", "A16Y16U16V16_F",
                            "A8B8G8R8", "A8R8G8B8", "B8G8R8A8", "R8G8B8A8", "X8B8G8R8", "X8R8G8B8", "B8G8R8X8", "R8G8B8X8", "A2B10G10R10", "A2R10G10B10", "B10G10R10A2", "R10G10B10A2",
                            "A2Y10U10V10", "V10U10Y10A2", "A8Y8U8V8", "V8U8Y8A8", "Y8___U8V8_N444", "Y8___V8U8_N444", "Y10___U10V10_N444", "Y10___V10U10_N444", "Y12___U12V12_N444",
                            "Y12___V12U12_N444", "Y16___U16V16_N444", "Y16___V16U16_N444", "FEATURE", "FEATURE_X8", "UNKNOWN"};
        NvU8 match = UNKNOWN;
        for (; i <= max(); ++i) {
            if (names[i] == find) {
                match = i;
                break;
            }
        }
        return match;
    }

    ENUM_CLASS_MEMBERS(PixelFormat);
};
template<> inline int EnumMax<PixelFormat::Enum>() { return PixelFormat::max() + 1; } // used by checkers


struct TensorDesc
{
    NvU64 bufferSize;
    Dims4 dims;
    DataFormat::UnderlyingType dataFormat;
    DataType::UnderlyingType   dataType;
    DataCategory::UnderlyingType dataCategory;
    PixelFormat::UnderlyingType  pixelFormat;
    PixelMapping::UnderlyingType pixelMapping;

    // valid iff pixelMapping == PITCH_LINEAR
    struct PitchLinearMappingDesc
    {
        NvU32 lineStride;
        NvU32 surfStride;
        NvU32 planeStride;
    } pitchLinear;

};


} // nvdla::

#undef ENUM_CLASS_MEMBERS

#endif // NVDLA_I_TYPE_H
