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

#ifndef NVDLA_PRIV_SURFACE_H
#define NVDLA_PRIV_SURFACE_H

#include "Type.h"
#include <cstring>
#include <unordered_set>

#include "Memory.h"

#include "SurfaceEnums.h"

#include "ErrorMacros.h"

#define GEN_SURFACE_STR(X, C, P, B, Ch, N)    #X,
#define GEN_SURFACE_FORMAT(X, C, P, B, Ch, N) surface::Format(surface ::X, C, P, B, Ch),
#define GEN_SURFACE_ENUM(X, C, P, B, Ch, N)   X = N,

#define SURFACE_ENUM_STATIC_MEMBERS(S, U, E, Z)                                             \
    template<>                                                                              \
    const char* const SurfaceEnum<S, U>::s_c_strs[] = {E(GEN_SURFACE_STR)};                 \
    template<>                                                                              \
    const surface::Format SurfaceEnum<S, U>::s_surface_formats[] = {E(GEN_SURFACE_FORMAT)}; \
    template<>                                                                              \
    const char* SurfaceEnum<S, U>::s_c_str = Z;                                             \
    template<>                                                                              \
    const size_t SurfaceEnum<S, U>::s_num_elements = sizeof(SurfaceEnum<S, U>::s_c_strs) / sizeof(SurfaceEnum<S, U>::s_c_strs[0]);

namespace nvdla {
namespace priv {
namespace engine_ast {
class Node;
class Edge;
} // namespace engine_ast
} // namespace priv
} // namespace nvdla

namespace nvdla {
namespace priv {

template<typename EnumClass, typename UnderlyingType>
class SurfaceEnum;

namespace surface {

class TensorSurfaceDesc;

enum SurfaceCategoryEnum
{
    SURFACE_CATEGORY_ENUMS(GEN_ENUM)
};
typedef SequenceEnum<SurfaceCategoryEnum, NvU8> SurfaceCategory;

enum SurfacePrecisionEnum
{
    SURFACE_PRECISION_ENUMS(GEN_ENUM)
};
typedef SequenceEnum<SurfacePrecisionEnum, NvU8> SurfacePrecision;

enum BiasDataCategoryEnum
{
    BIAS_DATA_CATEGORY_ENUMS(GEN_ENUM)
};
typedef SequenceEnum<BiasDataCategoryEnum, NvU8> BiasDataCategory;

enum BatchNormDataCategoryEnum
{
    BATCH_NORM_DATA_CATEGORY_ENUMS(GEN_ENUM)
};
typedef SequenceEnum<BatchNormDataCategoryEnum, NvU8> BatchNormDataCategory;

enum ScaleDataCategoryEnum
{
    SCALE_DATA_CATEGORY_ENUMS(GEN_ENUM)
};
typedef SequenceEnum<ScaleDataCategoryEnum, NvU8> ScaleDataCategory;

enum PixelMappingEnum
{
    PIXEL_MAPPING_ENUMS(GEN_ENUM)
};
typedef SequenceEnum<PixelMappingEnum, NvU8> PixelMapping;

enum SurfaceFormatEnum
{
    SURFACE_FORMAT_ENUMS(GEN_SURFACE_ENUM)
};
typedef SurfaceEnum<SurfaceFormatEnum, NvU8> SurfaceFormat;

struct PitchLinearSurfaceDesc
{
    static NvU32 lineStride(const TensorSurfaceDesc*);
    static NvU32 lineUVStride(const TensorSurfaceDesc*);
    static NvU64 size(const TensorSurfaceDesc*);
};

struct IMGDesc
{
    static PixelMapping pixelMapping(const TensorSurfaceDesc*);
    static NvU32 lineStride(const TensorSurfaceDesc*);
    static NvU32 surfaceStride(const TensorSurfaceDesc*);
    static NvU64 size(const TensorSurfaceDesc*);
};

struct WeightDesc
{
    /*
     * Weights for DC: {[RSKpCp]Cfg[RSKpCf]}Kfg{[RSKfCp]Cfg[RSKfCf]}
     * Weights for WG: [CfgKpRSCf]Kfg[CfgKfRSCf]
     * Weights for IMG:
     * Weights for Deconvolution:
     */

    // are weights compressed
    static bool iscompressed(const TensorSurfaceDesc*);
    // full channels per group    (For DC:-> =64 for int8; For WG:-> = 4)
    static NvU32 fullChnlsPerGrp(const TensorSurfaceDesc*);
    // full channel groups        (total_channels / Cf)
    static NvU32 fullChnlGroups(const TensorSurfaceDesc*);
    // partial channels per group (total_channels % Cf - which is <64 for int8; or <32 for int16/fp16)
    static NvU32 partialChnlsPerGrp(const TensorSurfaceDesc*);
    // full kernels per group     (=32 for int8; =16 for int16/fp16)
    static NvU32 fullKrnlsPerGrp(const TensorSurfaceDesc*);
    // full kernel groups         (total_kernels / Kf)
    static NvU32 fullKrnlGroups(const TensorSurfaceDesc*);
    // partial kernels per group  (total_kernels % Kf - which is <32 for int8; or <16 for int16/fp16)
    static NvU32 partialKrnlsPerGrp(const TensorSurfaceDesc*);

    static NvU32 wgs(const TensorSurfaceDesc*);
    static NvU32 wmb(const TensorSurfaceDesc*);

    static NvU64 size(const TensorSurfaceDesc*);
    static NvU32 bytesPerKernel(const TensorSurfaceDesc*);
    // get raw non-rounded weight tensor size
    static NvU64 rawSize(const TensorSurfaceDesc*);
};

struct BiasDataDesc
{
    /* Bias Data could be any of the 3 types:
     *   per-Layer:     1 x 1 x 1
     *   per-Channel:   1 x 1 x C
     *   per-Element:   W x H x C
     */
    static BiasDataCategory biasDataCategory(const TensorSurfaceDesc*);
    static NvU32 channelsPerGroup(const TensorSurfaceDesc*);
    static NvU32 lineStride(const TensorSurfaceDesc*);
    static NvU32 surfaceStride(const TensorSurfaceDesc*);
    static NvU64 size(const TensorSurfaceDesc*);
};

struct BatchNormDataDesc
{
    /* BatchNorm Data could be any of the 2 types:
     *    per-Layer:    1 x 1 x 1
     *    per-Channel:  1 x 1 x C
     */
    static BatchNormDataCategory batchNormDataCategory(const TensorSurfaceDesc*);
    static NvU32 lineStride(const TensorSurfaceDesc*);
    static NvU32 surfaceStride(const TensorSurfaceDesc*);
    static NvU64 size(const TensorSurfaceDesc*);
};

struct ScaleDataDesc
{
    /* Scale Data could be any of the 3 types:
     *    per-Layer:    1 x 1 x 1
     *    per-Channel:  1 x 1 x C
     *    per-Element:  W x H x C
     */
    static ScaleDataCategory scaleDataCategory(const TensorSurfaceDesc*);
    static NvU32 lineStride(const TensorSurfaceDesc*);
    static NvU32 surfaceStride(const TensorSurfaceDesc*);
    static NvU64 size(const TensorSurfaceDesc*);
};

struct FeatureDataDesc
{
    /* Format: N_CHWC' */

    // N  (used during multi-batch mode)
    static NvU32 numBatches(const TensorSurfaceDesc*);
    // C  (=ceil(total_channels/m_channels_per_group))
    static NvU32 channelGroups(const TensorSurfaceDesc*);
    // C' (=16 for int16/fp16; =32 for int8)
    static NvU32 channelsPerGroup(const TensorSurfaceDesc*);
    static NvU32 height(const TensorSurfaceDesc*);
    static NvU32 width(const TensorSurfaceDesc*);
    static NvU32 lineStride(const TensorSurfaceDesc*);
    static NvU32 surfaceStride(const TensorSurfaceDesc*);
    static NvU64 size(const TensorSurfaceDesc*);
};

struct EltwiseDataDesc
{
    static NvU32 channelGroups(const TensorSurfaceDesc*);
    static NvU32 channelsPerGroup(const TensorSurfaceDesc*);
    static NvU32 lineStride(const TensorSurfaceDesc*);
    static NvU32 surfaceStride(const TensorSurfaceDesc*);
    static NvU64 size(const TensorSurfaceDesc*);
};

class Format
{
public:
    Format()
        : m_enum(NVDLA_UNKNOWN_FORMAT),
          mSurfCategory(SURFACE_CATEGORY_UNKNOWN),
          mSurfPrecision(NVDLA_PRECISION_UNKNOWN),
          mBytesPerElement(0),
          mChannelsPerAtom(-1)
    {
    }
    Format(SurfaceFormatEnum en, SurfaceCategory sc, SurfacePrecision sp, NvU8 bpe, NvU8 cpa)
        : m_enum(en),
          mSurfCategory(sc),
          mSurfPrecision(sp),
          mBytesPerElement(bpe),
          mChannelsPerAtom(cpa)
    {
    }
    Format(const Format& other)
        : m_enum(other.m_enum),
          mSurfCategory(other.mSurfCategory),
          mSurfPrecision(other.mSurfPrecision),
          mBytesPerElement(other.mBytesPerElement),
          mChannelsPerAtom(other.mChannelsPerAtom)
    {
    }
    virtual ~Format()
    {
    }

    SurfaceFormatEnum formatEnum() const
    {
        return m_enum;
    }

    SurfaceCategory category() const
    {
        return mSurfCategory;
    }
    SurfacePrecision precision() const
    {
        return mSurfPrecision;
    }
    NvU8 bytesPerElement() const
    {
        return mBytesPerElement;
    }
    NvS8 channelsPerAtom() const
    {
        return mChannelsPerAtom;
    }

protected:
    SurfaceFormatEnum m_enum;
    SurfaceCategory mSurfCategory;   // img/weight/bias_data/FD/M-planar
    SurfacePrecision mSurfPrecision; // int8/int16/fp16
    NvU8 mBytesPerElement;           // 1/2
    NvS8 mChannelsPerAtom;           // channels/atom [channels/pixel for image; -1 for FD/wt/etc]
};

} // namespace surface
} // namespace priv
} // namespace nvdla

namespace nvdla {
namespace priv {

template<typename EnumClass, typename UnderlyingType = NvU8>
class SurfaceEnum
{
public:
    typedef UnderlyingType underlying_type;

protected:
    underlying_type m_e;
    static const char* const s_c_strs[];              // enum strings
    static const surface::Format s_surface_formats[]; // array of type Format

    static const char* s_c_str; // class name string
    static const size_t s_num_elements;

public:
    static const char* parameter_name_c_str()
    {
        return s_c_str;
    }
    const char* c_str() const
    {
        return s_c_strs[m_e];
    }

    underlying_type v() const
    {
        return m_e;
    }
    EnumClass e() const
    {
        return EnumClass(m_e);
    }
    surface::Format f() const
    {
        return surface::Format(s_surface_formats[m_e]);
    }
    surface::SurfaceCategory category() const
    {
        return s_surface_formats[m_e].category();
    }
    surface::SurfacePrecision precision() const
    {
        return s_surface_formats[m_e].precision();
    }
    NvU8 bytesPerElement() const
    {
        return s_surface_formats[m_e].bytesPerElement();
    }
    NvS8 channelsPerAtom() const
    {
        return s_surface_formats[m_e].channelsPerAtom();
    }

    static inline size_t num_elements()
    {
        return s_num_elements;
    }

    SurfaceEnum()
        : m_e(underlying_type(s_num_elements))
    { /* note, invalid!*/
    }
    SurfaceEnum(EnumClass p)
        : m_e(p)
    {
    }
    SurfaceEnum(underlying_type v)
    {
        if (v > s_num_elements)
        {
            v = s_num_elements; // FIXME: reverse?
            // throw: out of range!?! (prefer)
            // or require check on validity?
            // or coerce to valid?
        }
        m_e = v;
    }
    bool operator()(const SurfaceEnum& n)
    {
        return n.v() == m_e;
    }
    bool operator<(const SurfaceEnum& other) const
    {
        return (m_e < other.v());
    }
};

} // namespace priv
} // namespace nvdla

namespace nvdla {
namespace priv {

namespace surface {

class IsSurfacePrecisionDifferent
{
public:
    IsSurfacePrecisionDifferent(SurfacePrecision sp)
        : m_surf_prec(sp)
    {
    }
    bool operator()(SurfaceFormat sf)
    {
        if (sf.f().precision().e() != m_surf_prec.e())
        {
            return true;
        }
        return false;
    }

    bool operator()(SurfacePrecision sp)
    {
        if (sp.e() != m_surf_prec.e())
        {
            return true;
        }
        return false;
    }

protected:
    SurfacePrecision m_surf_prec;
};

class TensorSurfaceDesc
{
public:
    class SurfaceDescState
    {
    public:
        SurfaceDescState()
            : m_addressId(-2), // (so if you see -2 you know something's wrong)
              m_addressIdOffset(0),
              m_offset_in_buffer(0)
        {
        }
        virtual ~SurfaceDescState()
        {
        }

        void setAddressId(NvS16 id)
        {
            m_addressId = id;
        }
        NvS16 addressId() const
        {
            return m_addressId;
        }

        void setAddressIdOffset(NvU32 o)
        {
            m_addressIdOffset = o;
        }
        NvU32 addressIdOffset() const
        {
            return m_addressIdOffset;
        }

        void setBufferOffset(NvU64 offset)
        {
            m_offset_in_buffer = offset;
        }
        NvU64 bufferOffset() const
        {
            return m_offset_in_buffer;
        }
        void resetBufferOffset()
        {
            m_offset_in_buffer = 0;
        }

    protected:
        NvS16 m_addressId;
        NvU32 m_addressIdOffset;
        NvU64 m_offset_in_buffer;
    };

    TensorSurfaceDesc(NvU16 numBatches = 1)
        : m_size(0),
          m_line_stride(0),
          m_surface_stride(0),
          m_tensor_category(memory::TensorCategoryEnum::UNKNOWN_TENSOR),
          m_data_format(nvdla::DataFormat::UNKNOWN),
          m_surface_format(SurfaceFormatEnum::NVDLA_UNKNOWN_FORMAT),
          m_buffer_desc(0),
          m_copyOutDebugSurface(false),
          m_content(false),
          m_align_ls(false),
          m_activeProducers(0),
          m_activeConsumers(0),
          m_bindId(-1),          // non-bindable
          m_bindDomain(IOD_Max), // n/a
          m_edge(NULL)
    {
        m_mb_tsd_state = new MultiBatchState<SurfaceDescState>(numBatches);
    };
    ~TensorSurfaceDesc();

    /* for clone */
    TensorSurfaceDesc(const TensorSurfaceDesc& other)
        : m_dims(other.m_dims),
          m_size(other.m_size),
          m_line_stride(other.m_line_stride),
          m_surface_stride(other.m_surface_stride),
          m_tensor_category(other.m_tensor_category),
          m_data_format(other.m_data_format),
          m_surface_format(other.m_surface_format),
          m_buffer_desc(other.m_buffer_desc),
          m_copyOutDebugSurface(other.m_copyOutDebugSurface),
          m_content(other.m_content),
          m_align_ls(other.m_align_ls),
          m_activeProducers(other.m_activeProducers),
          m_activeConsumers(other.m_activeConsumers),
          m_bindId(other.m_bindId),
          m_bindDomain(other.m_bindDomain),
          m_edge(other.m_edge)
    {
        m_mb_tsd_state = other.m_mb_tsd_state; // Is this correct?
    }

    static inline bool debugBinding()
    {
        return true;
    }

    void setName(const std::string name)
    {
        m_name = name;
    }
    const std::string name() const
    {
        return m_name;
    }

    void setId(const std::string id)
    {
        m_id = id;
    }
    const std::string id() const
    {
        return m_id;
    }

    void setDimensions(Dims4 dims)
    {
        m_dims = dims;
    }
    Dims4 dimensions() const
    {
        return m_dims;
    }

    void setTensorCategory(memory::TensorCategory cat)
    {
        m_tensor_category = cat;
    }
    memory::TensorCategory tensorCategory() const
    {
        return m_tensor_category;
    }

    void setTensorBufferDesc(memory::TensorBufferDesc* buff_desc)
    {
        m_buffer_desc = buff_desc;
    }
    memory::TensorBufferDesc* tensorBufferDesc() const
    {
        return m_buffer_desc;
    }

    nvdla::DataFormat dataFormat() const
    {
        return m_data_format;
    }
    void setDataFormat(nvdla::DataFormat dataFormat)
    {
        m_data_format = dataFormat;
    }

    SurfaceFormat surfaceFormat() const
    {
        return m_surface_format;
    }
    void setSurfaceFormat(SurfaceFormat surfaceFormat)
    {
        m_surface_format = surfaceFormat;
    }
    void resetSurfaceFormat()
    {
        m_surface_format = surface::SurfaceFormatEnum::NVDLA_UNKNOWN_FORMAT;
    }

    NvU64 size();
    NvU32 lineStride();
    NvU32 surfaceStride();
    NvU32 planeStride() const;

    void setSize(NvU64 size)
    {
        m_size = size;
    }
    void resetSize()
    {
        m_size = 0;
    }
    void setLineStride(NvU32 lineStride)
    {
        m_line_stride = lineStride;
    }
    void resetLineStride()
    {
        m_line_stride = 0;
    }
    void setSurfaceStride(NvU32 surfaceStride)
    {
        m_surface_stride = surfaceStride;
    }
    void resetSurfaceStride()
    {
        m_surface_stride = 0;
    }

    bool copyOutDebugSurface() const
    {
        return m_copyOutDebugSurface;
    }
    void setCopyOutDebugSurface(bool b)
    {
        m_copyOutDebugSurface = b;
    }
    bool isSurfaceSymmetricTo(TensorSurfaceDesc* other);

    void addProducer(nvdla::priv::engine_ast::Node* prod)
    {
        m_producers.insert(prod);
    }
    void removeProducer(nvdla::priv::engine_ast::Node* prod)
    {
        if (m_producers.find(prod) != m_producers.end()) m_producers.erase(prod);
    }
    const std::unordered_set<nvdla::priv::engine_ast::Node*>& producers() const
    {
        return m_producers;
    }
    void clearProducers()
    {
        m_producers.clear();
    }

    void addConsumer(nvdla::priv::engine_ast::Node* cons)
    {
        m_consumers.insert(cons);
    }
    void removeConsumer(nvdla::priv::engine_ast::Node* cons)
    {
        if (m_consumers.find(cons) != m_consumers.end()) m_consumers.erase(cons);
    }
    const std::unordered_set<nvdla::priv::engine_ast::Node*>& consumers() const
    {
        return m_consumers;
    }
    void clearConsumers()
    {
        m_consumers.clear();
    }

    void setAddressId(NvS16 id, NvU16 batchId = 0)
    {
        m_mb_tsd_state->batch(batchId).setAddressId(id);
    }
    NvS16 addressId(NvU16 batchId = 0)
    {
        return m_mb_tsd_state->batch(batchId).addressId();
    }

    void setAddressIdOffset(NvU32 o, NvU16 batchId = 0)
    {
        m_mb_tsd_state->batch(batchId).setAddressIdOffset(o);
    }
    NvU32 addressIdOffset(NvU16 batchId = 0)
    {
        return m_mb_tsd_state->batch(batchId).addressIdOffset();
    }

    bool bindable() const
    {
        bool isBindable = m_bindId >= 0;
        if (debugBinding())
        {
            gLogInfo << "\t\t\t\t::Surface surface=" << id()
                     << ": bindable=" << isBindable << std::endl;
        }
        return isBindable;
    }
    NvS16 bindId() const
    {
        if (debugBinding())
        {
            gLogInfo << "\t\t\t\t::Surface bindId(" << id() << ":" << m_bindId << std::endl;
        }
        return m_bindId;
    }

    void setBindId(NvS16 bid, enum IOD bindDomain)
    {
        if (debugBinding())
        {
            gLogInfo << "\t\t\t\t::Surface bindId(" << id() << ", "
                     << (int)bindDomain << ") -> " << bid << std::endl;
        }
        m_bindDomain = bindDomain;
        m_bindId = bid;
    }
    NvS16 bindId(enum IOD& bindDomain) const
    {
        if (debugBinding())
        {
            gLogInfo << "\t\t\t\t::Surface bindId(" << id() << ", "
                     << (int)m_bindDomain << ") -> " << m_bindId << std::endl;
        }
        bindDomain = m_bindDomain;
        return m_bindId;
    }

    // address comes from the buffer + offset into it
    template<typename T>
    T* address(NvU16 batchId = 0) const
    {
        return m_buffer_desc ? ((T*)(m_buffer_desc->address<NvU8>(batchId) + bufferOffset(batchId))) : 0;
    }

    void setBufferOffset(NvU64 offset, NvU16 batchId = 0)
    {
        m_mb_tsd_state->batch(batchId).setBufferOffset(offset);
    }
    NvU64 bufferOffset(NvU16 batchId = 0) const
    {
        return m_mb_tsd_state->batch(batchId).bufferOffset();
    }
    void resetBufferOffset(NvU16 batchId = 0)
    {
        m_mb_tsd_state->batch(batchId).resetBufferOffset();
    }

    // setcontent only happens for aux tensors, whose there's only 1 copy for all batches
    void setContent(const void* data)
    {
        m_content = true;
        std::memcpy(address<void>(/*batchId*/ 0), data, size());
    }
    bool content() const
    {
        return m_content;
    }

    void setAlignLineStride(bool enb)
    {
        m_align_ls = enb;
    }
    bool alignLineStride() const
    {
        return m_align_ls;
    }

    bool referencedByEMU() const;

    nvdla::priv::engine_ast::Edge* parentEdge() const
    {
        return m_edge;
    }
    void setParentEdge(nvdla::priv::engine_ast::Edge* edg)
    {
        m_edge = edg;
    }

protected:
    std::string m_name;
    std::string m_id;
    Dims4 m_dims;
    NvU64 m_size;
    NvU32 m_line_stride;
    NvU32 m_surface_stride;
    memory::TensorCategory m_tensor_category; // Global/Local/Stream
    nvdla::DataFormat m_data_format;
    SurfaceFormat m_surface_format;
    memory::TensorBufferDesc* m_buffer_desc;
    bool m_copyOutDebugSurface;

    std::unordered_set<nvdla::priv::engine_ast::Node*> m_producers; // producers of this surface
    std::unordered_set<nvdla::priv::engine_ast::Node*> m_consumers; // consumers of this surface

    bool m_content;
    bool m_align_ls;
    size_t m_activeProducers;
    size_t m_activeConsumers;
    NvS16 m_bindId;
    enum IOD m_bindDomain;

    // tsd state for each of the batches in multi-batch case
    MultiBatchState<SurfaceDescState>* m_mb_tsd_state;

    nvdla::priv::engine_ast::Edge* m_edge; //associated edge
};

}; // namespace surface
} // namespace priv
} // namespace nvdla

#endif // NVDLA_PRIV_SURFACE_H
