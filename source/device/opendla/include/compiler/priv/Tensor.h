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

#ifndef NVDLA_PRIV_TENSOR_H
#define NVDLA_PRIV_TENSOR_H

#include <string>

#include "Network.h"
#include "Type.h"
#include "nvdla/INetwork.h"
#include "nvdla/ITensor.h"

#define MAX_TENSOR_SIZE (1 << 30)

namespace nvdla {

namespace priv {

class WisdomContainerEntry;

class Tensor;

class TensorFactory
{
public:
    typedef PrivPair<ITensor*, Tensor*> TensorPrivPair;

    static TensorPrivPair newTensor();

    static Tensor* priv(ITensor*);
    static ITensor* i(Tensor*);
    static ITensor* self(void*);

    static ITensor* deserializeFrom(WisdomContainerEntry*);

protected:
    static BiMap<ITensor*, Tensor*> s_priv;
    static BiMap<void*, ITensor*> s_self;
    static ITensor* deserializeTensor(WisdomContainerEntry*);
};

class Tensor : public ITensor
{
public: // externally facing
    Tensor(INetwork* network, const std::string name);
    virtual ~Tensor();

    virtual NvU16 getFactoryType() const;

    virtual const char* getName() const;
    virtual void setName(const char* n);

    virtual Dims4 getDimensions() const;
    virtual void setDimensions(Dims4 dimensions);

    virtual bool isNetworkInput() const;
    virtual bool isNetworkOutput() const;

    virtual DataFormat getDataFormat() const;
    virtual void setDataFormat(DataFormat);

    virtual DataType getDataType() const;
    virtual void setDataType(DataType);

    TensorType getTensorType() const;
    void setTensorType(TensorType);

    virtual INetwork* getNetwork() const;

    virtual Tensor* clone()
    {
        return new Tensor(*this);
    }

    virtual NvDlaError setChannelDynamicRange(NvS32 chnlIndx, NvF32 min, NvF32 max);
    virtual NvDlaError setChannelOffset(NvS32 chnlIndx, NvF32 offset);

    const std::vector<NvF32>& getChannelScales() const
    {
        return mChnlScales;
    }
    void setChannelScales(std::vector<NvF32> chnlScales)
    {
        mChnlScales = chnlScales;
    }

    const std::vector<NvF32>& getChannelOffsets() const
    {
        return mChnlOffsets;
    }
    void setChannelOffsets(std::vector<NvF32> chnlOffsets)
    {
        mChnlOffsets = chnlOffsets;
    }

#if 0
    virtual ILayer *getProducerLayer()          const;
    virtual int     getNumConsumerLayers()       const;
    virtual ILayer *getConsumerLayer(int index) const;
#endif

public: // internally facing
    Tensor()
        : mDimensions({0, 0, 0, 0}),
          mNetwork(NULL),
          mName(""),
          mDataFormat(DataFormat::UNKNOWN),
          mDataType(DataType::UNKNOWN),
          mTensorType(TensorType::kUNKNOWN){};

    void setNetwork(INetwork* network);
    // void setName(const std::string name);
    Tensor(const Tensor& other)
        : mDimensions(other.mDimensions),
          mNetwork(other.mNetwork),
          mName(other.mName),
          mDataFormat(other.mDataFormat),
          mDataType(other.mDataType),
          mTensorType(other.mTensorType),
          mChnlScales(other.mChnlScales),
          mChnlOffsets(other.mChnlOffsets){};

    virtual bool serializeTo(WisdomContainerEntry*) const;
    virtual bool deserializeFrom(WisdomContainerEntry*);

protected:
    Dims4 mDimensions;
    INetwork* mNetwork;
    std::string mName; // the user name if the user provided one, else
    DataFormat mDataFormat;
    DataType mDataType;
    TensorType mTensorType;          // the type of surface this tensor represents: image/i-o/kernel/bias
    std::vector<NvF32> mChnlScales;  // per-channel scaling factors
    std::vector<NvF32> mChnlOffsets; // per-channel offsets
};

} // namespace priv

} // namespace nvdla

#endif // NVDLA_PRIV_TENSOR_H
