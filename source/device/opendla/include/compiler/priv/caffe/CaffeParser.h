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

#ifndef NVDLA_PRIV_CAFFE_PARSER_H
#define NVDLA_PRIV_CAFFE_PARSER_H

#include <iostream>
#include <istream>
#include <vector>
#include <map>
#include <memory>
#include <string>

#include "compiler/priv/Layer.h"
#include "compiler/priv/Network.h"
#include "compiler/priv/Type.h"
#include "nvdla/caffe/ICaffeParser.h"

namespace ditcaffe {
class NetParameter;
}

namespace nvdla {
namespace caffe {
namespace priv {

class BlobNameToTensor : public IBlobNameToTensor
{
public:
    virtual void add(const std::string& name, ITensor* tensor);

    virtual ITensor* find(const char* name) const;
    virtual ITensor*& operator[](const std::string& name);

    virtual void setTensorNames();
    virtual ~BlobNameToTensor();

private:
    std::map<std::string, ITensor*> mMap;
};

class BinaryProtoBlob : public IBinaryProtoBlob
{
public:
    BinaryProtoBlob(void* memory, DataType type, Dims4 dimensions);

    const void* getData();
    Dims4 getDimensions();
    void destroy();

protected:
    void* mMemory;
    DataType mDataType;
    Dims4 mDimensions;
    virtual ~BinaryProtoBlob();
};

class CaffeParser;

class CaffeParserFactory
{
public:
    typedef nvdla::priv::PrivPair<ICaffeParser*, CaffeParser*> CaffeParserPrivPair;

    static CaffeParserPrivPair newCaffeParser();
    static NvDlaError deleteCaffeParser(ICaffeParser* parser);

    static CaffeParser* priv(ICaffeParser*);
    static ICaffeParser* i(CaffeParser*);
    static ICaffeParser* self(void*);

protected:
    static nvdla::priv::BiMap<ICaffeParser*, CaffeParser*> s_priv;
    static nvdla::priv::BiMap<void*, ICaffeParser*> s_self;
};

class CaffeParser : public ICaffeParser
{
public:
    CaffeParser()
        : ICaffeParser(),
          mDeploy(NULL),
          mModel(NULL),
          mTmpAllocs(),
          mDimsCallback(NULL),
          mBlobNameToTensor(NULL),
          mProtobufBufferSize(1024 << 20)
    {
    }

    virtual const IBlobNameToTensor* parse(const char* deploy,
                                           const char* model,
                                           INetwork* network);
    virtual int identifyOutputs(INetwork* network);
    virtual ~CaffeParser();

    void setProtobufBufferSize(size_t size)
    {
        mProtobufBufferSize = size;
    }

    // read a blob from a protobuf file (typically a mean blob)
    static BinaryProtoBlob* parseBinaryProto(const char* fileName);

    static void shutdownProtobufLibrary();

private:
    ditcaffe::NetParameter* mDeploy;
    ditcaffe::NetParameter* mModel;
    std::vector<void*> mTmpAllocs;
    INetwork::OutputDimensionsFormula* mDimsCallback;
    IBlobNameToTensor* mBlobNameToTensor;
    size_t mProtobufBufferSize;
};

} // namespace priv
} // namespace caffe
} // namespace nvdla

#endif // NVDLA_PRIV_CAFFE_PARSER_H
