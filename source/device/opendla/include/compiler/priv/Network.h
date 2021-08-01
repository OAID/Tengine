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

#ifndef NVDLA_PRIV_NETWORK_H
#define NVDLA_PRIV_NETWORK_H

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ResourceEnums.h" // for tensor surface types
#include "Type.h"
#include "WisdomContainer.h"
#include "nvdla/INetwork.h"
#include "nvdla/ITensor.h"

// #include "priv/EngineAST.h"

namespace nvdla {
namespace priv {

class Network;

class NetworkFactory
{
public:
    // PrivPair 是一个模版类，用来描述接口和具体实现之间的映射关系
    typedef PrivPair<INetwork*, Network*> NetworkPrivPair;

    static NetworkPrivPair newNetwork();
    static NvDlaError deleteNetwork(INetwork* network);
    // 通过 INetwork 访问到对应的 Network
    static Network* priv(INetwork*);
    // 通过 Network 访问到对应的 INetwork
    static INetwork* i(Network*);
    static INetwork* self(void* s);

    static INetwork* deserializeFrom(WisdomContainerEntry*);

protected:
    static BiMap<INetwork*, Network*> s_priv;
    static BiMap<void*, INetwork*> s_self;

    static INetwork* deserializeNetwork(WisdomContainerEntry*);
};

class Network : public INetwork
{
public: // externally facing
    virtual ITensor* addInput(const char* name, Dims4 dimensions);

    //	virtual void markChanged(const ILayer*);
    virtual bool markInput(ITensor* tensor);
    virtual void markOutput(ITensor* tensor);

    virtual IConvolutionLayer* addConvolution(ITensor* input, int numOutputs, int paddingValue,
                                              Dims2 kernelSize, Dims2 tlPadding, Dims2 brPadding, Dims2 stride, Dims2 dilation,
                                              Weights kernelWeights, Weights biasWeights, BiasMode biasmode, int numGroups);
    virtual IFullyConnectedLayer* addFullyConnected(ITensor* input, int outputSize, Weights kernelWeights, Weights biasWeights, BiasMode biasMode);
    virtual IActivationLayer* addActivation(ITensor* input, ActivationType type);
    virtual IPoolingLayer* addPooling(ITensor* input, PoolingType type,
                                      Dims2 windowSize, Dims2 stride, Dims2 tlPadding, Dims2 brPadding);
    virtual ILRNLayer* addLRN(ITensor* input, int window, float alpha, float beta, float k);
    virtual IScaleLayer* addScale(ITensor* input, ScaleMode mode, Weights shift, Weights scale, Weights power);
    virtual IBatchNormLayer* addBatchNorm(ITensor* input, BatchNormMode mode, Weights mean, Weights variance, float epsilon);
    virtual ISoftMaxLayer* addSoftMax(ITensor* input);
    virtual IConcatenationLayer* addConcatenation(ITensor* const* inputs, int numInputs);
    virtual ISliceLayer* addSlice(ITensor* input, int numOutputs);
    virtual IDeconvolutionLayer* addDeconvolution(ITensor* input, int numOutputs, int paddingValue,
                                                  Dims2 kernelSize, Dims2 tlPadding, Dims2 brPadding, Dims2 stride, Dims2 dilation,
                                                  Weights kernelWeights, Weights biasWeights, BiasMode biasMode, int numGroups);
    virtual IElementWiseLayer* addElementWise(ITensor* input0, ITensor* input1, ElementWiseOperation op);

    virtual int getNumInputs() const;
    virtual int getNumOutputs() const;
    virtual int getNumLayers() const;

    virtual ILayer* getLayer(int index) const;
    virtual ITensor* getOutput(int index) const;
    virtual ITensor* getInput(int index) const;

    virtual void setPoolingOutputDimensionsFormula(OutputDimensionsFormula* callback);
    virtual void setConvolutionOutputDimensionsFormula(OutputDimensionsFormula* callback);
    virtual void setDeconvolutionOutputDimensionsFormula(OutputDimensionsFormula* callback);

    virtual OutputDimensionsFormula& getPoolingOutputDimensionsFormula() const;
    virtual OutputDimensionsFormula& getConvolutionOutputDimensionsFormula() const;
    virtual OutputDimensionsFormula& getDeconvolutionOutputDimensionsFormula() const;

    virtual const std::vector<ITensor*>& getInputs() const;
    virtual const std::vector<ILayer*>& getLayers() const;
    virtual const std::vector<ITensor*>& getOutputs() const;

    virtual NvU16 getFactoryType() const;

public: // internally facing
    Network();
    virtual ~Network();
    virtual bool serializeTo(WisdomContainerEntry*) const;
    virtual bool deserializeFrom(WisdomContainerEntry*);
    virtual bool assignSymbols(Wisdom*);

protected:
    friend class Wisdom;
    friend class NetworkFactory;

    void destroy();

private:
    std::string newLayerName() const;
    std::string newTensorName() const;

    ITensor* addTensor(const std::string& s);
    const ILayer* findLayer(const std::string& name) const;
    bool checkNames(const char* name);

    std::vector<ITensor*> mTensors;
    std::vector<ILayer*> mLayers;
    std::vector<ITensor*> mInputs;
    std::vector<ITensor*> mOutputs;

    // provides layer dimension caching. Layers can be mutated in any order and dimensions queried at any point.
    // So mutating a layer trims this, and querying always refills the cache up to the queried layer
    //	mutable std::vector<Dims3> mDimensions;

    // internal flags used by the builder that are not accessible through the API
    // int mInternalBuildFlags{ InternalBuildFlags::kENABLE_GRAPH_OPTIMIZATIONS };
    OutputDimensionsFormula *mConvDims, *mDeconvDims, *mPoolDims;
};

extern std::map<LayerType, std::string> layerTypeNames;
extern std::map<ActivationType, std::string> activationTypeNames;

} // namespace priv
} // namespace nvdla

#endif /* NVDLA_NETWORK_PRIV_H */
