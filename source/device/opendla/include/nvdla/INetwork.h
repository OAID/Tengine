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

#ifndef NVDLA_I_NETWORK_H
#define NVDLA_I_NETWORK_H

#include <map>
#include <vector>
#include <memory>

#include "nvdla/IType.h"

namespace nvdla
{

class INetwork;
class ILayer;
class IConvolutionLayer;
class IFullyConnectedLayer;
class IActivationLayer;
class IPoolingLayer;
class ILRNLayer;
class IScaleLayer;
class IBatchNormLayer;
class ISoftMaxLayer;
class IConcatenationLayer;
class ISliceLayer;
class IDeconvolutionLayer;
class IElementWiseLayer;

class IWisdomContainerEntry;
class ITensor;


class INetwork
{
public:
    virtual ITensor* addInput(const char * name, Dims4 dimensions) = 0;

    //	virtual void markChanged(const ILayer *) = 0;
    //  指定网络的 Input 和 Output Tensor
    virtual bool markInput(ITensor * tensor) = 0;
    virtual void markOutput(ITensor * tensor) = 0;
    // 构建网络的API函数，理论上通过以下这组add函数，就可以不使用caffe模型，手工的创建一个网络，类似大多数框架提供的网络构造API函数。
    // 但NVDLA似乎没有对外开放这组接口用于手工构造网络，TVM框架就对外开放了这组接口
    virtual IConvolutionLayer *    addConvolution   (ITensor * input, int numOutputs, int paddingValue, Dims2 kernelSize,
                                                    Dims2 tlPadding, Dims2 brPadding, Dims2 stride, Dims2 dilation,
                                                    Weights kernelWeights, Weights biasWeights, BiasMode biasMode, int numGroups) = 0;
    virtual IFullyConnectedLayer * addFullyConnected(ITensor * input, int outputSize, Weights kernelWeights, Weights biasWeights, BiasMode biasMode) = 0;
    virtual IActivationLayer *     addActivation    (ITensor * input, ActivationType type) = 0;
    virtual IPoolingLayer *        addPooling       (ITensor * input, PoolingType type,
                                                    Dims2 windowSize, Dims2 stride, Dims2 tlPadding, Dims2 brPadding) = 0;
    virtual ILRNLayer *            addLRN           (ITensor * input, int window, float alpha, float beta, float k) = 0;
    virtual IScaleLayer *          addScale         (ITensor * input, ScaleMode mode, Weights shift, Weights scale, Weights power) = 0;
    virtual IBatchNormLayer *      addBatchNorm     (ITensor * input, BatchNormMode mode, Weights mean, Weights variance, float epsilon) = 0;
    virtual ISoftMaxLayer *        addSoftMax       (ITensor*input) = 0;
    virtual IConcatenationLayer *  addConcatenation (ITensor*const*inputs, int numInputs) = 0;
    virtual ISliceLayer *          addSlice         (ITensor*input, int numOutputs) = 0;
    virtual IDeconvolutionLayer *  addDeconvolution (ITensor * input, int numOutputs, int paddingValue,
                                                    Dims2 kernelSize, Dims2 tlPadding, Dims2 brPadding, Dims2 stride, Dims2 dilation,
                                                    Weights kernelWeights, Weights biasWeights, BiasMode biasMode, int numGroups) = 0;
    virtual IElementWiseLayer   *  addElementWise   (ITensor *input0, ITensor* input1, ElementWiseOperation op) = 0;

    virtual int getNumInputs()  const  = 0;
    virtual int getNumOutputs() const  = 0;
    virtual int getNumLayers()  const  = 0;

    virtual ILayer  * getLayer(int index)  const = 0;
    virtual ITensor * getOutput(int index) const = 0;
    virtual ITensor * getInput(int index)  const = 0;


    class OutputDimensionsFormula
    {
    public:
        virtual Dims2 compute(Dims2 inputDims, Dims2 kernelSize,  Dims2 stride, Dims2 tlPadding, Dims2 brPadding, const char* layerName) const = 0;
        virtual Dims2 compute(Dims2 inputDims, Dims2 kernelSize,  Dims2 stride, Dims2 tlPadding, Dims2 brPadding, Dims2 dilation, const char* layerName) const = 0;
        virtual ~OutputDimensionsFormula() { }
    };

    class NetworkDefaultConvolutionFormula : public OutputDimensionsFormula
    {
    public:
        virtual Dims2 compute(Dims2 input, Dims2 kernel, Dims2 stride, Dims2 tlPadding, Dims2 brPadding, const char*) const;
        virtual Dims2 compute(Dims2 input, Dims2 kernel, Dims2 stride, Dims2 tlPadding, Dims2 brPadding, Dims2 dilation, const char*) const;
    };

    class NetworkDefaultDeconvolutionFormula : public OutputDimensionsFormula
    {
    public:
        virtual Dims2 compute(Dims2 input, Dims2 kernel, Dims2 stride, Dims2 tlPadding, Dims2 brPadding, const char*) const;
        virtual Dims2 compute(Dims2 input, Dims2 kernel, Dims2 stride, Dims2 tlPadding, Dims2 brPadding, Dims2 dilation, const char*) const;
    };

    class NetworkDefaultPoolingFormula : public OutputDimensionsFormula
    {
    public:
        virtual Dims2 compute(Dims2 input, Dims2 kernel, Dims2 stride, Dims2 tlPadding, Dims2 brPadding, const char*) const;
        virtual Dims2 compute(Dims2 /*input*/, Dims2 /*kernel*/, Dims2 /*stride*/, Dims2 /*tlPadding*/, Dims2 /*brPadding*/, Dims2 /*dilation*/, const char*) const
        {
            return Dims2(-1, -1);
        }
    };

    virtual void setPoolingOutputDimensionsFormula      (OutputDimensionsFormula* callback) = 0;
    virtual void setConvolutionOutputDimensionsFormula  (OutputDimensionsFormula* callback) = 0;
    virtual void setDeconvolutionOutputDimensionsFormula(OutputDimensionsFormula* callback) = 0;

    virtual OutputDimensionsFormula& getPoolingOutputDimensionsFormula()       const = 0;
    virtual OutputDimensionsFormula& getConvolutionOutputDimensionsFormula()   const = 0;
    virtual OutputDimensionsFormula& getDeconvolutionOutputDimensionsFormula() const = 0;

    virtual const std::vector<ITensor *> & getInputs()  const = 0;
    virtual const std::vector<ILayer * > & getLayers()  const = 0;
    virtual const std::vector<ITensor *> & getOutputs() const = 0;

protected:
    INetwork();
    virtual ~INetwork();

};

INetwork *createNetwork();
NvDlaError destroyNetwork(INetwork *network);

}

#endif // NVDLA_I_NETWORK_H
