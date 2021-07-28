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

#ifndef NVDLA_I_LAYER_H
#define NVDLA_I_LAYER_H

#include "nvdla/IType.h"


namespace nvdla
{

class INetwork;
class ITensor;

class ILayer
{
public:

    virtual Dims4 getOutputDimensions() const = 0;

    virtual void        setName(const char* name) = 0;
    virtual const char* getName() const = 0;

    virtual LayerType getType() const = 0;

    virtual int      getNumInputs() const = 0;
    virtual ITensor* getInput(int i)  const = 0;

    virtual int      getNumOutputs() const = 0;
    virtual ITensor* getOutput(int i) const = 0;

    virtual void  setKernelSize(Dims2 kernelSize) = 0;
    virtual Dims2 getKernelSize() const = 0;

    virtual void setNumOutputMaps(int numOutputMaps) = 0;
    virtual int  getNumOutputMaps() const = 0;

    virtual void  setStride(Dims2 stride) = 0;
    virtual Dims2 getStride() const = 0;

    virtual  Dims2 getDilation() const = 0;
    virtual  void  setDilation(Dims2) = 0;

    virtual int getPaddingValue() const = 0;
    virtual void setPaddingValue(int) = 0;

    virtual Dims2 getBottomRightPadding() const = 0;
    virtual void  setBottomRightPadding(Dims2) = 0;

    virtual Dims2 getTopLeftPadding() const = 0;
    virtual void  setTopLeftPadding(Dims2) = 0;

    virtual void setNumGroups(int numGroups) = 0;
    virtual int  getNumGroups() const = 0;

    virtual void    setKernelWeights(Weights weights) = 0;
    virtual Weights getKernelWeights() const = 0;
    virtual bool    hasKernelWeights() = 0;

    virtual void    setBiasWeights(Weights weights) = 0;
    virtual Weights getBiasWeights() const = 0;
    virtual bool    hasBiasWeights() = 0;

    virtual NvDlaError supportsPrecision(nvdla::DataType compPrec) = 0;
    virtual NvDlaError setComputePrecision(nvdla::DataType compPrec) = 0;

protected:
    ILayer();
    virtual ~ILayer();
};



class IConvolutionLayer : public virtual ILayer
{
public:

    class Parameters
    {
    public:
        Dims2 kernelSize;
        Dims2 topLeftPadding;
        Dims2 bottomRightPadding;
        Dims2 stride;
        Dims2 dilation;
        int numOutputMaps;
        int numGroups;
        Weights kernelWeights;
        Weights biasWeights;
        int paddingValue;
        BiasMode biasMode;
    };

    virtual Dims2 getKernelSize() const = 0;
    virtual void  setKernelSize(Dims2) = 0;

    virtual int getPaddingValue() const = 0;
    virtual void setPaddingValue(int) = 0;

    virtual Dims2 getBottomRightPadding() const = 0;
    virtual void  setBottomRightPadding(Dims2) = 0;

    virtual Dims2 getTopLeftPadding() const = 0;
    virtual void  setTopLeftPadding(Dims2) = 0;

    virtual  Dims2 getStride() const = 0;
    virtual  void  setStride(Dims2) = 0;

    virtual  Dims2 getDilation() const = 0;
    virtual  void  setDilation(Dims2) = 0;

    virtual int  getNumGroups() const = 0;
    virtual void setNumGroups(int) = 0;

    virtual int  getNumOutputMaps() const = 0;
    virtual void setNumOutputMaps(int) = 0;

    virtual Weights getKernelWeights() const = 0;
    virtual void    setKernelWeights(Weights) = 0;
    virtual bool    hasKernelWeights() = 0;

    virtual Weights getBiasWeights() const = 0;
    virtual void    setBiasWeights(Weights) = 0;
    virtual bool    hasBiasWeights() = 0;

    virtual BiasMode getBiasMode() const = 0;
    virtual void setBiasMode(BiasMode) = 0;

    // virtual Dims4 getOutputDimensions() const = 0;
    virtual const Parameters& getParams() const = 0;

protected:
    virtual ~IConvolutionLayer();
};

class IFullyConnectedLayer : public virtual ILayer
{
public:

    class Parameters
    {
    public:
        int numOutputChannels;
        Dims2 kernelSize;
        Weights kernelWeights;
        Weights biasWeights;
        BiasMode biasMode;
    };


    virtual int getNumOutputChannels() const = 0;
    virtual void setNumOutputChannels(int) = 0;

    virtual Weights getKernelWeights() const = 0;
    virtual void    setKernelWeights(Weights) = 0;
    virtual bool    hasKernelWeights() = 0;

    virtual Weights getBiasWeights() const = 0;
    virtual void    setBiasWeights(Weights) = 0;
    virtual bool    hasBiasWeights() = 0;

    virtual BiasMode getBiasMode() const = 0;
    virtual void setBiasMode(BiasMode) = 0;

    //    virtual Dims4 getOutputDimensions() const = 0;
    virtual const Parameters& getParams() const = 0;
protected:
    virtual ~IFullyConnectedLayer();
};

class IActivationLayer : public virtual ILayer
{
public:
    class Parameters
    {
    public:
        ActivationType activationType;
    };

    virtual ActivationType getActivationType() const = 0;
    virtual void setActivationType(ActivationType) = 0;

    //    virtual Dims4 getOutputDimensions() const = 0;
    virtual const Parameters& getParams() const = 0;
protected:
    virtual ~IActivationLayer();
};

class IPoolingLayer : public virtual ILayer
{
public:

    class Parameters
    {
    public:
        PoolingType poolingType;
        Dims2 windowSize;
        Dims2 topLeftPadding;
        Dims2 bottomRightPadding;
        Dims2 stride;
    };

    virtual PoolingType getPoolingType() const = 0;
    virtual void setPoolingType(PoolingType) = 0;

    virtual Dims2 getWindowSize() const = 0;
    virtual void  setWindowSize(Dims2) = 0;

    virtual Dims2 getBottomRightPadding() const = 0;
    virtual void  setBottomRightPadding(Dims2) = 0;

    virtual Dims2 getTopLeftPadding() const = 0;
    virtual void  setTopLeftPadding(Dims2) = 0;

    virtual  Dims2 getStride() const = 0;
    virtual  void  setStride(Dims2) = 0;

    //    virtual Dims4 getOutputDimensions() const = 0;
    virtual const Parameters& getParams() const = 0;

protected:
    virtual ~IPoolingLayer();
};


class ILRNLayer : public virtual ILayer
{
public:


    class Parameters
    {
    public:

        int windowSize;
        float alpha;
        float beta;
        float k;

        static inline int minWindowSize() { return 1; }
        static inline int maxWindowSize() { return 16; }
        static inline float maxAbsAlpha() { return 1e20f; }
        static inline float minBeta() { return 0.01f; }
        static inline float maxBeta() { return 1e5f; }
        static inline float minK() { return 1e-5f; }
        static inline float maxK() { return 1e10f; }
    };

    virtual float getAlpha() const = 0;
    virtual void setAlpha(float) = 0;

    virtual float getBeta() const = 0;
    virtual void setBeta(float) = 0;

    virtual float getK() const = 0;
    virtual void setK(float) = 0;

    virtual int getWindowSize() const = 0;
    virtual void setWindowSize(int) = 0;

    //    virtual Dims4 getOutputDimensions() const = 0;
    //virtual bool  validateParams(Parameters *) const = 0;
    virtual const Parameters& getParams() const = 0;

protected:
    virtual ~ILRNLayer();
};


class IScaleLayer : public virtual ILayer
{
public:

    class Parameters
    {
    public:
        ScaleMode mode;
        Weights shift;
        Weights scale;
        Weights power;
    };

    virtual Weights getScale() const = 0;
    virtual void setScale(Weights) = 0;

    virtual Weights getShift() const = 0;
    virtual void setShift(Weights) = 0;

    virtual Weights getPower() const = 0;
    virtual void setPower(Weights) = 0;

    virtual ScaleMode getMode() const = 0;
    virtual void setMode(ScaleMode) = 0;

    //    virtual Dims4 getOutputDimensions()  const = 0;
    virtual const Parameters& getParams() const = 0;
protected:
    virtual ~IScaleLayer();
};

class IBatchNormLayer : public virtual ILayer
{
public:

    class Parameters
    {
    public:
        BatchNormMode mode;
        Weights mean;
        Weights variance;
        float epsilon;
    };

    virtual float getEpsilon() const = 0;
    virtual void setEpsilon(float) = 0;

    virtual Weights getMean() const = 0;
    virtual void setMean(Weights) = 0;

    virtual Weights getVariance() const = 0;
    virtual void setVariance(Weights) = 0;

    virtual BatchNormMode getMode() const = 0;
    virtual void setMode(BatchNormMode) = 0;

    virtual const Parameters& getParams() const = 0;
protected:
    virtual ~IBatchNormLayer();
};


class ISoftMaxLayer : public virtual ILayer
{
public:

    class Parameters
    {
    public:
    };

    //    virtual Dims4 getOutputDimensions()  const = 0;
    virtual const Parameters& getParams() const = 0;
protected:
    virtual ~ISoftMaxLayer();
};


class IConcatenationLayer : public virtual ILayer
{
public:
    class Parameters
    {
    public:
    };

    //    virtual Dims4 getOutputDimensions() const = 0;
    virtual const Parameters& getParams() const = 0;
protected:
    virtual ~IConcatenationLayer();
};


class ISliceLayer : public virtual ILayer
{
public:
    class Parameters
    {
    public:
    };

    //    virtual Dims4 getOutputDimensions() const = 0;
    virtual const Parameters& getParams() const = 0;
protected:
    virtual ~ISliceLayer();
};


class IDeconvolutionLayer : public virtual ILayer
{
public:

    class Parameters
    {
    public:
        Dims2 kernelSize;
        Dims2 topLeftPadding;
        Dims2 bottomRightPadding;
        Dims2 stride;
        Dims2 dilation;
        int numOutputMaps;
        int numGroups;
        Weights kernelWeights;
        Weights biasWeights;
        int paddingValue;
        BiasMode biasMode;
    };


    virtual Dims2 getKernelSize() const = 0;
    virtual void  setKernelSize(Dims2) = 0;

    virtual int getPaddingValue() const = 0;
    virtual void setPaddingValue(int) = 0;

    virtual Dims2 getBottomRightPadding() const = 0;
    virtual void  setBottomRightPadding(Dims2) = 0;

    virtual Dims2 getTopLeftPadding() const = 0;
    virtual void  setTopLeftPadding(Dims2) = 0;

    virtual  Dims2 getStride() const = 0;
    virtual  void  setStride(Dims2) = 0;

    virtual  Dims2 getDilation() const = 0;
    virtual  void  setDilation(Dims2) = 0;

    virtual int  getNumGroups() const = 0;
    virtual void setNumGroups(int) = 0;

    virtual int  getNumOutputMaps() const = 0;
    virtual void setNumOutputMaps(int) = 0;

    virtual Weights getKernelWeights() const = 0;
    virtual void    setKernelWeights(Weights) = 0;
    virtual bool    hasKernelWeights() = 0;

    virtual Weights getBiasWeights() const = 0;
    virtual void    setBiasWeights(Weights) = 0;
    virtual bool    hasBiasWeights() = 0;

    virtual BiasMode getBiasMode() const = 0;
    virtual void setBiasMode(BiasMode) = 0;

    //    virtual Dims4 getOutputDimensions() const = 0;
    virtual const Parameters& getParams() const = 0;
protected:
    virtual ~IDeconvolutionLayer();
};


class IElementWiseLayer : public virtual ILayer
{
public:
    class Parameters
    {
    public:
        ElementWiseOperation operation;
    };

    virtual ElementWiseOperation getOperation() const = 0;
    virtual void setOperation(ElementWiseOperation) = 0;

    //    virtual Dims4 getOutputDimensions() const = 0;
    virtual const Parameters& getParams() const = 0;
protected:
    virtual ~IElementWiseLayer();
};


class IInputLayer : public virtual ILayer
{
public:
    class Parameters
    {
    public:
        // InputOperation operation;
    };

    // virtual InputOperation getOperation() const = 0;
    // virtual void setOperation(InputOperation) = 0;

    // virtual Dims4 getOutputDimensions() const = 0;
    virtual const Parameters& getParams() const = 0;
protected:
    virtual ~IInputLayer();
};




}
#endif // NVDLA_I_LAYER_H
