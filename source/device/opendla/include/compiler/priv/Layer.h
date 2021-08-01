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

#ifndef NVDLA_PRIV_LAYER_H
#define NVDLA_PRIV_LAYER_H

#include <map>
#include <string>
#include <vector>

#include "Type.h"
#include "nvdla/ILayer.h"

#define MAX_CONCAT_INPUTS       10000
#define MAX_SLICE_OUTPUTS       10000
#define MAX_KERNEL_DIMS_PRODUCT 100000
#define MAX_PADDING_SUM         100000
#define MAX_PADDING_VALUE       255
#define MAX_STRIDE_SUM          100000
#define MAX_DILATION_SUM        100000
#define MAX_OUTPUT_MAPS         (1 << 28)
#define MAX_GROUPS              10000
#define MAX_WEIGHTS_COUNT       (1 << 28)

#define MAX_BATCH_SIZE        (1 << 28)
#define MAX_FIND_ITERATIONS   1000
#define MAX_NUM_INPUT_LAYERS  10000
#define MAX_NUM_OUTPUT_LAYERS 10000

//
// note: preprocessor definitions must happen at global scope level per MISRA
//

//
// LayerFactory can instance these...
//
#define LAYER_FACTORY_TYPE_ENUMS(op)                              \
    op(CONVOLUTION, 0U)                                           \
        op(FULLY_CONNECTED, 1U)                                   \
            op(ACTIVATION, 2U)                                    \
                op(POOLING, 3U)                                   \
                    op(LRN, 4U)                                   \
                        op(SCALE, 5U)                             \
                            op(BATCH_NORM, 6U)                    \
                                op(SOFT_MAX, 7U)                  \
                                    op(CONCATENATION, 8U)         \
                                        op(DECONVOLUTION, 9U)     \
                                            op(ELEMENT_WISE, 10U) \
                                                op(INPUT, 11U)    \
                                                    op(SLICE, 12U)

namespace nvdla {
class INetwork;
class ITensor;
class ILayer;

namespace priv {

class Wisdom;
class WisdomContainerEntry;

class Layer;
class ConvolutionLayer;
class FullyConnectedLayer;
class ActivationLayer;
class PoolingLayer;
class LRNLayer;
class ScaleLayer;
class BatchNormLayer;
class SoftMaxLayer;
class ConcatenationLayer;
class SliceLayer;
class DeconvolutionLayer;
class ElementWiseLayer;
class InputLayer;

class LayerFactoryType
{
public:
    enum Enum
    {
        LAYER_FACTORY_TYPE_ENUMS(GEN_ENUM)
    };
};
typedef SequenceEnum<LayerFactoryType::Enum, NvU16> LayerTypeEnum;

typedef PrivDiamond<ILayer, Layer, IConvolutionLayer, ConvolutionLayer> ConvolutionLayerDiamond;
typedef PrivDiamond<ILayer, Layer, IFullyConnectedLayer, FullyConnectedLayer> FullyConnectedLayerDiamond;
typedef PrivDiamond<ILayer, Layer, IActivationLayer, ActivationLayer> ActivationLayerDiamond;
typedef PrivDiamond<ILayer, Layer, IPoolingLayer, PoolingLayer> PoolingLayerDiamond;
typedef PrivDiamond<ILayer, Layer, ILRNLayer, LRNLayer> LRNLayerDiamond;
typedef PrivDiamond<ILayer, Layer, IScaleLayer, ScaleLayer> ScaleLayerDiamond;
typedef PrivDiamond<ILayer, Layer, IBatchNormLayer, BatchNormLayer> BatchNormLayerDiamond;
typedef PrivDiamond<ILayer, Layer, ISoftMaxLayer, SoftMaxLayer> SoftMaxLayerDiamond;
typedef PrivDiamond<ILayer, Layer, IConcatenationLayer, ConcatenationLayer> ConcatenationLayerDiamond;
typedef PrivDiamond<ILayer, Layer, ISliceLayer, SliceLayer> SliceLayerDiamond;
typedef PrivDiamond<ILayer, Layer, IDeconvolutionLayer, DeconvolutionLayer> DeconvolutionLayerDiamond;
typedef PrivDiamond<ILayer, Layer, IElementWiseLayer, ElementWiseLayer> ElementWiseLayerDiamond;
typedef PrivDiamond<ILayer, Layer, IInputLayer, InputLayer> InputLayerDiamond;

class LayerFactory
{
public:
    // called before deserialization when data otherwise given to the constructor is not available.
    template<class D>
    static D newLayer();

    static ILayer* deserializeFrom(WisdomContainerEntry*);
    static Layer* priv(ILayer*);
    static ILayer* i(Layer*);
    static ILayer* self(void*);

    // these are called when constructor data is available.  would prefer to remove these
    // and replace with a scheme to unify deserialization and initial construction.
    static ConvolutionLayerDiamond newConvolutionLayer(INetwork* network, const std::string& name,
                                                       ITensor* input, ITensor* output,
                                                       int numOutputMaps, int paddingValue,
                                                       Dims2 kernelSize, Dims2 tlPadding, Dims2 brPadding, Dims2 stride, Dims2 dilation,
                                                       Weights kernelWeights, Weights biasWeights, BiasMode biasMode, int numGroups);
    static FullyConnectedLayerDiamond newFullyConnectedLayer(INetwork* network, const std::string& name,
                                                             ITensor* input, ITensor* output,
                                                             int numOutputChannels,
                                                             Weights kernelWeights, Weights biasWeights, BiasMode biasMode);
    static ActivationLayerDiamond newActivationLayer(INetwork* network, const std::string& name,
                                                     ITensor* input, ITensor* output,
                                                     ActivationType activationType);
    static PoolingLayerDiamond newPoolingLayer(INetwork* network, const std::string& name,
                                               ITensor* input, ITensor* output,
                                               PoolingType type,
                                               Dims2 windowSize, Dims2 stride, Dims2 tlPadding, Dims2 brPadding);
    static LRNLayerDiamond newLRNLayer(INetwork* network, const std::string& name,
                                       ITensor* input,
                                       ITensor* output,
                                       int windowSize, float alpha, float beta, float k);
    static ScaleLayerDiamond newScaleLayer(INetwork* network, const std::string& name,
                                           ITensor* input, ITensor* output,
                                           ScaleMode mode,
                                           Weights shift, Weights scale, Weights power);
    static BatchNormLayerDiamond newBatchNormLayer(INetwork* network, const std::string& name,
                                                   ITensor* input, ITensor* output,
                                                   BatchNormMode mode,
                                                   Weights mean, Weights variance, float epsilon);
    static SoftMaxLayerDiamond newSoftMaxLayer(INetwork* network, const std::string& name,
                                               ITensor* input, ITensor* output);
    static ConcatenationLayerDiamond newConcatenationLayer(INetwork* network, const std::string& name,
                                                           ITensor* const* inputs, int numInputs,
                                                           ITensor* output);
    static SliceLayerDiamond newSliceLayer(INetwork* network, const std::string& name,
                                           ITensor* input,
                                           ITensor* const* outputs, int numOutputs);
    static DeconvolutionLayerDiamond newDeconvolutionLayer(INetwork* network, const std::string& name,
                                                           ITensor* input, ITensor* output,
                                                           int numOutputMaps, int paddingValue,
                                                           Dims2 kernelSize, Dims2 tlPadding, Dims2 brPadding, Dims2 stride, Dims2 dilation,
                                                           Weights kernelWeights, Weights biasWeights, BiasMode biasMode, int numGroups);
    static ElementWiseLayerDiamond newElementWiseLayer(INetwork* network, const std::string& name,
                                                       ITensor* const* inputs, ITensor* output,
                                                       ElementWiseOperation op);

    static InputLayerDiamond newInputLayer(INetwork* network, const std::string& name,
                                           ITensor* const* inputs, ITensor* output);

    template<class D>
    static typename D::DerivedPrivType* derivedPriv(typename D::BasePrivType* base_priv);

protected:
    static BiMap<ILayer*, Layer*> s_priv;
    static BiMap<void*, ILayer*> s_self;

    template<class D>
    static typename D::BaseInterfaceType* deserializeLayer(WisdomContainerEntry*);

    // this would be nice, but you can't declare a templated member variable.  only functions.
    //     template <class PD> static std::map<Layer *, PD> s_priv_diamond;
};

class Network;

class Layer : public virtual ILayer
{
public: // externally facing
    Layer(Network* network);
    virtual Dims4 getOutputDimensions() const = 0;

    virtual void setName(const char* name);
    virtual const char* getName() const;

    virtual LayerType getType() const;

    virtual int getNumInputs() const;
    virtual ITensor* getInput(int i) const;

    virtual int getNumOutputs() const;
    virtual ITensor* getOutput(int i) const;

    virtual void setKernelSize(Dims2 kernelSize);
    virtual Dims2 getKernelSize() const;

    virtual void setNumOutputMaps(int numOutputMaps);
    virtual int getNumOutputMaps() const;

    virtual void setStride(Dims2 stride);
    virtual Dims2 getStride() const;

    virtual Dims2 getDilation() const;
    virtual void setDilation(Dims2);

    virtual int getPaddingValue() const;
    virtual void setPaddingValue(int);

    virtual Dims2 getBottomRightPadding() const;
    virtual void setBottomRightPadding(Dims2);

    virtual Dims2 getTopLeftPadding() const;
    virtual void setTopLeftPadding(Dims2);

    virtual void setNumGroups(int numGroups);
    virtual int getNumGroups() const;

    virtual void setKernelWeights(Weights weights);
    virtual Weights getKernelWeights() const;
    virtual bool hasKernelWeights()
    {
        return false;
    }

    virtual void setBiasWeights(Weights weights);
    virtual Weights getBiasWeights() const;
    virtual bool hasBiasWeights()
    {
        return false;
    }

    virtual NvDlaError supportsPrecision(nvdla::DataType compPrec);
    virtual NvDlaError setComputePrecision(nvdla::DataType compPrec)
    {
        return NvDlaError_NotSupported;
    }

public: // internally facing
    virtual NvU16 getFactoryType() const = 0;
    virtual bool serializeTo(WisdomContainerEntry*) const;
    virtual bool deserializeFrom(WisdomContainerEntry*);

    std::string getInputSymbol(int i) const;
    void setInput(int i, ITensor*);

    std::string getOutputSymbol(int o) const;
    void setOutput(int o, ITensor*);

    virtual bool assignSymbols(Wisdom* wisdom);

protected:
    INetwork* mNetwork;

    Layer(INetwork* n, LayerType type, const std::string& name,
          ITensor* const* inputs, int numInputs,
          ITensor* const* outputs, int numOutputs);

    Layer(INetwork* n, LayerType type, const std::string& name,
          std::vector<std::string>& input_symbols, int numInputs,
          std::vector<std::string>& output_symbols, int numOutputs);

    Layer(INetwork* n, LayerType type, const std::string& name,
          ITensor* input,
          ITensor* output);

    virtual ~Layer();

    const LayerType mType;
    std::string mName;
    std::vector<ITensor*> mInputs, mOutputs;
    std::vector<std::string> mInputSymbols, mOutputSymbols;
};

class ConvolutionLayer : public virtual IConvolutionLayer, public priv::Layer
{
public:
    ConvolutionLayer(INetwork* network, const std::string& name,
                     ITensor* input, ITensor* output,
                     int numOutputMaps, Dims2 kernelSize, Weights kernelWeights, Weights biasWeights, BiasMode biasMode, int numGroups);
    ConvolutionLayer(INetwork* network, const std::string& name,
                     ITensor* input, ITensor* output, int numOutputMaps, int paddingValue,
                     Dims2 kernelSize, Dims2 tlPadding, Dims2 brPadding, Dims2 stride, Dims2 dilation,
                     Weights kernelWeights, Weights biasWeights, BiasMode biasMode, int numGroups);
    virtual ~ConvolutionLayer();

    virtual NvU16 getFactoryType() const;
    virtual bool deserializeFrom(WisdomContainerEntry*);
    virtual bool serializeTo(WisdomContainerEntry*) const;

    virtual Dims2 getKernelSize() const;
    virtual void setKernelSize(Dims2 ksize);

    virtual int getPaddingValue() const;
    virtual void setPaddingValue(int);

    virtual Dims2 getBottomRightPadding() const;
    virtual void setBottomRightPadding(Dims2);

    virtual Dims2 getTopLeftPadding() const;
    virtual void setTopLeftPadding(Dims2);

    virtual Dims2 getStride() const;
    virtual void setStride(Dims2);

    virtual Dims2 getDilation() const;
    virtual void setDilation(Dims2);

    virtual int getNumGroups() const;
    virtual void setNumGroups(int);

    virtual int getNumOutputMaps() const;
    virtual void setNumOutputMaps(int);

    virtual Weights getKernelWeights() const;
    virtual void setKernelWeights(Weights);
    virtual bool hasKernelWeights()
    {
        return true;
    }

    virtual Weights getBiasWeights() const;
    virtual void setBiasWeights(Weights);
    virtual bool hasBiasWeights()
    {
        return true;
    }

    virtual BiasMode getBiasMode() const;
    virtual void setBiasMode(BiasMode);

    virtual Dims4 getOutputDimensions() const;
    const Parameters& getParams() const;

protected:
    friend class LayerFactory;
    ConvolutionLayer();

    Parameters mParams;
};

class FullyConnectedLayer : public virtual IFullyConnectedLayer, public priv::Layer
{
public:
    FullyConnectedLayer(INetwork* network, const std::string& name,
                        ITensor* input, ITensor* output,
                        int numOutputChannels,
                        Weights kernelWeights, Weights biasWeights, BiasMode biasMode);
    virtual ~FullyConnectedLayer();

    virtual NvU16 getFactoryType() const;
    virtual bool deserializeFrom(WisdomContainerEntry*);
    virtual bool serializeTo(WisdomContainerEntry*) const;

    virtual int getNumOutputChannels() const;
    virtual void setNumOutputChannels(int);

    virtual int getNumOutputMaps() const;
    virtual void setNumOutputMaps(int);

    virtual Dims2 getKernelSize() const;
    virtual void setKernelSize(Dims2 ksize);

    virtual Weights getKernelWeights() const;
    virtual void setKernelWeights(Weights);
    virtual bool hasKernelWeights()
    {
        return true;
    }

    virtual Weights getBiasWeights() const;
    virtual void setBiasWeights(Weights);
    virtual bool hasBiasWeights()
    {
        return true;
    }

    virtual BiasMode getBiasMode() const;
    virtual void setBiasMode(BiasMode);

    virtual Dims4 getOutputDimensions() const;

    const Parameters& getParams() const;

protected:
    friend class LayerFactory;
    FullyConnectedLayer();

    Parameters mParams;
};

class ActivationLayer : public virtual IActivationLayer, public priv::Layer
{
public:
    ActivationLayer(INetwork* network, const std::string& name,
                    ITensor* input, ITensor* output,
                    ActivationType activationType);
    virtual ~ActivationLayer();

    virtual NvU16 getFactoryType() const;

    virtual bool deserializeFrom(WisdomContainerEntry*);
    virtual bool serializeTo(WisdomContainerEntry*) const;

    virtual ActivationType getActivationType() const;
    virtual void setActivationType(ActivationType);

    virtual Dims4 getOutputDimensions() const;
    const Parameters& getParams() const;

protected:
    friend class LayerFactory;
    ActivationLayer();

    Parameters mParams;
};

class PoolingLayer : public virtual IPoolingLayer, public priv::Layer
{
public:
    PoolingLayer(INetwork* network, const std::string& name,
                 ITensor* input, ITensor* output,
                 PoolingType type, Dims2 windowSize);
    PoolingLayer(INetwork* network, const std::string& name,
                 ITensor* input, ITensor* output,
                 PoolingType type,
                 Dims2 windowSize, Dims2 stride,
                 Dims2 tlPadding, Dims2 brPadding);
    virtual ~PoolingLayer();

    virtual NvU16 getFactoryType() const;
    virtual bool deserializeFrom(WisdomContainerEntry*);
    virtual bool serializeTo(WisdomContainerEntry*) const;

    virtual PoolingType getPoolingType() const;
    virtual void setPoolingType(PoolingType);

    virtual Dims2 getWindowSize() const;
    virtual void setWindowSize(Dims2);

    virtual Dims2 getBottomRightPadding() const;
    virtual void setBottomRightPadding(Dims2);

    virtual Dims2 getTopLeftPadding() const;
    virtual void setTopLeftPadding(Dims2);

    virtual Dims2 getStride() const;
    virtual void setStride(Dims2);

    virtual Dims4 getOutputDimensions() const;
    const Parameters& getParams() const;

protected:
    friend class LayerFactory;
    PoolingLayer();

    Parameters mParams;
};

class LRNLayer : public virtual ILRNLayer, public priv::Layer
{
public:
    LRNLayer(INetwork* network, const std::string& name,
             ITensor* input,
             ITensor* output,
             int windowSize, float alpha, float beta, float k);
    virtual ~LRNLayer();

    virtual NvU16 getFactoryType() const;
    virtual bool deserializeFrom(WisdomContainerEntry*);
    virtual bool serializeTo(WisdomContainerEntry*) const;

    virtual float getAlpha() const;
    virtual void setAlpha(float);

    virtual float getBeta() const;
    virtual void setBeta(float);

    virtual float getK() const;
    virtual void setK(float);

    virtual int getWindowSize() const;
    virtual void setWindowSize(int);

    //	ACCESSOR_DECL(DataType, arithmeticType, ArithmeticType)

    virtual Dims4 getOutputDimensions() const;
    const Parameters& getParams() const;

    virtual NvDlaError supportsPrecision(nvdla::DataType compPrec)
    {
        return NvDlaError_NotSupported;
    }

protected:
    friend class LayerFactory;
    LRNLayer();

    Parameters mParams;
};

class ScaleLayer : public virtual IScaleLayer, public priv::Layer
{
public:
    ScaleLayer(INetwork* network, const std::string& name,
               ITensor* input, ITensor* output,
               ScaleMode mode,
               Weights shift, Weights scale, Weights power);
    virtual ~ScaleLayer();

    virtual NvU16 getFactoryType() const;
    virtual bool deserializeFrom(WisdomContainerEntry*);
    virtual bool serializeTo(WisdomContainerEntry*) const;

    virtual Weights getScale() const;
    virtual void setScale(Weights);

    virtual Weights getShift() const;
    virtual void setShift(Weights);

    virtual Weights getPower() const;
    virtual void setPower(Weights);

    virtual ScaleMode getMode() const;
    virtual void setMode(ScaleMode);

    virtual Dims4 getOutputDimensions() const;
    const Parameters& getParams() const;

    virtual NvDlaError supportsPrecision(nvdla::DataType compPrec)
    {
        return NvDlaError_NotSupported;
    }

protected:
    friend class LayerFactory;
    ScaleLayer();

    Parameters mParams;
};

class BatchNormLayer : public virtual IBatchNormLayer, public priv::Layer
{
public:
    BatchNormLayer(INetwork* network, const std::string& name,
                   ITensor* input, ITensor* output,
                   BatchNormMode mode,
                   Weights mean, Weights variance, float epsilon);
    virtual ~BatchNormLayer();

    virtual NvU16 getFactoryType() const;
    virtual bool deserializeFrom(WisdomContainerEntry*);
    virtual bool serializeTo(WisdomContainerEntry*) const;

    virtual float getEpsilon() const;
    virtual void setEpsilon(float);

    virtual Weights getMean() const;
    virtual void setMean(Weights);

    virtual Weights getVariance() const;
    virtual void setVariance(Weights);

    virtual BatchNormMode getMode() const;
    virtual void setMode(BatchNormMode);

    virtual Dims4 getOutputDimensions() const;
    const Parameters& getParams() const;

    virtual NvDlaError supportsPrecision(nvdla::DataType compPrec)
    {
        return NvDlaError_NotSupported;
    }

protected:
    friend class LayerFactory;
    BatchNormLayer();

    Parameters mParams;
};

class SoftMaxLayer : public virtual ISoftMaxLayer, public priv::Layer
{
public:
    SoftMaxLayer(INetwork* network, const std::string& name,
                 ITensor* input, ITensor* output);
    virtual ~SoftMaxLayer();

    virtual NvU16 getFactoryType() const;
    virtual bool deserializeFrom(WisdomContainerEntry*);
    virtual bool serializeTo(WisdomContainerEntry*) const;

    virtual Dims4 getOutputDimensions() const;
    const Parameters& getParams() const;

    virtual NvDlaError supportsPrecision(nvdla::DataType compPrec)
    {
        return NvDlaError_NotSupported;
    }

protected:
    friend class LayerFactory;
    SoftMaxLayer();

    Parameters mParams;
};

class ConcatenationLayer : public virtual IConcatenationLayer, public priv::Layer
{
public:
    ConcatenationLayer(INetwork* network, const std::string& name,
                       ITensor* const* inputs, int numInputs,
                       ITensor* output);
    virtual ~ConcatenationLayer();

    virtual NvU16 getFactoryType() const;
    virtual bool deserializeFrom(WisdomContainerEntry*);
    virtual bool serializeTo(WisdomContainerEntry*) const;

    virtual Dims4 getOutputDimensions() const;
    const Parameters& getParams() const;

    virtual NvDlaError supportsPrecision(nvdla::DataType compPrec)
    {
        return NvDlaError_NotSupported;
    }

protected:
    friend class LayerFactory;
    ConcatenationLayer();

    Weights mBiasWeights;
    Weights mFilterWeights;
    Parameters mParams;
};

class SliceLayer : public virtual ISliceLayer, public priv::Layer
{
public:
    SliceLayer(INetwork* network, const std::string& name,
               ITensor* input,
               ITensor* const* outputs,
               int numOutputs);
    virtual ~SliceLayer();

    virtual NvU16 getFactoryType() const;
    virtual bool deserializeFrom(WisdomContainerEntry*);
    virtual bool serializeTo(WisdomContainerEntry*) const;

    virtual Dims4 getOutputDimensions() const;
    const Parameters& getParams() const;

    virtual NvDlaError supportsPrecision(nvdla::DataType compPrec)
    {
        return NvDlaError_NotSupported;
    }

protected:
    friend class LayerFactory;
    SliceLayer();

    Parameters mParams;
};

class DeconvolutionLayer : public virtual IDeconvolutionLayer, public priv::Layer
{
public:
    DeconvolutionLayer(INetwork* network, const std::string& name,
                       ITensor* input,
                       ITensor* output, int numOutputMaps,
                       Dims2 kernelSize, Weights kernelWeights, Weights biasWeights);
    DeconvolutionLayer(INetwork* network, const std::string& name,
                       ITensor* input, ITensor* output, int numOutputMaps, int paddingValue,
                       Dims2 kernelSize, Dims2 tlPadding, Dims2 brPadding, Dims2 stride, Dims2 dilation,
                       Weights kernelWeights, Weights biasWeights, BiasMode biasMode, int numGroups);
    virtual ~DeconvolutionLayer();

    virtual NvU16 getFactoryType() const;
    virtual bool deserializeFrom(WisdomContainerEntry*);
    virtual bool serializeTo(WisdomContainerEntry*) const;

    virtual Dims2 getKernelSize() const;
    virtual void setKernelSize(Dims2 ksize);

    virtual int getPaddingValue() const;
    virtual void setPaddingValue(int);

    virtual Dims2 getBottomRightPadding() const;
    virtual void setBottomRightPadding(Dims2);

    virtual Dims2 getTopLeftPadding() const;
    virtual void setTopLeftPadding(Dims2);

    virtual Dims2 getStride() const;
    virtual void setStride(Dims2);

    virtual Dims2 getDilation() const;
    virtual void setDilation(Dims2);

    virtual int getNumGroups() const;
    virtual void setNumGroups(int numGroups);

    virtual int getNumOutputMaps() const;
    virtual void setNumOutputMaps(int);

    virtual Weights getKernelWeights() const;
    virtual void setKernelWeights(Weights);
    virtual bool hasKernelWeights()
    {
        return true;
    }

    virtual Weights getBiasWeights() const;
    virtual void setBiasWeights(Weights);
    virtual bool hasBiasWeights()
    {
        return true;
    }

    virtual BiasMode getBiasMode() const;
    virtual void setBiasMode(BiasMode);

    virtual Dims4 getOutputDimensions() const;
    const Parameters& getParams() const;

    virtual NvDlaError supportsPrecision(nvdla::DataType compPrec)
    {
        return NvDlaError_NotSupported;
    }

protected:
    friend class LayerFactory;
    DeconvolutionLayer();

    Parameters mParams;
};

class ElementWiseLayer : public virtual IElementWiseLayer, public priv::Layer
{
public:
    ElementWiseLayer(INetwork* network, const std::string& name,
                     ITensor* const* inputs, ITensor* output,
                     ElementWiseOperation op);
    virtual ~ElementWiseLayer();

    virtual NvU16 getFactoryType() const;
    virtual bool deserializeFrom(WisdomContainerEntry*);
    virtual bool serializeTo(WisdomContainerEntry*) const;

    virtual ElementWiseOperation getOperation() const;
    virtual void setOperation(ElementWiseOperation);

    virtual Dims4 getOutputDimensions() const;
    const Parameters& getParams() const;

    virtual NvDlaError supportsPrecision(nvdla::DataType compPrec)
    {
        return NvDlaError_NotSupported;
    }

protected:
    friend class LayerFactory;
    ElementWiseLayer();

    Parameters mParams;
};

class InputLayer : public virtual IInputLayer, public priv::Layer
{
public:
    InputLayer(INetwork* network, const std::string& name,
               ITensor* const* inputs, ITensor* output);
    virtual ~InputLayer();

    virtual NvU16 getFactoryType() const;
    virtual bool deserializeFrom(WisdomContainerEntry*);
    virtual bool serializeTo(WisdomContainerEntry*) const;

    //    virtual InputOperation getOperation() const;
    //    virtual void setOperation(InputOperation);

    virtual Dims4 getOutputDimensions() const;
    const Parameters& getParams() const;

protected:
    friend class LayerFactory;
    InputLayer();

    Parameters mParams;
};

} // namespace priv
} // namespace nvdla

#endif // NVDLA_PRIV_LAYER_H
