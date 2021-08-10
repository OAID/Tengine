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

#ifndef NVDLA_PRIV_CANONICAL_AST_H
#define NVDLA_PRIV_CANONICAL_AST_H

#include <algorithm>
#include <map>
#include <vector>
#include <string>

#include "AST.h"
#include "CanonicalEnums.h"
#include "Layer.h"
#include "Type.h" // for misc

//
// provide explicit hash fn specialization for std::pair<canonical_ast::Node *, canonical_ast::Edge *> (Elem)
//
namespace nvdla {
namespace priv {
namespace canonical_ast {
class Node;
class Edge;
} // namespace canonical_ast
} // namespace priv
} // namespace nvdla
namespace std {
template<>
struct hash<std::pair<nvdla::priv::canonical_ast::Node*, nvdla::priv::canonical_ast::Edge*> >
{
    std::size_t operator()(const std::pair<nvdla::priv::canonical_ast::Node*, nvdla::priv::canonical_ast::Edge*>& v) const noexcept
    {
        std::size_t h;
        // one side or the other will be non-zero.
        if (v.first)
        {
            h = std::hash<nvdla::priv::canonical_ast::Node*>{}(v.first);
        }
        else
        {
            h = std::hash<nvdla::priv::canonical_ast::Edge*>{}(v.second);
        }
        return h;
    }
};
} // namespace std

namespace nvdla {

class ITensor;

namespace priv {

namespace canonical_ast {

//
// The canonical parameter space which inherits parameters from the parsed network
// so that the UMD compiler has a unanimous interpretation of any network
// irrespective of which frameworks they come from.
//
// since compiler works with graphs/asts, this interface is called the
// canonical-ast-interface.
//

class CanonicalParams
{
public:
    CanonicalParams()
    {
    }
    virtual ~CanonicalParams()
    {
    }

    virtual bool hasBiasTerm() const;
    virtual void setHasBiasTerm(bool);
};

class ConvolutionParams : public CanonicalParams
{
public:
    ConvolutionParams()
        : m_bias_mode(BiasMode::bNONE),
          m_has_bias_term(false),
          m_padding_value(0),
          m_num_groups(1)
    {
    }
    virtual ~ConvolutionParams()
    {
    }

    nvdla::BiasMode biasMode() const
    {
        return m_bias_mode;
    }
    void setBiasMode(nvdla::BiasMode mode)
    {
        m_bias_mode = mode;
    }

    Dims2 dilation() const
    {
        return m_dilation;
    }
    void setDilation(Dims2 dilation)
    {
        m_dilation = dilation;
    }

    Dims2 stride() const
    {
        return m_stride;
    }
    void setStride(Dims2 stride)
    {
        m_stride = stride;
    }

    Dims2 topLeftPadding() const
    {
        return m_TL_padding;
    }
    void setTopLeftPadding(Dims2 tlPadding)
    {
        m_TL_padding = tlPadding;
    }

    Dims2 bottomRightPadding() const
    {
        return m_BR_padding;
    }
    void setBottomRightPadding(Dims2 brPadding)
    {
        m_BR_padding = brPadding;
    }

    virtual bool hasBiasTerm() const
    {
        return m_has_bias_term;
    }
    virtual void setHasBiasTerm(bool hasBiasTerm)
    {
        m_has_bias_term = hasBiasTerm;
    }

    NvU32 paddingValue() const
    {
        return m_padding_value;
    }
    void setPaddingValue(NvU32 paddingValue)
    {
        m_padding_value = paddingValue;
    }

    const Dims4& biasDims() const
    {
        return m_bias_dims;
    }
    void setBiasDims(const Dims4& bd)
    {
        m_bias_dims = bd;
    }

    const Dims4& weightDims() const
    {
        return m_weight_dims;
    }
    void setWeightDims(const Dims4& kd)
    {
        m_weight_dims = kd;
    }

    const Weights& biasData() const
    {
        return m_bias_data;
    }
    void setBiasData(const Weights& biasData)
    {
        m_bias_data = biasData;
    }

    const Weights& weights() const
    {
        return m_weights;
    }
    void setWeights(const Weights& weights)
    {
        m_weights = weights;
    }

    NvU32 numGroups() const
    {
        return m_num_groups;
    }
    void setNumGroups(NvU32 numGroups)
    {
        m_num_groups = numGroups;
    }

protected:
    BiasMode m_bias_mode;
    bool m_has_bias_term;
    Dims2 m_TL_padding;
    Dims2 m_BR_padding;
    Dims2 m_stride;
    Dims2 m_dilation;
    Dims4 m_bias_dims;
    Dims4 m_weight_dims;
    NvU32 m_padding_value;
    Weights m_weights;
    Weights m_bias_data;
    NvU32 m_num_groups;
};

// innerproduct/fullyconnected
class FullyConnectedParams : public CanonicalParams
{
public:
    FullyConnectedParams()
        : m_bias_mode(BiasMode::bNONE),
          m_has_bias_term(false)
    {
    }
    virtual ~FullyConnectedParams()
    {
    }

    nvdla::BiasMode biasMode() const
    {
        return m_bias_mode;
    }
    void setBiasMode(nvdla::BiasMode mode)
    {
        m_bias_mode = mode;
    }

    virtual bool hasBiasTerm() const
    {
        return m_has_bias_term;
    }
    virtual void setHasBiasTerm(bool hasBiasTerm)
    {
        m_has_bias_term = hasBiasTerm;
    }

    const Dims4& weightDims() const
    {
        return m_weight_dims;
    }
    void setWeightDims(const Dims4& weightDims)
    {
        m_weight_dims = weightDims;
    }

    const Weights& weights() const
    {
        return m_weights;
    }
    void setWeights(const Weights& weights)
    {
        m_weights = weights;
    }

    const Dims4& biasDims() const
    {
        return m_bias_dims;
    }
    void setBiasDims(const Dims4& biasDims)
    {
        m_bias_dims = biasDims;
    }

    const Weights& biasData() const
    {
        return m_bias_data;
    }
    void setBiasData(const Weights& bias)
    {
        m_bias_data = bias;
    }

protected:
    BiasMode m_bias_mode;
    bool m_has_bias_term;
    Dims4 m_weight_dims;
    Dims4 m_bias_dims;
    Weights m_weights;
    Weights m_bias_data;
};

// relu
// sigmoid
class ActivationParams : public CanonicalParams
{
public:
    ActivationParams()
        : m_activation_type(ActivationType::kRELU)
    {
    }
    virtual ~ActivationParams()
    {
    }

    nvdla::ActivationType activationType() const
    {
        return m_activation_type;
    }
    void setActivationType(nvdla::ActivationType activationType)
    {
        m_activation_type = activationType;
    }

protected:
    nvdla::ActivationType m_activation_type;
};

class PoolingParams : public CanonicalParams
{
public:
    PoolingParams()
        : m_pool_type(PoolingType::kMAX)
    {
    }
    virtual ~PoolingParams()
    {
    }

    nvdla::PoolingType poolType() const
    {
        return m_pool_type;
    }
    void setPoolType(nvdla::PoolingType poolType)
    {
        m_pool_type = poolType;
    }

    Dims2 kernelDims() const
    {
        return m_kernel_dims;
    }
    void setKernelDims(Dims2 kernelDims)
    {
        m_kernel_dims = kernelDims;
    }

    Dims2 stride() const
    {
        return m_stride;
    }
    void setStride(Dims2 stride)
    {
        m_stride = stride;
    }

    Dims2 topLeftPadding() const
    {
        return m_TL_padding;
    }
    void setTopLeftPadding(Dims2 tlPadding)
    {
        m_TL_padding = tlPadding;
    }

    Dims2 bottomRightPadding() const
    {
        return m_BR_padding;
    }
    void setBottomRightPadding(Dims2 brPadding)
    {
        m_BR_padding = brPadding;
    }

protected:
    nvdla::PoolingType m_pool_type;
    Dims2 m_TL_padding;
    Dims2 m_BR_padding;
    Dims2 m_kernel_dims;
    Dims2 m_stride;
};

class LRNParams : public CanonicalParams
{
public:
    LRNParams()
        : m_local_size(5),
          m_alpha(1.0f),
          m_beta(0.75f),
          m_k(1.0f)
    {
    }
    virtual ~LRNParams()
    {
    }

    NvF32 alpha() const
    {
        return m_alpha;
    }
    void setAlpha(NvF32 alpha)
    {
        m_alpha = alpha;
    }

    NvF32 beta() const
    {
        return m_beta;
    }
    void setBeta(NvF32 beta)
    {
        m_beta = beta;
    }

    NvF32 k() const
    {
        return m_k;
    }
    void setK(NvF32 k)
    {
        m_k = k;
    }

    NvU32 localSize() const
    {
        return m_local_size;
    }
    void setLocalSize(NvU32 localSize = 5)
    {
        m_local_size = localSize;
    }

protected:
    // these params are extracted from ditcaffe::LRNParameter
    NvU32 m_local_size; //or window_size [def = 5]
    NvF32 m_alpha;      // [def  = 1.0f]
    NvF32 m_beta;       // [def = 0.75f]
    NvF32 m_k;          // [def = 1.0f]
    //TODO: NvU32 m_norm_region = 0;
    //TODO: NvU32 m_engine = 0;
};

//scale layer
//power layer
class ScaleParams : public CanonicalParams
{
public:
    ScaleParams()
        : m_mode(ScaleMode::sUNIFORM),
          m_has_bias_term(false),
          m_power(),
          m_scale(),
          m_shift()
    {
    }
    virtual ~ScaleParams()
    {
    }

    nvdla::ScaleMode mode() const
    {
        return m_mode;
    }
    void setMode(nvdla::ScaleMode mode)
    {
        m_mode = mode;
    }

    const Weights power() const
    {
        return m_power;
    }
    void setPower(const Weights power)
    {
        m_power = power;
    }

    const Weights scale() const
    {
        return m_scale;
    }
    void setScale(const Weights scale)
    {
        m_scale = scale;
    }

    const Weights shift() const
    {
        return m_shift;
    }
    void setShift(const Weights shift)
    {
        m_shift = shift;
    }

    const Dims4& powerDims() const
    {
        return m_power_dims;
    }
    void setPowerDims(const Dims4& powerDims)
    {
        m_power_dims = powerDims;
    }

    const Dims4& scaleDims() const
    {
        return m_scale_dims;
    }
    void setScaleDims(const Dims4& scaleDims)
    {
        m_scale_dims = scaleDims;
    }

    const Dims4& shiftDims() const
    {
        return m_shift_dims;
    }
    void setShiftDims(const Dims4& shiftDims)
    {
        m_shift_dims = shiftDims;
    }

    virtual bool hasBiasTerm() const
    {
        return m_has_bias_term;
    }
    virtual void setHasBiasTerm(bool hasBiasTerm)
    {
        m_has_bias_term = hasBiasTerm;
    }

protected:
    nvdla::ScaleMode m_mode; // [def = nvdla::kUNIFORM]
    bool m_has_bias_term;

    // these params are extracted from ditcaffe::PowerParameter
    Weights m_power; // [def = 1.0f]
    Weights m_scale; // [def = 1.0f]
    Weights m_shift; // [def = 0.0f]

    // extra params
    Dims4 m_scale_dims;
    Dims4 m_shift_dims;
    Dims4 m_power_dims;
};

//batch norm layer
class BatchNormParams : public CanonicalParams
{
public:
    BatchNormParams()
        : m_mode(BatchNormMode::bnUNIFORM),
          m_epsilon(0.0f),
          m_mean(),
          m_variance()
    {
    }
    virtual ~BatchNormParams()
    {
    }

    nvdla::BatchNormMode mode() const
    {
        return m_mode;
    }
    void setMode(nvdla::BatchNormMode mode)
    {
        m_mode = mode;
    }

    float epsilon() const
    {
        return m_epsilon;
    }
    void setEpsilon(const float e)
    {
        m_epsilon = e;
    }

    const Weights mean() const
    {
        return m_mean;
    }
    void setMean(const Weights mean)
    {
        m_mean = mean;
    }

    const Weights variance() const
    {
        return m_variance;
    }
    void setVariance(const Weights variance)
    {
        m_variance = variance;
    }

    const Dims4& meanDims() const
    {
        return m_mean_dims;
    }
    void setMeanDims(const Dims4& md)
    {
        m_mean_dims = md;
    }

    const Dims4& varianceDims() const
    {
        return m_variance_dims;
    }
    void setVarianceDims(const Dims4& vd)
    {
        m_variance_dims = vd;
    }

protected:
    nvdla::BatchNormMode m_mode; // [def = nvdla::bUNIFORM]
    Dims4 m_mean_dims;
    Dims4 m_variance_dims;
    float m_epsilon;
    Weights m_mean;
    Weights m_variance;
};

// softmax
// softmaxwithloss
class SoftMaxParams : public CanonicalParams
{
public:
    SoftMaxParams()
        : m_engine(0),
          m_axis(1)
    {
    }
    virtual ~SoftMaxParams()
    {
    }

protected:
    // these params are extracted from ditcaffe::SoftMaxParameter
    NvU32 m_engine;
    NvS32 m_axis;
};

class ConcatenationParams : public CanonicalParams
{
public:
    ConcatenationParams()
        : m_axis(1),
          m_concat_dim(1),
          m_num_inputs(0)
    {
    }
    virtual ~ConcatenationParams()
    {
    }

    NvU32 numInputs() const
    {
        return m_num_inputs;
    }
    void setNumInputs(NvU8 numInputs)
    {
        m_num_inputs = numInputs;
    }

public:
    // these params are extracted from ditcaffe::ConcatParameter
    NvS32 m_axis;
    NvU32 m_concat_dim; // deprecated
    NvU32 m_num_inputs;
};

class SplitParams : public CanonicalParams
{
public:
    SplitParams()
        : m_axis(1),
          m_num_outputs(0)
    {
    }
    virtual ~SplitParams()
    {
    }

    NvU32 numOutputs() const
    {
        return m_num_outputs;
    }
    void setNumOutputs(NvU8 numOutputs)
    {
        m_num_outputs = numOutputs;
    }

protected:
    // these params are extracted from ditcaffe::SliceParameter
    NvS32 m_axis;
    NvU32 m_num_outputs;
};

class DeconvolutionParams : public CanonicalParams
{
public:
    DeconvolutionParams()
        : m_bias_mode(BiasMode::bNONE),
          m_has_bias_term(false),
          m_padding_value(0),
          m_num_groups(1)
    {
    }
    virtual ~DeconvolutionParams()
    {
    }

    nvdla::BiasMode biasMode() const
    {
        return m_bias_mode;
    }
    void setBiasMode(nvdla::BiasMode mode)
    {
        m_bias_mode = mode;
    }

    Dims2 dilation() const
    {
        return m_dilation;
    }
    void setDilation(Dims2 dilation)
    {
        m_dilation = dilation;
    }

    Dims2 stride() const
    {
        return m_stride;
    }
    void setStride(Dims2 stride)
    {
        m_stride = stride;
    }

    Dims2 topLeftPadding() const
    {
        return m_TL_padding;
    }
    void setTopLeftPadding(Dims2 tlPadding)
    {
        m_TL_padding = tlPadding;
    }

    Dims2 bottomRightPadding() const
    {
        return m_BR_padding;
    }
    void setBottomRightPadding(Dims2 brPadding)
    {
        m_BR_padding = brPadding;
    }

    virtual bool hasBiasTerm() const
    {
        return m_has_bias_term;
    }
    virtual void setHasBiasTerm(bool hasBiasTerm)
    {
        m_has_bias_term = hasBiasTerm;
    }

    NvU32 paddingValue() const
    {
        return m_padding_value;
    }
    void setPaddingValue(NvU32 paddingValue)
    {
        m_padding_value = paddingValue;
    }

    const Dims4& biasDims() const
    {
        return m_bias_dims;
    }
    void setBiasDims(const Dims4& bd)
    {
        m_bias_dims = bd;
    }

    const Dims4& weightDims() const
    {
        return m_weight_dims;
    }
    void setWeightDims(const Dims4& kd)
    {
        m_weight_dims = kd;
    }

    const Weights& biasData() const
    {
        return m_bias_data;
    }
    void setBiasData(const Weights& biasData)
    {
        m_bias_data = biasData;
    }

    const Weights& weights() const
    {
        return m_weights;
    }
    void setWeights(const Weights& weights)
    {
        m_weights = weights;
    }

    NvU32 numGroups() const
    {
        return m_num_groups;
    }
    void setNumGroups(NvU32 numGroups)
    {
        m_num_groups = numGroups;
    }

protected:
    BiasMode m_bias_mode;
    bool m_has_bias_term;
    Dims2 m_TL_padding;
    Dims2 m_BR_padding;
    Dims2 m_stride;
    Dims2 m_dilation;
    Dims4 m_bias_dims;
    Dims4 m_weight_dims;
    NvU32 m_padding_value;
    Weights m_weights;
    Weights m_bias_data;
    NvU32 m_num_groups;
};

class ElementWiseParams : public CanonicalParams
{
public:
    ElementWiseParams()
        : m_type(ElementWiseOperation::kSUM),
          m_has_stable_prod_grad(true)
    {
    }
    virtual ~ElementWiseParams()
    {
    }

    nvdla::ElementWiseOperation type() const
    {
        return m_type;
    }
    void setType(nvdla::ElementWiseOperation type)
    {
        m_type = type;
    }

protected:
    // these params are extracted from ditcaffe::EltWiseParameter
    nvdla::ElementWiseOperation m_type; // [def = nvdla::kSUM]
    bool m_has_stable_prod_grad;
};

}; // namespace canonical_ast

}; // namespace priv

}; // namespace nvdla

namespace nvdla {

class INetwork;
class ITensor;

namespace priv {

class Network;
class Tensor;
class Wisdom;
class WisdomContainerEntry;

namespace canonical_ast {

enum CanonicalOpTypeEnum
{
    CANONICAL_OPERATION_TYPE_ENUMS(GEN_ENUM)
};
typedef SequenceEnum<CanonicalOpTypeEnum, NvU16> CanonicalOpType;

class Node;
class Edge;

class Graph : public ast::Graph<Node, Edge>
{
public:
    Graph()
        : m_next_node_id(0), m_next_edge_id(0)
    {
        m_scored_ordering = new ast::ScoredGraphOrdering<Graph>(this);
    }
    ast::ScoredGraphOrdering<Graph>* scoredOrdering()
    {
        return m_scored_ordering;
    }
    virtual ~Graph()
    {
        delete m_scored_ordering;
    }

    Graph* clone();

    std::string nextNodeId()
    {
        return std::string("n-") + toString(m_next_node_id++);
    }
    std::string nextEdgeId()
    {
        return std::string("e-") + toString(m_next_edge_id++);
    }

protected:
    int m_next_node_id;
    int m_next_edge_id;

    ast::ScoredGraphOrdering<Graph>* m_scored_ordering;
};

class Node
{
public:
    virtual std::string className() const
    {
        return std::string("Node");
    }
    virtual std::string pretty() const
    {
        return "<pretty-node>";
    } // override in specific nodes to add. careful...

    static std::string prettyId(const Node* n, NvU8 flags = ast::PrettyId_Default)
    {
        if (!n) { return std::string("(null)"); }
        std::stringstream r;
        std::string sep("");
        if (flags & ast::PrettyId_Id)
        {
            r << sep << n->id();
            sep = " ";
        }
        if (flags & ast::PrettyId_ClassName)
        {
            r << sep << n->className();
            sep = " ";
        }
        if (flags & ast::PrettyId_Name)
        {
            r << sep << n->id();
            sep = " ";
        } // same as id
        if (flags & (ast::PrettyId_Verbose))
        {
            r << sep << n->pretty();
            sep = " ";
        }
        return r.str();
    }

    typedef canonical_ast::Graph::EdgeSequence EdgeSequence;

    Node()
        : m_containing_graph(0)
    {
        m_unique_id = m_next_id++;
    }
    virtual ~Node()
    {
    }

    const std::string id() const
    {
        return m_id;
    }
    void setId(const std::string id)
    {
        m_id = id;
    }

    NvU32 uniqueId() const
    {
        return m_unique_id;
    }

    const std::string name() const
    {
        return m_name;
    }
    void setName(const std::string name)
    {
        m_name = name;
    }

    Graph* graph()
    {
        return m_containing_graph;
    }
    void setGraph(Graph* g)
    {
        m_containing_graph = g;
    }

    CanonicalOpType canonicalOpType() const
    {
        return m_can_op_type;
    }

    virtual CanonicalParams& params()
    {
        return m_basic_can_params;
    }

    const EdgeSequence& inputEdges() const
    {
        return m_input_edges;
    }
    void markInputEdge(Edge* input)
    {
        if (std::find(m_input_edges.begin(), m_input_edges.end(), input) == m_input_edges.end())
            m_input_edges.push_back(input);
    }

    const EdgeSequence& outputEdges() const
    {
        return m_output_edges;
    }
    void markOutputEdge(Edge* output)
    {
        if (std::find(m_output_edges.begin(), m_output_edges.end(), output) == m_output_edges.end())
            m_output_edges.push_back(output);
    }

protected:
    std::string m_id;  // unique within the graph
    NvU32 m_unique_id; // id for graph ordering. u32 instead of string.
    static NvU32 m_next_id;
    std::string m_name;
    Graph* m_containing_graph;
    CanonicalOpType m_can_op_type;
    CanonicalParams m_basic_can_params;
    EdgeSequence m_input_edges;
    EdgeSequence m_output_edges;
};

// scale (when power non-trivial)
// power
class ScaleNode : public Node
{
public:
    virtual std::string className() const
    {
        return std::string("ScaleNode");
    }
    ScaleNode()
        : Node()
    {
        m_can_op_type = SCALE;
    }
    virtual ~ScaleNode()
    {
    }

    virtual void captureNetworkParams(ScaleLayer* scale);
    virtual ScaleParams& params()
    {
        return m_can_params;
    }

protected:
    ScaleParams m_can_params;
};

// batch-norm
class BatchNormNode : public Node
{
public:
    virtual std::string className() const
    {
        return std::string("BatchNormNode");
    }
    BatchNormNode()
        : Node()
    {
        m_can_op_type = BATCH_NORM;
    }
    virtual ~BatchNormNode()
    {
    }

    virtual void captureNetworkParams(BatchNormLayer* bn);
    virtual BatchNormParams& params()
    {
        return m_can_params;
    }

protected:
    BatchNormParams m_can_params;
};

// softmax
class SoftMaxNode : public Node
{
public:
    virtual std::string className() const
    {
        return std::string("SoftMaxNode");
    }
    SoftMaxNode()
        : Node()
    {
        m_can_op_type = SOFTMAX;
    }
    virtual ~SoftMaxNode()
    {
    }

    virtual void captureNetworkParams(SoftMaxLayer* sm);
    virtual SoftMaxParams& params()
    {
        return m_can_params;
    }

protected:
    SoftMaxParams m_can_params;
};

// convolution layer
// deconvolve 1st
class ConvolutionNode : public Node
{
public:
    virtual std::string className() const
    {
        return std::string("ConvolutionNode");
    }
    ConvolutionNode()
        : Node()
    {
        m_can_op_type = CONVOLUTION;
    }
    virtual ~ConvolutionNode()
    {
    }

    virtual void captureNetworkParams(ConvolutionLayer* conv);
    virtual ConvolutionParams& params()
    {
        return m_can_params;
    }

protected:
    ConvolutionParams m_can_params;
};

// fully connected
class FullyConnectedNode : public Node
{
public:
    virtual std::string className() const
    {
        return std::string("FullyConnectedNode");
    }
    FullyConnectedNode()
        : Node()
    {
        m_can_op_type = FULLY_CONNECTED;
    }
    virtual ~FullyConnectedNode()
    {
    }

    virtual void captureNetworkParams(FullyConnectedLayer* fc);
    virtual FullyConnectedParams& params()
    {
        return m_can_params;
    }

protected:
    FullyConnectedParams m_can_params;
};

// deconvolve 2nd
class DeconvolutionNode : public Node
{
public:
    virtual std::string className() const
    {
        return std::string("DeconvolutionNode");
    }
    DeconvolutionNode()
        : Node()
    {
        m_can_op_type = DECONVOLUTION;
    }
    virtual ~DeconvolutionNode()
    {
    }

    virtual void captureNetworkParams(DeconvolutionLayer* deconv);
    virtual DeconvolutionParams& params()
    {
        return m_can_params;
    }

protected:
    DeconvolutionParams m_can_params;
};

// activation
class ActivationNode : public Node
{
public:
    virtual std::string className() const
    {
        return std::string("ActivationNode");
    }
    ActivationNode()
        : Node()
    {
        m_can_op_type = ACTIVATION;
    }
    virtual ~ActivationNode()
    {
    }

    virtual void captureNetworkParams(ActivationLayer* act);
    virtual ActivationParams& params()
    {
        return m_can_params;
    }

protected:
    ActivationParams m_can_params;
};

// elementwise
class ElementWiseNode : public Node
{
public:
    virtual std::string className() const
    {
        return std::string("ElementWiseNode");
    }
    ElementWiseNode()
        : Node()
    {
        m_can_op_type = ELEMENTWISE;
    }
    virtual ~ElementWiseNode()
    {
    }

    virtual void captureNetworkParams(ElementWiseLayer* ew);
    virtual ElementWiseParams& params()
    {
        return m_can_params;
    }

protected:
    ElementWiseParams m_can_params;
};

// pooling
class PoolingNode : public Node
{
public:
    virtual std::string className() const
    {
        return std::string("PoolingNode");
    }
    PoolingNode()
        : Node()
    {
        m_can_op_type = POOLING;
    }
    virtual ~PoolingNode()
    {
    }

    virtual void captureNetworkParams(PoolingLayer* pool);
    virtual PoolingParams& params()
    {
        return m_can_params;
    }

protected:
    PoolingParams m_can_params;
};

// lrn
class LRNNode : public Node
{
public:
    virtual std::string className() const
    {
        return std::string("LRNNode");
    }
    LRNNode()
        : Node()
    {
        m_can_op_type = LRN;
    }
    virtual ~LRNNode()
    {
    }

    virtual void captureNetworkParams(LRNLayer* lrn);
    virtual LRNParams& params()
    {
        return m_can_params;
    }

protected:
    LRNParams m_can_params;
};

// concat
class ConcatenationNode : public Node
{
public:
    virtual std::string className() const
    {
        return std::string("ConcatenationNode");
    }
    ConcatenationNode()
        : Node()
    {
        m_can_op_type = CONCATENATION;
    }
    virtual ~ConcatenationNode()
    {
    }

    virtual void captureNetworkParams(ConcatenationLayer* concat);
    virtual ConcatenationParams& params()
    {
        return m_can_params;
    }

protected:
    ConcatenationParams m_can_params;
};

// split
class SplitNode : public Node
{
public:
    virtual std::string className() const
    {
        return std::string("SplitNode");
    }
    SplitNode()
        : Node()
    {
        m_can_op_type = SPLIT;
    }
    virtual ~SplitNode()
    {
    }

    virtual void captureNetworkParams(SliceLayer* split);
    virtual SplitParams& getParams()
    {
        return m_can_params;
    }

protected:
    SplitParams m_can_params;
};

class Edge
{
public:
    virtual std::string className() const
    {
        return std::string("Edge");
    }
    static std::string prettyId(const Edge* e, NvU8 flags = ast::PrettyId_All)
    {
        if (!e) { return std::string("(null)"); }
        std::stringstream r;
        std::string sep("");
        if (flags & ast::PrettyId_Id)
        {
            r << sep << e->id();
            sep = " ";
        }
        if (flags & ast::PrettyId_Name)
        {
            r << sep << e->id();
            sep = " ";
        } // same as name
        if (flags & ast::PrettyId_ClassName)
        {
            r << sep << e->className();
            sep = " ";
        }
        if (flags & ast::PrettyId_Type)
        {
            r << sep << "canonical_ast::" << e->className();
            sep = " ";
        }

        return r.str();
    }

    Edge()
        : m_containing_graph(0), m_original_tensor(0)
    {
        m_unique_id = m_next_id++;
    }
    virtual ~Edge()
    {
    }

    const std::string id() const
    {
        return m_id;
    }
    void setId(const std::string id)
    {
        m_id = id;
    }

    NvU32 uniqueId() const
    {
        return m_unique_id;
    }

    Graph* graph()
    {
        return m_containing_graph;
    }
    void setGraph(Graph* g)
    {
        m_containing_graph = g;
    }

    Tensor* originalTensor() const
    {
        return m_original_tensor;
    }
    void setOriginalTensor(Tensor* tensor)
    {
        m_original_tensor = tensor;
    }

protected:
    std::string m_id;  // unique within the graph
    NvU32 m_unique_id; // id for graph ordering. u32 instead of string.
    static NvU32 m_next_id;
    Graph* m_containing_graph;
    Tensor* m_original_tensor;
};

Graph* generateGraph(Network*);

Node* newCanonicalNode(Layer* orig_nw_layer);

void traverseGraph(Graph*);

// for sending to debug tools
std::ostream& outputJson(Graph*, std::ostream&);
std::ostream& outputJson(Graph*, Edge*, std::ostream&);
std::ostream& outputJson(Graph*, Node*, std::ostream&);

INetwork* original_network();

bool serializeTo(WisdomContainerEntry*);
bool deserializeFrom(WisdomContainerEntry*);
//    virtual bool assignSymbols(Wisdom *);

class NodeFactory
{
public:
    static ConvolutionNode* newConvNode(ConvolutionLayer* orig_nw_layer);
    static FullyConnectedNode* newFCNode(FullyConnectedLayer* orig_nw_layer);
    static ActivationNode* newActivationNode(ActivationLayer* orig_nw_layer);
    static PoolingNode* newPoolingNode(PoolingLayer* orig_nw_layer);
    static LRNNode* newLRNNode(LRNLayer* orig_nw_layer);
    static ScaleNode* newScaleNode(ScaleLayer* orig_nw_layer);
    static BatchNormNode* newBatchNormNode(BatchNormLayer* orig_nw_layer);
    static SoftMaxNode* newSoftMaxNode(SoftMaxLayer* orig_nw_layer);
    static ConcatenationNode* newConcatNode(ConcatenationLayer* orig_nw_layer);
    static SplitNode* newSplitNode(SliceLayer* orig_nw_layer);
    static DeconvolutionNode* newDeconvNode(DeconvolutionLayer* orig_nw_layer);
    static ElementWiseNode* newEWNode(ElementWiseLayer* orig_nw_layer);

    template<typename T>
    static T nodeCast(Node*);

public:
    static std::map<Node*, ConvolutionNode*> s_conv_priv;
    static std::map<Node*, FullyConnectedNode*> s_fc_priv;
    static std::map<Node*, ActivationNode*> s_act_priv;
    static std::map<Node*, PoolingNode*> s_pool_priv;
    static std::map<Node*, LRNNode*> s_lrn_priv;
    static std::map<Node*, ScaleNode*> s_scale_priv;
    static std::map<Node*, BatchNormNode*> s_bn_priv;
    static std::map<Node*, SoftMaxNode*> s_sm_priv;
    static std::map<Node*, ConcatenationNode*> s_concat_priv;
    static std::map<Node*, SplitNode*> s_split_priv;
    static std::map<Node*, DeconvolutionNode*> s_deconv_priv;
    static std::map<Node*, ElementWiseNode*> s_ew_priv;
};

typedef Graph::NodeSequence NodeSequence;

}; // namespace canonical_ast

typedef ast::Graph<canonical_ast::Node, canonical_ast::Edge> CanonicalGraph;
typedef ast::Graph<canonical_ast::Node, canonical_ast::Edge>::ElemSequence CanonicalGraphElemSequence;
typedef ast::Graph<canonical_ast::Node, canonical_ast::Edge>::NodeSequence CanonicalGraphNodeSequence;
typedef ast::Graph<canonical_ast::Node, canonical_ast::Edge>::EdgeSequence CanonicalGraphEdgeSequence;
typedef ast::GraphVisitor<CanonicalGraph> CanonicalGraphVisitor;

}; // namespace priv

}; // namespace nvdla

#endif // NVDLA_PRIV_CANONICAL_AST_H
