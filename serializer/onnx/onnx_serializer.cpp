/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * License); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*
 * Copyright (c) 2017, Open AI Lab
 * Author: haitao@openailab.com
 */

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/message.h>

#include "data_type.hpp"
#include "tengine_errno.hpp"
#include "operator_manager.hpp"
#include "operator/conv_param.hpp"
#include "operator/pool_param.hpp"
#include "operator/concat_param.hpp"
#include "operator/softmax_param.hpp"
#include "operator/relu_param.hpp"
#include "operator/batch_norm_param.hpp"
#include "operator/flatten_param.hpp"
#include "operator/eltwise_param.hpp"
#include "operator/gemm_param.hpp"
#include "operator/fc_param.hpp"

#include "type_name.hpp"
#include "compiler.hpp"

#include "onnx_serializer.hpp"

namespace TEngine {

using op_load_t = std::function<bool(StaticGraph* graph, StaticNode* node, const onnx::NodeProto&)>;

bool OnnxSerializer::LoadModel(const std::vector<std::string>& file_list, StaticGraph* graph)
{
    if(file_list.size() != GetFileNum())
        return false;

    onnx::ModelProto model;

    if(!LoadModelFile(file_list[0].c_str(), model))
        return false;

    SetGraphSource(graph, file_list[0]);
    SetGraphSourceFormat(graph, "onnx");
    SetGraphConstTensorFile(graph, file_list[0]);

    return LoadGraph(model, graph);
}

bool OnnxSerializer::LoadModelFile(const char* fname, onnx::ModelProto& model)
{
    std::ifstream is(fname, std::ios::in | std::ios::binary);

    if(!is.is_open())
    {
        LOG_ERROR() << "cannot open file: " << fname << "\n";
        set_tengine_errno(ENOENT);
        return false;
    }

    google::protobuf::io::IstreamInputStream input_stream(&is);
    google::protobuf::io::CodedInputStream coded_input(&input_stream);

    coded_input.SetTotalBytesLimit(1024 << 20, 512 << 20);

    bool ret = model.ParseFromCodedStream(&coded_input);

    is.close();

    if(!ret)
    {
        LOG_ERROR() << "onnx serializer: parse file: " << fname << " failed\n";
        set_tengine_errno(EINVAL);
        return false;
    }

    return ret;
}

bool OnnxSerializer::LoadConstTensor(StaticGraph* graph, const onnx::GraphProto& onnx_graph)
{
    int const_tensor_number = onnx_graph.initializer_size();

    for(int i = 0; i < const_tensor_number; i++)
    {
        const onnx::TensorProto& onnx_tensor = onnx_graph.initializer(i);

        StaticTensor* tensor = CreateStaticConstTensor(graph, onnx_tensor.name());

        std::vector<int> dims;

        int dim_size = onnx_tensor.dims_size();
        int tensor_size = 1;

        for(int l = 0; l < dim_size; l++)
        {
            tensor_size *= onnx_tensor.dims(l);
            dims.push_back(onnx_tensor.dims(l));
        }

        SetTensorDim(tensor, dims);

        // TODO: how to parser from ONNX?
        SetTensorDataType(tensor, DataType::GetTypeID("float32"));

        // Note: the const tensor layout will be set in operator load function

        tensor_size = 4 * tensor_size;

        SetTensorSize(tensor, tensor_size);

        uint8_t* mem_buf = ( uint8_t* )std::malloc(tensor_size);
        uint8_t* raw_data = ( uint8_t* )onnx_tensor.raw_data().c_str();

        /* load data */
        for(int i = 0; i < tensor_size; i++)
            mem_buf[i] = raw_data[i];

        SetConstTensorBuffer(tensor, mem_buf);
        SetConstTensorFileLocation(tensor, -1, 0);

        /* Now, create the node .... */

        StaticOp* op = CreateStaticOp(graph, "Const");
        StaticNode* node = CreateStaticNode(graph, GetTensorName(tensor));

        SetNodeOp(node, op);

        AddNodeOutputTensor(node, tensor);
    }

    return true;
}

void OnnxSerializer::CreateInputNode(StaticGraph* graph, const onnx::GraphProto& onnx_graph)
{
    int input_number = onnx_graph.input_size();

    for(int i = 0; i < input_number; i++)
    {
        const onnx::ValueInfoProto& val = onnx_graph.input(i);

        if(FindConstTensor(graph, val.name()) != nullptr)
            continue;

        // now, catch an input tensor

        const onnx::TypeProto& type = val.type();

        const onnx::TypeProto::Tensor& tensor_type = type.tensor_type();

        const onnx::TensorShapeProto& shape = tensor_type.shape();

        int has_shape = 1;

        std::vector<int> dims;

        for(int i = 0; i < shape.dim_size(); i++)
        {
            const onnx::TensorShapeProto::Dimension& dim = shape.dim(i);

            if(dim.has_dim_param())
            {
                has_shape = 0;
                break;
            }

            dims.push_back(dim.dim_value());
        }

        StaticTensor* tensor = CreateStaticTensor(graph, val.name());

        SetTensorDataType(tensor, DataType::GetTypeID("float32"));
        SetTensorDataLayout(tensor, "NCHW");

        if(has_shape)
            SetTensorDim(tensor, dims);

        StaticNode* node = CreateStaticNode(graph, val.name());
        StaticOp* op = CreateStaticOp(graph, "InputOp");

        SetNodeOp(node, op);

        AddNodeOutputTensor(node, tensor);

        /*add this node into graph input node list */

        AddGraphInputNode(graph, node);
    }
}

static bool onnx_skip_output_for_test(const std::string& op_type, int idx)
{
    if(op_type == "Dropout" && idx > 0)
        return true;
    return false;
}

bool OnnxSerializer::LoadNode(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    for(int i = 0; i < onnx_node.input_size(); i++)
    {
        const std::string& input_name = onnx_node.input(i);

        StaticTensor* tensor = FindTensor(graph, input_name);

        AddNodeInputTensor(node, tensor);
    }

    for(int i = 0; i < onnx_node.output_size(); i++)
    {
        const std::string& onnx_op_name = onnx_node.op_type();

        if(onnx_skip_output_for_test(onnx_op_name, i))
            continue;

        const std::string& output_name = onnx_node.output(i);

        StaticTensor* tensor = CreateStaticTensor(graph, output_name);

        SetTensorDataType(tensor, DataType::GetTypeID("float32"));
        SetTensorDataLayout(tensor, "NCHW");
        AddNodeOutputTensor(node, tensor);
    }

    return true;
}

bool OnnxSerializer::LoadGraph(onnx::ModelProto& model, StaticGraph* graph)
{
    const onnx::GraphProto& onnx_graph = model.graph();

    SetGraphIdentity(graph, model.domain(), onnx_graph.name(), std::to_string(( int )model.model_version()));

    LoadConstTensor(graph, onnx_graph);
    CreateInputNode(graph, onnx_graph);

    int i;

    for(i = 0; i < onnx_graph.node_size(); i++)
    {
        const onnx::NodeProto& onnx_node = onnx_graph.node(i);
        const std::string& onnx_op_name = onnx_node.op_type();

        if(!FindOpLoadMethod(onnx_op_name))
        {
            LOG_ERROR() << "cannot find load function for operator: " << onnx_op_name << "\n";
            continue;
        }

        StaticNode* node = CreateStaticNode(graph, onnx_node.output(0));

        if(!LoadNode(graph, node, onnx_node))
            break;

        op_load_t op_func = any_cast<op_load_t>(GetOpLoadMethod(onnx_op_name));

        if(!op_func(graph, node, onnx_node))
            break;
    }

    if(i < onnx_graph.node_size())
        return false;

    return true;
}

/* Global functions to load indiviual operator */

static bool LoadOnnxConvolutionOp(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    ConvParam param = any_cast<ConvParam>(OpManager::GetOpDefParam("Convolution"));

    for(int k = 0; k < onnx_node.attribute_size(); k++)
    {
        const onnx::AttributeProto& attr = onnx_node.attribute(k);

        if(attr.name() == "kernel_shape")
        {
            param.kernel_h = attr.ints(0);
            param.kernel_w = attr.ints(1);
        }
        else if(attr.name() == "strides")
        {
            param.stride_h = attr.ints(0);
            param.stride_w = attr.ints(1);
        }
        else if(attr.name() == "pads")
        {
            param.pad_h = attr.ints(0);
            param.pad_w = attr.ints(1);
        }
    }

    /* update the input tensor data layout */

    for(int k = 0; k < onnx_node.input_size(); k++)
    {
        const std::string& input_name = onnx_node.input(k);
        StaticTensor* tensor = FindTensor(graph, input_name);

        if(k == 1)    // weight
        {
            const std::vector<int>& dim = GetTensorDim(tensor);

            SetTensorDataLayout(tensor, "NCHW");

            /* onnx hide the output channel in weight ..*/
            param.output_channel = dim[0];
        }
        else if(k == 2)
            SetTensorDataLayout(tensor, "W");
    }

    StaticOp* op = CreateStaticOp(graph, "Convolution");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);

    return true;
}

static bool LoadOnnxBN(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    BatchNormParam param = any_cast<BatchNormParam>(OpManager::GetOpDefParam("BatchNormalization"));

    // get espilon

    for(int k = 0; k < onnx_node.attribute_size(); k++)
    {
        const onnx::AttributeProto& attr = onnx_node.attribute(k);

        if(attr.name() == "epsilon")
            param.eps = attr.f();
    }

    for(int k = 1; k < onnx_node.input_size(); k++)
    {
        const std::string& input_name = onnx_node.input(k);
        StaticTensor* tensor = FindTensor(graph, input_name);
        SetTensorDataLayout(tensor, "W");
    }

    StaticOp* op = CreateStaticOp(graph, "BatchNormalization");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);

    return true;
}

static bool LoadOnnxRelu(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    ReLuParam param = any_cast<ReLuParam>(OpManager::GetOpDefParam("ReLu"));
    param.negative_slope = 0.f;

    StaticOp* op = CreateStaticOp(graph, "ReLu");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);

    return true;
}

static bool LoadOnnxPooling(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    PoolParam param = any_cast<PoolParam>(OpManager::GetOpDefParam("Pooling"));

    const std::string& onnx_op = onnx_node.op_type();

    if(onnx_op == "GlobalAveragePool")
    {
        param.global = 1;
        param.alg = kPoolAvg;
    }
    else if(onnx_op == "MaxPool" || onnx_op == "AveragePool")
    {
        param.global = 0;

        if(onnx_op == "AveragePool")
            param.alg = kPoolAvg;
        else
            param.alg = kPoolMax;

        for(int k = 0; k < onnx_node.attribute_size(); k++)
        {
            const onnx::AttributeProto& attr = onnx_node.attribute(k);

            if(attr.name() == "kernel_shape")
            {
                param.kernel_h = attr.ints(0);
                param.kernel_w = attr.ints(1);
            }
            else if(attr.name() == "strides")
            {
                param.stride_h = attr.ints(0);
                param.stride_w = attr.ints(1);
            }
            else if(attr.name() == "pads")
            {
                param.pad_h = attr.ints(0);
                param.pad_w = attr.ints(1);
            }
        }
    }
    else
    {
        LOG_ERROR() << "UKNOWN POOLING: " << onnx_op << "\n";
        return false;
    }

    param.kernel_shape.resize(2);
    param.kernel_shape[0] = param.kernel_h;
    param.kernel_shape[1] = param.kernel_w;

    param.pads.resize(4);
    param.pads[0] = param.pad_h;
    param.pads[1] = param.pad_w;
    param.pads[2] = param.pad_h;
    param.pads[3] = param.pad_w;

    param.strides.resize(2);
    param.strides[0] = param.stride_h;
    param.strides[1] = param.stride_w;

    StaticOp* op = CreateStaticOp(graph, "Pooling");

    SetOperatorParam(op, param);

    SetNodeOp(node, op);

    return true;
}

static bool LoadOnnxFlatten(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    FlattenParam param = any_cast<FlattenParam>(OpManager::GetOpDefParam("Flatten"));

    const onnx::AttributeProto& attr = onnx_node.attribute(0);

    param.axis = attr.i();

    StaticOp* op = CreateStaticOp(graph, "Flatten");

    SetOperatorParam(op, param);

    SetNodeOp(node, op);

    return true;
}

static bool LoadOnnxGemm(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    GemmParam param = any_cast<GemmParam>(OpManager::GetOpDefParam("Gemm"));

    for(int k = 0; k < onnx_node.attribute_size(); k++)
    {
        const onnx::AttributeProto& attr = onnx_node.attribute(k);

        if(attr.name() == "alpha")
            param.alpha = attr.f();
        else if(attr.name() == "beta")
            param.beta = attr.f();
        else if(attr.name() == "transA")
            param.transA = attr.i();
        else if(attr.name() == "transB")
            param.transB = attr.i();
    }

    StaticTensor* weight_tensor = FindTensor(graph, onnx_node.input(1));

    StaticTensor* bias_tensor = FindTensor(graph, onnx_node.input(2));

    SetTensorDataLayout(weight_tensor, "HW");
    SetTensorDataLayout(bias_tensor, "W");

    if(param.transA)
    {
        StaticOp* op = CreateStaticOp(graph, "Gemm");

        SetOperatorParam(op, param);

        SetNodeOp(node, op);

        return true;
    }

    // create fc instead
    if(!param.transB)
    {
        // swap shape
        int k = weight_tensor->dims[0];
        int n = weight_tensor->dims[1];

        weight_tensor->dims[0] = n;
        weight_tensor->dims[1] = k;

        float* tmp = ( float* )malloc(k * n * sizeof(float));
        float* data = ( float* )GetConstTensorBuffer(weight_tensor);

        for(int i = 0; i < n; i++)
            for(int j = 0; j < k; j++)
            {
                tmp[i * k + j] = data[j * n + i];
            }

        memcpy(data, tmp, n * k * sizeof(float));

        free(tmp);
    }

    if(param.alpha != 1)
    {
        float* data = ( float* )GetConstTensorBuffer(weight_tensor);
        int tensor_size = weight_tensor->dims[0] * weight_tensor->dims[1];

        for(int i = 0; i < tensor_size; i++)
            data[i] *= param.alpha;
    }

    if(param.beta != 1)
    {
        float* data = ( float* )GetConstTensorBuffer(bias_tensor);
        int tensor_size = weight_tensor->dims[0];

        for(int i = 0; i < tensor_size; i++)
            data[i] *= param.beta;
    }

    StaticOp* op = CreateStaticOp(graph, "FullyConnected");

    FCParam fc_param = any_cast<FCParam>(OpManager::GetOpDefParam("FullyConnected"));

    fc_param.num_output = weight_tensor->dims[0];

    SetOperatorParam(op, fc_param);

    SetNodeOp(node, op);

    return true;
}

static bool LoadOnnxConcat(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    ConcatParam param = any_cast<ConcatParam>(OpManager::GetOpDefParam("Concat"));

    /* ONNX does not set the concat axis..., while caffe does */

    StaticOp* op = CreateStaticOp(graph, "Concat");

    SetOperatorParam(op, param);

    SetNodeOp(node, op);

    return true;
}

static bool LoadOnnxDropout(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    StaticOp* op = CreateStaticOp(graph, "Dropout");

    SetNodeOp(node, op);

    return true;
}

static bool LoadOnnxAdd(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    EltwiseParam param = any_cast<EltwiseParam>(OpManager::GetOpDefParam("Eltwise"));

    param.type = ELT_SUM;

    StaticOp* op = CreateStaticOp(graph, "Eltwise");

    SetOperatorParam(op, param);

    SetNodeOp(node, op);

    return true;
}

static bool LoadOnnxSoftmax(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    StaticOp* op = CreateStaticOp(graph, "Softmax");

    SoftmaxParam param = any_cast<SoftmaxParam>(OpManager::GetOpDefParam("Softmax"));

    param.axis = 1;

    SetOperatorParam(op, param);

    SetNodeOp(node, op);

    AddGraphOutputNode(graph, node);

    return true;
}

// To register all op loader...
bool OnnxSerializerRegisterOpLoader(void)
{
    // first get the onnx_serializer object

    SerializerPtr serializer;

    if(!SerializerManager::SafeGet("onnx", serializer))
        return false;

    OnnxSerializer* p_onnx = dynamic_cast<OnnxSerializer*>(serializer.get());

    p_onnx->RegisterOpLoadMethod("Conv", op_load_t(LoadOnnxConvolutionOp));
    p_onnx->RegisterOpLoadMethod("Relu", op_load_t(LoadOnnxRelu));
    p_onnx->RegisterOpLoadMethod("MaxPool", op_load_t(LoadOnnxPooling));
    p_onnx->RegisterOpLoadMethod("GlobalAveragePool", op_load_t(LoadOnnxPooling));
    p_onnx->RegisterOpLoadMethod("AveragePool", op_load_t(LoadOnnxPooling));
    p_onnx->RegisterOpLoadMethod("Concat", op_load_t(LoadOnnxConcat));
    p_onnx->RegisterOpLoadMethod("Dropout", op_load_t(LoadOnnxDropout));
    p_onnx->RegisterOpLoadMethod("Softmax", op_load_t(LoadOnnxSoftmax));
    p_onnx->RegisterOpLoadMethod("BatchNormalization", op_load_t(LoadOnnxBN));
    p_onnx->RegisterOpLoadMethod("Add", op_load_t(LoadOnnxAdd));
    p_onnx->RegisterOpLoadMethod("Flatten", op_load_t(LoadOnnxFlatten));
    p_onnx->RegisterOpLoadMethod("Gemm", op_load_t(LoadOnnxGemm));

    return true;
}

}    // namespace TEngine
