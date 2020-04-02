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
#include <algorithm>
#include <vector>

#include "tengine_c_api.h"
#include "exec_attr.hpp"
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
#include "operator/reshape_param.hpp"
#include "operator/permute_param.hpp"
#include "operator/clip_param.hpp"
#include "operator/hardswish_param.hpp"
#include "operator/elu_param.hpp"
#include "operator/interp_param.hpp"
#include "operator/transpose_param.hpp"
#include "operator/slice_param.hpp"
#include "operator/split_param.hpp"


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

    //printf("file size: %d \n", (int)file_list[0].size());
    SetGraphSource(graph, file_list[0]);
    SetGraphSourceFormat(graph, "onnx");
    SetGraphConstTensorFile(graph, file_list[0]);
    SetGraphLayout(graph, TENGINE_LAYOUT_NCHW);
    SetModelLayout(graph, TENGINE_LAYOUT_NCHW);
    SetModelFormat(graph, MODEL_FORMAT_ONNX);

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
static onnx::TensorProto get_node_attr_tensor(const onnx::NodeProto& node, const char* key)
{
    for (int i=0; i<node.attribute_size(); i++)
    {
        const onnx::AttributeProto& attr = node.attribute(i);
        if (attr.name() == key)
        {
            return attr.t();
        }
    }

    return onnx::TensorProto();
}
/*
static std::vector<int> get_tensor_proto_reshape_shape(const onnx::TensorProto& tp)
{
    const int64_t* shape_data = 0;
    int size = 0;

    // int64
    if (tp.has_raw_data())
    {
        shape_data = (const int64_t*)tp.raw_data().data();
        size = tp.raw_data().size() / 8;
    }
    else if (tp.data_type() == 7)
    {
        shape_data = tp.int64_data().data();
        size = tp.int64_data_size();
    }

    std::vector<int> shape;
    for (int j=0; j<size; j++)
    {
        shape.push_back(shape_data[j]);
    }

    return shape;
}
*/

void OnnxSerializer::LoadConstNode(const onnx::GraphProto& onnx_graph, StaticGraph* graph){
    std::map<std::string, onnx::TensorProto> node_tensor;
    
    int node_count = onnx_graph.node_size();

    for(int i = 0; i < node_count; i++){
        const onnx::NodeProto& node = onnx_graph.node(i);
        const std::string& op = node.op_type();
        if(op == "Constant"){
            onnx::TensorProto node_attr = get_node_attr_tensor(node, "value");
            node_tensor.insert(std::pair<std::string, onnx::TensorProto >(node.output(0),node_attr));
        }

    }
    if(node_tensor.size()==0)
    {
        return;
    }
    for(int i = 0; i < node_count; i++){
        const onnx::NodeProto& node = onnx_graph.node(i);

        const std::string& op = node.op_type();
     
        if(op == "Reshape"){
            const onnx::TensorProto& shape_tensor = node_tensor[node.input(1)];
            StaticTensor* tensor = CreateStaticConstTensor(graph, node.input(1));
            std::vector<int> dims;
            int dim_size = shape_tensor.dims_size();
            int tensor_size = 1;
            for(int l = 0; l < dim_size; l++)
            {
                tensor_size *= shape_tensor.dims(l);
            }
            if (shape_tensor.has_raw_data())
            {
                SetTensorDataType(tensor, DataType::GetTypeID("float32"));
                tensor_size = tensor_size * sizeof(int64_t);
                SetTensorSize(tensor, tensor_size);       
                int64_t* raw_data = ( int64_t* )shape_tensor.raw_data().data();
                int64_t* mem_buf = ( int64_t* )std::malloc(tensor_size);
                for(int i = 0; i <tensor_size/ (int)sizeof(int64_t); i++){
                    mem_buf[i] = raw_data[i];
                }
                dims.push_back(tensor_size/ (int)sizeof(int64_t));
                SetTensorDim(tensor, dims);
                SetConstTensorBuffer(tensor, mem_buf);
            }
            SetConstTensorFileLocation(tensor, -1, 0);
            StaticOp* op = CreateStaticOp(graph, "Const");
            StaticNode* node_create = CreateStaticNode(graph, GetTensorName(tensor));
            SetNodeOp(node_create, op);
            AddNodeOutputTensor(node_create, tensor);

        }
    }

}

bool OnnxSerializer::LoadConstTensor(StaticGraph* graph, const onnx::GraphProto& onnx_graph)
{

    int const_tensor_number = onnx_graph.initializer_size();

    
    LoadConstNode(onnx_graph, graph);

    for(int i = 0; i < const_tensor_number; i++)
    {
        const onnx::TensorProto& onnx_tensor = onnx_graph.initializer(i);
 
        StaticTensor* tensor = CreateStaticConstTensor(graph, onnx_tensor.name());
        std::vector<int> dims;

        //printf("%s \n", onnx_tensor.name().c_str(), onn);
        int dim_size = onnx_tensor.dims_size();
        int tensor_size = 1;

        for(int l = 0; l < dim_size; l++)
        {
            tensor_size *= onnx_tensor.dims(l);
            dims.push_back(onnx_tensor.dims(l));
        }

        if(dim_size==0)
        {
            dims.push_back(1);
        }
        
        SetTensorDim(tensor, dims);

        // Note: the const tensor layout will be set in operator load function
        // int data_type = onnx_tensor.data_type();
       
        if(onnx_tensor.has_raw_data())
        {
            if(onnx_tensor.data_type()==7)
            {
                SetTensorDataType(tensor, DataType::GetTypeID("float32"));
                tensor_size = sizeof(int64_t) * tensor_size;
                SetTensorSize(tensor, tensor_size);

                int64_t* mem_buf = ( int64_t* )std::malloc(tensor_size);
                int64_t* raw_data = ( int64_t* )onnx_tensor.raw_data().data();

                for(int i = 0; i < tensor_size/sizeof(int64_t); i++)
                {
                    mem_buf[i] = raw_data[i];
                }
                SetConstTensorBuffer(tensor, mem_buf);
            }
            else if(onnx_tensor.data_type()==1)
            {
                SetTensorDataType(tensor, DataType::GetTypeID("float32"));
                tensor_size = 4 * tensor_size;
                SetTensorSize(tensor, tensor_size);

                float* mem_buf = ( float* )std::malloc(tensor_size);
                float* raw_data = ( float* )onnx_tensor.raw_data().c_str();

                for(int i = 0; i < tensor_size/4; i++)
                {
                    mem_buf[i] = raw_data[i];
                }
                SetConstTensorBuffer(tensor, mem_buf);
            }
            else
            {
                SetTensorDataType(tensor, DataType::GetTypeID("float32"));
                tensor_size = 4 * tensor_size;
                SetTensorSize(tensor, tensor_size);

                uint8_t* mem_buf = ( uint8_t* )std::malloc(tensor_size);
                uint8_t* raw_data = ( uint8_t* )onnx_tensor.raw_data().c_str();

                for(int i = 0; i < tensor_size/4; i++)
                {
                    mem_buf[i] = raw_data[i];
                }
                SetConstTensorBuffer(tensor, mem_buf);
            }
            

        } 
        else
        {
            if(onnx_tensor.data_type()==1)
            {
                SetTensorDataType(tensor, DataType::GetTypeID("float32"));
                SetTensorSize(tensor, tensor_size*sizeof(float));
                
                float* mem_buf = ( float* )std::malloc(tensor_size * sizeof(float));
                const float* float_data = onnx_tensor.float_data().data();
                // printf("onnx_tensor data_type : %d\n",onnx_tensor.data_type());
                // printf("onnx_tensor:%s\n",onnx_tensor.name().c_str());
                // printf("float_data:%p\n",&float_data);
                for(int i = 0; i < tensor_size; i++)
                    mem_buf[i] = float_data[i];
                    
                SetConstTensorBuffer(tensor, mem_buf);
            }
            else if(onnx_tensor.data_type()==7)
            {
                SetTensorDataType(tensor, DataType::GetTypeID("float32"));
                tensor_size = tensor_size * sizeof(int64_t);
                SetTensorSize(tensor, tensor_size);       
                int64_t* raw_data = ( int64_t* )onnx_tensor.int64_data().data();
                int64_t* mem_buf = ( int64_t* )std::malloc(tensor_size);
                for(int i = 0; i <tensor_size/ (int)sizeof(int64_t); i++){
                    mem_buf[i] = raw_data[i];
                }
                // dims.push_back(tensor_size/ (int)sizeof(int64_t));
                // SetTensorDim(tensor, dims);
                SetConstTensorBuffer(tensor, mem_buf);
            }
          
        }
        
        SetConstTensorFileLocation(tensor, -1, 0);

        /* Now, create the node .... */

        StaticOp* op = CreateStaticOp(graph, "Const");
        StaticNode* node = CreateStaticNode(graph, GetTensorName(tensor));
        //const std::string& name_ = GetTensorName(tensor);
        //printf("tensor name: %s \n", name_.c_str());
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

        if(FindConstTensor(graph, val.name()) != nullptr){
            continue;
        }

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
    std::vector<std::string> no_supported_op;
    for(i = 0; i < onnx_graph.node_size(); i++)
    {
        const onnx::NodeProto& onnx_node = onnx_graph.node(i);
        const std::string& onnx_op_name = onnx_node.op_type();

        if(!FindOpLoadMethod(onnx_op_name))
        {
            auto it = find(no_supported_op.begin(),no_supported_op.end(),onnx_op_name);
            if(it == no_supported_op.end())
            {
                if(onnx_op_name == "Constant")
                    continue;
                no_supported_op.push_back(onnx_op_name);
            }
     //       LOG_ERROR() << "cannot find load function for operator: " << onnx_op_name << "\n";
     //       continue;
        }
    }
    if(no_supported_op.size())
    {
        
        LOG_ERROR() << "These "<<no_supported_op.size() << "op are not supported\n";
        LOG_ERROR() << "{";
        for(int j = 0; j < (int) no_supported_op.size(); j++)
        {
            LOG_ERROR() << no_supported_op[j] <<",";
        }
        LOG_ERROR() << "}\n";
   
        return false;
    }
        
    for(i = 0; i < onnx_graph.node_size(); i++)
    {
        const onnx::NodeProto& onnx_node = onnx_graph.node(i);
        const std::string& onnx_op_name = onnx_node.op_type();

        //if(!FindOpLoadMethod(onnx_op_name))
       // {
      //      LOG_ERROR() << "cannot find load function for operator: " << onnx_op_name << "\n";
       //     continue;
       // }
        if(onnx_op_name == "Constant")
            continue;
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
            param.pad_h0 = attr.ints(0);
            param.pad_h1 = attr.ints(0);
            param.pad_w0 = attr.ints(1);
            param.pad_w1 = attr.ints(1);
        }
        else if(attr.name() == "group")
        {
            param.group = attr.i();
        }
        else if(attr.name() == "dilations")
        {
            param.dilation_h = attr.ints(0);
            param.dilation_w = attr.ints(0);
        }
        else
            LOG_ERROR() << "attr.name:" << attr.name() << "\n";
    }

    /* update the input tensor data layout */

    for(int k = 0; k < onnx_node.input_size(); k++)
    {
        const std::string& input_name = onnx_node.input(k);
        StaticTensor* tensor = FindTensor(graph, input_name);
        if(k == 1)    // weight
        {
            const std::vector<int>& dim = GetTensorDim(tensor);
            /* onnx hide the output channel in weight ..*/
            param.output_channel = dim[0];
        }
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

static bool LoadOnnxTanh(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{

    StaticOp* op = CreateStaticOp(graph, "Tanh");
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
                param.pad_h0 = attr.ints(0);
                param.pad_h1 = attr.ints(0);
                param.pad_w0 = attr.ints(1);
                param.pad_w1 = attr.ints(1);
            }
        }
    }
    else
    {
        LOG_ERROR() << "UKNOWN POOLING: " << onnx_op << "\n";
        return false;
    }

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
    for(int k = 0; k < onnx_node.attribute_size(); k++)
    {
        const onnx::AttributeProto& attr = onnx_node.attribute(k);
        if(attr.name() == "axis"){
            param.axis = attr.i();
        }
    }
    

    // param.axis = 1;

    SetOperatorParam(op, param);

    SetNodeOp(node, op);


    return true;
}

static bool LoadOnnxHardSwish(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    StaticOp* op = CreateStaticOp(graph, "Hardswish");

    HardswishParam param = any_cast<HardswishParam>(OpManager::GetOpDefParam("Hardswish"));

    for(int k = 0; k < onnx_node.attribute_size(); k++)
    {
        const onnx::AttributeProto& attr = onnx_node.attribute(k);

        if(attr.name() == "alpha")
            param.alpha = attr.f();
        else if(attr.name() == "beta")
            param.beta = attr.f();
    }

    SetOperatorParam(op, param);

    SetNodeOp(node, op);


    return true;
}

static bool LoadOnnxElu(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    StaticOp* op = CreateStaticOp(graph, "Elu");

    EluParam param = any_cast<EluParam>(OpManager::GetOpDefParam("Elu"));

    for(int k = 0; k < onnx_node.attribute_size(); k++)
    {
        const onnx::AttributeProto& attr = onnx_node.attribute(k);

        if(attr.name() == "alpha")
            param.alpha = attr.f();
    }

    SetOperatorParam(op, param);

    SetNodeOp(node, op);


    return true;
}

static bool LoadOnnxPRelu(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    StaticOp* op = CreateStaticOp(graph, "PReLU");

    SetNodeOp(node, op);

    return true;
}

static bool LoadOnnxInterp(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    StaticOp* op = CreateStaticOp(graph, "Interp");

    InterpParam param = any_cast<InterpParam>(OpManager::GetOpDefParam("Interp"));

    if(onnx_node.input_size() == 1)
    {
        for(int k = 0; k < onnx_node.attribute_size(); k++)
        {
            const onnx::AttributeProto& attr = onnx_node.attribute(k);
            if(attr.name() == "scales")
            {
                param.height_scale = attr.f();
                param.width_scale = attr.f();
            }
        }
    }
    else
    {
        const std::string& input_name = onnx_node.input(1);
        // std::cout<<"tensor name:"<<input_name<<"\n";
        StaticTensor* tensor = FindTensor(graph, input_name);
        float* data = ( float* )GetConstTensorBuffer(tensor);
        //int scales_size = tensor->dims[0];
        // printf("scale size:%d\n", scales_size);
        // printf("scale data:%f %f\n",data[0], data[1]);
        param.height_scale = data[2];
        param.width_scale = data[3];

    }

    std::string mode = "nearest";
    for(int k = 0; k < onnx_node.attribute_size(); k++)
    {
        const onnx::AttributeProto& attr = onnx_node.attribute(k);

        if(attr.name() == "mode")
        {
            mode = attr.s();
        }
    }

    if (mode == "nearest")
    {
        param.resize_type = 1;
    }
    else if (mode == "bilinear" || mode == "linear")
    {
        param.resize_type = 2;
    }

    SetOperatorParam(op, param);

    SetNodeOp(node, op);


    return true;
}

static bool LoadOnnxClip(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    ClipParam param = any_cast<ClipParam>(OpManager::GetOpDefParam("Clip"));

    int size = onnx_node.attribute_size(); 
 
    for(int i = 0; i < size; i++)
    {
        const onnx::AttributeProto& attr = onnx_node.attribute(i);
        if(attr.name() == "max")
        {
            param.max = attr.f();
            //std::cout<<"max:"<<param.max<<std::endl;
        }
        else if(attr.name() == "min")
        {
            param.min = attr.f();
            //std::cout<<"min:"<<param.min<<std::endl;
        }
    }

    StaticOp* op = CreateStaticOp(graph, "Clip");

    SetOperatorParam(op, param);

    SetNodeOp(node, op);

    return true;
}

static bool LoadOnnxMul(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    EltwiseParam param = any_cast<EltwiseParam>(OpManager::GetOpDefParam("Eltwise"));

    param.type = ELT_PROD;

    StaticOp* op = CreateStaticOp(graph, "Eltwise");

    SetOperatorParam(op, param);

    SetNodeOp(node, op);

    return true;
}

static bool LoadOnnxDiv(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    EltwiseParam param = any_cast<EltwiseParam>(OpManager::GetOpDefParam("Eltwise"));

    param.type = ELT_DIV;

    StaticOp* op = CreateStaticOp(graph, "Eltwise");

    SetOperatorParam(op, param);

    SetNodeOp(node, op);

    return true;
}
static bool LoadOnnxFloor(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    EltwiseParam param = any_cast<EltwiseParam>(OpManager::GetOpDefParam("Eltwise"));

    param.type = ELT_FLOOR;

    StaticOp* op = CreateStaticOp(graph, "Eltwise");

    SetOperatorParam(op, param);

    SetNodeOp(node, op);

    return true;
}

static bool LoadOnnxTranspose(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{

    TransposeParam param = any_cast<TransposeParam>(OpManager::GetOpDefParam("Transpose"));
    const onnx::AttributeProto& attr = onnx_node.attribute(0);

    int size = attr.ints_size();
    for(int i = 0; i < size; i++){
        param.tr_shape.push_back(attr.ints(i));
    }

    StaticOp* op = CreateStaticOp(graph, "Transpose");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);

    return true;
}

static bool LoadOnnxReshape(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    ReshapeParam param = any_cast<ReshapeParam>(OpManager::GetOpDefParam("Reshape"));


    StaticTensor* shape_tensor = FindTensor(graph, onnx_node.input(1));

    param.is_onnx = true;
    int size = shape_tensor->dims[0];
    int64_t* data = (int64_t*)GetConstTensorBuffer(shape_tensor);
    for(int i = 0; i < size; i++){
        param.re_shape.push_back(data[i]);
    }

    StaticOp* op = CreateStaticOp(graph, "Reshape");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);

    return true;
}
static bool LoadOnnxLeakyReLu(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    ReLuParam param = any_cast<ReLuParam>(OpManager::GetOpDefParam("ReLu"));
    const onnx::AttributeProto& attr = onnx_node.attribute(0);
    param.negative_slope = attr.f();

    StaticOp* op = CreateStaticOp(graph, "ReLu");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);

    return true;
}
static bool LoadOnnxSlice(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    SliceParam param = any_cast<SliceParam>(OpManager::GetOpDefParam("Slice"));
    
    for(int k = 0; k < onnx_node.attribute_size(); k++)
    {
        const onnx::AttributeProto& attr = onnx_node.attribute(k);
        if(attr.name() == "axes"){
            param.axis = attr.ints(0);
        } else if (attr.name() == "ends") {
            param.end = attr.ints(0);
        } else if (attr.name() == "starts"){
            param.begin = attr.ints(0);
        }
    }
    //printf("Slice param: %d %d %d\n", param.end, param.begin, param.axis);
    param.iscaffe = false;
    param.ismxnet = false;
    param.isonnx = true;
    StaticOp* op = CreateStaticOp(graph, "Slice");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);

    return true;
}

static bool LoadOnnxSigmod(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    //printf("Load Sigmod\n");
    StaticOp* op = CreateStaticOp(graph, "Sigmoid");
    SetNodeOp(node, op);

    return true;
}

static bool LoadOnnxSplit(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    SplitParam param = any_cast<SplitParam>(OpManager::GetOpDefParam("Split"));
    param.is_onnx = true;
    for(int k = 0; k < onnx_node.attribute_size(); k++)
    {
        const onnx::AttributeProto& attr = onnx_node.attribute(k);
        if(attr.name() == "axis"){
            param.axis = attr.i();
        } else if (attr.name() == "split") {
            int size = attr.ints_size();
            param.split_dim = size;
            for(int i = 0; i < size; i++){
                param.split_sizes_.push_back(attr.ints(i));
            }
        }
    }

    param.is_caffe = false;

    StaticOp* op = CreateStaticOp(graph, "Split");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);

    return true;
}

static bool LoadOnnxExp(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    EltwiseParam param = any_cast<EltwiseParam>(OpManager::GetOpDefParam("Eltwise"));

    param.type = ELT_EXP;

    StaticOp* op = CreateStaticOp(graph, "Eltwise");

    SetOperatorParam(op, param);

    SetNodeOp(node, op);

    return true;
}

static bool LoadOnnxSub(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    EltwiseParam param = any_cast<EltwiseParam>(OpManager::GetOpDefParam("Eltwise"));

    param.type = ELT_SUB;

    StaticOp* op = CreateStaticOp(graph, "Eltwise");

    SetOperatorParam(op, param);

    SetNodeOp(node, op);

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
    p_onnx->RegisterOpLoadMethod("HardSwish", op_load_t(LoadOnnxHardSwish));
	p_onnx->RegisterOpLoadMethod("Elu", op_load_t(LoadOnnxElu));
    p_onnx->RegisterOpLoadMethod("Tanh", op_load_t(LoadOnnxTanh));
	p_onnx->RegisterOpLoadMethod("PRelu", op_load_t(LoadOnnxPRelu));
    p_onnx->RegisterOpLoadMethod("Upsample", op_load_t(LoadOnnxInterp));
    p_onnx->RegisterOpLoadMethod("Clip", op_load_t(LoadOnnxClip));
    p_onnx->RegisterOpLoadMethod("Mul", op_load_t(LoadOnnxMul));
    p_onnx->RegisterOpLoadMethod("Div", op_load_t(LoadOnnxDiv));
    p_onnx->RegisterOpLoadMethod("Floor", op_load_t(LoadOnnxFloor));
    p_onnx->RegisterOpLoadMethod("Transpose", op_load_t(LoadOnnxTranspose));
    p_onnx->RegisterOpLoadMethod("Reshape", op_load_t(LoadOnnxReshape));
    p_onnx->RegisterOpLoadMethod("LeakyRelu", op_load_t(LoadOnnxLeakyReLu));
    p_onnx->RegisterOpLoadMethod("Transpose", op_load_t(LoadOnnxTranspose));
	p_onnx->RegisterOpLoadMethod("Slice", op_load_t(LoadOnnxSlice));
    p_onnx->RegisterOpLoadMethod("Sigmoid", op_load_t(LoadOnnxSigmod));
    p_onnx->RegisterOpLoadMethod("Split", op_load_t(LoadOnnxSplit));
    p_onnx->RegisterOpLoadMethod("Exp", op_load_t(LoadOnnxExp));
    p_onnx->RegisterOpLoadMethod("Sub", op_load_t(LoadOnnxSub));
    //p_onnx->RegisterOpLoadMethod("Constant", op_load_t(LoadOnnxConstant));
    return true;
}

}    // namespace TEngine
