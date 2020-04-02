#include <iostream>
#include <fstream>
#include <algorithm>

#include "tengine_c_api.h"
#include "exec_attr.hpp"
#include "tf_lite_serializer.hpp"
#include "logger.hpp"
#include "data_type.hpp"

#include "operator/conv_param.hpp"
#include "operator/pool_param.hpp"
#include "operator/concat_param.hpp"
#include "operator/reshape_param.hpp"
#include "operator/softmax_param.hpp"
#include "operator/detection_postprocess_param.hpp"
#include "operator/eltwise_param.hpp"
#include "flatbuffers/flexbuffers.h"
#include "operator/l2pool_param.hpp"
#include "operator/stridedslice_param.hpp"
#include "operator/log_softmax_param.hpp"
#include "operator/resize_param.hpp"
#include "operator/gather_param.hpp"
#include "operator/logical_param.hpp"
#include "operator/fc_param.hpp"
#include "operator/transpose.hpp"
#include "operator/reverse.hpp"
#include "operator/comparison_param.hpp"
#include "operator/depthtospace_param.hpp"
#include "operator/spacetodepth_param.hpp"
#include "operator/reduction_param.hpp"
#include "operator/squared_difference.hpp"
#include "operator/ceil.hpp"
#include "operator/round.hpp"
#include "operator/sparsetodense_param.hpp"

namespace TEngine {

using LiteNode = TFLiteSerializer::LiteNode;
using LiteTensor = TFLiteSerializer::LiteTensor;
using LiteGraph = TFLiteSerializer::LiteGraph;

using op_load_t = std::function<bool(LiteNode* node, LiteGraph* lite_graph, StaticGraph* graph)>;

bool TFLiteSerializer::LoadModel(const std::vector<std::string>& file_list, StaticGraph* graph)
{
    if(file_list.size() != GetFileNum())
        return false;

    std::ifstream input_file;

    input_file.open(file_list[0], std::ios::binary | std::ios::in);
    input_file.seekg(0, std::ios::end);

    int model_len = input_file.tellg();
    char* model_data = new char[model_len];

    input_file.seekg(0, std::ios::beg);
    input_file.read(model_data, model_len);
    input_file.close();

    SetGraphSource(graph, file_list[0]);
    SetGraphSourceFormat(graph, "tflite");
    SetGraphLayout(graph, TENGINE_LAYOUT_NHWC);
    SetModelLayout(graph, TENGINE_LAYOUT_NHWC);
    SetModelFormat(graph, MODEL_FORMAT_TFLITE);

    bool ret = LoadModelFromMem(model_data, model_len, graph);

    if(!ret)
        delete[] model_data;

    return ret;
}

bool TFLiteSerializer::LoadModelFromMem(char* mem_addr, int mem_size, StaticGraph* graph)
{
    ::flatbuffers::Verifier verifier(( const unsigned char* )mem_addr, mem_size);

    if(!::tflite::VerifyModelBuffer(verifier))
    {
        LOG_ERROR() << "bad tf lite model file\n";
        return false;
    }

    const LiteModel* lite_model = ::tflite::GetModel(mem_addr);

    if(!lite_model->subgraphs() || lite_model->subgraphs()->size() != 1)
    {
        LOG_ERROR() << "bad graph format\n";
        return false;
    }

    LiteGraph lite_graph;

    lite_graph.lite_model = lite_model;

    if(!ConstructGraph(lite_model, &lite_graph))
        return false;

    // DumpLiteGraph(&lite_graph);

    if(!OptimizeGraph(&lite_graph))
        return false;

    if(!GenerateStaticGraph(&lite_graph, graph))
        return false;

    return true;
}

bool TFLiteSerializer::ConstructGraph(const LiteModel* lite_model, LiteGraph* lite_graph)
{
    // load all tensors first

    auto tensors = (*lite_model->subgraphs())[0]->tensors();

    int i = 0;

    for(auto* tensor : *tensors)
    {
        LiteTensor* lite_tensor = new LiteTensor();

        lite_tensor->tf_tensor = tensor;
        lite_tensor->idx = i++;
        lite_tensor->name = tensor->name()->c_str();

        auto shape = tensor->shape();

        for(unsigned int i = 0; i < shape->Length(); ++i)
            lite_tensor->shape.push_back(shape->Get(i));

        int type = tensor->type();

        switch(type)
        {
            case ::tflite::TensorType_FLOAT32:
                lite_tensor->type = "FP32";
                break;
            case ::tflite::TensorType_UINT8:
                lite_tensor->type = "UINT8";
                break;
            case ::tflite::TensorType_INT32:
                lite_tensor->type = "INT32";
                break;
            default:
                lite_tensor->type = "unknown";
        }

        lite_graph->tensor_list.push_back(lite_tensor);
    }

    // load ops

    const auto ops = (*lite_model->subgraphs())[0]->operators();
    const auto opcodes = lite_model->operator_codes();

    i = 0;

    for(auto* op : *ops)
    {
        LiteNode* lite_node = new LiteNode();

        lite_node->lite_op = op;

        /* get op name */

        int op_code_idx = op->opcode_index();

        const auto* op_code = opcodes->Get(op_code_idx);

        if(op_code->builtin_code() == ::tflite::BuiltinOperator_CUSTOM)
            lite_node->op = op_code->custom_code()->c_str();
        else
            lite_node->op = EnumNameBuiltinOperator(op_code->builtin_code());

        /*inputs and outputs */
        auto inputs = op->inputs();

        for(unsigned int i = 0; i < inputs->Length(); i++)
        {
            auto input_idx = inputs->Get(i);

            if(input_idx != -1)
            {
                LiteTensor* lite_tensor = lite_graph->tensor_list.at(input_idx);
                lite_node->inputs.push_back(lite_tensor);
            }
            else
            {
                LiteTensor* lite_tensor = new LiteTensor();

                lite_tensor->name = "NoData";
                lite_tensor->idx = lite_graph->tensor_list.size();

                lite_graph->tensor_list.push_back(lite_tensor);

                lite_node->inputs.push_back(lite_tensor);
            }
        }

        auto outputs = op->outputs();

        for(unsigned int i = 0; i < outputs->Length(); i++)
        {
            auto output_idx = outputs->Get(i);
            LiteTensor* lite_tensor;

            if(output_idx != -1)
            {
                lite_tensor = lite_graph->tensor_list.at(output_idx);
                lite_node->outputs.push_back(lite_tensor);
            }
            else
            {
                lite_tensor = new LiteTensor();
                lite_node->outputs.push_back(lite_tensor);
            }

            lite_tensor->producer = lite_node;
        }

        lite_node->name = lite_node->outputs[0]->name;

        lite_graph->seq_nodes.push_back(lite_node);
    }

    // setup graph inputs/outputs
    auto inputs = (*lite_model->subgraphs())[0]->inputs();

    if(inputs)
    {
        for(int input : *inputs)
        {
            LiteTensor* tensor = lite_graph->tensor_list.at(input);
            lite_graph->input_tensors.push_back(tensor);
            tensor->graph_input = true;
        }
    }

    auto outputs = (*lite_model->subgraphs())[0]->outputs();

    if(outputs)
    {
        for(int output : *outputs)
        {
            LiteTensor* tensor = lite_graph->tensor_list.at(output);
            tensor->graph_output = true;
            lite_graph->output_tensors.push_back(tensor);
        }
    }

    return true;
}

bool TFLiteSerializer::OptimizeGraph(LiteGraph* lite_graph)
{
    return true;
}

bool TFLiteSerializer::LoadTensorScaleAndZero(StaticTensor* static_tensor, LiteTensor* lite_tensor)
{
    auto quantization = lite_tensor->tf_tensor->quantization();
    float scale = 1.f;
    int zero_point = 0;

    if(quantization->scale() && quantization->zero_point())
    {
        scale = quantization->scale()->Get(0);
        zero_point = quantization->zero_point()->Get(0);
    }

    static_tensor->scale.resize(0);
    static_tensor->zero_point.resize(0);
    static_tensor->scale.push_back(scale);
    static_tensor->zero_point.push_back(zero_point);

    return true;
}

bool TFLiteSerializer::LoadConstLiteTensor(StaticTensor* static_tensor, LiteTensor* tensor, LiteGraph* lite_graph,
                                           StaticGraph* graph)
{
    void* mem_buf;
    int shape_size = 1;
    int mem_size;
    const TFLiteTensor* tf_tensor = tensor->tf_tensor;

    auto* buffers = lite_graph->lite_model->buffers();
    int buf_idx = tf_tensor->buffer();

    auto* buffer = buffers->Get(buf_idx);
    auto* src_buf = buffer->data();

    for(unsigned int i = 0; i < tensor->shape.size(); i++)
        shape_size *= tensor->shape[i];

    int element_size = DataType::GetTypeSize(static_tensor->data_type);
    mem_size = shape_size * element_size;

    mem_buf = malloc(mem_size + 128);

    if(tensor->type == "UINT8")
    {
        const uint8_t* src_ptr = ( const uint8_t* )(src_buf->data());
        memcpy(mem_buf, src_ptr, mem_size);
    }
    else if(tensor->type == "INT32")
    {
        const int* src_ptr = ( const int* )src_buf->data();
        memcpy(mem_buf, src_ptr, mem_size);
    }
    else
    {
        const void* src_ptr = src_buf->data();
        memcpy(mem_buf, src_ptr, mem_size);
    }

    // DIM SWITCH WILL BE DELAYED to OP LOAD
    SetConstTensorBuffer(static_tensor, mem_buf);
    SetConstTensorFileLocation(static_tensor, -1, 0);

    StaticOp* op = CreateStaticOp(graph, "Const");
    StaticNode* node = CreateStaticNode(graph, tensor->name);

    SetNodeOp(node, op);

    AddNodeOutputTensor(node, static_tensor);

    return true;
}

bool TFLiteSerializer::LoadLiteTensor(LiteTensor* tensor, LiteGraph* lite_graph, StaticGraph* graph)
{
    StaticTensor* static_tensor;
    bool const_tensor = false;

    if(tensor->producer || tensor->graph_input)
    {
        static_tensor = CreateStaticTensor(graph, tensor->name);
    }
    else
    {
        const_tensor = true;
        static_tensor = CreateStaticConstTensor(graph, tensor->name);
    }
    int data_type;
    if(tensor->type == "UINT8")
        data_type = TENGINE_DT_UINT8;
    else if(tensor->type == "INT32")
        data_type = TENGINE_DT_INT32;
    else
    {
        data_type = TENGINE_DT_FP32;
    }
    SetTensorDataType(static_tensor, data_type);
    SetTensorDim(static_tensor, tensor->shape);

    LoadTensorScaleAndZero(static_tensor, tensor);

    tensor->static_tensor = static_tensor;

    // layout will be set during the op load

    // Load Const Tensor
    if(const_tensor)
        return LoadConstLiteTensor(static_tensor, tensor, lite_graph, graph);

    return true;
}

bool TFLiteSerializer::LoadLiteNode(LiteNode* node, LiteGraph* lite_graph, StaticGraph* graph)
{
    //if(!FindOpLoadMethod(node->op))
    //{
    //    LOG_ERROR() << "cannot find load method for op: " << node->op << "\n";
    //    return false;
   // }

    StaticNode* static_node = CreateStaticNode(graph, node->name);

    // handle input
    for(unsigned int i = 0; i < node->inputs.size(); i++)
    {
        LiteTensor* input = node->inputs.at(i);
        AddNodeInputTensor(static_node, input->static_tensor);
    }

    // handle output

    for(unsigned int i = 0; i < node->outputs.size(); i++)
    {
        LiteTensor* output = node->outputs.at(i);
        AddNodeOutputTensor(static_node, output->static_tensor);
    }

    // for each op, load the op
    op_load_t op_func = any_cast<op_load_t>(GetOpLoadMethod(node->op));

    node->static_node = static_node;

    if(!op_func(node, lite_graph, graph))
    {
        LOG_ERROR() << "failed to load node: " << node->name << " op: " << node->op << "\n";
        return false;
    }

    return true;
}

void TFLiteSerializer::CreateGraphInputNode(LiteTensor* tensor, StaticGraph* graph)
{
    StaticOp* op = CreateStaticOp(graph, "InputOp");
    StaticNode* node = CreateStaticNode(graph, tensor->name);

    SetNodeOp(node, op);

    AddNodeOutputTensor(node, tensor->static_tensor);

    AddGraphInputNode(graph, node);
}

bool TFLiteSerializer::GenerateStaticGraph(LiteGraph* lite_graph, StaticGraph* graph)
{
    // first load all tensor
    int tensor_number = lite_graph->tensor_list.size();

    for(int i = 0; i < tensor_number; i++)
    {
        LiteTensor* tensor = lite_graph->tensor_list.at(i);

        LoadLiteTensor(tensor, lite_graph, graph);
    }

    // create input node for graph_input tensor
    for(unsigned int i = 0; i < lite_graph->input_tensors.size(); i++)
    {
        LiteTensor* tensor = lite_graph->input_tensors.at(i);

        CreateGraphInputNode(tensor, graph);
    }

    // second load all nodes
    int node_number = lite_graph->seq_nodes.size();
    
	std::vector<std::string> no_supported_op;
    for(int i =0; i < node_number; i++)
    {   
        LiteNode* node = lite_graph->seq_nodes.at(i);
        if(!FindOpLoadMethod(node->op))
        {   
            auto it = find(no_supported_op.begin(),no_supported_op.end(),node->op);
            if(it == no_supported_op.end()) 
                no_supported_op.push_back(node->op);
        }
    }
    
	if(no_supported_op.size())
    {   
        LOG_ERROR() << "These " << no_supported_op.size() << "ops  are not supported\n";
        LOG_ERROR() << "{";
        for(int i = 0; i < (int)no_supported_op.size();i++)
        {   
            LOG_ERROR() << no_supported_op[i] << ",";
        }
        LOG_ERROR() << "}\n";
        return false;
    }
 

    for(int i = 0; i < node_number; i++)
    {
        LiteNode* node = lite_graph->seq_nodes.at(i);

        if(!LoadLiteNode(node, lite_graph, graph))
            return false;
    }

    return true;
}

void TFLiteSerializer::DumpLiteTensor(LiteTensor* tensor)
{
    std::cout << tensor->name << " " << tensor->type << " [";
    for(unsigned int i = 0; i < tensor->shape.size(); i++)
        std::cout << " " << tensor->shape[i];

    std::cout << "] ";

    if(!tensor->producer && !tensor->graph_input)
        std::cout << " Const ";
}

void TFLiteSerializer::DumpLiteGraph(LiteGraph* lite_graph)
{
    for(unsigned int i = 0; i < lite_graph->seq_nodes.size(); i++)
    {
        LiteNode* node = lite_graph->seq_nodes.at(i);

        std::cout << i << ":\t" << node->op << " \t" << node->name << "\n";
        std::cout << "\tInput: " << node->inputs.size() << " Output: " << node->outputs.size() << "\n";

        for(unsigned int j = 0; j < node->inputs.size(); j++)
        {
            LiteTensor* tensor = node->inputs[j];
            std::cout << "\t I" << j << ": ";
            DumpLiteTensor(tensor);
            std::cout << "\n";
        }

        for(unsigned int j = 0; j < node->outputs.size(); j++)
        {
            LiteTensor* tensor = node->outputs[j];
            std::cout << "\t O" << j << ": ";
            DumpLiteTensor(tensor);
            std::cout << "\n";
        }
    }
    std::cout << "\nGraph Inputs:\n";

    for(unsigned int i = 0; i < lite_graph->input_tensors.size(); i++)
    {
        LiteTensor* tensor = lite_graph->input_tensors.at(i);
        std::cout << "\t" << i << "\t" << tensor->name << "\n";
    }

    std::cout << "\nGraph Outputs:\n";

    for(unsigned int i = 0; i < lite_graph->output_tensors.size(); i++)
    {
        LiteTensor* tensor = lite_graph->output_tensors.at(i);
        std::cout << "\t" << i << "\t" << tensor->name << "\n";
    }
}

namespace tf_lite_serializer {

static void ExchangeNC(const std::vector<int>& shape, std::vector<int>& new_shape)
{
    new_shape.resize(4);

    new_shape[0] = shape[3];
    new_shape[1] = shape[1];
    new_shape[2] = shape[2];
    new_shape[3] = shape[0];
}

static bool LoadConv2D(LiteNode* node, LiteGraph* lite_graph, StaticGraph* graph)
{
    StaticNode* static_node = node->static_node;
    int kernel_h = 1, kernel_w = 1, output_channel = 1;
    LiteTensor* lite_tensor = node->inputs[1];

    output_channel = lite_tensor->shape[0];
    kernel_h = lite_tensor->shape[1];
    kernel_w = lite_tensor->shape[2];

    ConvParam param = any_cast<ConvParam>(OpManager::GetOpDefParam("Convolution"));
    const tflite::Conv2DOptions* lite_param = node->lite_op->builtin_options_as<tflite::Conv2DOptions>();

    int lite_activation = lite_param->fused_activation_function();
    switch(lite_activation)
    {
        case 0:
            param.activation = -1;
            break;
        case 1:
            param.activation = 0;
            break;
        case 2:
            param.activation = 1;
            break;
        case 3:
            param.activation = 6;
            break;
        default:
            param.activation = -4;
            break;
    }
    param.stride_h = lite_param->stride_h();
    param.stride_w = lite_param->stride_w();
    int padding = lite_param->padding();
    if(padding == 0)
    {
        param.pad_h0 = -1;
        param.pad_h1 = -1;
        param.pad_w0 = -1;
        param.pad_w1 = -1;
    }
    else
    {
        param.pad_h0 = 0;
        param.pad_h1 = 0;
        param.pad_w0 = 0;
        param.pad_w1 = 0;
    }
    param.dilation_h = 1;
    param.dilation_w = 1;
    param.group = 1;
    param.kernel_h = kernel_h;
    param.kernel_w = kernel_w;
    param.output_channel = output_channel;

    StaticOp* op = CreateStaticOp(graph, "Convolution");

    SetOperatorParam(op, param);
    SetNodeOp(static_node, op);

    // bias

    return true;
}

static bool LoadConv2DDepthwise(LiteNode* node, LiteGraph* lite_graph, StaticGraph* graph)
{
    StaticNode* static_node = node->static_node;
    int kernel_h = 1, kernel_w = 1, output_channel = 1;
    LiteTensor* lite_tensor = node->inputs[1];
    {
        output_channel = lite_tensor->static_tensor->dims[3];
        kernel_h = lite_tensor->static_tensor->dims[1];
        kernel_w = lite_tensor->static_tensor->dims[2];
    }
    ConvParam param = any_cast<ConvParam>(OpManager::GetOpDefParam("Convolution"));
    const tflite::DepthwiseConv2DOptions* lite_param =
        node->lite_op->builtin_options_as<tflite::DepthwiseConv2DOptions>();

    int lite_activation = lite_param->fused_activation_function();
    switch(lite_activation)
    {
        case 0:
            param.activation = -1;
            break;
        case 1:
            param.activation = 0;
            break;
        case 2:
            param.activation = 1;
            break;
        case 3:
            param.activation = 6;
            break;
        default:
            param.activation = -4;
            break;
    }

    param.stride_h = lite_param->stride_h();
    param.stride_w = lite_param->stride_w();
    param.group = output_channel / lite_param->depth_multiplier();
    int padding = lite_param->padding();
    if(padding == 0)
    {
        param.pad_h0 = -1;
        param.pad_h1 = -1;
        param.pad_w0 = -1;
        param.pad_w1 = -1;
    }
    else
    {
        param.pad_h0 = 0;
        param.pad_h1 = 0;
        param.pad_w0 = 0;
        param.pad_w1 = 0;
    }

    param.dilation_h = 1;
    param.dilation_w = 1;
    param.kernel_h = kernel_h;
    param.kernel_w = kernel_w;
    param.output_channel = output_channel;

    StaticOp* op = CreateStaticOp(graph, "Convolution");

    SetOperatorParam(op, param);
    SetNodeOp(static_node, op);

    std::vector<int> new_shape;
    ExchangeNC(node->inputs[1]->shape, new_shape);
    SetTensorDim(node->inputs[1]->static_tensor, new_shape);

    return true;
}

static bool LoadFullyConnected(LiteNode* node, LiteGraph* lite_graph, StaticGraph* graph)
{
    StaticNode* static_node = node->static_node;
    LiteTensor* lite_tensor = node->inputs[1];

    int M = lite_tensor->shape[0];

    FCParam param = any_cast<FCParam>(OpManager::GetOpDefParam("FullyConnected"));
    const tflite::FullyConnectedOptions* lite_param = node->lite_op->builtin_options_as<tflite::FullyConnectedOptions>();

    int lite_activation = lite_param->fused_activation_function();
    if(lite_activation != 0)
        return false;

    param.num_output = M;
    StaticOp* op = CreateStaticOp(graph, "FullyConnected");

    SetOperatorParam(op, param);
    SetNodeOp(static_node, op);

    return true;
}

static bool LoadPooling(LiteNode* node, LiteGraph* lite_graph, StaticGraph* graph)
{
    StaticNode* static_node = node->static_node;
    PoolParam param = any_cast<PoolParam>(OpManager::GetOpDefParam("Pooling"));
    const tflite::Pool2DOptions* lite_param = node->lite_op->builtin_options_as<tflite::Pool2DOptions>();

    param.kernel_h = lite_param->filter_height();
    param.kernel_w = lite_param->filter_width();

    param.stride_h = lite_param->stride_h();
    param.stride_w = lite_param->stride_w();

    if(lite_param->padding() == 0)
    {
        param.pad_h0 = -1;
        param.pad_h1 = -1;
        param.pad_w0 = -1;
        param.pad_w1 = -1;
    }
    else
    {
        param.pad_h0 = 0;
        param.pad_h1 = 0;
        param.pad_w0 = 0;
        param.pad_w1 = 0;
    }

    if(node->op == "AVERAGE_POOL_2D")
        param.alg = kPoolAvg;
    else if(node->op == "MAX_POOL_2D")
        param.alg = kPoolMax;

    StaticOp* op = CreateStaticOp(graph, "Pooling");
    SetOperatorParam(op, param);
    SetNodeOp(static_node, op);

    return true;
}

static bool LoadConcat(LiteNode* node, LiteGraph* lite_graph, StaticGraph* graph)
{
    StaticNode* static_node = node->static_node;
    ConcatParam param = any_cast<ConcatParam>(OpManager::GetOpDefParam("Concat"));
    const tflite::ConcatenationOptions* lite_param = node->lite_op->builtin_options_as<tflite::ConcatenationOptions>();
    int activation = lite_param->fused_activation_function();

    param.axis = lite_param->axis();

    StaticOp* op = CreateStaticOp(graph, "Concat");
    if(activation)
        AddOperatorAttr(op, "Activation", activation);
    SetOperatorParam(op, param);
    SetNodeOp(static_node, op);

    return true;
}

static bool LoadReshape(LiteNode* node, LiteGraph* lite_graph, StaticGraph* graph)
{
    StaticNode* static_node = node->static_node;
    ReshapeParam param = any_cast<ReshapeParam>(OpManager::GetOpDefParam("Reshape"));
    StaticTensor* output_tensor = node->outputs[0]->static_tensor;
    // const tflite::ReshapeOptions * lite_param =
    // node->lite_op->builtin_options_as<tflite::ReshapeOptions>();
    // set dims
    auto new_shape = output_tensor->dims;
    if(new_shape.size() == 6)
    {
        param.re_shape.push_back(new_shape[0]);
        param.re_shape.push_back(new_shape[1]);
        param.re_shape.push_back(new_shape[2]);
        param.re_shape.push_back(new_shape[3]);
        param.re_shape.push_back(new_shape[4]);
        param.re_shape.push_back(new_shape[5]);                        
    }
    else if(new_shape.size() == 5)
    {
        param.re_shape.push_back(new_shape[0]);
        param.re_shape.push_back(new_shape[1]);
        param.re_shape.push_back(new_shape[2]);
        param.re_shape.push_back(new_shape[3]);
        param.re_shape.push_back(new_shape[4]);
    }
    else if(new_shape.size() == 4)
    {
        param.re_shape.push_back(new_shape[0]);
        param.re_shape.push_back(new_shape[1]);
        param.re_shape.push_back(new_shape[2]);
        param.re_shape.push_back(new_shape[3]);                        
    }
    else if(new_shape.size() == 3)
    {
        param.re_shape.push_back(new_shape[0]);
        param.re_shape.push_back(new_shape[1]);
        param.re_shape.push_back(new_shape[2]);
    }
    else if(new_shape.size() == 2)
    {
        param.re_shape.push_back(new_shape[0]);
        param.re_shape.push_back(new_shape[1]);
    }
    else
        return false;

    StaticOp* op = CreateStaticOp(graph, "Reshape");
    SetOperatorParam(op, param);
    SetNodeOp(static_node, op);

    return true;

}

static bool LoadLogistic(LiteNode* node, LiteGraph* lite_graph, StaticGraph* graph)
{
    StaticNode* static_node = node->static_node;

    StaticOp* op = CreateStaticOp(graph, "Logistic");
    SetNodeOp(static_node, op);

    return true;
}

static bool LoadGather(LiteNode* node, LiteGraph* lite_graph, StaticGraph* graph)
{
    StaticNode* static_node = node->static_node;
    GatherParam param = any_cast<GatherParam>(OpManager::GetOpDefParam("Gather"));

    const tflite::GatherOptions* lite_param = node->lite_op->builtin_options_as<tflite::GatherOptions>();
    LiteTensor* lite_tensor = node->inputs[1];

    param.indices_num = lite_tensor->shape[0];
    param.axis = lite_param->axis();
    StaticOp* op = CreateStaticOp(graph, "Gather");
    SetOperatorParam(op, param);
    SetNodeOp(static_node, op);

    return true;
}

static bool LoadReverse(LiteNode* node, LiteGraph* lite_graph, StaticGraph* graph)
{
    StaticNode* static_node = node->static_node;

    StaticOp* op = CreateStaticOp(graph, "Reverse");
    SetNodeOp(static_node, op);
    return true;
}

static bool LoadSquaredDifference(LiteNode* node, LiteGraph* lite_graph, StaticGraph* graph)
{
    StaticNode* static_node = node->static_node;
    StaticOp* op = CreateStaticOp(graph, "SquaredDifference");
    SetNodeOp(static_node, op);
    return true;
}

static bool LoadCeil(LiteNode* node, LiteGraph* lite_graph, StaticGraph* graph)
{
    StaticNode* static_node = node->static_node;
    StaticOp* op = CreateStaticOp(graph, "Ceil");
    SetNodeOp(static_node, op);
    return true;
}

static bool LoadRound(LiteNode* node, LiteGraph* lite_graph, StaticGraph* graph)
{
    StaticNode* static_node = node->static_node;
    StaticOp* op = CreateStaticOp(graph, "Round");
    SetNodeOp(static_node, op);
    return true;
}
static bool LoadLogicalOr(LiteNode* node, LiteGraph* lite_graph, StaticGraph* graph)
{
    StaticNode* static_node = node->static_node;
    LogicalParam param = any_cast<LogicalParam>(OpManager::GetOpDefParam("Logical"));

    param.type = 1;
    StaticOp* op = CreateStaticOp(graph, "Logical");
    SetOperatorParam(op, param);
    SetNodeOp(static_node, op);

    return true;
}
static bool LoadLogicalAnd(LiteNode* node, LiteGraph* lite_graph, StaticGraph* graph)
{
    StaticNode* static_node = node->static_node;
    LogicalParam param = any_cast<LogicalParam>(OpManager::GetOpDefParam("Logical"));

    param.type = 0;
    StaticOp* op = CreateStaticOp(graph, "Logical");
    SetOperatorParam(op, param);
    SetNodeOp(static_node, op);

    return true;
}

static bool LoadSoftmax(LiteNode* node, LiteGraph* lite_graph, StaticGraph* graph)
{
    StaticNode* static_node = node->static_node;
    SoftmaxParam param = any_cast<SoftmaxParam>(OpManager::GetOpDefParam("Softmax"));

    param.axis = 3;
    StaticOp* op = CreateStaticOp(graph, "Softmax");
    SetOperatorParam(op, param);
    SetNodeOp(static_node, op);

    return true;
}

static bool LoadEltwise(LiteNode* node, LiteGraph* lite_graph, StaticGraph* graph)
{
    StaticNode* static_node = node->static_node;
    EltwiseParam param = any_cast<EltwiseParam>(OpManager::GetOpDefParam("Eltwise"));

    if(node->op == "ADD")
        param.type = ELT_SUM;
    else if(node->op == "SUB")
        param.type = ELT_SUB;
    else if(node->op == "PROD")
        param.type = ELT_PROD;
    else if(node->op == "RSQRT")
        param.type = ELT_RSQRT;
    else if(node->op == "DIV")
        param.type = ELT_DIV;
    else if(node->op == "LOG")
        param.type = ELT_LOG;
    else if(node->op == "EXP")
        param.type = ELT_EXP;
    else if(node->op == "POW")
        param.type = ELT_POW;
    else if(node->op == "SQRT")
        param.type = ELT_SQRT;
    else if(node->op == "FLOOR")
        param.type = ELT_FLOOR;
    else if(node->op == "MUL")
        param.type = ELT_PROD;        
    StaticOp* op = CreateStaticOp(graph, "Eltwise");
    SetOperatorParam(op, param);
    SetNodeOp(static_node, op);

    return true;
}

static bool LoadDetectionPostProcess(LiteNode* node, LiteGraph* lite_graph, StaticGraph* graph)
{
    StaticNode* static_node = node->static_node;

    DetectionPostProcessParam param =
        any_cast<DetectionPostProcessParam>(OpManager::GetOpDefParam("DetectionPostProcess"));
    const uint8_t* lite_buffer = node->lite_op->custom_options()->data();
    size_t lite_buffer_len = node->lite_op->custom_options()->size();

    const flexbuffers::Map& m = flexbuffers::GetRoot(lite_buffer, lite_buffer_len).AsMap();
    param.max_detections = m["max_detections"].AsInt32();
    param.max_classes_per_detection = m["max_classes_per_detection"].AsInt32();
    param.nms_score_threshold = m["nms_score_threshold"].AsFloat();
    param.nms_iou_threshold = m["nms_iou_threshold"].AsFloat();
    param.num_classes = m["num_classes"].AsInt32();
    param.scales.resize(4);
    param.scales[0] = m["y_scale"].AsFloat();
    param.scales[1] = m["x_scale"].AsFloat();
    param.scales[2] = m["h_scale"].AsFloat();
    param.scales[3] = m["w_scale"].AsFloat();

    StaticOp* op = CreateStaticOp(graph, "DetectionPostProcess");
    SetOperatorParam(op, param);
    SetOperatorDynamicShape(op);
    SetNodeOp(static_node, op);
    return true;
}

static bool LoadL2Normalization(LiteNode* node, LiteGraph* lite_graph, StaticGraph* graph)
{
    StaticNode* static_node = node->static_node;

    StaticOp* op = CreateStaticOp(graph, "L2Normalization");
    SetNodeOp(static_node, op);

    return true;
}

static bool LoadElu(LiteNode* node, LiteGraph* lite_graph, StaticGraph* graph)
{
    StaticNode* static_node = node->static_node;

    StaticOp* op = CreateStaticOp(graph, "Elu");
    SetNodeOp(static_node, op);
    return true;
}

static bool LoadReLU1(LiteNode* node, LiteGraph* lite_graph, StaticGraph* graph)
{
    StaticNode* static_node = node->static_node;

    StaticOp* op = CreateStaticOp(graph, "ReLU1");
    SetNodeOp(static_node, op);
    return true;
}

static bool LoadLogSoftmax(LiteNode* node, LiteGraph* lite_graph, StaticGraph* graph)
{
    StaticNode* static_node = node->static_node;

    LogSoftmaxParam param =
        any_cast<LogSoftmaxParam>(OpManager::GetOpDefParam("LogSoftmax"));

    param.axis=3;
    StaticOp* op = CreateStaticOp(graph, "LogSoftmax");
    SetOperatorParam(op, param);
    SetNodeOp(static_node, op);
    return true;
}

static bool LoadL2Pool(LiteNode* node, LiteGraph* lite_graph, StaticGraph* graph)
{
    StaticNode* static_node = node->static_node;
    L2PoolParam param = any_cast<L2PoolParam>(OpManager::GetOpDefParam("L2Pool"));
    const tflite::Pool2DOptions* lite_param = node->lite_op->builtin_options_as<tflite::Pool2DOptions>();

    param.kernel_h = lite_param->filter_height();
    param.kernel_w = lite_param->filter_width();

    param.stride_h = lite_param->stride_h();
    param.stride_w = lite_param->stride_w();
    if(0 == lite_param->padding())
    {
        param.padding = PaddingType::kSame;
    }
    else if(1 == lite_param->padding())
    {
        param.padding = PaddingType::kValid;
    }

    StaticOp* op = CreateStaticOp(graph, "L2Pool");
    SetOperatorParam(op, param);
    SetNodeOp(static_node, op);

    return true;
}
static bool LoadResizeNearestNeighbor(LiteNode* node, LiteGraph* lite_graph, StaticGraph* graph)
{
    StaticNode* static_node = node->static_node;
    ResizeParam param = any_cast<ResizeParam>(OpManager::GetOpDefParam("Resize"));
    // const tflite::ResizeNearestNeighborOptions* lite_param = node->lite_op->builtin_options_as<tflite::ResizeNearestNeighborOptions>();
    auto* buffers = lite_graph->lite_model->buffers();

    LiteTensor* in_tensor = node->inputs[0];

    int in_h= in_tensor->shape[0];
    int in_w= in_tensor->shape[1];
    LiteTensor* size_tensor = node->inputs[1];
    const TFLiteTensor* size_TFL_tensor = size_tensor->tf_tensor;

    int size_buf_idx = size_TFL_tensor->buffer();
    auto* size_buffer = buffers->Get(size_buf_idx);
    auto* size_buf = size_buffer->data();
    int* sizes = ( int* )(size_buf->data());
    // const tflite::ReshapeOptions * lite_param =
    // node->lite_op->builtin_options_as<tflite::ReshapeOptions>();
    // set dims
    param.scale_h=sizes[0]/in_h;
    param.scale_w=sizes[1]/in_w;
    param.type=0;

    StaticOp* op = CreateStaticOp(graph, "Resize");
    SetOperatorParam(op, param);
    SetNodeOp(static_node, op);

    return true;
}
static bool LoadStridedSlice(LiteNode* node, LiteGraph* lite_graph, StaticGraph* graph)
{
    StaticNode* static_node = node->static_node;
    const tflite::StridedSliceOptions* stridedslice_param = node->lite_op->builtin_options_as<tflite::StridedSliceOptions>();
    StridedSliceParam param = any_cast<StridedSliceParam>(OpManager::GetOpDefParam("StridedSlice"));
    
    auto* buffers = lite_graph->lite_model->buffers();

    param.begin_mask = stridedslice_param->begin_mask();
    param.end_mask = stridedslice_param->end_mask();
    param.ellipsis_mask = stridedslice_param->ellipsis_mask();
    param.new_axis_mask = stridedslice_param->new_axis_mask();
    param.shrink_axis_mask = stridedslice_param->shrink_axis_mask();
    LiteTensor* begins_tensor = node->inputs[1];
    LiteTensor* ends_tensor = node->inputs[2];
    LiteTensor* strides_tensor = node->inputs[3];
    const TFLiteTensor* begins_TFL_tensor = begins_tensor->tf_tensor;
    const TFLiteTensor* ends_TFL_tensor = ends_tensor->tf_tensor;
    const TFLiteTensor* strides_TFL_tensor = strides_tensor->tf_tensor;

    int begins_buf_idx = begins_TFL_tensor->buffer();
    int ends_buf_idx = ends_TFL_tensor->buffer();
    int strides_buf_idx = strides_TFL_tensor->buffer();

    auto* begins_buffer = buffers->Get(begins_buf_idx);
    auto* begins_buf = begins_buffer->data();
    auto* ends_buffer = buffers->Get(ends_buf_idx);
    auto* ends_buf = ends_buffer->data();
    auto* strideds_buffer = buffers->Get(strides_buf_idx);
    auto* strideds_buf = strideds_buffer->data();

    int* begins = ( int* )(begins_buf->data());
    int* ends = ( int* )(ends_buf->data());
    int* strides = ( int* )(strideds_buf->data());

    StaticTensor* inputs_tensor = node->inputs[0]->static_tensor;
    auto new_shape = inputs_tensor->dims;
    if(new_shape.size() == 4)
    {
        param.begin[0] = begins[0];
        param.end[0] = ends[0];
        param.stride[0] = strides[0];
        param.begin[1] = begins[1];
        param.end[1] = ends[1];
        param.stride[1] = strides[1];
        param.begin[2] = begins[2];
        param.end[2] = ends[2];
        param.stride[2] = strides[2];
        param.begin[3] = begins[3];
        param.end[3] = ends[3];
        param.stride[3] = strides[3];
    }
    else if(new_shape.size() == 3)
    {
        param.begin[0] = begins[0];
        param.end[0] = ends[0];
        param.stride[0] = strides[0];
        param.begin[1] = begins[1];
        param.end[1] = ends[1];
        param.stride[1] = strides[1];
        param.begin[2] = begins[2];
        param.end[2] = ends[2];
        param.stride[2] = strides[2];
        param.begin[3] = 0;
        param.end[3] = 0;
        param.stride[3] = 0;
    }else if(new_shape.size() == 2){
        param.begin[0] = 0;
        param.end[0] = 0;
        param.stride[0] = 0;
        param.begin[2] = begins[0];
        param.end[2] = ends[0];
        param.stride[2] = strides[0];
        param.begin[3] = begins[1];
        param.end[3] = ends[1];
        param.stride[3] = strides[1];
        param.begin[1] = 0;
        param.end[1] = 0;
        param.stride[1] = 0; 
    }

    StaticOp* op = CreateStaticOp(graph, "StridedSlice");
    SetOperatorParam(op, param);
    SetNodeOp(static_node, op);

    return true;
}

static bool LoadTranspose(LiteNode* node, LiteGraph* lite_graph, StaticGraph* graph)
{
    StaticNode* static_node = node->static_node;
    TransposeParam param = any_cast<TransposeParam>(OpManager::GetOpDefParam("Transpose"));
    StaticTensor* output_tensor = node->outputs[0]->static_tensor;

    auto new_shape = output_tensor->dims;

    if(new_shape.size() == 6){
        param.tr_shape.push_back(new_shape[0]);
        param.tr_shape.push_back(new_shape[1]);
        param.tr_shape.push_back(new_shape[2]);
        param.tr_shape.push_back(new_shape[3]);
        param.tr_shape.push_back(new_shape[4]);
        param.tr_shape.push_back(new_shape[5]);
    } else if(new_shape.size() == 5){
        param.tr_shape.push_back(new_shape[0]);
        param.tr_shape.push_back(new_shape[1]);
        param.tr_shape.push_back(new_shape[2]);
        param.tr_shape.push_back(new_shape[3]);
        param.tr_shape.push_back(new_shape[5]);
    } else if(new_shape.size() == 4){
        param.tr_shape.push_back(new_shape[0]);
        param.tr_shape.push_back(new_shape[1]);
        param.tr_shape.push_back(new_shape[2]);
        param.tr_shape.push_back(new_shape[3]);
    } else if(new_shape.size() == 3){
        param.tr_shape.push_back(new_shape[0]);
        param.tr_shape.push_back(new_shape[1]);
        param.tr_shape.push_back(new_shape[2]);
    } else if(new_shape.size() == 2){
        param.tr_shape.push_back(new_shape[0]);
        param.tr_shape.push_back(new_shape[1]);
    } else if(new_shape.size() == 1){
        param.tr_shape.push_back(new_shape[0]);
    }

    StaticOp* op = CreateStaticOp(graph, "Transpose");
    SetOperatorParam(op, param);
    SetNodeOp(static_node, op);

    return true;
}
static bool LoadComparison(LiteNode* node, LiteGraph* lite_graph, StaticGraph* graph)
{
    StaticNode* static_node = node->static_node;
    ComparisonParam param = any_cast<ComparisonParam>(OpManager::GetOpDefParam("Comparison"));
    if(node->op == "EQUAL")
        param.type = COMP_EQUAL;
    else if(node->op == "GREATER")
        param.type = COMP_GREATER;
    else if(node->op == "GREATER_EQUAL")
        param.type = COMP_GREATER_EQUAL;
    else if(node->op == "LESS")
        param.type = COMP_LESS;
    else if(node->op == "LESS_GREATER")
        param.type = COMP_LESS_EQUAL;
    StaticOp* op = CreateStaticOp(graph, "Comparison");
    SetOperatorParam(op, param);
    SetNodeOp(static_node, op);

    return true;
}

static bool LoadDepthToSpace(LiteNode* node, LiteGraph* lite_graph, StaticGraph* graph)
{
    StaticNode* static_node = node->static_node;
    DepthToSpaceParam param = any_cast<DepthToSpaceParam>(OpManager::GetOpDefParam("DepthToSpace"));
    //const tflite::SpaceToDepthOptions* spacetodepth_param = node->lite_op->builtin_options_as<tflite::SpaceToDepthOptions>();
 
    param.block_size = 2;

    StaticOp* op = CreateStaticOp(graph, "DepthToSpace");
    SetOperatorParam(op, param);
    SetNodeOp(static_node, op);

    return true;
}

static bool LoadSpaceToDepth(LiteNode* node, LiteGraph* lite_graph, StaticGraph* graph)
{
    StaticNode* static_node = node->static_node;
    SpaceToDepthParam param = any_cast<SpaceToDepthParam>(OpManager::GetOpDefParam("SpaceToDepth"));
    const tflite::SpaceToDepthOptions* spacetodepth_param = node->lite_op->builtin_options_as<tflite::SpaceToDepthOptions>();

    param.block_size = spacetodepth_param->block_size();

    StaticOp* op = CreateStaticOp(graph, "SpaceToDepth");
    SetOperatorParam(op, param);
    SetNodeOp(static_node, op);

    return true;
}

static bool LoadReduction(LiteNode* node, LiteGraph* lite_graph, StaticGraph* graph)
{
    StaticNode* static_node = node->static_node;
    ReductionParam param = any_cast<ReductionParam>(OpManager::GetOpDefParam("Reduction"));

    auto* buffers = lite_graph->lite_model->buffers();
    LiteTensor* in_tensor = node->inputs[1];
    int vector_size = in_tensor->shape[0];
    param.keepdim = vector_size;
    
    const TFLiteTensor* ends_TFL_tensor = in_tensor->tf_tensor;
    int tensor_idx = ends_TFL_tensor->buffer();
    auto* tensor_data = buffers->Get(tensor_idx);
    int* dataT = (int*)tensor_data->data();

    switch(vector_size){
        case 1:
            param.dim_0 = dataT[1];
            break;
        case 2:
            param.dim_0 = dataT[1];
            param.dim_1 = dataT[2];
            break;
        case 3:
            param.dim_0 = dataT[1];
            param.dim_1 = dataT[2];
            param.dim_2 = dataT[3];
            break;
        case 4:
            param.dim_0 = dataT[1];
            param.dim_1 = dataT[2];
            param.dim_2 = dataT[3];
            param.dim_3 = dataT[4];
            break;
        default:
            break;
    }
    
    if(node->op == "MEAN"){
        param.type = 1;
    }
    if(node->op == "SUM"){
        param.type = 0;
    }

    StaticOp* op = CreateStaticOp(graph, "Reduction");
    SetOperatorParam(op, param);
    SetNodeOp(static_node, op);
    return true;

}

static bool LoadSparseToDense(LiteNode* node, LiteGraph* lite_graph, StaticGraph* graph)
{
    StaticNode* static_node = node->static_node;
    SparseToDenseParam param = any_cast<SparseToDenseParam>(OpManager::GetOpDefParam("SparseToDense"));

    auto* buffers = lite_graph->lite_model->buffers();

    LiteTensor* default_value_tensor = node->inputs[3];

    const TFLiteTensor* default_value_TFL_tensor = default_value_tensor->tf_tensor;
    int default_value_buf_idx = default_value_TFL_tensor->buffer();
    auto* default_value_buffer = buffers->Get(default_value_buf_idx);
    auto* default_value_buf = default_value_buffer->data();
    int* default_value = ( int* )(default_value_buf->data());
    param.default_value = *default_value;

    LiteTensor* output_shape_tensor = node->inputs[1];
    const TFLiteTensor* output_shape_TFL_tensor = output_shape_tensor->tf_tensor;
    int output_shape_buf_idx = output_shape_TFL_tensor->buffer();
    auto* output_shape_buffer = buffers->Get(output_shape_buf_idx);
    auto* output_shape_buf = output_shape_buffer->data();
    int* output_shape = ( int* )(output_shape_buf->data());

    param.output_shape_size0 = output_shape[0];
    if(typeid(output_shape[1]) == typeid(int))
    {
        param.output_shape_size1 = output_shape[1];
    }
    StaticOp* op = CreateStaticOp(graph, "SparseToDense");
    SetOperatorParam(op, param);
    SetNodeOp(static_node, op);

    return true;
}

}    // namespace tf_lite_serializer

using namespace tf_lite_serializer;

bool TFLiteSerializerRegisterOpLoader(void)
{
    SerializerPtr serializer;

    if(!SerializerManager::SafeGet("tflite", serializer))
        return false;

    TFLiteSerializer* tf_lite = dynamic_cast<TFLiteSerializer*>(serializer.get());

    tf_lite->RegisterOpLoadMethod("CONV_2D", op_load_t(LoadConv2D));
    tf_lite->RegisterOpLoadMethod("AVERAGE_POOL_2D", op_load_t(LoadPooling));
    tf_lite->RegisterOpLoadMethod("MAX_POOL_2D", op_load_t(LoadPooling));
    tf_lite->RegisterOpLoadMethod("DEPTHWISE_CONV_2D", op_load_t(LoadConv2DDepthwise));
    tf_lite->RegisterOpLoadMethod("RESHAPE", op_load_t(LoadReshape));
    tf_lite->RegisterOpLoadMethod("SQUEEZE", op_load_t(LoadReshape));
    tf_lite->RegisterOpLoadMethod("CONCATENATION", op_load_t(LoadConcat));
    tf_lite->RegisterOpLoadMethod("LOGISTIC", op_load_t(LoadLogistic));
    tf_lite->RegisterOpLoadMethod("SOFTMAX", op_load_t(LoadSoftmax));
    tf_lite->RegisterOpLoadMethod("ADD", op_load_t(LoadEltwise));
    tf_lite->RegisterOpLoadMethod("TFLite_Detection_PostProcess", op_load_t(LoadDetectionPostProcess));
    tf_lite->RegisterOpLoadMethod("L2_NORMALIZATION", op_load_t(LoadL2Normalization));
    tf_lite->RegisterOpLoadMethod("L2_POOL_2D", op_load_t(LoadL2Pool));
    tf_lite->RegisterOpLoadMethod("ELU", op_load_t(LoadElu));
    tf_lite->RegisterOpLoadMethod("RELU_N1_TO_1", op_load_t(LoadReLU1));
    tf_lite->RegisterOpLoadMethod("STRIDED_SLICE", op_load_t(LoadStridedSlice));
    tf_lite->RegisterOpLoadMethod("LOG_SOFTMAX", op_load_t(LoadLogSoftmax));
    tf_lite->RegisterOpLoadMethod("RESIZE_NEAREST_NEIGHBOR", op_load_t(LoadResizeNearestNeighbor));
    tf_lite->RegisterOpLoadMethod("GATHER", op_load_t(LoadGather));
    tf_lite->RegisterOpLoadMethod("REVERSE_V2", op_load_t(LoadReverse));
    tf_lite->RegisterOpLoadMethod("LOGICALOR", op_load_t(LoadLogicalOr));
    tf_lite->RegisterOpLoadMethod("LOGICALAND", op_load_t(LoadLogicalAnd));
    tf_lite->RegisterOpLoadMethod("FULLY_CONNECTED", op_load_t(LoadFullyConnected));
    tf_lite->RegisterOpLoadMethod("TRANSPOSE", op_load_t(LoadTranspose));
    tf_lite->RegisterOpLoadMethod("DIV", op_load_t(LoadEltwise));
    tf_lite->RegisterOpLoadMethod("EQUAL", op_load_t(LoadComparison));
    tf_lite->RegisterOpLoadMethod("GREATER_EQUAL", op_load_t(LoadComparison));
    tf_lite->RegisterOpLoadMethod("GREATER", op_load_t(LoadComparison));
    tf_lite->RegisterOpLoadMethod("LESS", op_load_t(LoadComparison));
    tf_lite->RegisterOpLoadMethod("LESS_EQUAL", op_load_t(LoadComparison));
    tf_lite->RegisterOpLoadMethod("SPACE_TO_DEPTH", op_load_t(LoadSpaceToDepth));
    tf_lite->RegisterOpLoadMethod("DEPTH_TO_SPACE", op_load_t(LoadDepthToSpace));
    tf_lite->RegisterOpLoadMethod("MUL", op_load_t(LoadEltwise));   
    tf_lite->RegisterOpLoadMethod("MEAN", op_load_t(LoadReduction));
    tf_lite->RegisterOpLoadMethod("SUB", op_load_t(LoadEltwise)); 
    tf_lite->RegisterOpLoadMethod("SQUARED_DIFFERENCE", op_load_t(LoadSquaredDifference));
    tf_lite->RegisterOpLoadMethod("CEIL", op_load_t(LoadCeil));
    tf_lite->RegisterOpLoadMethod("ROUND", op_load_t(LoadRound));
    tf_lite->RegisterOpLoadMethod("SPARSE_TO_DENSE", op_load_t(LoadSparseToDense));
    return true;
}

}    // namespace TEngine
