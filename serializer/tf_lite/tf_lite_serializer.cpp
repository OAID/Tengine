#include <iostream>
#include <fstream>

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
    SetGraphLayout(graph,TENGINE_LAYOUT_NHWC);
    SetModelLayout(graph,TENGINE_LAYOUT_NHWC);
    SetModelFormat(graph,MODEL_FORMAT_TFLITE);

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
    static_tensor->scale = scale;
    static_tensor->zero_point = zero_point;

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

    mem_buf = malloc(mem_size);

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
    if(!FindOpLoadMethod(node->op))
    {
        LOG_ERROR() << "cannot find load method for op: " << node->op << "\n";
        return false;
    }

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
    if(new_shape.size() == 4)
    {
        param.dim_0 = new_shape[0];
        param.dim_1 = new_shape[1];
        param.dim_2 = new_shape[2];
        param.dim_3 = new_shape[3];
    }
    else if(new_shape.size() == 3)
    {
        param.dim_0 = new_shape[0];
        param.dim_1 = new_shape[1];
        param.dim_2 = new_shape[2];
    }
    else if(new_shape.size() == 2)
    {
        param.dim_0 = new_shape[0];
        param.dim_1 = new_shape[1];
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

static bool LoadSoftmax(LiteNode* node, LiteGraph* lite_graph, StaticGraph* graph)
{
    StaticNode* static_node = node->static_node;
    SoftmaxParam param = any_cast<SoftmaxParam>(OpManager::GetOpDefParam("Softmax"));

    param.axis = 1;
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

    return true;
}

}    // namespace TEngine
