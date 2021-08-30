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
 * Copyright (c) 2021, OPEN AI LAB
 * Author: qwang02@openailab.com
 */

#include "tflite2tengine.hpp"

const int OP_VERSION = 1;
static int op_set;

static ir_tensor_t* find_tensor(ir_graph_t* graph, const std::string& tensor_name)
{
    for (uint16_t i = 0; i < graph->tensor_num; i++)
    {
        ir_tensor_t* tensor = get_ir_graph_tensor(graph, i);
        if (tensor->name == tensor_name)
        {
            return tensor;
        }
    }
    return nullptr;
}

int tflite_serializer::set_graph_output(ir_graph_t* graph)
{
    std::vector<int16_t> output_nodes;
    for (int i = 0; i < graph->tensor_num; i++)
    {
        ir_tensor_t* tensor = get_ir_graph_tensor(graph, i);
        if (tensor->tensor_type == TENSOR_TYPE_VAR && tensor->consumer_num == 0)
        {
            output_nodes.push_back(tensor->producer);
        }
    }

    set_ir_graph_output_node(graph, output_nodes.data(), output_nodes.size());
    return 0;
}

bool tflite_serializer::find_op_load_method(const std::string& op_name)
{
    if (op_load_map.count(op_name))
        return true;

    return false;
}

bool tflite_serializer::construct_graph(const LiteModel* lite_model, LiteGraph_t* lite_graph)
{
    // load all tensors first

    auto tensors = (*lite_model->subgraphs())[0]->tensors();

    int i = 0;

    for (auto* tensor : *tensors)
    {
        LiteTensor_t* lite_tensor = new LiteTensor_t();

        lite_tensor->tf_tensor = tensor;
        lite_tensor->idx = i++;
        lite_tensor->name = tensor->name()->c_str();

        auto shape = tensor->shape();

        for (unsigned int i = 0; i < shape->Length(); ++i)
            lite_tensor->shape.push_back(shape->Get(i));

        int type = tensor->type();

        switch (type)
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

    for (auto* op : *ops)
    {
        LiteNode* lite_node = new LiteNode();

        lite_node->lite_op = op;

        /* get op name */

        int op_code_idx = op->opcode_index();

        const auto* op_code = opcodes->Get(op_code_idx);

        if (op_code->builtin_code() == ::tflite::BuiltinOperator_CUSTOM)
            lite_node->op = op_code->custom_code()->c_str();
        else
            lite_node->op = EnumNameBuiltinOperator(op_code->builtin_code());

        /*inputs and outputs */
        auto inputs = op->inputs();

        for (unsigned int i = 0; i < inputs->Length(); i++)
        {
            auto input_idx = inputs->Get(i);

            if (input_idx != -1)
            {
                LiteTensor_t* lite_tensor = lite_graph->tensor_list.at(input_idx);
                lite_node->inputs.push_back(lite_tensor);
            }
            else
            {
                LiteTensor_t* lite_tensor = new LiteTensor_t();

                lite_tensor->name = "NoData";
                lite_tensor->idx = lite_graph->tensor_list.size();

                lite_graph->tensor_list.push_back(lite_tensor);

                lite_node->inputs.push_back(lite_tensor);
            }
        }

        auto outputs = op->outputs();

        for (unsigned int i = 0; i < outputs->Length(); i++)
        {
            auto output_idx = outputs->Get(i);
            LiteTensor_t* lite_tensor;

            if (output_idx != -1)
            {
                lite_tensor = lite_graph->tensor_list.at(output_idx);
                lite_node->outputs.push_back(lite_tensor);
            }
            else
            {
                lite_tensor = new LiteTensor_t();
                lite_node->outputs.push_back(lite_tensor);
            }

            lite_tensor->producer = lite_node;
        }

        lite_node->name = lite_node->outputs[0]->name;

        lite_graph->seq_nodes.push_back(lite_node);
    }

    // setup graph inputs/outputs
    auto inputs = (*lite_model->subgraphs())[0]->inputs();

    if (inputs)
    {
        for (int input : *inputs)
        {
            LiteTensor_t* tensor = lite_graph->tensor_list.at(input);
            lite_graph->input_tensors.push_back(tensor);
            tensor->graph_input = true;
        }
    }

    auto outputs = (*lite_model->subgraphs())[0]->outputs();

    if (outputs)
    {
        for (int output : *outputs)
        {
            LiteTensor_t* tensor = lite_graph->tensor_list.at(output);
            tensor->graph_output = true;
            lite_graph->output_tensors.push_back(tensor);
        }
    }

    return true;
}

bool tflite_serializer::optimize_graph(LiteGraph_t* lite_graph)
{
    return true;
}

int tflite_serializer::get_lite_tensor_data_type(const std::string type)
{
    int tensor_data_type = -1;
    if (type == "UINT8")
        tensor_data_type = TENGINE_DT_UINT8;
    else if (type == "INT32")
        tensor_data_type = TENGINE_DT_INT32;
    else
        tensor_data_type = TENGINE_DT_FP32;
    return tensor_data_type;
}

int tflite_serializer::load_tensor_scale_and_zero(ir_tensor_t* ir_tensor, LiteTensor_t* lite_tensor)
{
    auto quantization = lite_tensor->tf_tensor->quantization();
    float scale = 1.f;
    int zero_point = 0;

    if (quantization->scale() && quantization->zero_point())
    {
        scale = quantization->scale()->Get(0);
        zero_point = quantization->zero_point()->Get(0);
    }

    ir_tensor->quant_param_num = 1;
    ir_tensor->scale = scale;
    ir_tensor->zero_point = zero_point;

    return 0;
}

int tflite_serializer::load_lite_tensor(ir_graph_t* graph, LiteGraph_t* lite_graph)
{
    // first load all lite_tensor
    int tensor_number = lite_graph->tensor_list.size();
    for (int i = 0; i < tensor_number; i++)
    {
        LiteTensor_t* lite_tensor = lite_graph->tensor_list.at(i);

        int data_type = get_lite_tensor_data_type(lite_tensor->type);
        if (data_type < 0)
        {
            return -1;
        }
        if (lite_tensor->producer || lite_tensor->graph_input)
        {
            continue;
        }

        ir_tensor_t* ir_tensor = create_ir_tensor(graph, lite_tensor->name.c_str(), data_type);
        if (ir_tensor == NULL)
        {
            fprintf(stderr, "create ir tensor failed!\n");
            return -1;
        }

        set_ir_tensor_shape(ir_tensor, lite_tensor->shape.data(), lite_tensor->shape.size());
        ir_tensor->tensor_type = TENSOR_TYPE_CONST;
        load_tensor_scale_and_zero(ir_tensor, lite_tensor);

        void* mem_buf;
        int shape_size = 1;
        int mem_size;
        const TFLiteTensor* tf_tensor = lite_tensor->tf_tensor;

        auto* buffers = lite_graph->lite_model->buffers();
        int buf_idx = tf_tensor->buffer();

        auto* buffer = buffers->Get(buf_idx);
        auto* src_buf = buffer->data();

        for (int j = 0; j < lite_tensor->shape.size(); j++)
            shape_size *= lite_tensor->shape[j];

        int element_size = ir_tensor->elem_size;
        mem_size = shape_size * element_size;

        mem_buf = malloc(mem_size + 128);

        if (data_type == TENGINE_DT_UINT8)
        {
            const uint8_t* src_ptr = (const uint8_t*)(src_buf->data());
            memcpy(mem_buf, src_ptr, mem_size);
        }
        else if (data_type == TENGINE_DT_INT32)
        {
            const int* src_ptr = (const int*)src_buf->data();
            memcpy(mem_buf, src_ptr, mem_size);
        }
        else if (data_type == TENGINE_DT_FP32)
        {
            const float* src_ptr = (const float*)src_buf->data();
            memcpy(mem_buf, src_ptr, mem_size);
        }
        else
        {
            const void* src_ptr = src_buf->data();
            memcpy(mem_buf, src_ptr, mem_size);
        }
        ir_tensor->data = mem_buf;
        ir_node_t* ir_node = create_ir_node(graph, lite_tensor->name.c_str(), OP_CONST, OP_VERSION);
        set_ir_node_output_tensor(ir_node, 0, ir_tensor);
    }
    return 0;
}

int tflite_serializer::set_graph_input(ir_graph_t* graph, LiteGraph_t* lite_graph)
{
    std::vector<int16_t> input_nodes;
    int tensor_number = lite_graph->tensor_list.size();
    for (int i = 0; i < tensor_number; i++)
    {
        LiteTensor_t* lite_tensor = lite_graph->tensor_list.at(i);
        int data_type = get_lite_tensor_data_type(lite_tensor->type);
        if (data_type < 0)
        {
            return -1;
        }
        if (!lite_tensor->graph_input)
        {
            continue;
        }
        ir_tensor_t* ir_tensor = create_ir_tensor(graph, lite_tensor->name.c_str(), data_type);
        set_ir_tensor_shape(ir_tensor, lite_tensor->shape.data(), lite_tensor->shape.size());
        ir_tensor->tensor_type = TENSOR_TYPE_INPUT;
        load_tensor_scale_and_zero(ir_tensor, lite_tensor);

        ir_node_t* ir_node = create_ir_node(graph, lite_tensor->name.c_str(), OP_INPUT, OP_VERSION);
        set_ir_node_output_tensor(ir_node, 0, ir_tensor);
        input_nodes.push_back(ir_node->index);
    }
    std::vector<int16_t> node_idx;
    for (int i = 0; i < input_nodes.size(); i++)
    {
        node_idx.push_back(input_nodes[i]);
    }
    set_ir_graph_input_node(graph, node_idx.data(), input_nodes.size());
    return 0;
}

int tflite_serializer::load_graph_node(ir_graph_t* graph, LiteGraph_t* lite_graph)
{
    // second load all nodes
    int node_number = lite_graph->seq_nodes.size();

    std::vector<std::string> no_supported_op;
    for (int i = 0; i < node_number; i++)
    {
        LiteNode* node = lite_graph->seq_nodes.at(i);
        if (!find_op_load_method(node->op))
        {
            unsupport_op.push_back(node->op);
        }
        else
        {
            support_op.push_back(node->op);
        }
    }

    if (unsupport_op.size())
    {
        for (int i = 0; i < (int)unsupport_op.size(); i++)
        {
            fprintf(stderr, "unsupport_op[%d]\n", i);
        }
        return -1;
    }
    for (int i = 0; i < node_number; i++)
    {
        LiteNode* node = lite_graph->seq_nodes.at(i);
        if (node->op == "null" || node->op == "_zeros")
            continue;

        std::vector<std::string>::iterator iter = std::find(support_op.begin(), support_op.end(), node->op);
        if (iter == support_op.end())
        {
            std::vector<std::string>::iterator uniter = std::find(unsupport_op.begin(), unsupport_op.end(), node->op);
            if (uniter == unsupport_op.end())
            {
                unsupport_op.push_back(node->op);
            }
            else
            {
                continue;
            }
        }
        else
        {
            continue;
        }
    }
    if (unsupport_op.size() != 0)
    {
        printf("These ops are not in tflite serializer: \n");
        for (int i = 0; i < (int)unsupport_op.size(); i++)
        {
            printf("[ %s ]\n", unsupport_op[i].c_str());
        }
        printf("\n");
        return -1;
    }
    for (int i = 0; i < node_number; i++)
    {
        LiteNode_t* lite_node = lite_graph->seq_nodes.at(i);
        const std::string op_name = lite_node->op;
        ir_node_t* ir_node = create_ir_node(graph, lite_node->name.c_str(), op_load_map[op_name].first, OP_VERSION);
        if (ir_node == NULL)
        {
            return -1;
        }

        // handle input
        for (int i = 0; i < lite_node->inputs.size(); i++)
        {
            LiteTensor_t* input = lite_node->inputs.at(i);
            int tensor_id = get_ir_tensor_index_from_name(graph, input->name.c_str());
            ir_tensor_t* tensor = get_ir_graph_tensor(graph, tensor_id);
            set_ir_node_input_tensor(ir_node, i, tensor);
        }

        // handle output

        for (int i = 0; i < lite_node->outputs.size(); i++)
        {
            LiteTensor_t* lite_tensor = lite_node->outputs.at(i);
            ir_tensor_t* ir_tensor = create_ir_tensor(graph, lite_tensor->name.c_str(), TENGINE_DT_UINT8);
            set_ir_node_output_tensor(ir_node, i, ir_tensor);
            load_tensor_scale_and_zero(ir_tensor, lite_tensor);
        }

        // for each op, load the op
        op_load_t loader = op_load_map[op_name].second;

        if (loader(graph, ir_node, lite_node) != 0)
        {
            fprintf(stderr, "failed to load node: %s, op : %s\n", lite_node->name.c_str(), lite_node->op.c_str());
            return -1;
        }
    }
    return 0;
}

bool tflite_serializer::load_model_from_mem(char* mem_addr, int mem_size, ir_graph_t* graph)
{
    ::flatbuffers::Verifier verifier((const unsigned char*)mem_addr, mem_size);
    if (!::tflite::VerifyModelBuffer(verifier))
    {
        return false;
    }

    const LiteModel* lite_model = ::tflite::GetModel(mem_addr);

    if (!lite_model->subgraphs() || lite_model->subgraphs()->size() != 1)
    {
        return false;
    }

    LiteGraph_t lite_graph;

    lite_graph.lite_model = lite_model;

    if (!construct_graph(lite_model, &lite_graph))
        return false;

    if (!optimize_graph(&lite_graph))
        return false;

    if (load_lite_tensor(graph, &lite_graph) < 0)
        return false;
    if (set_graph_input(graph, &lite_graph) < 0)
        return false;
    if (load_graph_node(graph, &lite_graph) < 0)
        return false;
    if (set_graph_output(graph) < 0)
        return false;

    return true;
}

int tflite_serializer::load_model(ir_graph_t* graph, std::string model_file)
{
    register_op_load();

    std::ifstream input_file;

    input_file.open(model_file, std::ios::binary | std::ios::in);
    input_file.seekg(0, std::ios::end);

    int model_len = input_file.tellg();
    char* model_data = new char[model_len];

    input_file.seekg(0, std::ios::beg);
    input_file.read(model_data, model_len);
    input_file.close();

    if (load_model_from_mem(model_data, model_len, graph) < 0)
        return -1;

    graph->model_format = MODEL_FORMAT_TFLITE;
    graph->graph_layout = TENGINE_LAYOUT_NCHW;
    graph->model_layout = TENGINE_LAYOUT_NHWC;
    return 0;
}

graph_t tflite_serializer::tflite2tengine(std::string model_file)
{
    fprintf(stderr, "----------tflite2tengine begin----------\n");
    context_t context = create_context(NULL, 1);
    ir_graph_t* ir_graph = create_ir_graph((struct context*)context);
    if (ir_graph == NULL)
    {
        destroy_context(context);
        return NULL;
    }
    ir_graph->attribute->private_context = 1; // new context

    int ret = load_model(ir_graph, model_file);
    if (0 != ret)
    {
        destroy_graph(ir_graph);
        return NULL;
    }
    ir_graph->device = find_default_device();

    fprintf(stderr, "----------tflite2tengine done.----------\n");
    return ir_graph;
}

static int LoadConv2D(ir_graph_t* graph, ir_node_t* ir_node, LiteNode_t* lite_node)
{
    int kernel_h = 1, kernel_w = 1, output_channel = 1, input_channel = 1;
    LiteTensor_t* lite_tensor = lite_node->inputs[1];

    output_channel = lite_tensor->shape[0];
    kernel_h = lite_tensor->shape[1];
    kernel_w = lite_tensor->shape[2];
    input_channel = lite_tensor->shape[3];

    struct conv_param* param = (struct conv_param*)ir_node->op.param_mem;
    const tflite::Conv2DOptions* lite_param = lite_node->lite_op->builtin_options_as<tflite::Conv2DOptions>();

    int lite_activation = lite_param->fused_activation_function();
    switch (lite_activation)
    {
    case 0:
        param->activation = -1;
        break;
    case 1:
        param->activation = 0;
        break;
    case 2:
        param->activation = 1;
        break;
    case 3:
        param->activation = 6;
        break;
    default:
        param->activation = -4;
        break;
    }
    param->stride_h = lite_param->stride_h();
    param->stride_w = lite_param->stride_w();
    int padding = lite_param->padding();
    if (padding == 0)
    {
        param->pad_h0 = -1;
        param->pad_h1 = -1;
        param->pad_w0 = -1;
        param->pad_w1 = -1;
    }
    else
    {
        param->pad_h0 = 0;
        param->pad_h1 = 0;
        param->pad_w0 = 0;
        param->pad_w1 = 0;
    }
    param->dilation_h = 1;
    param->dilation_w = 1;
    param->group = 1;
    param->kernel_h = kernel_h;
    param->kernel_w = kernel_w;
    param->output_channel = output_channel;
    param->input_channel = input_channel;

    return 0;
}

static int LoadConv2DDepthwise(ir_graph_t* graph, ir_node_t* ir_node, LiteNode_t* lite_node)
{
    int kernel_h = 1, kernel_w = 1, output_channel = 1, input_channel = 1;
    LiteTensor_t* lite_tensor = lite_node->inputs[1];

    output_channel = lite_tensor->shape[3];
    kernel_h = lite_tensor->shape[1];
    kernel_w = lite_tensor->shape[2];
    input_channel = lite_tensor->shape[0];

    struct conv_param* param = (struct conv_param*)ir_node->op.param_mem;
    const tflite::DepthwiseConv2DOptions* lite_param = lite_node->lite_op->builtin_options_as<tflite::DepthwiseConv2DOptions>();

    int lite_activation = lite_param->fused_activation_function();
    switch (lite_activation)
    {
    case 0:
        param->activation = -1;
        break;
    case 1:
        param->activation = 0;
        break;
    case 2:
        param->activation = 1;
        break;
    case 3:
        param->activation = 6;
        break;
    default:
        param->activation = -4;
        break;
    }

    param->stride_h = lite_param->stride_h();
    param->stride_w = lite_param->stride_w();
    param->group = output_channel / lite_param->depth_multiplier();
    int padding = lite_param->padding();
    if (padding == 0)
    {
        param->pad_h0 = -1;
        param->pad_h1 = -1;
        param->pad_w0 = -1;
        param->pad_w1 = -1;
    }
    else
    {
        param->pad_h0 = 0;
        param->pad_h1 = 0;
        param->pad_w0 = 0;
        param->pad_w1 = 0;
    }

    param->dilation_h = 1;
    param->dilation_w = 1;
    param->kernel_h = kernel_h;
    param->kernel_w = kernel_w;
    param->output_channel = output_channel;
    param->input_channel = input_channel;

    int dims[4] = {1};
    dims[0] = lite_tensor->shape[3];
    dims[1] = lite_tensor->shape[1];
    dims[2] = lite_tensor->shape[2];
    dims[3] = lite_tensor->shape[0];

    struct tensor* weight = get_ir_graph_tensor(graph, ir_node->input_tensors[1]);
    set_ir_tensor_shape(weight, dims, 4);

    return 0;
}

static int LoadPooling(ir_graph_t* graph, ir_node_t* ir_node, LiteNode_t* lite_node)
{
    struct pool_param* pool_param = (struct pool_param*)ir_node->op.param_mem;
    const tflite::Pool2DOptions* lite_param = lite_node->lite_op->builtin_options_as<tflite::Pool2DOptions>();

    pool_param->kernel_h = lite_param->filter_height();
    pool_param->kernel_w = lite_param->filter_width();

    pool_param->stride_h = lite_param->stride_h();
    pool_param->stride_w = lite_param->stride_w();

    if (lite_param->padding() == 0)
    {
        pool_param->pad_h0 = -1;
        pool_param->pad_h1 = -1;
        pool_param->pad_w0 = -1;
        pool_param->pad_w1 = -1;
    }
    else
    {
        pool_param->pad_h0 = 0;
        pool_param->pad_h1 = 0;
        pool_param->pad_w0 = 0;
        pool_param->pad_w1 = 0;
    }

    if (lite_node->op == "AVERAGE_POOL_2D")
        pool_param->pool_method = POOL_AVG;
    else if (lite_node->op == "MAX_POOL_2D")
        pool_param->pool_method = POOL_MAX;

    pool_param->pad_h0_org = pool_param->pad_h0;
    pool_param->pad_h1_org = pool_param->pad_h1;
    pool_param->pad_w0_org = pool_param->pad_w0;
    pool_param->pad_w1_org = pool_param->pad_w1;

    return 0;
}

static int LoadReshape(ir_graph_t* graph, ir_node_t* ir_node, LiteNode_t* lite_node)
{
    struct reshape_param* reshape_param = (struct reshape_param*)ir_node->op.param_mem;
    ir_tensor_t* shape_tensor = find_tensor(graph, lite_node->inputs[1]->name);
    if (shape_tensor == nullptr)
    {
        fprintf(stderr, "find shape tensor of reshape node failed.\n");
        return -1;
    }
    reshape_param->is_onnx = 0;
    int size = shape_tensor->elem_num;
    reshape_param->dim_size = size;
    reshape_param->re_shape = (int*)sys_malloc(sizeof(int) * size);

    int* data = (int*)shape_tensor->data;
    for (int i = 0; i < size; i++)
    {
        reshape_param->re_shape[i] = data[i];
    }

    return 0;
}

static int LoadSoftmax(ir_graph_t* graph, ir_node_t* ir_node, LiteNode_t* lite_node)
{
    struct softmax_param* softmax_param = (struct softmax_param*)ir_node->op.param_mem;

    softmax_param->axis = 1; // default

    return 0;
}

void tflite_serializer::register_op_load(void)
{
    op_load_map["CONV_2D"] = std::pair<int, op_load_t>(OP_CONV, LoadConv2D);
    op_load_map["AVERAGE_POOL_2D"] = std::pair<int, op_load_t>(OP_POOL, LoadPooling);
    op_load_map["MAX_POOL_2D"] = std::pair<int, op_load_t>(OP_POOL, LoadPooling);
    op_load_map["DEPTHWISE_CONV_2D"] = std::pair<int, op_load_t>(OP_CONV, LoadConv2DDepthwise);
    op_load_map["RESHAPE"] = std::pair<int, op_load_t>(OP_RESHAPE, LoadReshape);
    //     op_load_map["SQUEEZE"] = std::pair<int, op_load_t>(OP_SQUEEZE, LoadReshape);
    //     op_load_map["CONCATENATION"] = std::pair<int, op_load_t>(OP_CONCAT, LoadConcat);
    //     op_load_map["LOGISTIC"] = std::pair<int, op_load_t>(OP_LOGISTIC, LoadLogistic);
    op_load_map["SOFTMAX"] = std::pair<int, op_load_t>(OP_SOFTMAX, LoadSoftmax);
    //     op_load_map["ADD"] = std::pair<int, op_load_t>(OP_ELTWISE, LoadEltwise);
    //     // op_load_map["TFLite_Detection_PostProcess"] = std::pair<int, op_load_t>(LoadDetectionPostProcess);
    //     op_load_map["L2_NORMALIZATION"] = std::pair<int, op_load_t>(OP_L2NORMALIZATION, LoadL2Normalization);
    //     // op_load_map["L2_POOL_2D"] = std::pair<int, op_load_t>(OP_L2POOL, LoadL2Pool);
    //     op_load_map["ELU"] = std::pair<int, op_load_t>(OP_ELU, LoadElu);
    //     // op_load_map["RELU_N1_TO_1"] = std::pair<int, op_load_t>(LoadReLU1);
    //     op_load_map["STRIDED_SLICE"] = std::pair<int, op_load_t>(OP_STRIDED_SLICE, LoadStridedSlice);
    //     op_load_map["LOG_SOFTMAX"] = std::pair<int, op_load_t>(OP_LOGSOFTMAX, LoadLogSoftmax);
    //     // op_load_map["RESIZE_NEAREST_NEIGHBOR"] = std::pair<int, op_load_t>(LoadResizeNearestNeighbor);
    //     op_load_map["GATHER"] = std::pair<int, op_load_t>(OP_GATHER, LoadGather);
    //     op_load_map["REVERSE_V2"] = std::pair<int, op_load_t>(OP_REVERSE, LoadReverse);
    //     // op_load_map["LOGICALOR"] = std::pair<int, op_load_t>(LoadLogicalOr);
    //     // op_load_map["LOGICALAND"] = std::pair<int, op_load_t>(LoadLogicalAnd);
    //     op_load_map["FULLY_CONNECTED"] = std::pair<int, op_load_t>(OP_FC, LoadFullyConnected);
    //     op_load_map["TRANSPOSE"] = std::pair<int, op_load_t>(OP_TRANSPOSE, LoadTranspose);
    //     op_load_map["DIV"] = std::pair<int, op_load_t>(OP_ELTWISE, LoadEltwise);
    //     // op_load_map["EQUAL"] = std::pair<int, op_load_t>(LoadComparison);
    //     // op_load_map["GREATER_EQUAL"] = std::pair<int, op_load_t>(LoadComparison);
    //     // op_load_map["GREATER"] = std::pair<int, op_load_t>(LoadComparison);
    //     // op_load_map["LESS"] = std::pair<int, op_load_t>(LoadComparison);
    //     // op_load_map["LESS_EQUAL"] = std::pair<int, op_load_t>(LoadComparison);
    //     op_load_map["SPACE_TO_DEPTH"] = std::pair<int, op_load_t>(OP_SPACETODEPTH, LoadSpaceToDepth);
    //     op_load_map["DEPTH_TO_SPACE"] = std::pair<int, op_load_t>(OP_DEPTHTOSPACE, LoadDepthToSpace);
    //     op_load_map["MUL"] = std::pair<int, op_load_t>(OP_ELTWISE, LoadEltwise);
    //     op_load_map["MEAN"] = std::pair<int, op_load_t>(OP_MEAN, LoadReduction);
    //     op_load_map["SUB"] = std::pair<int, op_load_t>(OP_ELTWISE, LoadEltwise);
    //     op_load_map["SQUARED_DIFFERENCE"] = std::pair<int, op_load_t>(OP_SQUAREDDIFFERENCE, LoadSquaredDifference);
    //     op_load_map["CEIL"] = std::pair<int, op_load_t>(OP_CEIL, LoadCeil);
    //     op_load_map["ROUND"] = std::pair<int, op_load_t>(OP_ROUND, LoadRound);
    //     op_load_map["SPARSE_TO_DENSE"] = std::pair<int, op_load_t>(OP_SPARSETODENSE, LoadSparseToDense);
}
