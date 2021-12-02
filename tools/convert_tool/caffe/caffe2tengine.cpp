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
 * Author: bzhang@openailab.com
 */

#include "caffe2tengine.hpp"

#include <assert.h>

/*
*   SELF DEFINE VARIABLE
*   FOR CAFFE SERIALIZER
*/
const int OP_VERSION = 1;

int caffe_serializer::load_text_file(std::string model_file, te_caffe::NetParameter& caffe_net)
{
    std::ifstream is(model_file.c_str(), std::ios::in);

    if (!is.is_open())
    {
        TLOG_ERR("cannot open file: %s \n", model_file.c_str());
        return -1;
    }
    google::protobuf::io::IstreamInputStream input_stream(&is);
    bool ret = google::protobuf::TextFormat::Parse(&input_stream, &caffe_net);
    is.close();

    if (!ret)
        TLOG_ERR("model file:  %s failed\n", model_file.c_str());

    return 0;
}

int caffe_serializer::load_binary_file(std::string model_file, te_caffe::NetParameter& caffe_net)
{
    std::ifstream is(model_file.c_str(), std::ios::in | std::ios::binary);

    if (!is.is_open())
    {
        TLOG_ERR("cannot open file: %s \n", model_file.c_str());
        return -1;
    }

    google::protobuf::io::IstreamInputStream input_stream(&is);
    google::protobuf::io::CodedInputStream coded_input(&input_stream);
    // SetTotalBytesLimit(max_limit, warning_threshold)
#if GOOGLE_PROTOBUF_VERSION >= 3011000
    coded_input.SetTotalBytesLimit(INT_MAX);
#else
    coded_input.SetTotalBytesLimit(INT_MAX, INT_MAX / 2);
#endif

    bool ret = caffe_net.ParseFromCodedStream(&coded_input);

    is.close();

    if (!ret)
        TLOG_ERR("parse file:  %s failed\n", model_file.c_str());

    return 0;
}
bool caffe_serializer::find_op_load_method(const std::string& op_name)
{
    if (op_load_map.count(op_name))
        return true;

    return false;
}

ir_tensor_t* find_caffe_tensor(ir_graph_t* graph, const std::string& tensor_name)
{
    for (uint16_t i = 0; i < graph->tensor_num; i++)
    {
        ir_tensor_t* tensor = get_ir_graph_tensor(graph, i);
        if (tensor->name == tensor_name)
            return tensor;
    }

    return nullptr;
}

static void load_blob(ir_graph_t* ir_graph, std::string node_name, const std::vector<std::string>& name_list,
                      const std::vector<std::string>& layout_list, const te_caffe::LayerParameter& layer_param)
{
    unsigned int blob_num = layer_param.blobs_size();
    for (unsigned int n = 0; n < blob_num && n < name_list.size(); n++)
    {
        std::string new_tensor_name = node_name + "/" + name_list[n];
        ir_tensor_t* ir_tensor = create_ir_tensor(ir_graph, new_tensor_name.c_str(), TENGINE_DT_FP32);
        /* load tensor data*/
        const te_caffe::BlobProto& blob = layer_param.blobs(n);

        int dim_num = 0;
        int* dims;
        int data_num = 1;
        int dim_idx;
        if (blob.has_shape())
        {
            dim_num = blob.shape().dim_size();
            dims = (int*)malloc(sizeof(int) * dim_num);
            memset(dims, 0, sizeof(int) * dim_num);
            for (int i = 0; i < dim_num; i++)
            {
                dims[i] = blob.shape().dim(i);
                data_num *= dims[i];
            }
        }
        else
        {
            std::vector<int> temp;
            temp.push_back(blob.num());
            temp.push_back(blob.channels());
            temp.push_back(blob.height());
            temp.push_back(blob.width());

            int start = 0;

            while (temp[start] == 1)
            {
                start++;
            }
            dim_num = temp.size() - start;
            dims = (int*)malloc(sizeof(int) * dim_num);
            memset(dims, 0, sizeof(int) * dim_num);
            for (unsigned int i = start; i < temp.size(); i++)
            {
                if (i - start >= 0)
                {
                    dims[i - start] = temp[i];
                    data_num *= dims[i - start];
                }
            }
        }
        int res = set_ir_tensor_shape(ir_tensor, dims, dim_num);
        free(dims);
        dims = nullptr;
        assert(res == 0);
        ir_tensor->tensor_type = TENSOR_TYPE_CONST;
        int tensor_size = data_num * sizeof(float);
        ir_tensor->data = sys_malloc(tensor_size);
        float* ptr = (float*)ir_tensor->data;
        for (int i = 0; i < data_num; i++)
        {
            ptr[i] = blob.data(i);
        }

        ir_node_t* ir_node = create_ir_node(ir_graph, new_tensor_name.c_str(), OP_CONST, OP_VERSION);
        set_ir_node_output_tensor(ir_node, 0, ir_tensor);
    }
}

int create_tensor(ir_graph_t* graph, const te_caffe::LayerParameter& layer_param, float val, std::string blob_name)
{
    ir_tensor_t* blob_tensor = create_ir_tensor(graph, blob_name.c_str(), TENGINE_DT_FP32);
    const te_caffe::BlobProto& mean_blob = layer_param.blobs(0);
    std::vector<int> temp;
    temp.push_back(mean_blob.shape().dim(0));
    int dim_num = temp.size();
    if (dim_num > 0)
    {
        int* dims = (int*)malloc(sizeof(int) * dim_num);
        memset(dims, 0, sizeof(int) * dim_num);
        int elem_size = 1;

        for (unsigned int i = 0; i < dim_num; i++)
        {
            elem_size *= temp[i];
            dims[i] = temp[i];
        }

        set_ir_tensor_shape(blob_tensor, dims, dim_num);
        blob_tensor->tensor_type = TENSOR_TYPE_CONST;
        int tensor_size = mean_blob.data_size() * sizeof(float);
        blob_tensor->data = (float*)malloc(tensor_size);

        float* ptr = (float*)blob_tensor->data;
        for (int i = 0; i < mean_blob.data_size(); i++)
            ptr[i] = val;
    }
    ir_node_t* ir_node = create_ir_node(graph, blob_name.c_str(), OP_CONST, OP_VERSION);
    set_ir_node_output_tensor(ir_node, 0, blob_tensor);
    return 0;
}
int caffe_serializer::load_tensor_data(ir_graph_t* graph, const te_caffe::NetParameter test_net, const te_caffe::NetParameter train_net)
{
    name_map_t tensor_name_map;
    int layer_number = train_net.layer_size();
    for (int i = 0; i < layer_number; i++)
    {
        const te_caffe::LayerParameter& layer_param = train_net.layer(i);

        train_name_map[layer_param.name()] = &layer_param;
    }
    layer_number = test_net.layer_size();

    int size = (int)op_load_map.size();

    int n;
    for (n = 0; n < layer_number; n++)
    {
        const te_caffe::LayerParameter& layer_param = test_net.layer(n);
        const std::string& caffe_op_name = layer_param.type();

        for (int i = 0; i < layer_param.bottom_size(); i++)
        {
            if (train_name_map.count(layer_param.name()))
            {
                std::string tr_name = layer_param.name();
                const te_caffe::LayerParameter* train_layer_param = train_name_map[layer_param.name()];
                if (caffe_op_name == "Convolution" || caffe_op_name == "InnerProduct"
                    || caffe_op_name == "DeConvolution" || caffe_op_name == "DepthwiseConvolution"
                    || caffe_op_name == "ConvolutionDepthwise")
                {
                    std::vector<std::string> name_list = {"weight", "bias"};
                    std::vector<std::string> layout_list = {"NCHW", "W"};
                    load_blob(graph, tr_name, name_list, layout_list, *train_layer_param);
                }
                if (caffe_op_name == "BatchNorm")
                {
                    std::vector<std::string> name_list = {"means", "vars"};
                    std::vector<std::string> layout_list = {"W", "W"};
                    create_tensor(graph, *train_layer_param, 1.0f, tr_name + "/" + "gamma");
                    create_tensor(graph, *train_layer_param, 0.0f, tr_name + "/" + "beta");
                    load_blob(graph, tr_name, name_list, layout_list, *train_layer_param);
                }
                if (caffe_op_name == "Scale")
                {
                    std::vector<std::string> name_list = {"gamma", "beta"};
                    std::vector<std::string> layout_list = {"CHW", "W"};
                    load_blob(graph, tr_name, name_list, layout_list, *train_layer_param);
                }
                if (caffe_op_name == "PReLU")
                {
                    std::vector<std::string> name_list = {"slope"};
                    std::vector<std::string> layout_list = {"W"};
                    load_blob(graph, tr_name, name_list, layout_list, *train_layer_param);
                }
                if (caffe_op_name == "Bias")
                {
                    std::vector<std::string> name_list = {"weight"};
                    std::vector<std::string> layout_list = {"NCHW"};
                    load_blob(graph, tr_name, name_list, layout_list, *train_layer_param);
                }
                if (caffe_op_name == "Normalize")
                {
                    std::vector<std::string> name_list = {"scale"};
                    std::vector<std::string> layout_list = {"W"};
                    load_blob(graph, tr_name, name_list, layout_list, *train_layer_param);
                }
            }
        }
    }
    return 0;
}
int caffe_serializer::set_graph_input(ir_graph_t* graph, const te_caffe::NetParameter test_net, const te_caffe::NetParameter train_net)
{
    int layer_number = test_net.layer_size();
    std::vector<int16_t> input_nodes;

    for (int i = 0; i < layer_number; i++)
    {
        const te_caffe::LayerParameter& layer_param = test_net.layer(i);
        const te_caffe::InputParameter& input_param = layer_param.input_param();
        if (layer_param.type() != "Input")
            continue;

        std::string val = (layer_param.name() == "input" ? "data" : layer_param.name());
        ir_tensor_t* tensor = create_ir_tensor(graph, val.c_str(), TENGINE_DT_FP32);
        int has_shape = 1;

        std::vector<int> dim;
        if (input_param.shape_size())
        {
            const te_caffe::BlobShape& blob_shape = input_param.shape(0);

            for (int i = 0; i < blob_shape.dim_size(); i++)
            {
                dim.push_back(blob_shape.dim(i));
            }
        }

        int dim_num = (int)dim.size();
        if (dim_num == 0)
            has_shape = 0;

        if (has_shape)
        {
            int* dims = (int*)malloc(sizeof(int) * dim_num);
            memset(dims, 0, sizeof(int) * dim_num);
            for (int i = 0; i < dim_num; i++)
                dims[i] = dim[i];
            set_ir_tensor_shape(tensor, dims, dim_num);
        }

        tensor->tensor_type = TENSOR_TYPE_INPUT;
        ir_node_t* node = create_ir_node(graph, val.c_str(), OP_INPUT, OP_VERSION);

        set_ir_node_output_tensor(node, 0, tensor);
        input_nodes.push_back(node->index);
    }
    int16_t* node_idx = (int16_t*)sys_malloc(sizeof(int16_t) * input_nodes.size());
    for (int i = 0; i < input_nodes.size(); i++)
    {
        node_idx[i] = input_nodes[i];
    }
    set_ir_graph_input_node(graph, node_idx, input_nodes.size());
    return 0;
}

int caffe_serializer::load_graph_node(ir_graph_t* graph, const te_caffe::NetParameter test_net, const te_caffe::NetParameter train_net)
{
    name_map_t tensor_name_map;
    int layer_number = train_net.layer_size();

    std::vector<std::string> conv_name_list = {"weight", "bias"};
    std::vector<std::string> bn_name_list = {"gamma", "beta", "means", "vars"};
    std::vector<std::string> sc_name_list = {"gamma", "beta"};
    std::vector<std::string> pr_name_list = {"slope"};
    std::vector<std::string> bs_name_list = {"weight"};
    std::vector<std::string> nr_name_list = {"scale"};

    for (int i = 0; i < layer_number; i++)
    {
        const te_caffe::LayerParameter& layer_param = train_net.layer(i);
        train_name_map[layer_param.name()] = &layer_param;
    }
    layer_number = test_net.layer_size();
    int size = (int)op_load_map.size();
    std::vector<std::string> no_supported_op;
    for (int i = 0; i < layer_number; i++)
    {
        const te_caffe::LayerParameter& layer_param = test_net.layer(i);
        const std::string& caffe_op_name = layer_param.type();
        if (!find_op_load_method(caffe_op_name))
        {
            auto it = find(no_supported_op.begin(), no_supported_op.end(), caffe_op_name);
            if (it == no_supported_op.end())
            {
                if (caffe_op_name == "Constant" || caffe_op_name == "Input")
                    continue;
                no_supported_op.push_back(caffe_op_name);
            }
        }
    }
    if (no_supported_op.size())
    {
        TLOG_ERR("These %d op are not supported\n{ ", no_supported_op.size());
        for (int j = 0; j < (int)no_supported_op.size(); j++)
        {
            TLOG_ERR("%s ", no_supported_op[j].c_str());
        }
        TLOG_ERR("}\n");
        return -1;
    }
    int n;
    for (n = 0; n < layer_number; n++)
    {
        const te_caffe::LayerParameter& layer_param = test_net.layer(n);
        const std::string& caffe_op_name = layer_param.type();
        if (caffe_op_name == "data" || caffe_op_name == "Input")
        {
            continue;
        }
        ir_node_t* ir_node = create_ir_node(graph, layer_param.name().c_str(), op_load_map[caffe_op_name].first, OP_VERSION);
        for (int i = 0; i < layer_param.bottom_size(); i++)
        {
            const std::string& orig_name = layer_param.bottom(i);
            std::string& tensor_name = tensor_name_map[orig_name];

            int tensor_id = get_ir_tensor_index_from_name(graph, tensor_name.c_str());
            ir_tensor_t* tensor = NULL;
            if (tensor_id < 0)
            {
                tensor_id = get_ir_tensor_index_from_name(graph, orig_name.c_str());
            }

            tensor = get_ir_graph_tensor(graph, tensor_id);
            set_ir_node_input_tensor(ir_node, i, tensor);
            input_tensors.push_back(tensor->name);
            const te_caffe::LayerParameter* train_layer_param = train_name_map[layer_param.name()];
            if (train_name_map[layer_param.name()] == nullptr)
            {
                continue;
            }
            std::vector<std::string> name_list;

            int blob_size = train_layer_param->blobs_size();

            if (train_layer_param->type() == "Convolution" || train_layer_param->type() == "InnerProduct"
                || train_layer_param->type() == "DeConvolution" || train_layer_param->type() == "DepthwiseConvolution"
                || train_layer_param->type() == "ConvolutionDepthwise")
            {
                name_list = conv_name_list;
            }
            else if (train_layer_param->type() == "BatchNorm")
            {
                name_list = bn_name_list;
                blob_size = 2 + train_layer_param->blobs_size();
            }
            else if (train_layer_param->type() == "Scale")
            {
                name_list = sc_name_list;
            }
            else if (train_layer_param->type() == "PReLu")
            {
                name_list = pr_name_list;
            }
            else if (train_layer_param->type() == "Bias")
            {
                name_list = bs_name_list;
            }
            else if (train_layer_param->type() == "Normalize")
            {
                name_list = nr_name_list;
            }
            const std::string& orig_name_top = layer_param.name();

            for (int nx = 0; nx < blob_size && nx < name_list.size(); nx++)
            {
                std::string blob_name = orig_name_top + "/" + name_list[nx];
                int blod_id = get_ir_tensor_index_from_name(graph, blob_name.c_str());
                ir_tensor_t* blob_tensor = get_ir_graph_tensor(graph, blod_id);
                if (blod_id < 0)
                {
                    blob_tensor = create_ir_tensor(graph, blob_name.c_str(), TENGINE_DT_FP32);
                }
                else
                {
                    blob_tensor = get_ir_graph_tensor(graph, blod_id);
                }

                set_ir_node_input_tensor(ir_node, nx + 1 + i, blob_tensor);
            }
        }
        std::vector<std::string> same_name;
        for (int i = 0; i < layer_param.top_size(); i++)
        {
            const std::string& orig_name = layer_param.top(i);
            std::string tensor_name;
            if (tensor_name_map.count(orig_name))
            {
                char* ir_node_name = ir_node->name;
                std::string node_name = ir_node_name;
                tensor_name = node_name + "/" + std::to_string(i);
            }
            else
                tensor_name = orig_name;

            ir_tensor_t* tensor = create_ir_tensor(graph, tensor_name.c_str(), TENGINE_DT_FP32);
            set_ir_node_output_tensor(ir_node, i, tensor);
            tensor_name_map[orig_name] = tensor_name;
            output_tensors.push_back(tensor_name);
        }
        op_load_t loader = op_load_map[caffe_op_name].second;
        if (loader(graph, ir_node, layer_param) < 0)
        {
            TLOG_ERR("load op %s func failed in node %s .\n", caffe_op_name.c_str(), ir_node->name);
            return -1;
        }
    }

    if (n < layer_number)
    {
        fprintf(stderr, "Check layer number error ! \n");
        return -1;
    }
    return 0;
}
int caffe_serializer::set_graph_output(ir_graph_t* graph, const te_caffe::NetParameter test_net, const te_caffe::NetParameter train_net)
{
    int layer_number = test_net.layer_size();
    std::vector<int16_t> output_nodes;
    name_map_t tensor_name_map;
    for (int n = 0; n < layer_number; n++)
    {
        const te_caffe::LayerParameter& layer_param = test_net.layer(n);
        const std::string& caffe_op_name = layer_param.type();
        if (caffe_op_name == "data" || caffe_op_name == "Input")
        {
            continue;
        }
    }
    std::vector<std::string> graph_outputs;
    for (int i = 0; i < output_tensors.size(); i++)
    {
        int check_flag = true;

        auto it = find(input_tensors.begin(), input_tensors.end(), output_tensors[i]);
        if (it == input_tensors.end())
        {
            graph_outputs.push_back(output_tensors[i]);
        }
    }

    for (int i = 0; i < graph_outputs.size(); i++)
    {
        int tensor_id = get_ir_tensor_index_from_name(graph, graph_outputs[i].c_str());
        ir_tensor_t* ir_tensor = get_ir_graph_tensor(graph, tensor_id);
        ir_node_t* node = get_ir_graph_node(graph, ir_tensor->producer);
        set_ir_node_output_tensor(node, 0, ir_tensor);
        output_nodes.push_back(node->index);
    }

    std::vector<int16_t> node_idx;
    for (int i = 0; i < output_nodes.size(); i++)
    {
        node_idx.push_back(output_nodes[i]);
    }
    set_ir_graph_output_node(graph, node_idx.data(), output_nodes.size());
    return 0;
}

int caffe_serializer::load_model(ir_graph_t* graph, std::string model_file, std::string proto_file)
{
    register_op_load();
    te_caffe::NetParameter test_net;
    te_caffe::NetParameter train_net;
    if (load_text_file(proto_file, test_net) < 0)
        return -1;
    fprintf(stderr, "Process 1: Finish load model file \n");
    if (load_binary_file(model_file, train_net) < 0)
        return -1;
    fprintf(stderr, "Process 2: Finish load protobuf file \n");
    if (load_tensor_data(graph, test_net, train_net) < 0)
        return -1;
    fprintf(stderr, "Process 3: Finish load graph node \n");
    if (set_graph_input(graph, test_net, train_net) < 0)
        return -1;
    fprintf(stderr, "Process 4: Finish set graph input \n");
    if (load_graph_node(graph, test_net, train_net) < 0)
        return -1;
    fprintf(stderr, "Process 5: Finish load graph node \n");
    if (set_graph_output(graph, test_net, train_net) < 0)
        return -1;
    fprintf(stderr, "Process 6: Finish set graph output \n");
    return 0;
}

graph_t caffe_serializer::caffe2tengine(std::string model_file, std::string proto_file)
{
    fprintf(stderr, "----------caffe2tengine begin----------\n");

    context_t context = create_context(NULL, 1);
    ir_graph_t* ir_graph = create_ir_graph((struct context*)context);
    if (ir_graph == NULL)
    {
        destroy_context(context);
        return NULL;
    }
    ir_graph->attribute->private_context = 1; // new context

    int ret = load_model(ir_graph, model_file, proto_file);
    if (0 != ret)
    {
        destroy_graph(ir_graph);
        return NULL;
    }
    ir_graph->device = find_default_device();

    fprintf(stderr, "----------caffe2tengine done.----------\n");
    return ir_graph;
}

int load_batchnorm(ir_graph_t* graph, ir_node_t* node, const te_caffe::LayerParameter& layer_param)
{
    struct batchnorm_param* batchnorm_param = (struct batchnorm_param*)node->op.param_mem;

    const te_caffe::BatchNormParameter& bn_param = layer_param.batch_norm_param();

    batchnorm_param->eps = bn_param.eps();
    batchnorm_param->caffe_flavor = 1;

    return 0;
}

int load_softmax(ir_graph_t* graph, ir_node_t* node, const te_caffe::LayerParameter& layer_param)
{
    const te_caffe::SoftmaxParameter& softmax_param = layer_param.softmax_param();

    struct softmax_param* param = (struct softmax_param*)node->op.param_mem;

    if (softmax_param.has_axis())
        param->axis = softmax_param.axis();
    else
        param->axis = 1;

    return 0;
}

int load_conv(ir_graph_t* graph, ir_node_t* node, const te_caffe::LayerParameter& layer_param)
{
    const te_caffe::ConvolutionParameter& conv_param = layer_param.convolution_param();
    const std::string& caffe_op_name = layer_param.type();
    struct conv_param* param = (struct conv_param*)node->op.param_mem;

    if (conv_param.has_kernel_h() && conv_param.has_kernel_w())
    {
        param->kernel_h = conv_param.kernel_h();
        param->kernel_w = conv_param.kernel_w();
    }
    else
    {
        param->kernel_h = conv_param.kernel_size(0);
        param->kernel_w = conv_param.kernel_size(0);
    }

    if (conv_param.has_stride_h() && conv_param.has_stride_w())
    {
        param->stride_h = conv_param.stride_h();
        param->stride_w = conv_param.stride_w();
    }
    else if (conv_param.stride_size())
    {
        param->stride_h = conv_param.stride(0);
        param->stride_w = conv_param.stride(0);
    }

    if (conv_param.has_pad_h() && conv_param.has_pad_w())
    {
        param->pad_h0 = conv_param.pad_h();
        param->pad_h1 = conv_param.pad_h();
        param->pad_w0 = conv_param.pad_w();
        param->pad_w1 = conv_param.pad_w();
    }
    else if (conv_param.pad_size())
    {
        param->pad_h0 = conv_param.pad(0);
        param->pad_h1 = conv_param.pad(0);
        param->pad_w0 = conv_param.pad(0);
        param->pad_w1 = conv_param.pad(0);
    }

    param->output_channel = conv_param.num_output();

    if (conv_param.has_group())
        param->group = conv_param.group();
    if (caffe_op_name == "ConvolutionDepthwise")
    {
        param->group = conv_param.num_output();
    }

    if (conv_param.dilation_size())
    {
        param->dilation_h = conv_param.dilation(0);
        param->dilation_w = conv_param.dilation(0);
    }

    return 0;
}

int load_deconv(ir_graph_t* graph, ir_node_t* node, const te_caffe::LayerParameter& layer_param)
{
    const te_caffe::ConvolutionParameter& conv_param = layer_param.convolution_param();

    struct deconv_param* param = (struct deconv_param*)node->op.param_mem;

    if (conv_param.has_kernel_h() && conv_param.has_kernel_w())
    {
        param->kernel_h = conv_param.kernel_h();
        param->kernel_w = conv_param.kernel_w();
    }
    else
    {
        param->kernel_h = conv_param.kernel_size(0);
        param->kernel_w = conv_param.kernel_size(0);
    }

    if (conv_param.has_stride_h() && conv_param.has_stride_w())
    {
        param->stride_h = conv_param.stride_h();
        param->stride_w = conv_param.stride_w();
    }
    else if (conv_param.stride_size())
    {
        param->stride_h = conv_param.stride(0);
        param->stride_w = conv_param.stride(0);
    }

    if (conv_param.has_pad_h() && conv_param.has_pad_w())
    {
        param->pad_h0 = conv_param.pad_h();
        param->pad_h1 = conv_param.pad_h();
        param->pad_w0 = conv_param.pad_w();
        param->pad_w1 = conv_param.pad_w();
    }
    else if (conv_param.pad_size())
    {
        param->pad_h0 = conv_param.pad(0);
        param->pad_w0 = conv_param.pad(0);
        param->pad_h1 = conv_param.pad(0);
        param->pad_w1 = conv_param.pad(0);
    }
    param->num_output = conv_param.num_output();

    if (conv_param.has_group())
        param->group = conv_param.group();

    if (conv_param.dilation_size())
    {
        param->dilation_h = conv_param.dilation(0);
        param->dilation_w = conv_param.dilation(0);
    }

    return 0;
}

int load_fc(ir_graph_t* graph, ir_node_t* node, const te_caffe::LayerParameter& layer_param)
{
    const te_caffe::InnerProductParameter& ip_param = layer_param.inner_product_param();

    struct fc_param* param = (struct fc_param*)node->op.param_mem;
    param->num_output = ip_param.num_output();

    return 0;
}
int load_prelu(ir_graph_t* graph, ir_node_t* node, const te_caffe::LayerParameter& layer_param)
{
    return 0;
}
int load_normalize(ir_graph_t* graph, ir_node_t* node, const te_caffe::LayerParameter& layer_param)
{
    const te_caffe::NormalizeParameter& normalize_param = layer_param.norm_param();

    struct normalize_param* param = (struct normalize_param*)node->op.param_mem;

    param->across_spatial = normalize_param.across_spatial();
    param->channel_shared = normalize_param.channel_shared();

    return 0;
}

int load_scale(ir_graph_t* graph, ir_node_t* node, const te_caffe::LayerParameter& layer_param)
{
    struct scale_param* param = (struct scale_param*)node->op.param_mem;

    const te_caffe::ScaleParameter& scale_param = layer_param.scale_param();

    if (scale_param.has_axis())
        param->axis = scale_param.axis();

    if (scale_param.has_num_axes())
        param->num_axes = scale_param.num_axes();

    if (scale_param.has_bias_term())
        param->bias_term = scale_param.bias_term();

    return 0;
}

int load_relu(ir_graph_t* graph, ir_node_t* node, const te_caffe::LayerParameter& layer_param)
{
    struct relu_param* param = (struct relu_param*)node->op.param_mem;

    const te_caffe::ReLUParameter& caffe_param = layer_param.relu_param();

    if (caffe_param.has_negative_slope())
        param->negative_slope = static_cast<float>(caffe_param.negative_slope());
    else
        param->negative_slope = 0.f;

    return true;
}

int load_split(ir_graph_t* graph, ir_node_t* node, const te_caffe::LayerParameter& layer_param)
{
    struct split_param* param = (struct split_param*)node->op.param_mem;
    param->is_caffe = true;

    return 0;
}

int load_pool(ir_graph_t* graph, ir_node_t* node, const te_caffe::LayerParameter& layer_param)
{
    const te_caffe::PoolingParameter& pool_param = layer_param.pooling_param();

    struct pool_param* param = (struct pool_param*)node->op.param_mem;

    if (pool_param.pool() == te_caffe::PoolingParameter_PoolMethod_AVE)
    {
        param->pool_method = kPoolAvg;
    }
    else if (pool_param.pool() == te_caffe::PoolingParameter_PoolMethod_STOCHASTIC)
    {
        param->pool_method = kPoolRand;
    }
    else
    {
        param->pool_method = kPoolMax;
    }

    if (pool_param.has_kernel_size())
    {
        param->kernel_h = pool_param.kernel_size();
        param->kernel_w = pool_param.kernel_size();
    }
    else if (pool_param.has_kernel_h() && pool_param.has_kernel_w())
    {
        param->kernel_h = pool_param.kernel_h();
        param->kernel_w = pool_param.kernel_w();
    }

    param->global = pool_param.global_pooling();
    if (pool_param.has_pad())
    {
        param->pad_h0 = pool_param.pad();
        param->pad_h1 = pool_param.pad();
        param->pad_w0 = pool_param.pad();
        param->pad_w1 = pool_param.pad();
    }
    else if (pool_param.has_pad_h() && pool_param.has_pad_w())
    {
        param->pad_h0 = pool_param.pad_h();
        param->pad_h1 = pool_param.pad_h();
        param->pad_w0 = pool_param.pad_w();
        param->pad_w1 = pool_param.pad_w();
    }

    if (pool_param.has_stride())
    {
        param->stride_h = pool_param.stride();
        param->stride_w = pool_param.stride();
    }
    else if (pool_param.has_stride_h() && pool_param.has_stride_w())
    {
        param->stride_h = pool_param.stride_h();
        param->stride_w = pool_param.stride_w();
    }

    param->caffe_flavor = 1;

    return 0;
}
static EltType ConvertCaffeEltwise(te_caffe::EltwiseParameter_EltwiseOp method)
{
    if (method == te_caffe::EltwiseParameter_EltwiseOp_PROD)
        return ELT_PROD;
    else if (method == te_caffe::EltwiseParameter_EltwiseOp_MAX)
        return ELT_MAX;

    /* for others, return SUM */
    return ELT_SUM;
}
int load_eltwise(ir_graph_t* graph, ir_node_t* node, const te_caffe::LayerParameter& layer_param)
{
    const te_caffe::EltwiseParameter& eltwise_param = layer_param.eltwise_param();
    struct eltwise_param* param = (struct eltwise_param*)node->op.param_mem;
    // defalt: SUM
    param->type = ELT_SUM;
    if (eltwise_param.has_operation())
        param->type = ConvertCaffeEltwise(eltwise_param.operation());

    param->caffe_flavor = 1;
    param->shift = eltwise_param.shift();
    param->scale = eltwise_param.scale();
    param->power = eltwise_param.power();

    return 0;
}

int load_slice(ir_graph_t* graph, ir_node_t* node, const te_caffe::LayerParameter& layer_param)
{
    const te_caffe::SliceParameter& slice_param = layer_param.slice_param();
    struct slice_param* param = (struct slice_param*)node->op.param_mem;
    if (slice_param.has_axis())
        param->axis = slice_param.axis();
    else
        param->axis = 1;

    param->begin = 0;
    param->end = 0;
    param->begin_ = nullptr;
    param->size_ = nullptr;

    int size = slice_param.slice_point().size();
    int32_t* data = (int32_t*)malloc(size * sizeof(int32_t));
    for (int i = 0; i < size; i++)
    {
        data[i] = slice_param.slice_point()[i];
    }

    param->slice_point_ = create_vector(sizeof(float), NULL);
    param->slice_point_->elem_num = size;
    param->slice_point_->elem_size = sizeof(int32_t);
    set_vector_data(param->slice_point_, 0, data);

    param->iscaffe = 1;
    param->ismxnet = 0;
    param->isonnx = 0;
    return 0;
}
int load_concat(ir_graph_t* graph, ir_node_t* node, const te_caffe::LayerParameter& layer_param)
{
    const te_caffe::ConcatParameter& concat_param = layer_param.concat_param();
    struct concat_param* param = (struct concat_param*)node->op.param_mem;
    if (concat_param.has_concat_dim())
        param->axis = static_cast<int>(concat_param.concat_dim());
    else
        param->axis = concat_param.axis();
    return 0;
}
int load_dropout(ir_graph_t* graph, ir_node_t* node, const te_caffe::LayerParameter& layer_param)
{
    return 0;
}
int load_lrn(ir_graph_t* graph, ir_node_t* node, const te_caffe::LayerParameter& layer_param)
{
    const ::te_caffe::LRNParameter& caffe_param = layer_param.lrn_param();
    struct lrn_param* param = (struct lrn_param*)node->op.param_mem;
    if (caffe_param.norm_region() == te_caffe::LRNParameter_NormRegion_WITHIN_CHANNEL)
        param->norm_region = LRN_WITHIN_CHANNEL;
    else
        param->norm_region = LRN_ACROSS_CHANNELS;

    param->k = caffe_param.k();
    param->alpha = caffe_param.alpha();
    param->beta = caffe_param.beta();
    param->local_size = caffe_param.local_size();

    return 0;
}
int load_relu6(ir_graph_t* graph, ir_node_t* node, const te_caffe::LayerParameter& layer_param)
{
    return 0;
}
int load_permute(ir_graph_t* graph, ir_node_t* node, const te_caffe::LayerParameter& layer_param)
{
    const te_caffe::PermuteParameter& permute_param = layer_param.permute_param();
    struct permute_param* param = (struct permute_param*)node->op.param_mem;

    param->order0 = permute_param.order(0);
    param->order1 = permute_param.order(1);
    param->order2 = permute_param.order(2);
    param->order3 = permute_param.order(3);
    return 0;
}
int load_clip(ir_graph_t* graph, ir_node_t* node, const te_caffe::LayerParameter& layer_param)
{
    const ::te_caffe::ClipParameter& caffe_param = layer_param.clip_param();
    struct clip_param* param = (struct clip_param*)node->op.param_mem;
    param->min = caffe_param.min();
    param->max = caffe_param.max();
    return 0;
}
int load_flatten(ir_graph_t* graph, ir_node_t* node, const te_caffe::LayerParameter& layer_param)
{
    const te_caffe::FlattenParameter& flatten_param = layer_param.flatten_param();
    struct flatten_param* param = (struct flatten_param*)node->op.param_mem;
    param->axis = flatten_param.axis();

    return 0;
}
int load_priorbox(ir_graph_t* graph, ir_node_t* node, const te_caffe::LayerParameter& layer_param)
{
    const te_caffe::PriorBoxParameter& caffe_param = layer_param.prior_box_param();
    struct priorbox_param* param = (struct priorbox_param*)node->op.param_mem;
    param->offset = caffe_param.offset();
    // img_size
    if (caffe_param.has_img_h() && caffe_param.has_img_w())
    {
        param->image_h = caffe_param.img_h();
        param->image_w = caffe_param.img_w();
    }
    else if (caffe_param.has_img_size())
    {
        param->image_h = caffe_param.img_size();
        param->image_w = caffe_param.img_size();
    }
    else
    {
        param->image_h = 0;
        param->image_w = 0;
    }

    // step
    if (caffe_param.has_step_h() && caffe_param.has_step_w())
    {
        param->step_h = caffe_param.step_h();
        param->step_w = caffe_param.step_w();
    }
    else if (caffe_param.has_step())
    {
        param->step_h = caffe_param.step();
        param->step_w = caffe_param.step();
    }
    else
    {
        param->step_h = 0;
        param->step_w = 0;
    }

    // min_size, max_size
    param->min_size = (float*)sys_malloc(sizeof(float) * caffe_param.min_size_size());
    param->max_size = (float*)sys_malloc(sizeof(float) * caffe_param.max_size_size());
    param->variance = (float*)sys_malloc(sizeof(float) * caffe_param.variance_size());
    param->aspect_ratio = (float*)sys_malloc(sizeof(float) * caffe_param.aspect_ratio_size());

    param->min_size_num = caffe_param.min_size_size();
    for (int i = 0; i < caffe_param.min_size_size(); ++i)
    {
        param->min_size[i] = caffe_param.min_size(i);
    }
    param->max_size_num = caffe_param.max_size_size();
    for (int i = 0; i < caffe_param.max_size_size(); ++i)
    {
        param->max_size[i] = caffe_param.max_size(i);
    }

    // variance
    for (int i = 0; i < caffe_param.variance_size(); ++i)
    {
        param->variance[i] = caffe_param.variance(i);
    }
    // clip
    param->clip = caffe_param.clip();
    // flip
    param->flip = caffe_param.flip();
    // aspect_ratio
    param->aspect_ratio_size = caffe_param.aspect_ratio_size();
    for (int i = 0; i < caffe_param.aspect_ratio_size(); ++i)
    {
        param->aspect_ratio[i] = caffe_param.aspect_ratio(i);
    }

    return 0;
}
int load_roipooling(ir_graph_t* graph, ir_node_t* node, const te_caffe::LayerParameter& layer_param)
{
    const te_caffe::ROIPoolingParameter& caffe_param = layer_param.roi_pooling_param();
    struct roipooling_param* param = (struct roipooling_param*)node->op.param_mem;
    param->pooled_h = caffe_param.pooled_h();
    param->pooled_w = caffe_param.pooled_w();
    param->spatial_scale = caffe_param.spatial_scale();

    return 0;
}
int load_resize(ir_graph_t* graph, ir_node_t* node, const te_caffe::LayerParameter& layer_param)
{
    const te_caffe::Resize1_Parameter& caffe_param = layer_param.resize1_param();
    struct resize_param* param = (struct resize_param*)node->op.param_mem;
    param->scale_h = caffe_param.out_height_scale();
    param->scale_w = caffe_param.out_width_scale();
    param->type = caffe_param.resize_type();

    return 0;
}
int load_rpn(ir_graph_t* graph, ir_node_t* node, const te_caffe::LayerParameter& layer_param)
{
    const te_caffe::RPNParameter& caffe_param = layer_param.rpn_param();
    struct rpn_param* param = (struct rpn_param*)node->op.param_mem;
    param->feat_stride = caffe_param.feat_stride();
    param->basesize = caffe_param.basesize();
    param->min_size = caffe_param.boxminsize();
    param->per_nms_topn = caffe_param.per_nms_topn();
    param->post_nms_topn = caffe_param.post_nms_topn();
    param->nms_thresh = caffe_param.nms_thresh();

    struct vector* ratios_tmp = create_vector(sizeof(float), NULL);
    struct vector* anchor_scales_tmp = create_vector(sizeof(float), NULL);
    struct vector* anchor_tmp = create_vector(sizeof(float), NULL);

    for (int i = 0; i < caffe_param.scale_size(); ++i)
    {
        float tmp = caffe_param.scale(i);
        push_vector_data(anchor_scales_tmp, &tmp);
    }
    param->anchor_scales = anchor_scales_tmp;
    for (int i = 0; i < caffe_param.ratio_size(); ++i)
    {
        float tmp = caffe_param.ratio(i);
        push_vector_data(ratios_tmp, &tmp);
    }
    param->ratios = ratios_tmp;
    param->anchors_ = anchor_tmp;
    return 0;
}
int load_detectionoutput(ir_graph_t* graph, ir_node_t* node, const te_caffe::LayerParameter& layer_param)
{
    const te_caffe::DetectionOutputParameter& caffe_param = layer_param.detection_output_param();
    struct detection_output_param* param = (struct detection_output_param*)node->op.param_mem;
    param->num_classes = caffe_param.num_classes();
    param->confidence_threshold = caffe_param.confidence_threshold();
    param->keep_top_k = caffe_param.keep_top_k();
    param->nms_threshold = caffe_param.nms_param().nms_threshold();
    if (caffe_param.nms_param().has_top_k())
    {
        param->nms_top_k = caffe_param.nms_param().top_k();
    }
    return 0;
}
int load_reshape(ir_graph_t* graph, ir_node_t* node, const te_caffe::LayerParameter& layer_param)
{
    const te_caffe::ReshapeParameter& caffe_param = layer_param.reshape_param();
    struct reshape_param* param = (struct reshape_param*)node->op.param_mem;
    param->re_shape = (int*)sys_malloc(sizeof(int) * caffe_param.shape().dim_size());
    for (int i = 0; i < caffe_param.shape().dim_size(); i++)
    {
        param->re_shape[i] = caffe_param.shape().dim(i);
    }
    param->dim_size = caffe_param.shape().dim_size();
    param->is_onnx = 1; // ？？？ 此处保持与原 convert tools 一致
    param->is_mxnet = 0;
    return 0;
}
int load_upsample(ir_graph_t* graph, ir_node_t* node, const te_caffe::LayerParameter& layer_param)
{
    const ::te_caffe::UpsampleParameter& caffe_param = layer_param.upsample_param();
    struct upsample_param* param = (struct upsample_param*)node->op.param_mem;
    if (caffe_param.has_scale())
    {
        param->scale = caffe_param.scale();
    }

    return 0;
}
int load_sigmoid(ir_graph_t* graph, ir_node_t* node, const te_caffe::LayerParameter& layer_param)
{
    return 0;
}
int load_tanh(ir_graph_t* graph, ir_node_t* node, const te_caffe::LayerParameter& layer_param)
{
    return 0;
}

int load_tile(ir_graph_t* graph, ir_node_t* node, const te_caffe::LayerParameter& layer_param)
{
    const ::te_caffe::TileParameter& caffe_param = layer_param.tile_param();
    struct tile_param* param = (struct tile_param*)node->op.param_mem;
    param->frame_flag = 0;
    param->reps = (int*)malloc(sizeof(int));
    param->reps[0] = caffe_param.index();

    return 0;
}
int load_shufflechannel(ir_graph_t* graph, ir_node_t* node, const te_caffe::LayerParameter& layer_param)
{
    const ::te_caffe::ShuffleChannelParameter& caffe_param = layer_param.shuffle_channel_param();
    struct shuffle_channel_param* param = (struct shuffle_channel_param*)node->op.param_mem;
    param->group = caffe_param.group();

    return 0;
}
int load_crop(ir_graph_t* graph, ir_node_t* node, const te_caffe::LayerParameter& layer_param)
{
    const ::te_caffe::CropParameter& caffe_param = layer_param.crop_param();
    struct crop_param* param = (struct crop_param*)node->op.param_mem;
    if (caffe_param.has_crop_h())
        param->crop_h = caffe_param.crop_h();
    else
    {
        param->crop_h = 0;
    }
    if (caffe_param.has_crop_w())
        param->crop_w = caffe_param.crop_w();
    else
    {
        param->crop_h = 0;
    }

    if (caffe_param.has_center_crop())
        param->center_crop = caffe_param.center_crop();
    else
    {
        param->center_crop = false;
    }
    //param.offset_h = caffe_param.offset_h();
    //param.offset_w = caffe_param.offset_w();
    if (caffe_param.has_num_args())
        param->num_args = caffe_param.num_args();
    else
    {
        param->num_args = 0;
    }

    //int size = caffe_param.

    int size = 0;

    if (caffe_param.offset_size())
    {
        size = caffe_param.offset_size();
        if (size == 1)
        {
            param->offset_c = caffe_param.offset(0);
            param->offset_h = caffe_param.offset(0);
            param->offset_w = caffe_param.offset(0);
        }
        if (size == 2)
        {
            param->offset_h = caffe_param.offset(0);
            param->offset_w = caffe_param.offset(1);
        }
        if (size == 3)
        {
            param->offset_c = caffe_param.offset(0);
            param->offset_h = caffe_param.offset(1);
            param->offset_w = caffe_param.offset(2);
        }
    }
    else
    {
        param->offset_c = 0;
        param->offset_h = 0;
        param->offset_w = 0;
    }

    if (caffe_param.has_axis())
    {
        param->axis = caffe_param.axis();
    }
    else
    {
        param->axis = 2;
    }
    param->flag = 0;
    return 0;
}
int load_absval(ir_graph_t* graph, ir_node_t* node, const te_caffe::LayerParameter& layer_param)
{
    return 0;
}
int load_interp(ir_graph_t* graph, ir_node_t* node, const te_caffe::LayerParameter& layer_param)
{
    const ::te_caffe::InterpParameter& caffe_param = layer_param.interp_param();
    struct interp_param* param = (struct interp_param*)node->op.param_mem;
    param->resize_type = 2; //1=nearest  2=bilinear  3=bicubic
    param->width_scale = (float)caffe_param.zoom_factor();
    param->height_scale = (float)caffe_param.zoom_factor();
    param->output_width = caffe_param.width();
    param->output_height = caffe_param.height();

    return 0;
}
int load_elu(ir_graph_t* graph, ir_node_t* node, const te_caffe::LayerParameter& layer_param)
{
    const ::te_caffe::ELUParameter& caffe_param = layer_param.elu_param();
    struct elu_param* param = (struct elu_param*)node->op.param_mem;
    param->alpha = caffe_param.alpha();

    return 0;
}
int load_threshold(ir_graph_t* graph, ir_node_t* node, const te_caffe::LayerParameter& layer_param)
{
    const ::te_caffe::ThresholdParameter& caffe_param = layer_param.threshold_param();
    struct threshold_param* param = (struct threshold_param*)node->op.param_mem;
    param->threshold = caffe_param.threshold();

    return 0;
}
int load_embedding(ir_graph_t* graph, ir_node_t* node, const te_caffe::LayerParameter& layer_param)
{
    const ::te_caffe::EmbedParameter& caffe_param = layer_param.embed_param();
    struct embedding_param* param = (struct embedding_param*)node->op.param_mem;
    param->num_output = caffe_param.num_output();
    param->input_dim = caffe_param.input_dim();
    param->bias_term = caffe_param.bias_term();
    param->weight_data_size = caffe_param.weight_data_size();

    return 0;
}
int load_mvn(ir_graph_t* graph, ir_node_t* node, const te_caffe::LayerParameter& layer_param)
{
    const ::te_caffe::MVNParameter& caffe_param = layer_param.mvn_param();
    struct mvn_param* param = (struct mvn_param*)node->op.param_mem;

    if (caffe_param.normalize_variance() == true)
        param->normalize_variance = 1;
    else
        param->normalize_variance = 0;

    if (caffe_param.across_channels() == true)
        param->across_channels = 1;
    else
        param->across_channels = 0;

    param->eps = caffe_param.eps();
    return 0;
}
int load_reduction(ir_graph_t* graph, ir_node_t* node, const te_caffe::LayerParameter& layer_param)
{
    const ::te_caffe::ReductionParameter& caffe_param = layer_param.reduction_param();
    struct reduction_param* param = (struct reduction_param*)node->op.param_mem;

    param->dim_0 = caffe_param.dim_0();
    param->dim_1 = caffe_param.dim_1();
    param->dim_2 = caffe_param.dim_2();
    param->dim_3 = caffe_param.dim_3();
    param->type = caffe_param.operation();
    param->keepdim = 1;
    return 0;
}
int load_reorg(ir_graph_t* graph, ir_node_t* node, const te_caffe::LayerParameter& layer_param)
{
    const te_caffe::ReorgParameter& caffe_param = layer_param.reorg_param();
    struct reorg_param* param = (struct reorg_param*)node->op.param_mem;

    param->stride = caffe_param.stride();

    return 0;
}

/*
*   OPERAOTR REGISTER FUNCTION DEFINE FOR CAFFE SERIALIZER START
*/
void caffe_serializer::register_op_load()
{
    op_load_map["BatchNorm"] = std::pair<int, op_load_t>(OP_BATCHNORM, load_batchnorm);
    op_load_map["Convolution"] = std::pair<int, op_load_t>(OP_CONV, load_conv);
    op_load_map["DeConvolution"] = std::pair<int, op_load_t>(OP_DECONV, load_deconv);
    op_load_map["Softmax"] = std::pair<int, op_load_t>(OP_SOFTMAX, load_softmax);
    op_load_map["PReLU"] = std::pair<int, op_load_t>(OP_PRELU, load_prelu);
    op_load_map["InnerProduct"] = std::pair<int, op_load_t>(OP_FC, load_fc);
    op_load_map["SoftmaxWithLoss"] = std::pair<int, op_load_t>(OP_SOFTMAX, load_softmax);
    op_load_map["Normalize"] = std::pair<int, op_load_t>(OP_NORMALIZE, load_normalize);
    op_load_map["Scale"] = std::pair<int, op_load_t>(OP_SCALE, load_scale);
    op_load_map["ReLU"] = std::pair<int, op_load_t>(OP_RELU, load_relu);
    op_load_map["Split"] = std::pair<int, op_load_t>(OP_SPLIT, load_split);
    op_load_map["Pooling"] = std::pair<int, op_load_t>(OP_POOL, load_pool);
    op_load_map["Eltwise"] = std::pair<int, op_load_t>(OP_ELTWISE, load_eltwise);
    op_load_map["DepthwiseConvolution"] = std::pair<int, op_load_t>(OP_DECONV, load_deconv);
    op_load_map["ConvolutionDepthwise"] = std::pair<int, op_load_t>(OP_CONV, load_conv);
    op_load_map["Slice"] = std::pair<int, op_load_t>(OP_SLICE, load_slice);
    op_load_map["Concat"] = std::pair<int, op_load_t>(OP_CONCAT, load_concat);
    op_load_map["Dropout"] = std::pair<int, op_load_t>(OP_DROPOUT, load_dropout);
    op_load_map["LRN"] = std::pair<int, op_load_t>(OP_LRN, load_lrn);
    op_load_map["ReLU6"] = std::pair<int, op_load_t>(OP_RELU6, load_relu6);
    op_load_map["Permute"] = std::pair<int, op_load_t>(OP_PERMUTE, load_permute);
    op_load_map["Flatten"] = std::pair<int, op_load_t>(OP_FLATTEN, load_flatten);
    op_load_map["Clip"] = std::pair<int, op_load_t>(OP_CLIP, load_clip);
    op_load_map["PriorBox"] = std::pair<int, op_load_t>(OP_PRIORBOX, load_priorbox);
    op_load_map["Reshape"] = std::pair<int, op_load_t>(OP_RESHAPE, load_reshape);
    op_load_map["Resize"] = std::pair<int, op_load_t>(OP_RESIZE, load_resize);
    op_load_map["DetectionOutput"] = std::pair<int, op_load_t>(OP_DETECTION_OUTPUT, load_detectionoutput);
    op_load_map["RPN"] = std::pair<int, op_load_t>(OP_RPN, load_rpn);
    op_load_map["ROIPooling"] = std::pair<int, op_load_t>(OP_ROIPOOLING, load_roipooling);
    op_load_map["Reorg"] = std::pair<int, op_load_t>(OP_REORG, load_reorg);
    op_load_map["Sigmoid"] = std::pair<int, op_load_t>(OP_SIGMOID, load_sigmoid);
    op_load_map["TanH"] = std::pair<int, op_load_t>(OP_TANH, load_tanh);
    op_load_map["Upsample"] = std::pair<int, op_load_t>(OP_UPSAMPLE, load_upsample);
    // op_load_map["Power"]                        = std::pair<int, op_load_t>(OP_POWER,         load_power);
    op_load_map["Tile"] = std::pair<int, op_load_t>(OP_TILE, load_tile);
    op_load_map["ShuffleChannel"] = std::pair<int, op_load_t>(OP_SHUFFLECHANNEL, load_shufflechannel);
    op_load_map["Crop"] = std::pair<int, op_load_t>(OP_CROP, load_crop);
    op_load_map["AbsVal"] = std::pair<int, op_load_t>(OP_ABSVAL, load_absval);
    op_load_map["Interp"] = std::pair<int, op_load_t>(OP_INTERP, load_interp);
    op_load_map["ELU"] = std::pair<int, op_load_t>(OP_ELU, load_elu);
    op_load_map["Threshold"] = std::pair<int, op_load_t>(OP_THRESHOLD, load_threshold);
    op_load_map["Embedding"] = std::pair<int, op_load_t>(OP_EMBEDDING, load_embedding);
    op_load_map["MVN"] = std::pair<int, op_load_t>(OP_MVN, load_mvn);
    op_load_map["Reduction"] = std::pair<int, op_load_t>(OP_REDUCTION, load_reduction);
}
/*
*   OPERATOR REGISTER FUNCTION DEFINE FOR CAFFE SERIALIZER END
*/
