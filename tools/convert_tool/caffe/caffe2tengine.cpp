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
    bool                                     ret = google::protobuf::TextFormat::Parse(&input_stream, &caffe_net);
    is.close();

    if (!ret)
        TLOG_ERR("model file:  %s failed\n", model_file.c_str());

    return 0;
}

int caffe_serializer::load_binary_file(std::string model_file, te_caffe::NetParameter& caffe_net)
{
    // printf("binary : %s \n", model_file.c_str());
    std::ifstream is(model_file.c_str(), std::ios::in | std::ios::binary);

    if (!is.is_open())
    {
        TLOG_ERR("cannot open file: %s \n", model_file.c_str());
        return -1;
    }

    google::protobuf::io::IstreamInputStream input_stream(&is);
    google::protobuf::io::CodedInputStream   coded_input(&input_stream);
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

int caffe_serializer::load_graph_node(ir_graph_t* graph, const te_caffe::NetParameter test_net,
                                      const te_caffe::NetParameter train_net)
{
    name_map_t tensor_name_map;
    int        layer_number = train_net.layer_size();

    // fprintf(stderr, "op train_net map size: %d \n", layer_number);
    for (int i = 0; i < layer_number; i++)
    {
        const te_caffe::LayerParameter& layer_param = train_net.layer(i);

        train_name_map[layer_param.name()]          = &layer_param;
    }
    layer_number = test_net.layer_size();

    int size     = (int)op_load_map.size();
    // fprintf(stderr, "op test_net map size: %d \n", layer_number);

    std::vector<std::string> no_supported_op;
    for (int i = 0; i < layer_number; i++)
    {
        const te_caffe::LayerParameter& layer_param   = test_net.layer(i);
        const std::string&              caffe_op_name = layer_param.type();
        if (!find_op_load_method(caffe_op_name))
        {
            // printf("%s \n", caffe_op_name.c_str());
            auto it = find(no_supported_op.begin(), no_supported_op.end(), caffe_op_name);
            if (it == no_supported_op.end())
            {
                if (caffe_op_name == "Constant")
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
    // printf("layer number : %d \n", layer_number);
    for (n = 0; n < layer_number; n++)
    {
        const te_caffe::LayerParameter& layer_param   = test_net.layer(n);
        const std::string&              caffe_op_name = layer_param.type();
        fprintf(stderr, "%s \n", caffe_op_name.c_str());
        ir_node_t* ir_node = create_ir_node(graph, caffe_op_name.c_str(), op_load_map[caffe_op_name].first, OP_VERSION);
        // if (ir_node == NULL)
        //     return -1;
        // fprintf(stderr, "layer_param.bottom_size: %d \n", layer_param.bottom_size());
        for (int i = 0; i < layer_param.bottom_size(); i++)
        {
            const std::string& orig_name = layer_param.bottom(i);
            // fprintf(stderr, "%s \n", orig_name.c_str());
            std::string& tensor_name = tensor_name_map[orig_name];
            // printf("%s \n", tensor_name.c_str());

            // ir_tensor_t* tensor = find_caffe_tensor(graph, orig_name);

            int          tensor_id = get_ir_tensor_index_from_name(graph, orig_name.c_str());
            ir_tensor_t* tensor    = get_ir_graph_tensor(graph, tensor_id);
            // fprintf(stderr, "input tensor : %s \n", tensor->name);

            set_ir_node_input_tensor(ir_node, i, tensor);

            if (train_name_map.count(layer_param.name()))
            {
                // printf("train data copy in: %s \n", layer_param.name().c_str());

                // fprintf(stderr, "train_name_map : %s \n", layer_param.name().c_str());
                const te_caffe::LayerParameter* p_train;

                p_train = train_name_map[layer_param.name()];
                // printf("train_set : %s \n", layer_param.name().c_str());
                if (p_train->blobs_size())
                {
                    blob_load_t func = blob_load_map[caffe_op_name];
                    if (!func(graph, ir_node, *p_train))
                    {
                        break;
                    }
                }
            }
            // printf("finish tensor input\n");
            // output_tensors.push_back(tensor);
            // fprintf(stderr, "output_tensors num: %d %s\n", (int)output_tensors.size(), output_tensors[(int)output_tensors.size()-1]->name);
        }


        // fprintf(stderr, "layer_param.top_size() %d %s \n", layer_param.top_size(), caffe_op_name.c_str());
        for (int i = 0; i < layer_param.top_size(); i++)
        {
            const std::string& orig_name = layer_param.top(i);
            std::string        tensor_name;

            if (tensor_name_map.count(orig_name))
            {
                std::string ir_node_name = ir_node->name;
                tensor_name              = ir_node_name + "/" + std::to_string(i);
            }
            else
            {
                tensor_name = orig_name;
            }
            // fprintf(stderr, "%s \n", orig_name.c_str());
            // fprintf(stderr, "output tensor %s \n", tensor_name.c_str());

            ir_tensor_t* tensor = create_ir_tensor(graph, tensor_name.c_str(), TENGINE_DT_FP32);
            set_ir_node_output_tensor(ir_node, i, tensor);
            tensor_name_map[orig_name] = tensor_name;
            // output_tensors.push_back(tensor);
            // record the name mapping
            // printf("output_tensors num: %d %s\n", (int)output_tensors.size(), output_tensors[(int)output_tensors.size()-1]->name);
        }
        op_load_t loader = op_load_map[caffe_op_name].second;
        if (loader(graph, ir_node, layer_param) < 0)
        {
            TLOG_ERR("load op %s func failed in node %s .\n", caffe_op_name.c_str(), ir_node->name);
            return -1;
        }
#if 0
        if(train_name_map.count(layer_param.name()))
        {
            // fprintf(stderr, "train_name_map : %s \n", layer_param.name().c_str());
            const te_caffe::LayerParameter* p_train;

            p_train = train_name_map[layer_param.name()];
            // printf("train_set : %s \n", layer_param.name().c_str());
            if (p_train->blobs_size())
            {
                blob_load_t func = blob_load_map[caffe_op_name];
                if (!func(graph, ir_node, *p_train)){
                    break;
                }
            }
        }
#endif
    }
    // printf("tensor \n");
    if (n < layer_number)
    {
        fprintf(stderr, "Check layer number error ! \n");
        return -1;
    }
}
int caffe_serializer::load_tensor_data(ir_graph_t* graph, const te_caffe::NetParameter test_net,
                                       const te_caffe::NetParameter train_net)
{
    name_map_t tensor_name_map;
    int        layer_number = train_net.layer_size();

    // fprintf(stderr, "op train_net map size: %d \n", layer_number);
    for (int i = 0; i < layer_number; i++)
    {
        const te_caffe::LayerParameter& layer_param = train_net.layer(i);

        train_name_map[layer_param.name()]          = &layer_param;
    }
    layer_number = test_net.layer_size();

    int size     = (int)op_load_map.size();

    int n;
    // printf("layer number : %d \n", layer_number);
    for (n = 0; n < layer_number; n++)
    {
        const te_caffe::LayerParameter& layer_param   = test_net.layer(n);
        const std::string&              caffe_op_name = layer_param.type();
        // fprintf(stderr, "%s \n", caffe_op_name.c_str());
        ir_node_t* ir_node = create_ir_node(graph, caffe_op_name.c_str(), op_load_map[caffe_op_name].first, OP_VERSION);
        if (ir_node == NULL)
            return -1;

        if (train_name_map.count(layer_param.name()))
        {
            const te_caffe::LayerParameter* p_train;

            p_train = train_name_map[layer_param.name()];
            if (p_train->blobs_size())
            {
                blob_load_t func = blob_load_map[caffe_op_name];
                if (!func(graph, ir_node, *p_train))
                {
                    break;
                }
            }
        }
    }
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
    // if (load_tensor_data(graph, test_net, train_net) < 0)
    // return -1;
    fprintf(stderr, "Process 3: Finish load graph node \n");
    if (load_graph_node(graph, test_net, train_net) < 0)
        return -1;
    fprintf(stderr, "Process 3: Finish load graph node \n");

    // if (load_constant_tensor(graph, onnx_graph) < 0)
    //     return -1;
    // if (set_graph_input(graph, onnx_graph) < 0)
    //     return -1;
    // if (load_graph_node(graph, onnx_graph) < 0)
    //     return -1;
    // if (set_graph_output(graph, onnx_graph) < 0)
    //     return -1;

    return 0;
}

graph_t caffe_serializer::caffe2tengine(std::string model_file, std::string proto_file)
{
    fprintf(stderr, "----------caffe2tengine begin----------\n");

    context_t   context  = create_context(NULL, 1);
    ir_graph_t* ir_graph = create_ir_graph((struct context*)context);
    if (ir_graph == NULL)
    {
        destroy_context(context);
        return NULL;
    }
    ir_graph->attribute->private_context = 1;    // new context

    int ret                              = load_model(ir_graph, model_file, proto_file);
    if (0 != ret)
    {
        destroy_graph(ir_graph);
        return NULL;
    }
    ir_graph->device = find_default_device();

    fprintf(stderr, "----------caffe2tengine done.----------\n");
    return ir_graph;
}


static void LoadCaffeBlob(ir_graph_t* ir_graph, ir_node_t* ir_node, const std::vector<std::string>& name_list,
                          const std::vector<std::string>& layout_list, const te_caffe::LayerParameter& layer_param)

{
    unsigned int blob_num = layer_param.blobs_size();
    // printf("blob num: %d, name list: %d \n", blob_num, (int)name_list.size());
    for (unsigned int i = 0; i < blob_num && i < name_list.size(); i++)
    {
        std::string node_name       = ir_node->name;
        std::string new_tensor_name = node_name + "/" + name_list[i];

        ir_tensor_t* ir_tensor      = create_ir_tensor(ir_graph, new_tensor_name.c_str(), TENGINE_DT_FP32);

        /* load tensor data*/

        const te_caffe::BlobProto& blob = layer_param.blobs(i);

        int  dim_num                    = 0;
        int* dims;
        if (blob.has_shape())
        {
            dim_num = blob.shape().dim_size();
            dims    = (int*)malloc(sizeof(int) * dim_num);
            memset(dims, 0, sizeof(int) * dim_num);
            for (int i = 0; i < dim_num; i++)
            {
                dims[i] = blob.shape().dim(i);
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
                start++;

            dim_num = temp.size() - start;
            dims    = (int*)malloc(sizeof(int) * dim_num);
            memset(dims, 0, sizeof(int) * dim_num);
            for (unsigned int i = start; i < temp.size(); i++)
                dims[i] = temp[i];
        }
        if (dim_num > 0)
        {
            set_ir_tensor_shape(ir_tensor, dims, dim_num);
            ir_tensor->tensor_type = TENSOR_TYPE_CONST;
            int tensor_size        = ir_tensor->elem_num * sizeof(float);
            ir_tensor->data        = sys_malloc(tensor_size);
            float* ptr             = (float*)ir_tensor->data;

            for (int i = 0; i < blob.data_size(); i++)
            {
                ptr[i] = blob.data(i);
            }
        }

        ir_node_t* new_ir_node = create_ir_node(ir_graph, new_tensor_name.c_str(), OP_CONST, OP_VERSION);

        // int  index = get_ir_node_index_from_name(ir_graph, new_tensor_name.c_str());

        set_ir_node_output_tensor(new_ir_node, 0, ir_tensor);
        set_ir_node_input_tensor(ir_node, i + 1, ir_tensor);
    }
}



static void CreatePresetNode(ir_graph_t* graph, ir_node_t* ir_node, const char* name, const char* layout,
                             std::vector<int>& temp, float val, int index)
{
    std::string  node_name       = ir_node->name;
    std::string  new_tensor_name = node_name + "/" + name;
    ir_tensor_t* ir_tensor       = create_ir_tensor(graph, new_tensor_name.c_str(), TENGINE_DT_FP32);

    int dim_num                  = temp.size();
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
        set_ir_tensor_shape(ir_tensor, dims, dim_num);
        ir_tensor->tensor_type = TENSOR_TYPE_CONST;
        int tensor_size        = elem_size * sizeof(float);
        ir_tensor->data        = sys_malloc(tensor_size);

        float* ptr             = (float*)ir_tensor->data;
        for (int i = 0; i < elem_size; i++)
            ptr[i] = val;
    }

    ir_node_t* new_ir_node = create_ir_node(graph, new_tensor_name.c_str(), OP_CONST, OP_VERSION);

    set_ir_node_output_tensor(new_ir_node, 0, ir_tensor);
    set_ir_node_input_tensor(new_ir_node, 0, ir_tensor);
}


bool load_batchnorm_blob(ir_graph_t* graph, ir_node_t* node, const te_caffe::LayerParameter& layer_param)
{
    const te_caffe::BlobProto& rescale_blob = layer_param.blobs(2);


    struct batchnorm_param* batchnorm_param = (struct batchnorm_param*)node->op.param_mem;

    batchnorm_param->rescale_factor         = rescale_blob.data(0);

    /* for compatible reason, create the two tensors: gamma (1.0) and beta (0.0) */

    /* get the dim, i.e., channel size */

    const te_caffe::BlobProto& mean_blob = layer_param.blobs(0);

    std::vector<int> temp;
    temp.push_back(mean_blob.shape().dim(0));

    CreatePresetNode(graph, node, "gamma", "W", temp, 1.0f, 0);
    CreatePresetNode(graph, node, "beta", "W", temp, 0.0f, 0);
    if (layer_param.blobs_size())
    {
        std::vector<std::string> name_list   = { "means", "vars" };
        std::vector<std::string> layout_list = { "W", "W" };

        LoadCaffeBlob(graph, node, name_list, layout_list, layer_param);
    }
    return 0;
}

int load_batchnorm(ir_graph_t* graph, ir_node_t* node, const te_caffe::LayerParameter& layer_param)
{
    struct batchnorm_param* batchnorm_param      = (struct batchnorm_param*)node->op.param_mem;

    const te_caffe::BatchNormParameter& bn_param = layer_param.batch_norm_param();

    batchnorm_param->eps                         = bn_param.eps();
    batchnorm_param->caffe_flavor                = 1;

    if (layer_param.blobs_size())
    {
        load_batchnorm_blob(graph, node, layer_param);
    }

    return 0;
}

int load_softmax(ir_graph_t* graph, ir_node_t* node, const te_caffe::LayerParameter& layer_param)
{
    const te_caffe::SoftmaxParameter& softmax_param = layer_param.softmax_param();

    struct softmax_param* param                     = (struct softmax_param*)node->op.param_mem;

    if (softmax_param.has_axis())
        param->axis = softmax_param.axis();
    else
        param->axis = 1;

    return 0;
}


int load_conv(ir_graph_t* graph, ir_node_t* node, const te_caffe::LayerParameter& layer_param)
{
    const te_caffe::ConvolutionParameter& conv_param = layer_param.convolution_param();
    // const te_caffe::LayerParameter& layer_param = caffe_net.layer(i);
    const std::string& caffe_op_name = layer_param.type();
    struct conv_param* param         = (struct conv_param*)node->op.param_mem;

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
    // printf("conv param: %d %d %d %d %d %d \n", param->stride_h, param->stride_w, param->pad_h0, param->pad_h1, param->pad_w0, param->pad_w1);
    /* create new Node and tensor for pre-trained weights */
    if (layer_param.blobs_size())
    {
        std::vector<std::string> name_list   = { "weight", "bias" };
        std::vector<std::string> layout_list = { "NCHW", "W" };

        LoadCaffeBlob(graph, node, name_list, layout_list, layer_param);
    }
    return 0;
}

int load_deconv(ir_graph_t* graph, ir_node_t* node, const te_caffe::LayerParameter& layer_param)
{
    const te_caffe::ConvolutionParameter& conv_param = layer_param.convolution_param();

    struct deconv_param* param                       = (struct deconv_param*)node->op.param_mem;

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

    /* create new Node and tensor for pre-trained weights */
    if (layer_param.blobs_size())
    {
        std::vector<std::string> name_list   = { "weight", "bias" };
        std::vector<std::string> layout_list = { "NCHW", "C" };

        LoadCaffeBlob(graph, node, name_list, layout_list, layer_param);
    }
    return 0;
}
PoolArg ConvertCaffePool(te_caffe::PoolingParameter_PoolMethod method)
{
    if (method == te_caffe::PoolingParameter_PoolMethod_AVE)
        return kPoolAvg;
    else if (method == te_caffe::PoolingParameter_PoolMethod_STOCHASTIC)
        return kPoolRand;

    /* for others, return MAX */

    return kPoolMax;
}


int load_fc(ir_graph_t* graph, ir_node_t* node, const te_caffe::LayerParameter& layer_param)
{
    const te_caffe::InnerProductParameter& ip_param = layer_param.inner_product_param();

    struct fc_param* param                          = (struct fc_param*)node->op.param_mem;
    param->num_output                               = ip_param.num_output();

    /* Load weight and bias blob */
    if (layer_param.blobs_size())
    {
        std::vector<std::string> name_list   = { "weight", "bias" };
        std::vector<std::string> layout_list = { "HW", "W" };

        LoadCaffeBlob(graph, node, name_list, layout_list, layer_param);
    }
    return 0;
}
int load_prelu(ir_graph_t* graph, ir_node_t* node, const te_caffe::LayerParameter& layer_param)
{
    if (layer_param.blobs_size())
    {
        std::vector<std::string> name_list   = { "slope" };
        std::vector<std::string> layout_list = { "W" };
        LoadCaffeBlob(graph, node, name_list, layout_list, layer_param);
    }

    return 0;
}
int load_normalize(ir_graph_t* graph, ir_node_t* node, const te_caffe::LayerParameter& layer_param)
{
    const te_caffe::NormalizeParameter& normalize_param = layer_param.norm_param();

    struct normalize_param* param                       = (struct normalize_param*)node->op.param_mem;

    param->across_spatial                               = normalize_param.across_spatial();
    param->channel_shared                               = normalize_param.channel_shared();

    return 0;
}


int load_scale(ir_graph_t* graph, ir_node_t* node, const te_caffe::LayerParameter& layer_param)
{
    struct scale_param* param                   = (struct scale_param*)node->op.param_mem;

    const te_caffe::ScaleParameter& scale_param = layer_param.scale_param();

    if (scale_param.has_axis())
        param->axis = scale_param.axis();

    if (scale_param.has_num_axes())
        param->num_axes = scale_param.num_axes();

    if (scale_param.has_bias_term())
        param->bias_term = scale_param.bias_term();


    if (layer_param.blobs_size())
    {
        std::vector<std::string> name_list   = { "gamma", "beta" };
        std::vector<std::string> layout_list = { "CHW", "W" };

        LoadCaffeBlob(graph, node, name_list, layout_list, layer_param);
    }

    return 0;
}

int load_relu(ir_graph_t* graph, ir_node_t* node, const te_caffe::LayerParameter& layer_param)
{
    struct relu_param* param                   = (struct relu_param*)node->op.param_mem;

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
    param->is_caffe           = true;

    return 0;
}


#if 0
int load_data(ir_graph_t* graph, ir_node_t* node, const te_caffe::LayerParameter& layer_param)
{

    const te_caffe::InputParameter& input_param = layer_param.input_param();

    if (input_param.shape_size())
    {
        std::vector<int> dim;
        const te_caffe::BlobShape& blob_shape = input_param.shape(0);

        for (int i = 0; i < blob_shape.dim_size(); i++)
        {
            dim.push_back(blob_shape.dim(i));
        }
    }

    // AddGraphInputNode(graph, node);

    return 0;
}
#endif

int load_pool(ir_graph_t* graph, ir_node_t* node, const te_caffe::LayerParameter& layer_param)
{
    const te_caffe::PoolingParameter& pool_param = layer_param.pooling_param();

    struct pool_param* param                     = (struct pool_param*)node->op.param_mem;

    // param.alg = ConvertCaffePool(pool_param.pool());
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
    struct eltwise_param*             param         = (struct eltwise_param*)node->op.param_mem;
    // defalt: SUM
    param->type = ELT_SUM;
    if (eltwise_param.has_operation())
        param->type = ConvertCaffeEltwise(eltwise_param.operation());

    param->caffe_flavor = 1;
    param->shift        = eltwise_param.shift();
    param->scale        = eltwise_param.scale();
    param->power        = eltwise_param.power();

    return 0;
}

int load_input(ir_graph_t* graph, ir_node_t* ir_node, const te_caffe::LayerParameter& layer_param)
{
    std::vector<int16_t>            input_nodes;
    const te_caffe::InputParameter& input_param = layer_param.input_param();

    std::string val                             = layer_param.type();
    // if(get_ir_tensor_index_from_name(graph, layer_param.c_str()) != -1)
    //     continue;
    printf("graph input tensor name: %s \n", val.c_str());
    ir_tensor_t* tensor = create_ir_tensor(graph, val.c_str(), TENGINE_DT_FP32);

    int has_shape       = 1;

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

#if 1
    if (has_shape)
    {
        int* dims = (int*)malloc(sizeof(int) * dim_num);
        memset(dims, 0, sizeof(int) * dim_num);
        for (int i = 0; i < dim_num; i++)
            dims[i] = dim[i];
        set_ir_tensor_shape(tensor, dims, dim_num);
    }
#endif

    ir_node_t* node = create_ir_node(graph, val.c_str(), OP_INPUT, OP_VERSION);
    set_ir_node_output_tensor(node, 0, tensor);
    input_nodes.push_back(node->index);

    int16_t* node_idx = (int16_t*)sys_malloc(sizeof(int16_t) * input_nodes.size());
    for (int i = 0; i < input_nodes.size(); i++)
    {
        node_idx[i] = input_nodes[i];
    }
    // printf("input index: %d \n", node_idx[0]);
    set_ir_graph_input_node(graph, node_idx, input_nodes.size());

    return true;
}
int LoadConvolutionBlob(ir_graph_t* graph, ir_node_t* node, const te_caffe::LayerParameter& layer_param)
{
    std::vector<std::string> name_list   = { "weight", "bias" };
    std::vector<std::string> layout_list = { "NCHW", "W" };

    LoadCaffeBlob(graph, node, name_list, layout_list, layer_param);

    return true;
}
int LoadDeconvolutionBlob(ir_graph_t* graph, ir_node_t* node, const te_caffe::LayerParameter& layer_param)
{
    if (layer_param.blobs_size())
    {
        std::vector<std::string> name_list   = { "weight", "bias" };
        std::vector<std::string> layout_list = { "NCHW", "C" };

        LoadCaffeBlob(graph, node, name_list, layout_list, layer_param);
    }
    return true;
}


int LoadBiasBlob(ir_graph_t* graph, ir_node_t* node, const te_caffe::LayerParameter& layer_param)
{
    if (layer_param.blobs_size())
    {
        std::vector<std::string> name_list   = { "weight" };
        std::vector<std::string> layout_list = { "NCHW" };

        LoadCaffeBlob(graph, node, name_list, layout_list, layer_param);
    }
    return true;
}


int LoadFullyConnectedBlob(ir_graph_t* graph, ir_node_t* node, const te_caffe::LayerParameter& layer_param)
{
    if (layer_param.blobs_size())
    {
        std::vector<std::string> name_list   = { "weight", "bias" };
        std::vector<std::string> layout_list = { "HW", "W" };

        LoadCaffeBlob(graph, node, name_list, layout_list, layer_param);
    }
    return true;
}

int LoadScaleBlob(ir_graph_t* graph, ir_node_t* node, const te_caffe::LayerParameter& layer_param)
{
    std::vector<std::string> name_list   = { "gamma", "beta" };
    std::vector<std::string> layout_list = { "CHW", "W" };

    LoadCaffeBlob(graph, node, name_list, layout_list, layer_param);

    return true;
}
int LoadPReLuBlob(ir_graph_t* graph, ir_node_t* node, const te_caffe::LayerParameter& layer_param)
{
    if (layer_param.blobs_size())
    {
        std::vector<std::string> name_list   = { "slope" };
        std::vector<std::string> layout_list = { "W" };
        LoadCaffeBlob(graph, node, name_list, layout_list, layer_param);
    }
    return true;
}
int LoadNormalizeBlob(ir_graph_t* graph, ir_node_t* node, const te_caffe::LayerParameter& layer_param)
{
    if (layer_param.blobs_size())
    {
        std::vector<std::string> name_list   = { "scale" };
        std::vector<std::string> layout_list = { "W" };
        LoadCaffeBlob(graph, node, name_list, layout_list, layer_param);
    }
    return true;
}

int LoadBatchNormBlob(ir_graph_t* graph, ir_node_t* node, const te_caffe::LayerParameter& layer_param)
{
    const te_caffe::BlobProto& rescale_blob = layer_param.blobs(2);

    struct batchnorm_param* param           = (struct batchnorm_param*)node->op.param_mem;

    param->rescale_factor                   = rescale_blob.data(0);


    /* for compatible reason, create the two tensors: gamma (1.0) and beta (0.0) */

    /* get the dim, i.e., channel size */

    const te_caffe::BlobProto& mean_blob = layer_param.blobs(0);

    std::vector<int> dims;
    dims.push_back(mean_blob.shape().dim(0));

    CreatePresetNode(graph, node, "gamma", "W", dims, 1.0f, 0);
    CreatePresetNode(graph, node, "beta", "W", dims, 0.0f, 0);
    if (layer_param.blobs_size())
    {
        std::vector<std::string> name_list   = { "means", "vars" };
        std::vector<std::string> layout_list = { "W", "W" };

        LoadCaffeBlob(graph, node, name_list, layout_list, layer_param);
    }
    return true;
}
/*
*   OPERAOTR REGISTER FUNCTION DEFINE FOR ONNX SERIALIZER START
*/
void caffe_serializer::register_op_load()
{
    op_load_map["BatchNorm"]       = std::pair<int, op_load_t>(OP_UNARY, load_batchnorm);
    op_load_map["Convolution"]     = std::pair<int, op_load_t>(OP_CONV, load_conv);
    op_load_map["DeConvolution"]   = std::pair<int, op_load_t>(OP_DECONV, load_deconv);
    op_load_map["Softmax"]         = std::pair<int, op_load_t>(OP_SOFTMAX, load_softmax);
    op_load_map["PReLU"]           = std::pair<int, op_load_t>(OP_PRELU, load_prelu);
    op_load_map["InnerProduct"]    = std::pair<int, op_load_t>(OP_FC, load_fc);
    op_load_map["SoftmaxWithLoss"] = std::pair<int, op_load_t>(OP_SOFTMAX, load_softmax);
    op_load_map["Normalize"]       = std::pair<int, op_load_t>(OP_NORMALIZE, load_normalize);
    op_load_map["Scale"]           = std::pair<int, op_load_t>(OP_SCALE, load_scale);
    op_load_map["ReLU"]            = std::pair<int, op_load_t>(OP_RELU, load_relu);
    op_load_map["Split"]           = std::pair<int, op_load_t>(OP_SPLIT, load_split);
    op_load_map["Pooling"]         = std::pair<int, op_load_t>(OP_POOL, load_pool);
    op_load_map["Eltwise"]         = std::pair<int, op_load_t>(OP_ELTWISE, load_eltwise);
    op_load_map["Input"]           = std::pair<int, op_load_t>(OP_INPUT, load_input);
    op_load_map["Data"]            = std::pair<int, op_load_t>(OP_INPUT, load_input);


    blob_load_map["Convolution"]   = LoadConvolutionBlob;
    // blob_load_map["Deconvolution"]              = LoadDeconvolutionBlob;
    blob_load_map["InnerProduct"] = LoadFullyConnectedBlob;
    blob_load_map["BatchNorm"]    = LoadBatchNormBlob;
    blob_load_map["Scale"]        = LoadScaleBlob;
    // blob_load_map["PReLU"]                      = LoadPReLuBlob;
    // blob_load_map["Normalize"]                  = LoadNormalizeBlob;
    // blob_load_map["ConvolutionDepthwise"]       = LoadConvolutionBlob;
    // blob_load_map["DepthwiseConvolution"]       = LoadConvolutionBlob;
    blob_load_map["Bias"] = LoadBiasBlob;
#if 0
    op_load_map["Data"]                         = std::pair<int, op_load_t>(OP_INPUT,           load_data);
    op_load_map["Slice"]                        = std::pair<int, op_load_t>(OP_SLICE,           load_slice);
    op_load_map["Concat"]                       = std::pair<int, op_load_t>(OP_CONCAT,          load_concat);
    op_load_map["Dropout"]                      = std::pair<int, op_load_t>(OP_DROPOUT,         load_dropout);
    op_load_map["BatchNorm"]                    = std::pair<int, op_load_t>(OP_BATCHNORM,       load_batchnorm);
    op_load_map["LRN"]                          = std::pair<int, op_load_t>(OP_LRN,             load_lrn);
    op_load_map["Permute"]                      = std::pair<int, op_load_t>(OP_PERMUTE,         load_permute);
    op_load_map["Flatten"]                      = std::pair<int, op_load_t>(OP_FLATTEN,         load_flatten);
    op_load_map["PriorBox"]                     = std::pair<int, op_load_t>(OP_PRIORBOX,        load_priorbox);
    op_load_map["Reshape"]                      = std::pair<int, op_load_t>(OP_RESHAPE,         load_reshape);
    op_load_map["DetectionOutput"]              = std::pair<int, op_load_t>(OP_DETECTION_OUTPUT,load_detectionoutput);
    op_load_map["RPN"]                          = std::pair<int, op_load_t>(OP_RPN,             load_rpn);
    op_load_map["ROIPooling"]                   = std::pair<int, op_load_t>(OP_ROIPOOLING,      load_roipooling);
    op_load_map["Reorg"]                        = std::pair<int, op_load_t>(OP_REORG,           load_reorg);
    op_load_map["Resize"]                       = std::pair<int, op_load_t>(OP_RESIZE,          load_resize);
    op_load_map["Sigmoid"]                      = std::pair<int, op_load_t>(OP_SIGMOID,         load_sigmoid);
    op_load_map["TanH"]                         = std::pair<int, op_load_t>(OP_TANH,            load_tanh);
    op_load_map["Upsample"]                     = std::pair<int, op_load_t>(OP_UPSAMPLE,        load_upsample);
    op_load_map["Power"]                        = std::pair<int, op_load_t>(OP_ELTWISE,         load_eltwise);
    op_load_map["ReLU6"]                        = std::pair<int, op_load_t>(OP_RELU6,           load_relu6);
    op_load_map["DepthwiseConvolution"]         = std::pair<int, op_load_t>(OP_DECONV,          load_deconv);
    op_load_map["ConvolutionDepthwise"]         = std::pair<int, op_load_t>(OP_CONV,            load_conv);
    op_load_map["Clip"]                         = std::pair<int, op_load_t>(OP_CLIP,            load_clip);
    op_load_map["Tile"]                         = std::pair<int, op_load_t>(OP_TILE,            load_tile);
    op_load_map["ShuffleChannel"]               = std::pair<int, op_load_t>(OP_SHUFFLECHANNEL,  load_shufflechannel);
    op_load_map["Crop"]                         = std::pair<int, op_load_t>(OP_CROP,            load_crop);
    op_load_map["AbsVal"]                       = std::pair<int, op_load_t>(OP_ABSVAL,          load_absval);
    op_load_map["Interp"]                       = std::pair<int, op_load_t>(OP_INTERP,          load_interp);
    op_load_map["ELU"]                          = std::pair<int, op_load_t>(OP_ELU,             load_elu);
    op_load_map["Threshold"]                    = std::pair<int, op_load_t>(OP_THRESHOLD,       load_threshold);
    op_load_map["Embedding"]                    = std::pair<int, op_load_t>(OP_EMBEDDING,       load_embedding);
    op_load_map["MVN"]                          = std::pair<int, op_load_t>(OP_MVN,             load_mvn);
    op_load_map["Reduction"]                    = std::pair<int, op_load_t>(OP_REDUCTION,       load_reduction);
    op_load_map["Bias"]                         = std::pair<int, op_load_t>(OP_BIAS,            load_bias);
#endif
}
/*
*   OPERAOTR REGISTER FUNCTION DEFINE FOR ONNX SERIALIZER END
*/