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
 * Author: xlchen@openailab.com
 */

#include "darknet2tengine.hpp"

const int OP_VERSION = 1;

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

int darknet_serializer::load_weight_file(ir_graph_t* graph, const char* weight_file, list* sections)
{
    FILE* fp = fopen(weight_file, "rb");
    int major;
    int minor;
    int revision;
    int seen;
    std::vector<std::string> tensor_name_map;
    if (0 == fread(&major, sizeof(int), 1, fp))
    {
        fprintf(stderr, "read major failed\n");
        return -1;
    }
    if (0 == fread(&minor, sizeof(int), 1, fp))
    {
        fprintf(stderr, "read minor failed\n");
        return -1;
    }
    if (0 == fread(&revision, sizeof(int), 1, fp))
    {
        fprintf(stderr, "read revision failed\n");
        return -1;
    }
    if ((major * 10 + minor) >= 2 && major < 1000 && minor < 1000)
    {
        double iseen = 0;
        if (0 == fread(&iseen, sizeof(double), 1, fp))
        {
            fprintf(stderr, "read iseen failed\n");
            return -1;
        }
        seen = (int)iseen;
    }
    else
    {
        if (0 == fread(&seen, sizeof(int), 1, fp))
        {
            fprintf(stderr, "read seen failed\n");
            return -1;
        }
    }
    int transpose = (major > 1000) || (minor > 1000);

    fprintf(stderr, "major: %d, minor: %d, revision: %d, seen: %d, transpose: %d \n", major, minor, revision, seen, transpose);

    dk_node* n = sections->front;
    section* s = (section*)n->val;
    list* options = s->options;
    ir_node_t* input_node = create_ir_node(graph, "input", OP_INPUT, OP_VERSION);
    ir_tensor_t* input_tensor = create_ir_tensor(graph, "input_0", TENGINE_DT_FP32);
    set_ir_node_output_tensor(input_node, 0, input_tensor);

    std::vector<int> dim;
    int input_h = option_find_int_quiet(options, (char*)"height", 0);
    int input_w = option_find_int_quiet(options, (char*)"width", 0);
    int input_c = option_find_int_quiet(options, (char*)"channels", 0);
    int batch_num = option_find_int(options, (char*)"batch", 1);

    dim.push_back(batch_num);
    dim.push_back(input_c);
    dim.push_back(input_h);
    dim.push_back(input_w);
    set_ir_tensor_shape(input_tensor, dim.data(), dim.size());
    input_tensor->tensor_type = TENSOR_TYPE_INPUT;
    tensor_name_map.push_back("input_0");

    std::vector<int16_t> input_nodes;
    input_nodes.push_back(input_node->index);
    set_ir_graph_input_node(graph, input_nodes.data(), input_nodes.size());

    free_section(s);
    n = n->next;
    int count = 1;
    while (n)
    {
        s = (section*)n->val;
        options = s->options;
        fprintf(stderr, "s type:%d %s\n", count, s->type);
        std::string op_name = s->type;
        std::string node_name = s->type + std::to_string(count);
        std::string a = "[";
        std::string b = "]";
        std::string c = "_";

        remove_str(node_name, a);
        replace_str(node_name, b, c);
        ir_node_t* node = create_ir_node(graph, node_name.c_str(), op_load_map[op_name].first, OP_VERSION);

        std::string tensor_name = node_name + "_0";
        tensor_name_map.push_back(tensor_name);
        ir_tensor_t* tensor = create_ir_tensor(graph, tensor_name.c_str(), TENGINE_DT_FP32);
        set_ir_node_output_tensor(node, 0, tensor);

        op_load_t loader = op_load_map[op_name].second;
        if (loader(graph, node, tensor_name_map, options, count, fp) < 0)
        {
            fprintf(stderr, "load op %s func failed in node %s .\n", op_name.c_str(), node_name.c_str());
            return -1;
        }
        free_section(s);
        count++;
        n = n->next;
    }
    fclose(fp);

    return true;
}

int darknet_serializer::set_graph_output(ir_graph_t* graph)
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

int darknet_serializer::load_model(ir_graph_t* graph, std::string model_file, std::string proto_file)
{
    register_op_load();

    const char* cfg_file = proto_file.c_str();
    const char* weight_file = model_file.c_str();

    // load cfg
    list* sections = read_cfg(cfg_file);

    // load weight
    if (load_weight_file(graph, weight_file, sections) < 0)
        return -1;
    if (set_graph_output(graph) < 0)
        return -1;

    graph->model_format = MODEL_FORMAT_DARKNET;
    graph->graph_layout = TENGINE_LAYOUT_NCHW;
    graph->model_layout = TENGINE_LAYOUT_NCHW;
    return 0;
}

graph_t darknet_serializer::darknet2tengine(std::string model_file, std::string proto_file)
{
    fprintf(stderr, "----------darknet2tengine begin----------\n");

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

    fprintf(stderr, "----------darknet2tengine done.----------\n");
    return ir_graph;
}

static int load_conv_blob(ir_graph_t* graph, ir_node_t* node, std::vector<int>& weight_dims, int batch_norm, FILE* fp)
{
    if (fp == NULL)
        return -1;
    // Add the weight tensor
    std::string weight_tensor_name = std::string{node->name} + "_1";
    int weight_node_id = add_const_node_above(graph, node->index, weight_tensor_name.c_str());
    ir_node_t* weight_node = get_ir_graph_node(graph, weight_node_id);
    ir_tensor_t* weight_tensor = get_ir_graph_tensor(graph, weight_node->output_tensors[0]);
    set_ir_tensor_shape(weight_tensor, weight_dims.data(), weight_dims.size());
    set_ir_node_input_tensor(node, 1, weight_tensor);

    // Add the bias tensor
    std::vector<int> bias_dims;
    bias_dims.push_back(1);
    bias_dims.push_back(1);
    bias_dims.push_back(1);
    bias_dims.push_back(weight_dims[0]);
    std::string bias_tensor_name = std::string{node->name} + "_2";
    int bias_node_id = add_const_node_above(graph, node->index, bias_tensor_name.c_str());
    ir_node_t* bias_node = get_ir_graph_node(graph, bias_node_id);
    ir_tensor_t* bias_tensor = get_ir_graph_tensor(graph, bias_node->output_tensors[0]);
    set_ir_tensor_shape(bias_tensor, bias_dims.data(), bias_dims.size());
    set_ir_node_input_tensor(node, 2, bias_tensor);

    int out_channel = weight_dims[0];
    bias_tensor->data = (void*)sys_malloc(out_channel * sizeof(float));
    float* bias_data = (float*)bias_tensor->data;
    if (0 == fread(bias_data, sizeof(float), out_channel, fp))
    {
        printf("Read bias data failed\n");
        return -1;
    }
    float* scales = NULL;
    float* means = NULL;
    float* variances = NULL;
    if (batch_norm)
    {
        scales = (float*)sys_malloc(sizeof(float) * out_channel);
        means = (float*)sys_malloc(sizeof(float) * out_channel);
        variances = (float*)sys_malloc(sizeof(float) * out_channel);
        if (0 == fread(scales, sizeof(float), out_channel, fp))
            printf("Read scales failed\n");
        if (0 == fread(means, sizeof(float), out_channel, fp))
            printf("Read means failed\n");
        if (0 == fread(variances, sizeof(float), out_channel, fp))
            printf("Read variances failed\n");
    }
    int weight_size = weight_dims[0] * weight_dims[1] * weight_dims[2] * weight_dims[3];
    weight_tensor->data = (void*)sys_malloc(weight_size * sizeof(float) + 128);
    float* weight_data = (float*)weight_tensor->data;
    if (0 == fread(weight_data, sizeof(float), weight_size, fp))
        printf("Read weight data failed\n");

    // fuse the batchnorm
    if (batch_norm)
    {
        int kernel_size = weight_dims[1] * weight_dims[2] * weight_dims[3];
        for (int i = 0; i < out_channel; ++i)
        {
            float scale = scales[i] / sqrt(variances[i] + .00001);
            for (int j = 0; j < kernel_size; ++j)
            {
                weight_data[i * kernel_size + j] *= scale;
            }
            bias_data[i] -= means[i] * scale;
        }
    }

    return 0;
}

static int load_conv(ir_graph_t* graph, ir_node_t* node, std::vector<std::string>& tensor_name_map, list* options, int index, FILE* fp)
{
    ir_tensor_t* tensor = find_tensor(graph, tensor_name_map[index - 1]);
    if (tensor == nullptr)
        return -1;
    set_ir_node_input_tensor(node, 0, tensor);
    conv_param* param = (conv_param*)node->op.param_mem;
    int n = option_find_int(options, (char*)"filters", 1);
    int size = option_find_int(options, (char*)"size", 1);
    int stride = option_find_int(options, (char*)"stride", 1);
    int pad = option_find_int_quiet(options, (char*)"pad", 0);
    int padding = option_find_int_quiet(options, (char*)"padding", 0);
    int groups = option_find_int_quiet(options, (char*)"groups", 1);
    int batch_normalize = option_find_int_quiet(options, (char*)"batch_normalize", 0);
    char* activation_s = option_find_str(options, (char*)"activation", (char*)"logistic");
    // ACTIVATION activation = get_activation(activation_s);
    if (pad)
        padding = size / 2;

    param->kernel_h = size;
    param->kernel_w = size;
    param->stride_h = stride;
    param->stride_w = stride;
    param->pad_h0 = padding;
    param->pad_h1 = padding;
    param->pad_w0 = padding;
    param->pad_w1 = padding;
    param->group = groups;
    param->output_channel = n;

    if (tensor->dim_num != 4)
        return -1;
    int batch = tensor->dims[0];
    int in_c = tensor->dims[1];
    int in_h = tensor->dims[2];
    int in_w = tensor->dims[3];
    // Create the weight tensor
    std::vector<int> weight_dims;
    weight_dims.push_back(n);
    weight_dims.push_back(in_c / groups);
    weight_dims.push_back(size);
    weight_dims.push_back(size);
    if (load_conv_blob(graph, node, weight_dims, batch_normalize, fp) < 0)
        return -1;
    // Set the Ouput Tensor Dim
    std::vector<int> out_dims;
    int out_c = n;
    int out_h = (in_h + 2 * padding - size) / stride + 1;
    int out_w = (in_w + 2 * padding - size) / stride + 1;
    out_dims.push_back(batch);
    out_dims.push_back(out_c);
    out_dims.push_back(out_h);
    out_dims.push_back(out_w);

    ir_tensor_t* out_tensor = get_ir_graph_tensor(graph, node->output_tensors[0]);
    set_ir_tensor_shape(out_tensor, out_dims.data(), out_dims.size());

    if (strcmp(activation_s, "leaky") == 0)
    {
        std::string relu_name = "leaky_" + std::to_string(index);
        ir_node_t* relu_node = create_ir_node(graph, relu_name.c_str(), OP_RELU, OP_VERSION);
        relu_param* param = (relu_param*)relu_node->op.param_mem;
        param->negative_slope = 0.1f;
        set_ir_node_input_tensor(relu_node, 0, out_tensor);
        std::string relu_tensor_name = relu_name + "_0";
        ir_tensor_t* relu_out_tensor = create_ir_tensor(graph, relu_tensor_name.c_str(), TENGINE_DT_FP32);
        set_ir_tensor_shape(relu_out_tensor, out_dims.data(), out_dims.size());
        set_ir_node_output_tensor(relu_node, 0, relu_out_tensor);

        // update the tensor name map;
        tensor_name_map[index] = relu_tensor_name;
    }
    if (strcmp(activation_s, "mish") == 0)
    {
        std::string mish_name = "mish_" + std::to_string(index);
        ir_node_t* mish_node = create_ir_node(graph, mish_name.c_str(), OP_MISH, OP_VERSION);
        set_ir_node_input_tensor(mish_node, 0, out_tensor);
        std::string mish_tensor_name = mish_name + "_0";
        ir_tensor_t* mish_out_tensor = create_ir_tensor(graph, mish_tensor_name.c_str(), TENGINE_DT_FP32);
        set_ir_tensor_shape(mish_out_tensor, out_dims.data(), out_dims.size());
        set_ir_node_output_tensor(mish_node, 0, mish_out_tensor);

        // update the tensor name map;
        tensor_name_map[index] = mish_tensor_name;
    }
    return 0;
}

static int load_shortcut(ir_graph_t* graph, ir_node_t* node, std::vector<std::string>& tensor_name_map, list* options, int index, FILE* fp)
{
    ir_tensor_t* tensor = find_tensor(graph, tensor_name_map[index - 1]);
    if (tensor == nullptr)
    {
        printf("tensor is null \n");
        return -1;
    }
    set_ir_node_input_tensor(node, 0, tensor);
    char* l = option_find(options, (char*)"from");
    int from_index = atoi(l);
    if (from_index < 0)
        from_index = index + from_index;

    ir_tensor_t* tensor1 = find_tensor(graph, tensor_name_map[from_index]);
    if (tensor1 == NULL)
        return -1;
    set_ir_node_input_tensor(node, 1, tensor1);

    eltwise_param* param = (eltwise_param*)node->op.param_mem;
    param->type = ELT_SUM;
    param->caffe_flavor = 1;

    std::vector<int> out_dims;
    out_dims.push_back(tensor->dims[0]);
    out_dims.push_back(tensor->dims[1]);
    out_dims.push_back(tensor->dims[2]);
    out_dims.push_back(tensor->dims[3]);

    ir_tensor_t* out_tensor = get_ir_graph_tensor(graph, node->output_tensors[0]);
    set_ir_tensor_shape(out_tensor, out_dims.data(), out_dims.size());

    return 0;
}

static int load_yolo(ir_graph_t* graph, ir_node_t* node, std::vector<std::string>& tensor_name_map, list* options, int index, FILE* fp)
{
    ir_tensor_t* tensor = find_tensor(graph, tensor_name_map[index - 1]);
    if (tensor == NULL)
        return -1;
    set_ir_node_input_tensor(node, 0, tensor);
    std::vector<int> out_dims;
    out_dims.push_back(tensor->dims[0]);
    out_dims.push_back(tensor->dims[1]);
    out_dims.push_back(tensor->dims[2]);
    out_dims.push_back(tensor->dims[3]);

    ir_tensor_t* out_tensor = get_ir_graph_tensor(graph, node->output_tensors[0]);

    return 0;
}

static int load_route(ir_graph_t* graph, ir_node_t* node, std::vector<std::string>& tensor_name_map, list* options, int index, FILE* fp)
{
    //check layers option
    char* layers = option_find(options, (char*)("layers"));
    int layers_len = strlen(layers);
    int n_layers = 1;
    std::vector<int> layers_arr;
    for (int i = 0; i < layers_len; ++i)
    {
        if (layers[i] == ',')
            ++n_layers;
    }
    for (int i = 0; i < n_layers; ++i)
    {
        int from_index = atoi(layers);
        layers = strchr(layers, ',') + 1;
        if (from_index < 0)
            from_index = index + from_index;
        else
            from_index = from_index + 1;
        layers_arr.push_back(from_index);
    }
    //check groups option
    char* groups = option_find(options, (char*)("groups"));
    int groups_len = (groups != nullptr) ? strlen(groups) : 0;
    int n_groups = 0;
    std::vector<int> groups_arr;
    if (groups_len > 0)
    {
        n_groups = 1;
        for (int i = 0; i < groups_len; ++i)
        {
            if (groups[i] == ',')
                ++n_groups;
        }
        for (int i = 0; i < n_groups; ++i)
        {
            int group_count = atoi(groups);
            groups = strchr(groups, ',') + 1;
            groups_arr.push_back(group_count);
        }
    }

    //check group_id option
    char* group_id = option_find(options, (char*)("group_id"));
    int group_id_len = (group_id != nullptr) ? strlen(group_id) : 0;
    int n_group_id = 0;
    std::vector<int> group_id_arr;
    if (group_id_len > 0)
    {
        n_group_id = 1;
        for (int i = 0; i < group_id_len; ++i)
        {
            if (group_id[i] == ',')
                ++n_group_id;
        }
        for (int i = 0; i < n_group_id; ++i)
        {
            int group_index = atoi(group_id);
            group_id = strchr(group_id, ',') + 1;
            group_id_arr.push_back(group_index);
        }
    }
    if (groups_arr.size() == 0)
    {
        for (int i = 0; i < layers_arr.size(); i++)
            groups_arr.push_back(1);
    }
    if (group_id_arr.size() == 0)
    {
        for (int i = 0; i < layers_arr.size(); i++)
            group_id_arr.push_back(0);
    }
    //split if need
    std::vector<ir_node_t*> slice_node_arr;
    for (int i = 0; i < layers_arr.size(); i++)
    {
        std::string slice_name = "route_slice_" + std::to_string(index) + std::to_string(i);
        int from_index = layers_arr[i];
        ir_tensor_t* input_tensor = find_tensor(graph, tensor_name_map[from_index]);
        if (groups_arr[i] == 1)
        {
            slice_node_arr.push_back((ir_node_t*)nullptr);
        }
        else
        {
            std::vector<int> out_dims;
            for (int j = 0; j < input_tensor->dim_num; j++)
            {
                out_dims.push_back(input_tensor->dims[j]);
            }

            int step = input_tensor->dims[1] / groups_arr[i];
            out_dims[1] = step;
            int slice_node_id = add_node_above(graph, node->index, OP_SLICE, slice_name.c_str());
            ir_node_t* slice_node = get_ir_graph_node(graph, slice_node_id);
            slice_param* param = (slice_param*)slice_node->op.param_mem;
            param->axis = 1;
            param->iscaffe = 0;
            param->ismxnet = 0;
            param->isonnx = 1;
            param->begin = step * group_id_arr[i];
            param->end = step * (group_id_arr[i] + 1);
            param->slice_point_ = NULL;
            param->begin_ = NULL;
            param->size_ = NULL;
            set_ir_node_input_tensor(slice_node, 0, input_tensor);

            std::string slice_tensor_name = slice_name + "_" + std::to_string(0);
            ir_tensor_t* slice_out_tensor = get_ir_graph_tensor(graph, slice_node->output_tensors[0]);
            set_ir_tensor_shape(slice_out_tensor, out_dims.data(), out_dims.size());
            slice_node_arr.push_back(slice_node);
        }
    }
    //concat
    int output_c = 0;
    for (int i = 0; i < layers_arr.size(); i++)
    {
        if (slice_node_arr[i] != nullptr)
        {
            ir_node_t* slice_node = slice_node_arr[i];
            ir_tensor_t* slice_out_tensor = get_ir_graph_tensor(graph, slice_node->output_tensors[0]);
            output_c += slice_out_tensor->dims[1];
            set_ir_node_input_tensor(node, node->input_num, slice_out_tensor);
        }
        else
        {
            int from_index = layers_arr[i];
            ir_tensor_t* tensor = find_tensor(graph, tensor_name_map[from_index]);
            output_c += tensor->dims[1];
            set_ir_node_input_tensor(node, node->input_num, tensor);
        }
    }
    ir_tensor_t* tensor = find_tensor(graph, tensor_name_map[layers_arr[0]]);
    std::string concat_name = "route_concat" + std::to_string(index);
    std::string concat_tensor_name = concat_name + "_0";
    std::vector<int> concat_out_dims;
    concat_out_dims.push_back(tensor->dims[0]);
    concat_out_dims.push_back(output_c);
    concat_out_dims.push_back(tensor->dims[2]);
    concat_out_dims.push_back(tensor->dims[3]);
    ir_tensor_t* concat_out_tensor = get_ir_graph_tensor(graph, node->output_tensors[0]);
    set_ir_tensor_shape(concat_out_tensor, concat_out_dims.data(), concat_out_dims.size());

    concat_param* param = (concat_param*)node->op.param_mem;
    param->axis = 1; // may cause fault

    return 0;
}

static int load_upsample(ir_graph_t* graph, ir_node_t* node, std::vector<std::string>& tensor_name_map, list* options, int index, FILE* fp)
{
    ir_tensor_t* tensor = find_tensor(graph, tensor_name_map[index - 1]);
    set_ir_node_input_tensor(node, 0, tensor);
    upsample_param* param = (upsample_param*)node->op.param_mem;
    int scale = option_find_int(options, (char*)"stride", 2);
    param->scale = scale;

    std::vector<int> out_dims;
    out_dims.push_back(tensor->dims[0]);
    out_dims.push_back(tensor->dims[1]);
    out_dims.push_back(tensor->dims[2] * scale);
    out_dims.push_back(tensor->dims[3] * scale);

    ir_tensor_t* out_tensor = get_ir_graph_tensor(graph, node->output_tensors[0]);
    set_ir_tensor_shape(out_tensor, out_dims.data(), out_dims.size());

    return 0;
}

static int load_max_pooling(ir_graph_t* graph, ir_node_t* node, std::vector<std::string>& tensor_name_map, list* options, int index, FILE* fp)
{
    ir_tensor_t* tensor = find_tensor(graph, tensor_name_map[index - 1]);
    if (tensor == nullptr)
    {
        printf("tensor is null \n");
        return -1;
    }
    set_ir_node_input_tensor(node, 0, tensor);
    int stride = option_find_int(options, (char*)"stride", 1);
    int size = option_find_int(options, (char*)"size", stride);
    int padding = option_find_int_quiet(options, (char*)"padding", size - 1);

    pool_param* param = (pool_param*)node->op.param_mem;
    param->kernel_h = size;
    param->kernel_w = size;

    param->pad_h0 = padding;
    param->pad_h1 = padding;
    param->pad_w0 = padding;
    param->pad_w1 = padding;
    param->stride_h = stride;
    param->stride_w = stride;
    param->caffe_flavor = 2;
    param->pool_method = POOL_MAX;
    std::vector<int> out_dims;
    out_dims.push_back(tensor->dims[0]);
    out_dims.push_back(tensor->dims[1]);
    int in_h = tensor->dims[2];
    int in_w = tensor->dims[3];
    int out_h = (in_h + padding - size) / stride + 1;
    int out_w = (in_w + padding - size) / stride + 1;
    out_dims.push_back(out_h);
    out_dims.push_back(out_w);

    ir_tensor_t* out_tensor = get_ir_graph_tensor(graph, node->output_tensors[0]);
    set_ir_tensor_shape(out_tensor, out_dims.data(), out_dims.size());

    return 0;
}

static int load_reorg(ir_graph_t* graph, ir_node_t* node, std::vector<std::string>& tensor_name_map, list* options, int index, FILE* fp)
{
    ir_tensor_t* tensor = find_tensor(graph, tensor_name_map[index - 1]);
    set_ir_node_input_tensor(node, 0, tensor);
    reorg_param* param = (reorg_param*)node->op.param_mem;
    int stride = option_find_int(options, (char*)"stride", 1);
    param->stride = stride;

    std::vector<int> out_dims;
    out_dims.push_back(tensor->dims[0]);
    out_dims.push_back(tensor->dims[1] * stride * stride);
    out_dims.push_back(tensor->dims[2] / stride);
    out_dims.push_back(tensor->dims[3] / stride);

    ir_tensor_t* out_tensor = get_ir_graph_tensor(graph, node->output_tensors[0]);
    set_ir_tensor_shape(tensor, out_dims.data(), out_dims.size());

    return 0;
}

static int load_region(ir_graph_t* graph, ir_node_t* node, std::vector<std::string>& tensor_name_map, list* options, int index, FILE* fp)
{
    ir_tensor_t* tensor = find_tensor(graph, tensor_name_map[index - 1]);
    set_ir_node_input_tensor(node, 0, tensor);
    region_param* param = (region_param*)node->op.param_mem;

    int coords = option_find_int(options, (char*)"coords", 4);
    int classes = option_find_int(options, (char*)"classes", 20);
    int num = option_find_int(options, (char*)"num", 1);
    char* a = option_find_str(options, (char*)"anchors", 0);
    float thresh = option_find_float(options, (char*)"thresh", .5);

    param->num_classes = classes;
    param->num_box = num;
    param->coords = coords;
    param->nms_threshold = thresh;

    // get the bias;
    if (a)
    {
        int len = strlen(a);
        int n = 1;
        int i;
        for (i = 0; i < len; ++i)
        {
            if (a[i] == ',')
                ++n;
        }
        param->biases = (float*)sys_malloc(sizeof(float) * n);
        for (i = 0; i < n; ++i)
        {
            float bias = atof(a);
            param->biases[i] = bias;
            a = strchr(a, ',') + 1;
        }
    }
    std::vector<int> out_dims;
    out_dims.push_back(tensor->dims[0]);
    out_dims.push_back(tensor->dims[1]);
    out_dims.push_back(tensor->dims[2]);
    out_dims.push_back(tensor->dims[3]);

    ir_tensor_t* out_tensor = get_ir_graph_tensor(graph, node->output_tensors[0]);
    set_ir_tensor_shape(out_tensor, out_dims.data(), out_dims.size());

    return 0;
}

static int load_dropout(ir_graph_t* graph, ir_node_t* node, std::vector<std::string>& tensor_name_map, list* options, int index, FILE* fp)
{
    ir_tensor_t* tensor = find_tensor(graph, tensor_name_map[index - 1]);
    if (tensor == nullptr)
        return -1;
    set_ir_node_input_tensor(node, 0, tensor);
    std::vector<int> out_dims;
    out_dims.push_back(tensor->dims[0]);
    out_dims.push_back(tensor->dims[1]);
    out_dims.push_back(tensor->dims[2]);
    out_dims.push_back(tensor->dims[3]);

    ir_tensor_t* out_tensor = get_ir_graph_tensor(graph, node->output_tensors[0]);
    set_ir_tensor_shape(out_tensor, out_dims.data(), out_dims.size());

    return 0;
}

void darknet_serializer::register_op_load()
{
    op_load_map["[convolutional]"] = std::pair<int, op_load_t>(OP_CONV, load_conv);
    op_load_map["[shortcut]"] = std::pair<int, op_load_t>(OP_ELTWISE, load_shortcut);
    op_load_map["[yolo]"] = std::pair<int, op_load_t>(OP_DROPOUT, load_yolo);
    op_load_map["[route]"] = std::pair<int, op_load_t>(OP_CONCAT, load_route);
    op_load_map["[upsample]"] = std::pair<int, op_load_t>(OP_UPSAMPLE, load_upsample);
    op_load_map["[maxpool]"] = std::pair<int, op_load_t>(OP_POOL, load_max_pooling);
    op_load_map["[reorg]"] = std::pair<int, op_load_t>(OP_REORG, load_reorg);
    op_load_map["[region]"] = std::pair<int, op_load_t>(OP_REGION, load_region);
    op_load_map["[dropout]"] = std::pair<int, op_load_t>(OP_DROPOUT, load_dropout);
}
