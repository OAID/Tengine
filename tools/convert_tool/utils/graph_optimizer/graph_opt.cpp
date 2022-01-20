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

#include "graph_opt.hpp"

static int erase_tensor_id(ir_graph_t* graph, int16_t id)
{
    ir_tensor_t* tensor_del = get_ir_graph_tensor(graph, id);
    std::map<int16_t, int16_t> old_new_id;
    int16_t j = 0;
    for (size_t i = 0; i < graph->tensor_num; i++)
    {
        if (i == id) continue;

        ir_tensor_t* tensor = get_ir_graph_tensor(graph, i);
        tensor->index = j;
        graph->tensor_list[j] = graph->tensor_list[i];
        old_new_id[i] = j++;
    }
    for (size_t i = 0; i < graph->node_num; i++)
    {
        ir_node_t* node = get_ir_graph_node(graph, i);
        for (size_t j = 0; j < node->input_num; j++)
        {
            node->input_tensors[j] = old_new_id[node->input_tensors[j]];
        }
        for (size_t j = 0; j < node->output_num; j++)
        {
            node->output_tensors[j] = old_new_id[node->output_tensors[j]];
        }
    }

    ir_tensor_t** new_tensor_list = (ir_tensor_t**)sys_realloc(graph->tensor_list, sizeof(ir_tensor_t*) * (graph->tensor_num - 1));
    graph->tensor_list = new_tensor_list;
    graph->tensor_num--;

    destroy_ir_tensor(graph, tensor_del);
    return 0;
}

static int erase_node_id(ir_graph_t* graph, int16_t id)
{
    ir_node_t* node_del = get_ir_graph_node(graph, id);

    std::map<int16_t, int16_t> old_new_id;
    int16_t j = 0;
    for (size_t i = 0; i < graph->node_num; i++)
    {
        if (i == id) continue;

        ir_node_t* node = get_ir_graph_node(graph, i);
        node->index = j;
        graph->node_list[j] = graph->node_list[i];
        old_new_id[i] = j++;
    }

    for (size_t i = 0; i < graph->tensor_num; i++)
    {
        ir_tensor_t* tensor = get_ir_graph_tensor(graph, i);
        tensor->producer = old_new_id[tensor->producer];
        for (size_t j = 0; j < tensor->consumer_num; j++)
        {
            tensor->consumer[j] = old_new_id[tensor->consumer[j]];
        }
    }

    for (size_t i = 0; i < graph->input_num; i++)
    {
        graph->input_nodes[i] = old_new_id[graph->input_nodes[i]];
    }
    for (size_t i = 0; i < graph->output_num; i++)
    {
        graph->output_nodes[i] = old_new_id[graph->output_nodes[i]];
    }

    ir_node_t** new_node_list = (ir_node_t**)sys_realloc(graph->node_list, sizeof(ir_node_t*) * (graph->node_num - 1));
    graph->node_list = new_node_list;
    graph->node_num--;

    destroy_ir_node(graph, node_del);

    return 0;
}

int delete_node(ir_graph_t* graph, int16_t pre_node_id, int16_t del_node_id)
{
    ir_node_t* pre_node = get_ir_graph_node(graph, pre_node_id);
    ir_node_t* del_node = get_ir_graph_node(graph, del_node_id);

    /* setup new connection */
    ir_tensor_t* pre_output_tensor = get_ir_graph_tensor(graph, pre_node->output_tensors[0]);
    ir_tensor_t* del_output_tensor = get_ir_graph_tensor(graph, del_node->output_tensors[0]);
    for (size_t i = 0; i < del_output_tensor->consumer_num; i++)
    {
        int16_t consumer_id = del_output_tensor->consumer[i];
        pre_output_tensor->consumer[i] = consumer_id;
        ir_node_t* consumer_node = get_ir_graph_node(graph, consumer_id);
        for (size_t j = 0; j < consumer_node->input_num; j++)
        {
            ir_tensor_t* input = get_ir_graph_tensor(graph, consumer_node->input_tensors[j]);
            if (input->producer == del_node_id)
                consumer_node->input_tensors[j] = pre_output_tensor->index;
        }
    }
    pre_output_tensor->consumer_num = del_output_tensor->consumer_num;

    /* check if graph output */
    for (int i = 0; i < graph->output_num; ++i)
    {
        if (del_node_id == graph->output_nodes[i])
        {
            graph->output_nodes[i] = pre_node_id;
        }
    }

    /* delete const tensor&node of input */
    for (int i = 1; i < del_node->input_num; i++)
    {
        ir_tensor_t* tensor = get_ir_graph_tensor(graph, del_node->input_tensors[i]);
        ir_node_t* node = get_ir_graph_node(graph, tensor->producer);

        if (tensor->tensor_type == TENSOR_TYPE_CONST)
        {
            if (erase_tensor_id(graph, tensor->index) < 0 || erase_node_id(graph, node->index) < 0)
            {
                return -1;
            }
        }
    }

    /* delete node */
    if (erase_tensor_id(graph, del_node->output_tensors[0]) < 0 || erase_node_id(graph, del_node->index) < 0)
    {
        fprintf(stderr, "delete node:%s failed.\n", del_node->name);
        return -1;
    }
    return 0;
}

static int insert_node_id(ir_graph_t* graph, int16_t insert_node_id, int16_t inserted_node_id)
{
    ir_node_t* add_node = get_ir_graph_node(graph, insert_node_id);

    /* insert node id */
    std::map<int16_t, int16_t> old_new_id;
    int16_t tmp = graph->node_num - 1;
    for (int i = graph->node_num - 2; i >= 0; i--)
    {
        ir_node_t* node = get_ir_graph_node(graph, i);
        node->index = tmp;
        graph->node_list[tmp] = node;
        old_new_id[i] = tmp--;
        if (i == inserted_node_id)
        {
            old_new_id[add_node->index] = i;
            tmp--;
        }
    }
    graph->node_list[inserted_node_id] = add_node;
    add_node->index = inserted_node_id;

    for (size_t i = 0; i < graph->tensor_num; i++)
    {
        ir_tensor_t* tensor = get_ir_graph_tensor(graph, i);
        tensor->producer = old_new_id[tensor->producer];
        for (size_t j = 0; j < tensor->consumer_num; j++)
        {
            tensor->consumer[j] = old_new_id[tensor->consumer[j]];
        }
    }
    for (size_t i = 0; i < graph->input_num; i++)
    {
        graph->input_nodes[i] = old_new_id[graph->input_nodes[i]];
    }
    for (size_t i = 0; i < graph->output_num; i++)
    {
        graph->output_nodes[i] = old_new_id[graph->output_nodes[i]];
    }

    return 0;
}

static int insert_tensor_id(ir_graph_t* graph, int16_t insert_tensor_id, int16_t inserted_tensor_id)
{
    ir_tensor_t* add_tensor = get_ir_graph_tensor(graph, insert_tensor_id);

    /* insert tensor id */
    std::map<int16_t, int16_t> old_new_id;
    // int16_t inserted_tensor_id = down_node->output_tensors[0];
    int j = graph->tensor_num - 1;
    for (int i = graph->tensor_num - 2; i >= 0; i--)
    {
        ir_tensor_t* tensor = get_ir_graph_tensor(graph, i);
        tensor->index = j;
        graph->tensor_list[j] = tensor;
        old_new_id[i] = j--;
        if (i == inserted_tensor_id)
        {
            old_new_id[add_tensor->index] = i;
            j--;
        }
    }
    graph->tensor_list[inserted_tensor_id] = add_tensor;
    add_tensor->index = inserted_tensor_id;
    for (size_t i = 0; i < graph->node_num; i++)
    {
        ir_node_t* node = get_ir_graph_node(graph, i);
        for (size_t j = 0; j < node->input_num; j++)
        {
            node->input_tensors[j] = old_new_id[node->input_tensors[j]];
        }
        for (size_t j = 0; j < node->output_num; j++)
        {
            node->output_tensors[j] = old_new_id[node->output_tensors[j]];
        }
    }

    return 0;
}

int add_node_below(ir_graph_t* graph, int16_t up_node_id, int add_node_type, const char* name)
{
    /* get all down nodes */
    ir_node_t* up_node = get_ir_graph_node(graph, up_node_id);
    ir_tensor_t* up_node_output_tensor = get_ir_graph_tensor(graph, up_node->output_tensors[0]);
    std::vector<int16_t> down_nodes;
    for (size_t i = 0; i < up_node_output_tensor->consumer_num; i++)
    {
        down_nodes.push_back(up_node_output_tensor->consumer[i]);
    }

    /* create node and its own tensor */
    ir_node_t* add_node = create_ir_node(graph, name, add_node_type, 1);
    if (add_node == nullptr)
        return -1;
    ir_tensor_t* add_tensor = create_ir_tensor(graph, name, TENGINE_DT_FP32);
    if (add_tensor == nullptr)
        return -1;
    add_tensor->tensor_type = TENSOR_TYPE_VAR;
    set_ir_node_output_tensor(add_node, 0, add_tensor);

    /* setup new connection */
    for (short down_node_id : down_nodes)
    {
        ir_node_t* down_node = get_ir_graph_node(graph, down_node_id);
        set_ir_node_input_tensor(down_node, 0, add_tensor);
    }
    up_node_output_tensor->consumer_num = 0;
    set_ir_node_input_tensor(add_node, 0, up_node_output_tensor);

    if (down_nodes.empty()) // add node in tail
    {
        // exchange graph output
        for (int i = 0; i < graph->output_num; ++i)
        {
            if (graph->output_nodes[i] == up_node_id)
            {
                graph->output_nodes[i] = add_node->index;
            }
        }
        return add_node->index;
    }

    // insert id
    /* get min id from down nodes */
    int16_t down_node_id = graph->node_num;
    for (auto& id : down_nodes)
    {
        if (id < down_node_id)
        {
            down_node_id = id;
        }
    }

    ir_node_t* down_node = get_ir_graph_node(graph, down_node_id);
    int16_t down_tensor_id = down_node->output_tensors[0];

    /* insert node id */
    if (insert_node_id(graph, add_node->index, down_node_id) < 0)
        return -1;

    /* insert tensor id */
    if (insert_tensor_id(graph, add_tensor->index, down_tensor_id) < 0)
        return -1;

    return add_node->index;
}

int add_node_above(ir_graph_t* graph, int16_t down_node_id, int add_node_type, const char* name)
{
    /* get all up nodes */
    ir_node_t* down_node = get_ir_graph_node(graph, down_node_id);
    std::vector<int16_t> up_nodes;
    for (size_t i = 0; i < down_node->input_num; i++)
    {
        ir_tensor_t* tensor = get_ir_graph_tensor(graph, down_node->input_tensors[i]);
        if (tensor->tensor_type == TENSOR_TYPE_VAR)
            up_nodes.push_back(tensor->producer);
    }

    /* create node and its own tensor */
    ir_node_t* add_node = create_ir_node(graph, name, add_node_type, 1);
    if (add_node == nullptr)
        return -1;
    ir_tensor_t* add_tensor = create_ir_tensor(graph, name, TENGINE_DT_FP32);
    if (add_tensor == nullptr)
        return -1;
    add_tensor->tensor_type = TENSOR_TYPE_VAR;
    set_ir_node_output_tensor(add_node, 0, add_tensor);

    /* setup new connection */
    for (int i = 0; i < up_nodes.size(); i++)
    {
        ir_node_t* up_node = get_ir_graph_node(graph, up_nodes[i]);
        ir_tensor_t* up_node_output_tensor = get_ir_graph_tensor(graph, up_node->output_tensors[0]);
        for (size_t i = 0; i < up_node_output_tensor->consumer_num; i++)
        {
            if (up_node_output_tensor->consumer[i] == down_node_id)
                up_node_output_tensor->consumer[i] = add_node->index;
        }
        set_ir_node_input_tensor(add_node, i, up_node_output_tensor);
    }
    if (down_node->input_num != 0)
        down_node->input_tensors[0] = add_tensor->index;
    add_tensor->consumer[0] = down_node_id;
    add_tensor->consumer_num = 1;

    if (up_nodes.empty()) // add node in head
    {
        // exchange graph input
        for (int i = 0; i < graph->input_num; ++i)
        {
            if (graph->input_nodes[i] == down_node_id)
                graph->input_nodes[i] = add_node->index;
        }
    }

    /* insert node id */
    if (insert_node_id(graph, add_node->index, down_node_id) < 0)
        return -1;

    /* insert tensor id */
    if (insert_tensor_id(graph, add_tensor->index, down_node->output_tensors[0]) < 0)
        return -1;

    return add_node->index;
}

int add_const_node_above(ir_graph_t* graph, int16_t down_node_id, const char* name)
{
    /* get all up nodes */
    ir_node_t* down_node = get_ir_graph_node(graph, down_node_id);

    /* create const node and its own tensor */
    ir_node_t* add_node = create_ir_node(graph, name, OP_CONST, 1);
    if (add_node == nullptr)
        return -1;
    ir_tensor_t* add_tensor = create_ir_tensor(graph, name, TENGINE_DT_FP32);
    if (add_tensor == nullptr)
        return -1;
    add_tensor->tensor_type = TENSOR_TYPE_CONST;
    set_ir_node_output_tensor(add_node, 0, add_tensor);

    down_node->input_num++;
    down_node->input_tensors[down_node->input_num - 1] = add_tensor->index;
    add_tensor->consumer[0] = down_node_id;
    add_tensor->consumer_num = 1;

    /* insert node id */
    if (insert_node_id(graph, add_node->index, down_node_id) < 0)
        return -1;

    /* insert tensor id */
    if (insert_tensor_id(graph, add_tensor->index, down_node->output_tensors[0]) < 0)
        return -1;

    return add_node->index;
}

static int weight_bn(ir_graph_t* graph, ir_node_t* conv_node, float* mean, float* var, float* gamma, float* beta, float eps,
                     float rescale_factor, ir_tensor_t* bias_tensor)
{
    ir_tensor_t* kernel_tensor = get_ir_graph_tensor(graph, conv_node->input_tensors[1]);
    struct conv_param* param = (struct conv_param*)conv_node->op.param_mem;

    int group = param->group;
    int input_chan = kernel_tensor->dims[1];
    int output_chan = kernel_tensor->dims[0] / group;
    int kernel_x = param->kernel_w;
    int kernel_y = param->kernel_h;
    int kernel_size = input_chan * kernel_x * kernel_y;
    float* kernel_data = (float*)kernel_tensor->data;
    int channel_num = kernel_tensor->dims[0];

    std::vector<float> scale_mean(channel_num);
    std::vector<float> scale_var_inv(channel_num);

    float rescale_factor_tmp = rescale_factor;
    float* bias = NULL;
    if (bias_tensor != nullptr)
        bias = (float*)bias_tensor->data;

    /* create bias node and tensor */
    if (bias_tensor == nullptr)
    {
        std::string name = kernel_tensor->name;

        name = name + ".bias.bn";
        /* create */
        ir_node_t* bias_node = create_ir_node(graph, name.c_str(), OP_CONST, 1);
        if (bias_node == nullptr)
            return -1;
        ir_tensor_t* bias_tensor = create_ir_tensor(graph, name.c_str(), TENGINE_DT_FP32);
        if (bias_tensor == nullptr)
            return -1;

        bias_tensor->tensor_type = TENSOR_TYPE_CONST;
        int dim_num = 1;
        int* dims = (int*)sys_malloc(sizeof(int) * dim_num);
        dims[0] = channel_num;
        bias_tensor->data = (void*)sys_malloc(sizeof(float) * channel_num);
        set_ir_tensor_shape(bias_tensor, dims, dim_num);
        set_ir_node_output_tensor(bias_node, 0, bias_tensor);
        set_ir_node_input_tensor(conv_node, 2, bias_tensor);

        insert_node_id(graph, bias_node->index, kernel_tensor->producer);
        insert_tensor_id(graph, bias_tensor->index, kernel_tensor->index);
    }

    rescale_factor_tmp = rescale_factor_tmp ? 1 / rescale_factor_tmp : 0;

    if (NULL == bias)
    {
        for (int c = 0; c < channel_num; c++)
        {
            scale_var_inv[c] = 1.f / sqrt(var[c] * rescale_factor_tmp + eps);
            scale_mean[c] = -mean[c] * rescale_factor_tmp * scale_var_inv[c];
        }
    }
    else
    {
        for (int c = 0; c < channel_num; c++)
        {
            scale_var_inv[c] = 1.f / sqrt(var[c] * rescale_factor_tmp + eps);
            scale_mean[c] = (bias[c] - mean[c] * rescale_factor_tmp) * scale_var_inv[c];
        }
    }

    if (NULL != gamma)
    {
        for (int c = 0; c < channel_num; c++)
        {
            scale_var_inv[c] = gamma[c] * scale_var_inv[c];
            scale_mean[c] = gamma[c] * scale_mean[c];
        }
    }
    if (NULL != beta)
    {
        for (int c = 0; c < channel_num; c++)
        {
            scale_mean[c] = scale_mean[c] + beta[c];
        }
    }
    for (int g = 0; g < group; g++)
    {
        float* kernel = kernel_data + g * output_chan * kernel_size;
        for (int o_c = 0; o_c < output_chan; o_c++)
        {
            float w_scale = scale_var_inv[g * output_chan + o_c];
            for (int i = 0; i < kernel_size; i++)
            {
                kernel[o_c * kernel_size + i] = kernel[o_c * kernel_size + i] * w_scale;
            }
        }
    }

    bias_tensor = get_ir_graph_tensor(graph, conv_node->input_tensors[2]);
    float* bias_data = (float*)bias_tensor->data;
    for (int i = 0; i < channel_num; i++)
    {
        bias_data[i] = scale_mean[i];
    }

    return 0;
}

static int fc_weight_bn(ir_graph_t* graph, ir_node_t* fc_node, float* mean, float* var, float* gamma, float* beta, float eps,
                        float rescale_factor, ir_tensor_t* bias_tensor)
{
    ir_tensor_t* kernel_tensor = get_ir_graph_tensor(graph, fc_node->input_tensors[1]);
    struct fc_param* param = (struct fc_param*)fc_node->op.param_mem;

    int output_chan = param->num_output;
    float* kernel_data = (float*)kernel_tensor->data;
    int channel_num = kernel_tensor->dims[0];
    int total_size = kernel_tensor->dims[1];
    int kernel_size = total_size;

    float* scale_mean = (float*)malloc(channel_num * sizeof(float));
    float* scale_var_inv = (float*)malloc(channel_num * sizeof(float));

    float rescale_factor_tmp = rescale_factor;
    float* bias = NULL;
    if (bias_tensor != nullptr)
        bias = (float*)bias_tensor->data;

    /* create bias node and tensor */
    if (bias_tensor == nullptr)
    {
        std::string name = fc_node->name;
        name = name + ".bias.bn";
        /* create */
        ir_node_t* bias_node = create_ir_node(graph, name.c_str(), OP_CONST, 1);
        if (bias_node == nullptr)
            return -1;
        ir_tensor_t* bias_tensor = create_ir_tensor(graph, name.c_str(), TENGINE_DT_FP32);
        if (bias_tensor == nullptr)
            return -1;

        bias_tensor->tensor_type = TENSOR_TYPE_CONST;
        int dim_num = 1;
        int* dims = (int*)sys_malloc(sizeof(int) * dim_num);
        dims[0] = channel_num;
        bias_tensor->data = (void*)sys_malloc(sizeof(float) * channel_num);

        set_ir_tensor_shape(bias_tensor, dims, dim_num);
        set_ir_node_output_tensor(bias_node, 0, bias_tensor);
        set_ir_node_input_tensor(fc_node, 2, bias_tensor);

        insert_node_id(graph, bias_node->index, kernel_tensor->producer);
        insert_tensor_id(graph, bias_tensor->index, kernel_tensor->index);
    }

    rescale_factor_tmp = rescale_factor_tmp ? 1 / rescale_factor_tmp : 0;

    if (NULL == bias)
    {
        for (int c = 0; c < channel_num; c++)
        {
            scale_var_inv[c] = 1.f / sqrt(var[c] * rescale_factor_tmp + eps);
            scale_mean[c] = -mean[c] * rescale_factor_tmp * scale_var_inv[c];
        }
    }
    else
    {
        for (int c = 0; c < channel_num; c++)
        {
            scale_var_inv[c] = 1.f / sqrt(var[c] * rescale_factor_tmp + eps);
            scale_mean[c] = (bias[c] - mean[c] * rescale_factor_tmp) * scale_var_inv[c];
        }
    }

    if (NULL != gamma)
    {
        for (int c = 0; c < channel_num; c++)
        {
            scale_var_inv[c] = gamma[c] * scale_var_inv[c];
            scale_mean[c] = gamma[c] * scale_mean[c];
        }
    }
    if (NULL != beta)
    {
        for (int c = 0; c < channel_num; c++)
        {
            scale_mean[c] = scale_mean[c] + beta[c];
        }
    }
    for (int o_c = 0; o_c < output_chan; o_c++)
    {
        float w_scale = scale_var_inv[o_c];
        for (int i = 0; i < kernel_size; i++)
        {
            kernel_data[o_c * kernel_size + i] = kernel_data[o_c * kernel_size + i] * w_scale;
        }
    }

    bias_tensor = get_ir_graph_tensor(graph, fc_node->input_tensors[2]);
    float* bias_data = (float*)bias_tensor->data;
    for (int i = 0; i < channel_num; i++)
    {
        bias_data[i] = scale_mean[i];
    }

    free(scale_var_inv);
    free(scale_mean);

    return 0;
}

static int change_node_op(ir_node_t* node, int new_op_type)
{
    sys_free(node->op.param_mem);
    node->op.type = new_op_type;
    ir_method_t* ir_method = find_op_method(new_op_type, 1);
    if ((NULL != ir_method) && (NULL != ir_method->init) && (ir_method->init(&node->op) < 0))
    {
        return -1;
    }

    return 0;
}

static int fuse_conv_relu_common(ir_graph_t* graph)
{
    /* get all conv-relu chain */
    std::vector<std::pair<ir_node_t*, ir_node_t*> > conv_relu_v;
    for (size_t i = 0; i < graph->node_num; i++)
    {
        ir_node_t* relu_node = get_ir_graph_node(graph, i);
        if (relu_node->op.type != OP_RELU && relu_node->op.type != OP_RELU6 && relu_node->op.type != OP_CLIP)
            continue;
        if (relu_node->op.type == OP_RELU)
        {
            struct relu_param* relu_param = (struct relu_param*)relu_node->op.param_mem;
            if (relu_param->negative_slope != 0.f)
                continue;
        }
        if (relu_node->op.type == OP_CLIP)
        {
            struct clip_param* clip_param = (struct clip_param*)relu_node->op.param_mem;
            if (clip_param->min != 0.f && clip_param->max != 6.f)
                continue;
        }
        ir_tensor_t* conv_tensor = get_ir_graph_tensor(graph, relu_node->input_tensors[0]);
        ir_node_t* conv_node = get_ir_graph_node(graph, conv_tensor->producer);
        if (conv_node->op.type != OP_CONV)
            continue;
        if (conv_tensor->consumer_num != 1)
            continue;

        conv_relu_v.push_back(std::make_pair(conv_node, relu_node));
    }

    /* fused */
    for (auto& conv_relu : conv_relu_v)
    {
        ir_node_t* conv_node = conv_relu.first;
        ir_node_t* relu_node = conv_relu.second;
        struct conv_param* conv_param = (struct conv_param*)conv_node->op.param_mem;
        if (relu_node->op.type == OP_RELU)
            conv_param->activation = 0;
        if (relu_node->op.type == OP_RELU6 || relu_node->op.type == OP_CLIP)
            conv_param->activation = 6;

        /* delete relu node */
        if (delete_node(graph, conv_node->index, relu_node->index) < 0)
        {
            fprintf(stderr, "delete relu node:%s failed.\n", relu_node->name);
            return -1;
        }
    }

    return 0;
}

static int fuse_relu_eltwise(ir_graph_t* graph)
{
    /* get all relu-eltwise chain */
    std::vector<std::pair<ir_node_t*, ir_node_t*> > relu_eltwise_v;
    for (size_t i = 0; i < graph->node_num; i++)
    {
        ir_node_t* elt_node = get_ir_graph_node(graph, i);
        if (elt_node->op.type != OP_ELTWISE)
            continue;
        struct eltwise_param* elt_param = (struct eltwise_param*)elt_node->op.param_mem;
        if (elt_param->type != ELT_MIN_SCALAR)
            continue; // todo: verify 6

        /*Check if it is a  relu + minimum*/
        ir_tensor_t* relu_tensor = get_ir_graph_tensor(graph, elt_node->input_tensors[0]);
        ir_node_t* relu_node = get_ir_graph_node(graph, relu_tensor->producer);
        if (relu_node->op.type != OP_RELU)
            continue;
        relu_eltwise_v.push_back(std::make_pair(relu_node, elt_node));
    }

    /* fused */
    for (auto& relu_elt : relu_eltwise_v)
    {
        ir_node_t* relu_node = relu_elt.first;
        ir_node_t* elt_node = relu_elt.second;
        relu_node->op.type = OP_RELU6;

        /* delete elt node */
        if (delete_node(graph, relu_node->index, elt_node->index) < 0)
        {
            fprintf(stderr, "delete elt node:%s failed.\n", elt_node->name);
            return -1;
        }
    }

    return 0;
}

static int fuse_bn_scale(ir_graph_t* graph)
{
    /* get all bn-scale chain */
    std::vector<std::pair<ir_node_t*, ir_node_t*> > bn_scale_v;
    for (size_t i = 0; i < graph->node_num; i++)
    {
        ir_node_t* scale_node = get_ir_graph_node(graph, i);
        if (scale_node->op.type != OP_SCALE)
            continue;

        /*Check if it is a  bn + scale*/
        ir_tensor_t* bn_tensor = get_ir_graph_tensor(graph, scale_node->input_tensors[0]);
        ir_node_t* bn_node = get_ir_graph_node(graph, bn_tensor->producer);
        if (bn_node->op.type != OP_BATCHNORM)
            continue;
        bn_scale_v.push_back(std::make_pair(bn_node, scale_node));
    }

    /* fused */
    for (auto& bn_scale : bn_scale_v)
    {
        ir_node_t* bn_node = bn_scale.first;
        ir_node_t* scale_node = bn_scale.second;

        /* exchange gamma beta */
        int16_t tmp = bn_node->input_tensors[1];
        bn_node->input_tensors[1] = scale_node->input_tensors[1];
        scale_node->input_tensors[1] = tmp;
        tmp = bn_node->input_tensors[2];
        bn_node->input_tensors[2] = scale_node->input_tensors[2];
        scale_node->input_tensors[2] = tmp;

        struct batchnorm_param* param = (struct batchnorm_param*)bn_node->op.param_mem;
        param->caffe_flavor = 0;

        /* delete scale node */
        if (delete_node(graph, bn_node->index, scale_node->index) < 0)
        {
            fprintf(stderr, "delete scale node:%s failed.\n", scale_node->name);
            return -1;
        }
    }

    return 0;
}

static int fuse_conv_bn(ir_graph_t* graph)
{
    /* get all conv-bn chain */
    std::vector<std::pair<ir_node_t*, ir_node_t*> > conv_bn_v;
    for (size_t i = 0; i < graph->node_num; i++)
    {
        ir_node_t* bn_node = get_ir_graph_node(graph, i);
        if (bn_node->op.type != OP_BATCHNORM)
            continue;

        /*Check if it is a  conv + bn*/
        ir_tensor_t* conv_tensor = get_ir_graph_tensor(graph, bn_node->input_tensors[0]);
        ir_node_t* conv_node = get_ir_graph_node(graph, conv_tensor->producer);
        if (conv_node->op.type != OP_CONV)
            continue;

        conv_bn_v.push_back(std::make_pair(conv_node, bn_node));
    }

    /* fused */
    for (auto& conv_bn : conv_bn_v)
    {
        ir_node_t* conv_node = conv_bn.first;
        ir_node_t* bn_node = conv_bn.second;
        struct batchnorm_param* bn_param = (struct batchnorm_param*)bn_node->op.param_mem;
        ir_tensor_t* bn_mean = get_ir_graph_tensor(graph, bn_node->input_tensors[3]);
        ir_tensor_t* bn_var = get_ir_graph_tensor(graph, bn_node->input_tensors[4]);

        float* mean = (float*)bn_mean->data;
        float* var = (float*)bn_var->data;
        float* gamma = NULL;
        float* beta = NULL;

        if (!bn_param->caffe_flavor)
        {
            ir_tensor_t* bn_gamma = get_ir_graph_tensor(graph, bn_node->input_tensors[1]);
            ir_tensor_t* bn_beta = get_ir_graph_tensor(graph, bn_node->input_tensors[2]);
            gamma = (float*)bn_gamma->data;
            beta = (float*)bn_beta->data;
        }

        ir_tensor_t* bias_tensor = nullptr;
        if (conv_node->input_num > 2)
            bias_tensor = get_ir_graph_tensor(graph, conv_node->input_tensors[2]);

        weight_bn(graph, conv_node, mean, var, gamma, beta, bn_param->eps, bn_param->rescale_factor, bias_tensor);

        /* delete elt node */
        if (delete_node(graph, conv_node->index, bn_node->index) < 0)
        {
            fprintf(stderr, "delete elt node:%s failed.\n", bn_node->name);
            return -1;
        }
    }

    return 0;
}

static int fuse_fc_bn(ir_graph_t* graph)
{
    /* get all fc-bn chain */
    std::vector<std::pair<ir_node_t*, ir_node_t*> > fc_bn_v;
    for (size_t i = 0; i < graph->node_num; i++)
    {
        ir_node_t* bn_node = get_ir_graph_node(graph, i);
        if (bn_node->op.type != OP_BATCHNORM)
            continue;

        /*Check if it is a  fc + bn*/
        ir_tensor_t* fc_tensor = get_ir_graph_tensor(graph, bn_node->input_tensors[0]);
        ir_node_t* fc_node = get_ir_graph_node(graph, fc_tensor->producer);
        if (fc_node->op.type != OP_FC)
            continue;
        fc_bn_v.push_back(std::make_pair(fc_node, bn_node));
    }

    /* fused */
    for (auto& fc_bn : fc_bn_v)
    {
        ir_node_t* fc_node = fc_bn.first;
        ir_node_t* bn_node = fc_bn.second;
        struct batchnorm_param* bn_param = (struct batchnorm_param*)bn_node->op.param_mem;
        ir_tensor_t* bn_mean = get_ir_graph_tensor(graph, bn_node->input_tensors[3]);
        ir_tensor_t* bn_var = get_ir_graph_tensor(graph, bn_node->input_tensors[4]);

        float* mean = (float*)bn_mean->data;
        float* var = (float*)bn_var->data;
        float* gamma = NULL;
        float* beta = NULL;

        if (!bn_param->caffe_flavor)
        {
            ir_tensor_t* bn_gamma = get_ir_graph_tensor(graph, bn_node->input_tensors[1]);
            ir_tensor_t* bn_beta = get_ir_graph_tensor(graph, bn_node->input_tensors[2]);
            gamma = (float*)bn_gamma->data;
            beta = (float*)bn_beta->data;
        }

        ir_tensor_t* bias_tensor = nullptr;
        if (fc_node->input_num > 2)
            bias_tensor = get_ir_graph_tensor(graph, fc_node->input_tensors[2]);

        fc_weight_bn(graph, fc_node, mean, var, gamma, beta, bn_param->eps, bn_param->rescale_factor, bias_tensor);

        /* delete bn node */
        if (delete_node(graph, fc_node->index, bn_node->index) < 0)
        {
            fprintf(stderr, "delete bn node:%s failed.\n", bn_node->name);
            return -1;
        }
    }

    return 0;
}

static int fuse_conv_unsqueeze(ir_graph_t* graph)
{
    /* get all unsqueeze conv|fc eltwise chain */
    std::vector<std::vector<ir_node_t*> > fused_nodes;
    for (size_t i = 0; i < graph->node_num; i++)
    {
        ir_node_t* elt_node = get_ir_graph_node(graph, i);
        if (elt_node->op.type != OP_ELTWISE)
            continue;
        struct eltwise_param* param = (struct eltwise_param*)elt_node->op.param_mem;
        if (elt_node->input_num != 2 || param->type != ELT_SUM) // unsqueeze and conv|fc
            continue;

        /* Check if it is a  (unsqueeze conv|fc) + eltwise */
        ir_tensor_t* conv_tensor = get_ir_graph_tensor(graph, elt_node->input_tensors[0]);
        ir_tensor_t* unsq_tensor = get_ir_graph_tensor(graph, elt_node->input_tensors[1]);
        ir_node_t* conv_or_fc_node = get_ir_graph_node(graph, conv_tensor->producer);
        ir_node_t* unsq_node = get_ir_graph_node(graph, unsq_tensor->producer);
        if (unsq_node->op.type != OP_UNSQUEEZE || (conv_or_fc_node->op.type != OP_CONV && conv_or_fc_node->op.type != OP_FC))
            continue;
        std::vector<ir_node_t*> nodes{conv_or_fc_node, unsq_node, elt_node};
        fused_nodes.push_back(nodes);
    }

    /* fused */
    for (int i = 0; i < fused_nodes.size(); i++)
    {
        ir_node_t* conv_or_fc_node = fused_nodes[i][0];
        ir_node_t* unsq_node = fused_nodes[i][1];
        ir_node_t* elt_node = fused_nodes[i][2];

        ir_tensor_t* bias_tensor = get_ir_graph_tensor(graph, unsq_node->input_tensors[0]);
        set_ir_node_input_tensor(conv_or_fc_node, conv_or_fc_node->input_num, bias_tensor);
        bias_tensor->consumer[0] = conv_or_fc_node->index;
        bias_tensor->consumer_num--;

        /* delete unsqueeze node */
        if (erase_tensor_id(graph, unsq_node->output_tensors[0]) < 0 || erase_node_id(graph, unsq_node->index) < 0)
        {
            fprintf(stderr, "delete node:%s failed.\n", unsq_node->name);
            return -1;
        }

        /* delete elt node */
        if (delete_node(graph, conv_or_fc_node->index, elt_node->index) < 0)
        {
            fprintf(stderr, "delete elt node:%s failed.\n", elt_node->name);
            return -1;
        }
    }

    return 0;
}

int graph_opt(graph_t graph)
{
    fprintf(stderr, "graph opt begin\n");

    ir_graph_t* ir_graph = (ir_graph_t*)graph;

    if (fuse_conv_unsqueeze(ir_graph) < 0)
        return -1;
    if (fuse_relu_eltwise(ir_graph) < 0)
        return -1;
    if (fuse_bn_scale(ir_graph) < 0)
        return -1;
    if (fuse_conv_bn(ir_graph) < 0)
        return -1;
    if (fuse_fc_bn(ir_graph) < 0)
        return -1;
    if (fuse_conv_relu_common(ir_graph) < 0)
        return -1;

    fprintf(stderr, "graph opt done.\n");
    return 0;
}
