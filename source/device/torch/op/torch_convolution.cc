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
 * Copyright (c) 2021, Open AI Lab
 * Author: hhchen@openailab.com
 */

#include "torch_helper.hpp"

extern "C"
{
#include "operator/op.h"
#include "convolution_param.h"
}

bool Net::AddConvolutionNode(struct node* ir_node)
{
    struct conv_param* param = (struct conv_param*)ir_node->op.param_mem;

    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct tensor* weight_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    struct tensor* bias_tensor;

    bool bias = false;
    if (ir_node->input_num > 2)
    {
        bias = true;
        bias_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[2]);
    }

    torch::nn::Conv2d layer
        = torch::nn::Conv2d{create_conv_options(
            /*in_planes = */ input_tensor->dims[1], /*out_planes = */ output_tensor->dims[1],
            /*kerner_size = */ param->kernel_h, /*stride = */ param->stride_h, /*padding = */ param->pad_h0,
            /*groups = */ param->group, /*dilation = */ param->dilation_h, /*bias = */ bias)};
    register_module(std::to_string(ir_node->index), layer);
    torch_node_map[ir_node->index] = layer;

    {
        torch::Tensor t = torch::rand({weight_tensor->dims[0], weight_tensor->dims[1], weight_tensor->dims[2], weight_tensor->dims[3]});
        void* date_mem = t.data_ptr();
        memcpy(date_mem, weight_tensor->data, weight_tensor->elem_num * weight_tensor->elem_size);
        layer->weight = register_parameter(std::to_string(ir_node->index) + "_weight", t);
    }

    if (bias)
    {
        torch::Tensor t = torch::rand({output_tensor->dims[1]});
        void* date_mem = t.data_ptr();
        memcpy(date_mem, bias_tensor->data, bias_tensor->elem_num * bias_tensor->elem_size);
        layer->bias = register_parameter(std::to_string(ir_node->index) + "_bias", t);
    }


    return true;
}
