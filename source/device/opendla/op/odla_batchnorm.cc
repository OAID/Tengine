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
 * Copyright (c) 2021, Institute of Computing Technology
 * Author: wanglei21c@mails.ucas.ac.cn
 */

#include "odla_executor.hpp"

extern "C"
{
#include "operator/op.h"
#include "batchnorm_param.h"
}


nvdla::priv::canonical_ast::Node * ODLAEngine::AddBatchNormalizationNode(struct node* ir_node)
{
    struct graph* ir_graph = ir_node->graph;
    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct tensor* in_gamma_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    struct tensor* in_beta_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[2]);
    struct tensor* in_mean_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[3]);
    struct tensor* in_var_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[4]);
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    nvdla::priv::canonical_ast::Node * Node ;
    auto * batchNormNode = new nvdla::priv::canonical_ast::BatchNormNode();

    if (nullptr == input_tensor || nullptr == output_tensor)
    {
        fprintf(stderr, "Tengine: Get input & output for BatchNorm(id: %d, name: %s) layer failed.\n", ir_node->index, ir_node->name);
        return nullptr;
    }

    float* in_gamma = (float*)get_tensor_buffer(in_gamma_tensor);
    float* in_beta  = (float*)get_tensor_buffer(in_beta_tensor);
    float* in_mean  = (float*)get_tensor_buffer(in_mean_tensor);
    float* in_var   = (float*)get_tensor_buffer(in_var_tensor);

    float* in_scale = (float*)sys_malloc(in_gamma_tensor->elem_num * sizeof(float));
    float* in_shift = (float*)sys_malloc(in_gamma_tensor->elem_num * sizeof(float));
    this->host_buffer.push_back(in_scale);
    this->host_buffer.push_back(in_shift);

    int channel_num = input_tensor->dims[1];

    float* scale_mean = ( float* )sys_malloc(channel_num * sizeof(float));
    float* scale_var_inv = ( float* )sys_malloc(channel_num * sizeof(float));
    this->host_buffer.push_back(scale_mean);
    this->host_buffer.push_back(scale_var_inv);

    struct batchnorm_param* batchnorm_param = ( struct batchnorm_param* )ir_node->op.param_mem;
    float rescale_factor = batchnorm_param->rescale_factor ? 1 / batchnorm_param->rescale_factor : 0;

    for (int c = 0; c < channel_num; c++)
    {
        float tmp = sqrtf(in_var[c] * rescale_factor + batchnorm_param->eps);
        scale_var_inv[c] = ( float )(1.f / tmp);
        tmp = rescale_factor * scale_var_inv[c];
        scale_mean[c] = ( float )(-in_mean[c] * tmp);
    }

    nvdla::Weights meanBlob, varianceBlob;
    nvdla::Dims4 meanDims;
    nvdla::Dims4 varianceDims;
    batchNormNode->params().setMode(nvdla::BatchNormMode::bnm_CHANNEL);

    meanDims.c = channel_num;
    meanDims.h = 1;
    meanDims.w = 1;

    meanBlob.type = nvdla::DataType::FLOAT;
    meanBlob.values = scale_mean;
    meanBlob.count = channel_num;

    varianceBlob.type = nvdla::DataType::FLOAT;
    varianceBlob.values = scale_var_inv;
    varianceBlob.count = channel_num;

    varianceDims = meanDims;

    batchNormNode->params().setMean(meanBlob);
    batchNormNode->params().setVariance(varianceBlob);
    batchNormNode->params().setEpsilon(batchnorm_param->eps);
    batchNormNode->params().setMeanDims(meanDims);
    batchNormNode->params().setVarianceDims(varianceDims);

    Node = batchNormNode;
    nvdla::priv::canonical_ast::NodeFactory::s_bn_priv.insert(
        std::pair<nvdla::priv::canonical_ast::Node*, nvdla::priv::canonical_ast::BatchNormNode*>(Node, batchNormNode)
    );
    return Node;
}
