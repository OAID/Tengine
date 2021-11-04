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
#include "deconv_param.h"
}


nvdla::priv::canonical_ast::Node * ODLAEngine::AddDeconvlutionNode(struct node* ir_node)
{
    struct graph* ir_graph = ir_node->graph;
    struct subgraph* subgraph = get_ir_graph_subgraph(ir_graph, ir_node->subgraph_idx);
    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct tensor* conv_weight = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    struct deconv_param* param = (struct deconv_param*)ir_node->op.param_mem;
    nvdla::priv::canonical_ast::Node * Node ;
    auto * deconvolutionNode = new nvdla::priv::canonical_ast::DeconvolutionNode();

    // Init Node
    nvdla::Dims2 topLeftPadding(param->pad_h0, param->pad_w0);
    nvdla::Dims2 bottomRightPadding(param->pad_h1, param->pad_w1);
    nvdla::Dims2 kernel(param->kernel_h,param->kernel_w);
    nvdla::Dims2 stride(param->stride_h,param->stride_w);
    nvdla::Dims2 dilation(param->dilation_h,param->dilation_w);
    deconvolutionNode->params().setBiasMode(nvdla::bNONE);
    deconvolutionNode->params().setHasBiasTerm(false);
    deconvolutionNode->params().setTopLeftPadding(topLeftPadding);
    deconvolutionNode->params().setBottomRightPadding(bottomRightPadding);
    deconvolutionNode->params().setPaddingValue(0);
    deconvolutionNode->params().setStride(stride);
    deconvolutionNode->params().setDilation(dilation);
    deconvolutionNode->params().setNumGroups(param->group);


    nvdla::Weights kernelWeights;
    nvdla::Weights biasWeights;
    if (param->group == 1 || (param->group == conv_weight->dims[0] && param->group != 1))   // conv + dwconv
    {
        nvdla::Dims4 weightDims(conv_weight->dims[0], conv_weight->dims[1], conv_weight->dims[2], conv_weight->dims[3]);
        deconvolutionNode->params().setWeightDims(weightDims);

        switch (conv_weight->data_type)
        {
        case TENGINE_DT_FP32:
        {
            kernelWeights.values = conv_weight->data;
            kernelWeights.count = conv_weight->elem_num;
            kernelWeights.type = nvdla::DataType::FLOAT;
            break;
        }
        case TENGINE_DT_INT8:
        {

            kernelWeights.values = conv_weight->data;
            kernelWeights.count = conv_weight->elem_num;
            kernelWeights.type = nvdla::DataType::INT8;
            break;
        }
        case TENGINE_DT_UINT8:
        {
            float* weight_buffer = (float*)sys_malloc(conv_weight->elem_num * sizeof(float));
            this->host_buffer.push_back(weight_buffer);
            for (int i = 0; i < conv_weight->elem_num; i++)
            {
                weight_buffer[i] = (float)(((uint8_t*)conv_weight->data)[i] - conv_weight->zero_point) * conv_weight->scale;
            }

            kernelWeights.values = weight_buffer;
            kernelWeights.count = conv_weight->elem_num;
            kernelWeights.type = nvdla::DataType::FLOAT;
            break;
        }
        default:
            fprintf(stderr, "Tengine: Unsupported weight quant data type(%d) of deconv(id: %d, name: %s).\n", conv_weight->data_type, ir_node->index, ir_node->name);
            return nullptr;
        }

        if (ir_node->input_num > 2) // bias exist
        {
            struct tensor* conv_bias = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[2]);
            nvdla::Dims4 biasDims(odla_tensor_map[conv_bias->index]->getDimensions());

            switch (conv_bias->data_type)
            {
            case TENGINE_DT_FP32:
            {
                biasWeights.values = conv_bias->data;
                biasWeights.count = conv_bias->elem_num;
                biasWeights.type = nvdla::DataType::FLOAT;
                break;
            }
            case TENGINE_DT_INT32:
            {
                float * bias_buffer = (float *)sys_malloc(conv_bias->elem_num * sizeof(float));
                this->host_buffer.push_back(bias_buffer);
                if (1 == conv_bias->quant_param_num)
                {
                    for (uint32_t i = 0; i < conv_bias->elem_num; i++)
                    {
                        bias_buffer[i] = (float)(((int32_t*)conv_bias->data)[i]) * conv_bias->scale;
                    }
                }
                else
                {
                    for (uint32_t i = 0; i < conv_bias->elem_num; i++)
                    {
                        bias_buffer[i] = (float)(((int32_t*)conv_bias->data)[i]) * conv_bias->scale_list[i];
                    }
                }
                biasWeights.values = bias_buffer;
                biasWeights.count = conv_bias->elem_num;
                biasWeights.type = nvdla::DataType::FLOAT;
                break;
            }
            default:
                fprintf(stderr, "Tengine: Unsupported weight quant data type(%d) of deconv(id: %d, name: %s).\n", conv_bias->data_type, ir_node->index, ir_node->name);
                return nullptr;
            }

            deconvolutionNode->params().setHasBiasTerm(true);
            if(conv_bias->dim_num == 1) deconvolutionNode->params().setBiasMode(nvdla::bCHANNEL);
            deconvolutionNode->params().setBiasDims(biasDims);
        }
    }
    else    // conv group != 1
    {
        fprintf(stderr, "%s : can not support group convolution .\n", __func__);
    }

    deconvolutionNode->params().setWeights(kernelWeights);
    deconvolutionNode->params().setBiasData(biasWeights);


    Node = deconvolutionNode;
    nvdla::priv::canonical_ast::NodeFactory::s_deconv_priv.insert(
        std::pair<nvdla::priv::canonical_ast::Node*, nvdla::priv::canonical_ast::DeconvolutionNode*>(Node, deconvolutionNode)
    );

    return Node;
}
