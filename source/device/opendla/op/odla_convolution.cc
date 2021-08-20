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
#include "convolution_param.h"
}

nvdla::priv::canonical_ast::Node * ODLAEngine::AddConvolutionNode(struct node* ir_node)
{
    struct graph* ir_graph = ir_node->graph;
    struct conv_param* param = (struct conv_param*)ir_node->op.param_mem;

    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct tensor* conv_weight = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    struct subgraph* subgraph = get_ir_graph_subgraph(ir_graph, ir_node->subgraph_idx);

    nvdla::priv::canonical_ast::Node* Node ;
    auto * convolutionNode = new nvdla::priv::canonical_ast::ConvolutionNode();

    // Init Node
    nvdla::Dims2 topLeftPadding(param->pad_h0, param->pad_w0);
    nvdla::Dims2 bottomRightPadding(param->pad_h1, param->pad_w1);
    nvdla::Dims2 kernel(param->kernel_h,param->kernel_w);
    nvdla::Dims2 stride(param->stride_h,param->stride_w);
    nvdla::Dims2 dilation(param->dilation_h,param->dilation_w);
    convolutionNode->params().setBiasMode(nvdla::bNONE);
    convolutionNode->params().setHasBiasTerm(false);
    convolutionNode->params().setTopLeftPadding(topLeftPadding);
    convolutionNode->params().setBottomRightPadding(bottomRightPadding);
    convolutionNode->params().setPaddingValue(0);
    convolutionNode->params().setStride(stride);
    convolutionNode->params().setDilation(dilation);
    convolutionNode->params().setNumGroups(param->group);


    nvdla::Weights kernelWeights;
    nvdla::Weights biasWeights;
    if (param->group == 1 || (param->group == conv_weight->dims[0] && param->group != 1))   // conv + dwconv
    {
            nvdla::Dims4 weightDims(conv_weight->dims[0], conv_weight->dims[1], conv_weight->dims[2], conv_weight->dims[3]);
            convolutionNode->params().setWeightDims(weightDims);

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
                    if (conv_weight->quant_param_num != conv_weight->dims[0])
                    {
                        fprintf(stderr, "Tengine: Unsupported weight quant channel of conv(id: %d, name: %s).\n", ir_node->index, ir_node->name);
                        return nullptr;
                    }
                    float* weight_buffer = (float*)sys_malloc(conv_weight->elem_num * sizeof(float));
                    this->host_buffer.push_back(weight_buffer);
                    if (1 == conv_weight->quant_param_num)
                    {
                        for (uint32_t i = 0; i < conv_weight->elem_num; i++)
                        {
                            weight_buffer[i] = (float)(((int8_t*)conv_weight->data)[i]) * conv_weight->scale;
                        }
                    }else for (int ch = 0; ch < conv_weight->quant_param_num; ch++)
                    {
                        int block_size = conv_weight->dims[1] * conv_weight->dims[2] * conv_weight->dims[3];
                        for (int i = 0; i < block_size; i++)
                        {
                            int offset = block_size * ch;
                            weight_buffer[offset + i] = (float)(((int8_t*)conv_weight->data)[offset + i]) * conv_weight->scale_list[ch];
                        }
                        for (int i = 0; i < block_size; i++){
                            int offset = block_size * ch;
                            if(std::abs(weight_buffer[offset + i]) < 1e-4 && ((int8_t*)conv_weight->data)[offset + i] != 0) {
                                weight_buffer[offset + i] = 0;
                            }
                        }
                    }

                    kernelWeights.values = weight_buffer;
                    kernelWeights.count = conv_weight->elem_num;
                    kernelWeights.type = nvdla::DataType::FLOAT;
                    break;
                }
                case TENGINE_DT_UINT8:
                {
                    std::vector<float> weight_buffer;
                    weight_buffer.resize(conv_weight->elem_num);

                    for (int i = 0; i < conv_weight->elem_num; i++)
                    {
                        weight_buffer[i] = (float)(((uint8_t*)conv_weight->data)[i] - conv_weight->zero_point) * conv_weight->scale;
                    }

                    kernelWeights.values = weight_buffer.data();
                    kernelWeights.count = conv_weight->elem_num;
                    kernelWeights.type = nvdla::DataType::FLOAT;
                    break;
                }
                default:
                    fprintf(stderr, "Tengine: Unsupported weight quant data type(%d) of conv(id: %d, name: %s).\n", conv_weight->data_type, ir_node->index, ir_node->name);
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
                        fprintf(stderr, "Tengine: Unsupported weight quant data type(%d) of conv(id: %d, name: %s).\n", conv_bias->data_type, ir_node->index, ir_node->name);
                        return nullptr;
                }

                convolutionNode->params().setHasBiasTerm(true);
                if(conv_bias->dim_num == 1) convolutionNode->params().setBiasMode(nvdla::bCHANNEL);
                convolutionNode->params().setBiasDims(biasDims);
            }
    }
    else    // conv group != 1
    {
        fprintf(stderr, "%s : can not support group convolution .\n", __func__);
    }

    convolutionNode->params().setWeights(kernelWeights);
    convolutionNode->params().setBiasData(biasWeights);


    // Insert priv pair
    Node = convolutionNode;
    nvdla::priv::canonical_ast::NodeFactory::s_conv_priv.insert(
    std::pair<nvdla::priv::canonical_ast::Node*, nvdla::priv::canonical_ast::ConvolutionNode*>(Node, convolutionNode)
    );

    return Node;
}
