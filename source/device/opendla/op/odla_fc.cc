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


nvdla::priv::canonical_ast::Node * ODLAEngine::AddFullyConnectionNode(struct node* ir_node)
{
    struct graph* ir_graph = ir_node->graph;
    struct subgraph* subgraph = get_ir_graph_subgraph(ir_graph, ir_node->subgraph_idx);

    struct tensor* weight_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);

    nvdla::priv::canonical_ast::Node* Node ;
    auto * fullyConnectedNode = new nvdla::priv::canonical_ast::FullyConnectedNode();



    fullyConnectedNode->params().setBiasMode(nvdla::bNONE);
    fullyConnectedNode->params().setHasBiasTerm(false);


    nvdla::Weights kernelWeights;
    nvdla::Weights biasWeights;

    nvdla::Dims4 weightDims(odla_tensor_map[weight_tensor->index]->getDimensions());
    fullyConnectedNode->params().setWeightDims(weightDims);

    switch (weight_tensor->data_type)
    {
        case TENGINE_DT_FP32:
        {
            kernelWeights.values = weight_tensor->data;
            kernelWeights.count = weight_tensor->elem_num;
            kernelWeights.type = nvdla::DataType::FLOAT;
            break;
        }
        case TENGINE_DT_INT8:
        {
            if (weight_tensor->quant_param_num != weight_tensor->dims[0])
            {
                fprintf(stderr, "Tengine: Unsupported weight quant channel of fc(id: %d, name: %s).\n", ir_node->index, ir_node->name);
                return nullptr;
            }

            float* weight_buffer = (float*)malloc(weight_tensor->elem_num * sizeof(float));
            this->host_buffer.push_back(weight_buffer);

            for (int ch = 0; ch < weight_tensor->quant_param_num; ch++)
            {
                int block_size = weightDims.c * weightDims.h * weightDims.w;
                for (int i = 0; i < block_size; i++)
                {
                    int offset = block_size * ch;
                    weight_buffer[offset + i] = (float)(((int8_t*)weight_tensor->data)[offset + i]) * weight_tensor->scale_list[ch];
                }
            }

            kernelWeights.values = weight_buffer;
            kernelWeights.count = weight_tensor->elem_num;
            kernelWeights.type = nvdla::DataType::FLOAT;
            break;
        }
        case TENGINE_DT_UINT8:
        {
            float* weight_buffer = (float*)malloc(weight_tensor->elem_num * sizeof(float));
            this->host_buffer.push_back(weight_buffer);

            for (int i = 0; i < weight_tensor->elem_num; i++)
            {
                weight_buffer[i] = (float)(((uint8_t*)weight_tensor->data)[i] - weight_tensor->zero_point) * weight_tensor->scale;
            }

            kernelWeights.values = weight_buffer;
            kernelWeights.count = weight_tensor->elem_num;
            kernelWeights.type = nvdla::DataType::FLOAT;
            break;
        }
        default:
            fprintf(stderr, "Tengine: Unsupported weight quant data type(%d) of fc(id: %d, name: %s).\n", weight_tensor->data_type, ir_node->index, ir_node->name);
            return nullptr;
    }

    if (ir_node->input_num > 2)
    {
        struct tensor* bias_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[2]);
        nvdla::Dims4 biasDims(odla_tensor_map[bias_tensor->index]->getDimensions());

        switch (bias_tensor->data_type)
        {
            case TENGINE_DT_FP32:
            {
                biasWeights.values = bias_tensor->data;
                biasWeights.count = bias_tensor->elem_num;
                biasWeights.type = nvdla::DataType::FLOAT;
                break;
            }
            case TENGINE_DT_INT32:
            {
                float* bias_buffer = (float*)malloc(bias_tensor->elem_num * sizeof(float));
                this->host_buffer.push_back(bias_buffer);

                if (1 == bias_tensor->quant_param_num)
                {
                    for (uint32_t i = 0; i < bias_tensor->elem_num; i++)
                    {
                        bias_buffer[i] = (float)(((int32_t*)bias_tensor->data)[i]) * bias_tensor->scale;
                    }
                }
                else
                {
                    for (uint32_t i = 0; i < bias_tensor->elem_num; i++)
                    {
                        bias_buffer[i] = (float)(((int32_t*)bias_tensor->data)[i]) * bias_tensor->scale_list[i];
                    }
                }
                biasWeights.values = bias_buffer;
                biasWeights.count = bias_tensor->elem_num;
                biasWeights.type = nvdla::DataType::FLOAT;
                break;
            }
            default:
                fprintf(stderr, "Tengine: Unsupported weight quant data type(%d) of fc(id: %d, name: %s).\n", bias_tensor->data_type, ir_node->index, ir_node->name);
                return nullptr;
        }

        fullyConnectedNode->params().setHasBiasTerm(true);
        if(bias_tensor->dim_num == 1) fullyConnectedNode->params().setBiasMode(nvdla::bCHANNEL);
        fullyConnectedNode->params().setBiasDims(biasDims);
    }

    fullyConnectedNode->params().setWeights(kernelWeights);
    fullyConnectedNode->params().setBiasData(biasWeights);



    // Insert priv pair
    Node = fullyConnectedNode;
    nvdla::priv::canonical_ast::NodeFactory::s_fc_priv.insert(
        std::pair<nvdla::priv::canonical_ast::Node*, nvdla::priv::canonical_ast::FullyConnectedNode*>(Node, fullyConnectedNode)
    );

    return Node;
}
