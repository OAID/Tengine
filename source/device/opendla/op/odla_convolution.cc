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

    convolutionNode->setGraph(this->graph);
    this->graph->insertNode(convolutionNode);
    convolutionNode->setId(this->graph->nextNodeId());
    convolutionNode->setName(ir_node->name);


    nvdla::Weights kernelWeights;
    nvdla::Weights biasWeights;
    if (param->group == 1 || (param->group == conv_weight->dims[0] && param->group != 1))   // conv + dwconv
    {
        if (param->activation >= 0) // activation function exist
        {
            /*
            nvdla::priv::TensorFactory::TensorPrivPair t = nvdla::priv::TensorFactory::newTensor();
            nvdla::priv::Tensor* tmp_output_tensor = NULL;
            nvdla::Dims4 tensor_shape;
            nvdla::DataType datatype;
            switch(output_tensor->data_type)
            {
                // Why no Definition of DATATYPE?
            case TENGINE_DT_FP32:
                // float32
                datatype = nvdla::DataType::FLOAT;
                break;
            case TENGINE_DT_FP16:
                // float16
                datatype = nvdla::DataType::HALF;
                break;
            case TENGINE_DT_INT8:
                datatype = nvdla::DataType::INT8;
                break;
            case TENGINE_DT_UINT8:
                datatype = nvdla::DataType::UINT8;
                break;
            case TENGINE_DT_INT32:
                TLOG_ERR("Tensor date type: Tensor_name(%s) tensor_index(%d) tensor_data_type(%d) .\n",ir_tensor->name, ir_tensor->index, ir_tensor->data_type);
                break;
            default:
                TLOG_ERR("Tensor date type: Tensor_name(%s) tensor_index(%d) tensor_data_type(%d) .\n",ir_tensor->name, ir_tensor->index, ir_tensor->data_type);
                break;
            }

            tensor_shape.n = output_tensor->dims[0];
            tensor_shape.c = output_tensor->dims[1];
            tensor_shape.h = output_tensor->dims[2];
            tensor_shape.w = output_tensor->dims[3];
            t.i()->setDimensions(tensor_shape);
            t.i()->setDataFormat(NVDLA_DATA_FORMAT_NCHW);
            t.i()->setTensorType(nvdla::kIO);
            t.i()->setDataType(output_tensor->data_type);
            t.i()->setChannelDynamicRange(-1, -16129, 16129);

            auto tmp_output_Edge = new nvdla::priv::canonical_ast::Edge();
            t.i()->setName((std::string(input_tensor->name)+"activation").c_str());
            tmp_output_Edge->setGraph(this->graph);
            tmp_output_Edge->setId(graph->nextEdgeId());
            tmp_output_Edge->setOriginalTensor(tmp_output_tensor);

            if (ir_node->input_num > 2)    // bias exist
            {
                struct tensor* conv_bias = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[2]);
                if(conv_bias->dim_num);
                convolutionNode->params().setBiasMode(nvdla::bNONE);
                convolutionNode->params().setHasBiasTerm(true);
                (*conv)
                    .BindInputs({ this->vx_tensor_map[input_tensor->index], this->vx_tensor_map[conv_weight->index], this->vx_tensor_map[conv_bias->index] })
                    .BindOutputs({ tmp_output });
            }
            else
            {
                (*conv)
                    .BindInputs({ this->vx_tensor_map[input_tensor->index], this->vx_tensor_map[conv_weight->index] })
                    .BindOutputs({ tmp_output });
            }
            this->vx_tensor_map[output_tensor->index + ir_graph->tensor_num] = tmp_output;
            if (param->activation == 0) // Relu
            {
            }
            else if (param->activation == 6)    // Relu6
            {
            }
        */
        }
        else // activation function does not exist
        {
            nvdla::Dims4 weightDims(conv_weight->dims[0], conv_weight->dims[1], conv_weight->dims[2], conv_weight->dims[3]);
            convolutionNode->params().setWeightDims(weightDims);

            kernelWeights.count = conv_weight->elem_num;
            kernelWeights.values = conv_weight->data;
            kernelWeights.type = odla_tensor_map[conv_weight->index]->getDataType();

            if (ir_node->input_num > 2) // bias exist
            {
                struct tensor* conv_bias = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[2]);
                nvdla::Dims4 biasDims(odla_tensor_map[conv_bias->index]->getDimensions());

                biasWeights.count = conv_bias->elem_num;
                biasWeights.values = conv_bias->data;
                biasWeights.type = odla_tensor_map[conv_bias->index]->getDataType();

                convolutionNode->params().setHasBiasTerm(true);
                if(conv_bias->dim_num == 1) convolutionNode->params().setBiasMode(nvdla::bCHANNEL);
                convolutionNode->params().setBiasDims(biasDims);
            }
            else
            {
            }
        }
    }
    else    // conv group != 1
    {
        if (param->activation >= 0)
        {
            if (ir_node->input_num > 2)
            {

            }
            else
            {

            }
            if (param->activation == 0)
            {

            }
            else if (param->activation == 6)
            {

            }

        }
        else
        {
            if (ir_node->input_num > 2)
            {
            }
            else
            {
            }
        }
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
