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
 * Author: lswang@openailab.com
 */

#include "../trt_executor.hpp"

EXPORT_BEGIN
#include "eltwise_param.h"
EXPORT_FINISH

#include <NvInferRuntime.h>


// TODO: fix bug
bool TensorRTEngine::AddEltwiseLayer(struct graph* ir_graph, struct node* node)
{
    struct tensor* const_tensor = nullptr;

    int pi = 0;
    if (node->input_num > 1)
    {
        for (int i = 0; i < node->input_num; i++)
        {
            const_tensor = get_ir_graph_tensor(ir_graph, node->input_tensors[i]);
            if (const_tensor->tensor_type == TENSOR_TYPE_CONST)
            {
                if (i == 0)
                    pi = 1;
                break;
            }
        }
    }

    if (nullptr != const_tensor && const_tensor->tensor_type == TENSOR_TYPE_CONST && const_tensor->data_type == TENGINE_DT_FP32)
    {
        float* const_fp32 = ( float* )get_tensor_buffer(const_tensor);
        int const_size = get_tensor_buffer_size(const_tensor) / sizeof(float) ;
        if (const_size == 1 && const_fp32[0] == 0)
        {
            struct tensor* reshape_input = get_ir_graph_tensor(ir_graph, node->input_tensors[pi]);
            struct tensor* reshape_output = get_ir_graph_tensor(ir_graph, node->output_tensors[0]);
            if (nullptr == reshape_input || nullptr == reshape_output)
            {
                fprintf(stderr, "Tengine: Get input & output for Reshape(id: %d, name: %s) layer failed.\n", node->index, node->name);
                return false;
            }

            if (!check_if_input_in_map(reshape_input->index, this->tensor_swap_map))
            {
                fprintf(stderr, "Tengine: Query input for Reshape(id: %d, name: %s) layer failed.\n", node->index, node->name);
                return false;
            }

            nvinfer1::ITensor* trt_tensor = tensor_real_map[tensor_swap_map[reshape_input->index]];

            nvinfer1::IShuffleLayer* layer = this->network->addShuffle(*trt_tensor);
            if (nullptr == layer)
            {
                fprintf(stderr, "Tengine: Add Reshape(id: %d, name: %s) layer failed.\n", node->index, node->name);
                return false;
            }

            layer->setName(node->name);

            nvinfer1::Dims dims{};
            dims.nbDims = reshape_output->dim_num;

            for (int i = 0; i < dims.nbDims; i++)
                dims.d[i] = reshape_output->dims[i];

            layer->setReshapeDimensions(dims);

            this->layer_map[node->index] = layer;

            nvinfer1::ITensor* reshape_output_tensor = layer->getOutput(0);

            this->SetRange(reshape_output, reshape_output_tensor);

            this->tensor_real_map[node->output_tensors[0]] = reshape_output_tensor;
            this->tensor_swap_map[node->output_tensors[0]] = node->output_tensors[0];

            return true;
        }
        else if (const_size == 1)
        {
            float input1_data = ((float*)const_tensor->data)[0];
            struct tensor* input0 = get_ir_graph_tensor(ir_graph, node->input_tensors[pi]);

            nvinfer1::Weights weight{nvinfer1::DataType::kFLOAT, nullptr, 0};

            float* weight_buffer = (float*)sys_malloc(input0->elem_num * input0->elem_size);
            this->host_buffer.push_back(weight_buffer);

            for (int i = 0; i < input0->elem_num; i++)
                weight_buffer[i] = input1_data;

            weight.values = weight_buffer;
            weight.count = input0->elem_num;
            weight.type = nvinfer1::DataType::kFLOAT;

            nvinfer1::Dims4 dim4(input0->dims[0], input0->dims[1], input0->dims[2], input0->dims[3]);
            nvinfer1::IConstantLayer* layer_input1 = this->network->addConstant(dim4, weight);
            layer_input1->setName(const_tensor->name);

            nvinfer1::ITensor * trt_input1_tensor = layer_input1->getOutput(0);

            this->tensor_real_map[const_tensor->index] = trt_input1_tensor;
            this->tensor_swap_map[const_tensor->index] = const_tensor->index;
        }
    }


    std::vector<nvinfer1::ITensor*> input_list;

    for(uint8_t i = 0; i < node->input_num; i++)
    {
        struct tensor* ir_tensor = get_ir_graph_tensor(ir_graph, node->input_tensors[pi]);
        if (nullptr != ir_tensor && !check_if_input_in_map(ir_tensor->index, this->tensor_swap_map))
        {
            fprintf(stderr, "Tengine: Query input for Eltwise(id: %d, name: %s) layer failed.\n", node->index, node->name);
            return false;
        }

        input_list.push_back(tensor_real_map[tensor_swap_map[node->input_tensors[i]]]);
    }

    eltwise_param* param = (eltwise_param*)node->op.param_mem;

    nvinfer1::ElementWiseOperation trt_op_type;

    switch (param->type)
    {
        case ELT_PROD:
        case ELT_PROD_SCALAR:
            trt_op_type = nvinfer1::ElementWiseOperation::kPROD;
            break;
        case ELT_SUM:
        case ELT_SUM_SCALAR:
            trt_op_type = nvinfer1::ElementWiseOperation::kSUM;
            break;
        case ELT_SUB:
        case ELT_SUB_SCALAR:
            trt_op_type = nvinfer1::ElementWiseOperation::kSUB;
            break;
        case ELT_MAX:
            trt_op_type = nvinfer1::ElementWiseOperation::kMAX;
            break;
        //case ELT_RSQRT:
        //    trt_op_type = nvinfer1::ElementWiseOperation::;
        //    break;
        case ELT_MIN_SCALAR:
            trt_op_type = nvinfer1::ElementWiseOperation::kMIN;
            break;
        //case ELT_LAST:
        //    trt_op_type = nvinfer1::ElementWiseOperation::;
        //    break;
        case ELT_DIV:
            trt_op_type = nvinfer1::ElementWiseOperation::kDIV;
            break;
        //case ELT_LOG:
        //    trt_op_type = nvinfer1::ElementWiseOperation::;
        //    break;
//        case ELT_EXP:
//            trt_op_type = nvinfer1::UnaryOperation::kEXP;
//            break;
        //case ELT_SQRT:
        //    trt_op_type = nvinfer1::ElementWiseOperation::;
        //    break;
        //case ELT_FLOOR:
        //    trt_op_type = nvinfer1::ElementWiseOperation::kFLOOR_DIV;
        //    break;
        //case ELT_SQUARE:
        //    trt_op_type = nvinfer1::ElementWiseOperation::;
        //    break;
        case ELT_POW:
            trt_op_type = nvinfer1::ElementWiseOperation::kPOW;
            break;
        //case ELT_POWER:
        //    trt_op_type = nvinfer1::ElementWiseOperation::;
        //    break;
        default:
            fprintf(stderr, "Tengine: Eltwise(id: %d) type(%d) was not supported.\n", node->index, param->type);
            return false;
    }

    if (ELT_EXP == param->type)
    {
        nvinfer1::IUnaryLayer* layer = this->network->addUnary(*input_list[0], nvinfer1::UnaryOperation::kEXP);
        if (nullptr == layer)
        {
            fprintf(stderr, "Tengine: Add Eltwise(id: %d, name: %s) layer failed.\n", node->index, node->name);
            return false;
        }

        layer->setName(node->name);

        this->layer_map[node->index] = layer;

        nvinfer1::ITensor * trt_tensor = layer->getOutput(0);

        this->SetRange(ir_graph, node->output_tensors[0], trt_tensor);

        tensor_swap_map[node->output_tensors[0]] = node->output_tensors[0];
        tensor_real_map[tensor_swap_map[node->output_tensors[0]]] = trt_tensor;
    }
    else
    {
        nvinfer1::IElementWiseLayer* layer = this->network->addElementWise(*input_list[0], *input_list[1], trt_op_type);
        if (nullptr == layer)
        {
            fprintf(stderr, "Tengine: Add Eltwise(id: %d, name: %s) layer failed.\n", node->index, node->name);
            return false;
        }

        layer->setName(node->name);

        this->layer_map[node->index] = layer;

        nvinfer1::ITensor * trt_tensor = layer->getOutput(0);

        this->SetRange(ir_graph, node->output_tensors[0], trt_tensor);

        tensor_swap_map[node->output_tensors[0]] = node->output_tensors[0];
        tensor_real_map[tensor_swap_map[node->output_tensors[0]]] = trt_tensor;
    }




    return true;
}
