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
    std::vector<nvinfer1::ITensor*> input_list;

    for(uint8_t i = 0; i < node->input_num; i++)
    {
        struct tensor* ir_tensor = get_ir_graph_tensor(ir_graph, node->input_tensors[0]);
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
        //case ELT_EXP:
        //    trt_op_type = nvinfer1::ElementWiseOperation::;
        //    break;
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

    tensor_real_map[tensor_swap_map[node->output_tensors[0]]] = trt_tensor;
    tensor_swap_map[node->output_tensors[0]] = node->output_tensors[0];

    return true;
}
