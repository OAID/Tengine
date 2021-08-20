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
#include "eltwise_param.h"

#include "../common/compiler_fp16.h"
}


nvdla::priv::canonical_ast::Node * ODLAEngine::AddEltwiseNode(struct node* ir_node)
{
    struct graph* ir_graph = ir_node->graph;
    struct subgraph* subgraph = get_ir_graph_subgraph(ir_graph, ir_node->subgraph_idx);
    struct tensor* input_tensor0 = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);

    struct tensor* const_tensor = nullptr;
    if (ir_node->input_num > 1)
        const_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);

    nvdla::priv::canonical_ast::Node* Node ;
    auto * elementWiseNode = new nvdla::priv::canonical_ast::ElementWiseNode();

    eltwise_param* param = (eltwise_param*)ir_node->op.param_mem;
    if (nullptr != const_tensor && const_tensor->tensor_type == TENSOR_TYPE_CONST && const_tensor->data_type == TENGINE_DT_FP32)
    {
        float* const_fp32 = ( float* )get_tensor_buffer(const_tensor);
        int const_size = get_tensor_buffer_size(const_tensor) / sizeof(float) ;
        if (const_size == 1 && const_fp32[0] == 0)
        {
            struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
            std::vector<uint32_t> perm;
            for (int i = output_tensor->dim_num - 1; i >= 0; i--)
            {
                perm.push_back(output_tensor->dims[i]);
            }
        }
        else if (const_size == 1 && const_fp32[0] != 0)
        {
            struct tensor* input_tensor1 = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);

            float data_fp32 = ((float*)get_tensor_buffer(input_tensor1))[0];
            __fp16 data_fp16 = fp32_to_fp16(data_fp32);

            __fp16* fp16_data = (__fp16*)malloc(input_tensor0->elem_num * sizeof(__fp16) );
            for (int k = 0; k < input_tensor0->elem_num; k++)
            {
                fp16_data[k] = data_fp16;
            }

        }
        else if (const_size == input_tensor0->dims[1])
        {
            struct tensor* input_tensor1 = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);

            float* data_fp32 = (float*)get_tensor_buffer(input_tensor1);

            __fp16* fp16_data = (__fp16*)malloc(input_tensor0->elem_num * sizeof(__fp16) );
            for (int p = 0; p < input_tensor0->dims[1]; p++)
            {
                for (int k = 0; k < input_tensor0->elem_num / input_tensor0->dims[1]; k++)
                {
                    __fp16 data_fp16 = fp32_to_fp16(data_fp32[p]);
                    fp16_data[k] = data_fp16;
                }
            }

        }
        else
        {
            //to do
            fprintf(stderr,"To do!!\n");
        }

    }

    switch (param->type)
    {
        case ELT_MAX:
        {
            elementWiseNode->params().setType(nvdla::ElementWiseOperation::ew_kMAX);
            break;
        }
        case ELT_MIN_SCALAR:
        {
            elementWiseNode->params().setType(nvdla::ElementWiseOperation::kMIN);
            break;
        }
        case ELT_PROD:
        case ELT_PROD_SCALAR:
        {
            elementWiseNode->params().setType(nvdla::ElementWiseOperation::kPROD);
            break;
        }
        case ELT_SUM:
        case ELT_SUM_SCALAR:
        {
            elementWiseNode->params().setType(nvdla::ElementWiseOperation::kSUM);
            break;
        }
        case ELT_SUB:
        case ELT_SUB_SCALAR:
        case ELT_EXP:
        default:
            fprintf(stderr, "Opendla does not support %d element wise type. \n", param->type);
            break;
    }

    // Insert priv pair
    Node = elementWiseNode;
    nvdla::priv::canonical_ast::NodeFactory::s_ew_priv.insert(
        std::pair<nvdla::priv::canonical_ast::Node*, nvdla::priv::canonical_ast::ElementWiseNode*>(Node, elementWiseNode)
    );

    return Node;
}
