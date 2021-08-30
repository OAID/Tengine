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
#include "pooling_param.h"
}


nvdla::priv::canonical_ast::Node * ODLAEngine::AddPoolingNode(struct node* ir_node)
{
    struct graph* ir_graph = ir_node->graph;
    struct pool_param* param = (struct pool_param*)ir_node->op.param_mem;
    nvdla::PoolingType pooltype;
    nvdla::priv::canonical_ast::Node * Node ;
    auto * poolingNode = new nvdla::priv::canonical_ast::PoolingNode();
    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    // Init Node
    if (param->pool_method == 0)
    {
        pooltype = nvdla::PoolingType::kMAX;
    }
    else
    {
        pooltype = nvdla::PoolingType::kAVERAGE;
        if(1 == param->global){
            if(1 == input_tensor->quant_param_num){
                output_tensor->scale = input_tensor->scale;
                float tensor_min_val = output_tensor->scale * -127.0f;
                float tensor_max_val = output_tensor->scale * +127.0f;
                this->odla_tensor_map[output_tensor->index]->setChannelDynamicRange(-1, tensor_min_val, tensor_max_val);
            }else if (1 < input_tensor->quant_param_num){
                for (int ch = 0; ch < input_tensor->quant_param_num; ++ch)
                {
                    output_tensor->scale_list[ch] = input_tensor->scale_list[ch];
                    float tensor_min_val = output_tensor->scale_list[ch] * -127.0f;
                    float tensor_max_val = output_tensor->scale_list[ch] * +127.0f;
                    this->odla_tensor_map[output_tensor->index]->setChannelDynamicRange(ch, tensor_min_val, tensor_max_val);
                }
            }
        }
    }
    nvdla::Dims2 topLeftPadding(param->pad_h0, param->pad_w0);
    nvdla::Dims2 bottomRightPadding(param->pad_h1, param->pad_w1);
    nvdla::Dims2 kernel(param->kernel_h,param->kernel_w);
    nvdla::Dims2 stride(param->stride_h,param->stride_w);

    poolingNode->params().setPoolType(pooltype);
    poolingNode->params().setTopLeftPadding(topLeftPadding);
    poolingNode->params().setBottomRightPadding(bottomRightPadding);
    poolingNode->params().setKernelDims(kernel);
    poolingNode->params().setStride(stride);

    Node = poolingNode;
    nvdla::priv::canonical_ast::NodeFactory::s_pool_priv.insert(
        std::pair<nvdla::priv::canonical_ast::Node*, nvdla::priv::canonical_ast::PoolingNode*>(Node, poolingNode)
    );

    return Node;
}
