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
#include "pooling_param.h"
}


nvdla::priv::canonical_ast::Node * ODLAEngine::AddPoolingNode(struct node* ir_node)
{
    struct graph* ir_graph = ir_node->graph;
    struct pool_param* param = (struct pool_param*)ir_node->op.param_mem;
    nvdla::PoolingType pooltype;
    nvdla::priv::canonical_ast::Node * Node ;
    auto * poolingNode = new nvdla::priv::canonical_ast::PoolingNode();

    // Init Node
    if (param->pool_method == 0)
    {
        pooltype = nvdla::PoolingType::kMAX;
    }
    else
    {
        pooltype = nvdla::PoolingType::kAVERAGE;
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
