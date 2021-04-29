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

#include "acl_executor.hpp"

extern "C"
{
#include "operator/op.h"
}

bool CLGraph::AddInputLayer(struct node* node)
{
    TLOG_INFO("Tengine ACL: Support OP(%d) OP_INPUT.\n", node->index);
    /* output */
    struct graph* graph = node->graph;
    struct tensor* tensor = get_ir_graph_tensor(graph, node->output_tensors[0]);
    char* name = tensor->name;
    int* dim_w = tensor->dims;
    CLTensor* otensor = new CLTensor();
    TensorInfo ClTensorInfo = TensorInfo(TensorShape(dim_w[2], dim_w[3], dim_w[1], dim_w[0]), 1, data_type_);
    DataLayout aclDataLayout;
    aclDataLayout = (tensor->layout == 0) ? DataLayout::NCHW : DataLayout::NHWC;
    ClTensorInfo.set_data_layout(aclDataLayout);
    otensor->allocator()->init(ClTensorInfo);
    tensors_map_[name] = otensor;

    return true;
}
