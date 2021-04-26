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
 * Copyright (c) 2021, OPEN AI LAB
 * Author: qtang@openailab.com
 */


#include "acl_executor.hpp"

extern "C"
{
#include "operator/op.h"
}

bool CLGraph::AddCastLayer(struct node* node)
{
    struct graph* graph = node->graph;
    struct tensor* input_tensor = get_ir_graph_tensor(graph, node->input_tensors[0]);
    std::string name_in = input_tensor->name;

    /* set acl input tensor */
    CLTensor* itensor = nullptr;
    if (tensors_map_.count(name_in))
    {
        itensor = tensors_map_[name_in];
    }
    else
    {
        TLOG_ERR("can't find node [%s]tensor named :%s\n", node->name, name_in.c_str());
        return false;
    }

    /* set acl output tensor */
    struct tensor* out_tensor = get_ir_graph_tensor(graph, node->output_tensors[0]);
    int* dim_o = out_tensor->dims;
    std::string name_out = out_tensor->name;
    CLTensor* otensor = new CLTensor();

    TensorInfo ClTensorInfo_o = TensorInfo(TensorShape(dim_o[1], dim_o[3], dim_o[2], dim_o[0]), 1, data_type_);
    ClTensorInfo_o.set_data_layout(DataLayout::NHWC);
    otensor->allocator()->init(ClTensorInfo_o);
    tensors_map_[name_out] = otensor;

    /* add acl reshape layer into acl graph */
    std::shared_ptr<CLCast> cast = std::make_shared<CLCast>();
    cast->configure(itensor, otensor, ConvertPolicy::SATURATE);
    functions_map_.push_back(cast);

    return true;
}
