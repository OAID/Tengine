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

bool CLGraph::AddReshapeLayer(struct node* node)
{
    struct graph* graph = node->graph;
    struct tensor* input_tensor = get_ir_graph_tensor(graph, node->input_tensors[0]);
    int* dim_i = input_tensor->dims;
    std::string name_in = input_tensor->name;

    /* ugly code */
    for (int j=0;j<4;j++)
    {
        if (dim_i[j] == 0)
            dim_i[j] = 1;
    }

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

    /* ugly code */
    for (int j=0;j<4;j++)
    {
        if (dim_o[j] == 0)
            dim_o[j] = 1;
    }

    std::string name_out = out_tensor->name;
    CLTensor* otensor = new CLTensor();

    TensorInfo ClTensorInfo_o = TensorInfo(TensorShape(dim_o[1], dim_o[3], dim_o[2], dim_o[0]), 1, data_type_);
    ClTensorInfo_o.set_data_layout(DataLayout::NHWC);
    otensor->allocator()->init(ClTensorInfo_o);
    tensors_map_[name_out] = otensor;

    if (input_tensor->layout == TENGINE_LAYOUT_NCHW)
    {
        arm_compute::Strides dm0(1, 2, 0, 3);
        arm_compute::Strides dm1(2, 0, 1, 3);

        CLTensor* permute0_tensor = new CLTensor();
        TensorInfo ClTensorInfo_p0 = TensorInfo(TensorShape(dim_i[3], dim_i[2], dim_i[1], dim_i[0]), 1, data_type_);
        ClTensorInfo_p0.set_data_layout(DataLayout::NCHW);
        permute0_tensor->allocator()->init(ClTensorInfo_p0);
        tensors_map_[std::to_string(node->index+0.1).c_str()] = permute0_tensor;

        CLTensor* permute1_tensor = new CLTensor();
        TensorInfo ClTensorInfo_p1 = TensorInfo(TensorShape(dim_o[3], dim_o[2], dim_o[1], dim_o[0]), 1, data_type_);
        ClTensorInfo_p1.set_data_layout(DataLayout::NCHW);
        permute1_tensor->allocator()->init(ClTensorInfo_p1);
        tensors_map_[std::to_string(node->index+0.2).c_str()] = permute1_tensor;

        /* add acl reshape layer into acl graph */
        std::shared_ptr<CLPermute> permute0 = std::make_shared<CLPermute>();
        permute0->configure(itensor, permute0_tensor, dm0);
        functions_map_.push_back(permute0);

        std::shared_ptr<CLReshapeLayer> reshape = std::make_shared<CLReshapeLayer>();
        reshape->configure(permute0_tensor, permute1_tensor);
        functions_map_.push_back(reshape);

        std::shared_ptr<CLPermute> permute1 = std::make_shared<CLPermute>();
        permute1->configure(permute1_tensor, otensor, dm1);
        functions_map_.push_back(permute1);
    }
    else
    {
        std::shared_ptr<CLReshapeLayer> reshape = std::make_shared<CLReshapeLayer>();
        reshape->configure(itensor, otensor);
        functions_map_.push_back(reshape);
    }

    return true;
}
