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
#include "relu_param.h"
}

bool CLGraph::AddReLuLayer(struct node* node)
{
    struct graph* graph = node->graph;
    struct tensor* input_tensor = get_ir_graph_tensor(graph, node->input_tensors[0]);
    std::string name = input_tensor->name;
    struct relu_param* param = ( struct relu_param* )node->op.param_mem;

    float slop_param = param->negative_slope;
    CLTensor* itensor = nullptr;
    if (tensors_map_.count(name))
    {
        itensor = tensors_map_[name];
    }
    else
    {
        // TLOG_ERR("can't find node [%s]tensor named :%s\n", node->name, name);
        return false;
    }

    struct tensor* out_tensor = get_ir_graph_tensor(graph, node->output_tensors[0]);
    int* dim_o = out_tensor->dims;
    name = out_tensor->name;
    CLTensor* otensor = new CLTensor();
    int TengineDataLayOut = out_tensor->layout;

    if (bForcedNHWCMode_ == true && TengineDataLayOut == TENGINE_LAYOUT_NCHW)
    {
        // need to re init datalayout to nhwc
        TensorInfo ClTensorInfo_o = TensorInfo(TensorShape(dim_o[1], dim_o[3], dim_o[2], dim_o[0]), 1, data_type_);
        ClTensorInfo_o.set_data_layout(DataLayout::NHWC);
        otensor->allocator()->init(ClTensorInfo_o);
    }
    else
    {
        // keep  the same datalayout
        assert(TENGINE_LAYOUT_NCHW == TengineDataLayOut);
        TensorInfo ClTensorInfo_o = TensorInfo(TensorShape(dim_o[3], dim_o[2], dim_o[1], dim_o[0]), 1, data_type_);
        ClTensorInfo_o.set_data_layout(DataLayout::NCHW);
        otensor->allocator()->init(ClTensorInfo_o);
    }
    tensors_map_[name] = otensor;

    std::shared_ptr<CLActivationLayer> relu = std::make_shared<CLActivationLayer>();
    if (slop_param == 0)
    {
        relu->configure(itensor, otensor, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
    }
    else
    {
        relu->configure(itensor, otensor, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LEAKY_RELU,slop_param));
    }

    functions_map_.push_back(relu);
    return true;
}
