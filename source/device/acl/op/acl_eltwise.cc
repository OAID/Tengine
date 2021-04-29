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
#include "eltwise_param.h"
}

bool CLGraph::AddEltwiseLayer(struct node* node)
{
    TLOG_INFO("Tengine ACl: Support OP(%d) OP_ELTWISE.\n", node->index);
    struct graph* graph = node->graph;
    struct tensor* input_tensor0 = get_ir_graph_tensor(graph, node->input_tensors[0]);
    std::string name = input_tensor0->name;
    CLTensor* itensor0 = nullptr;
    if (tensors_map_.count(name))
    {
        itensor0 = tensors_map_[name];
        if (bForcedNHWCMode_ == true)    //
        {
            TensorInfo* pClTensorInfo = itensor0->info();
            if (pClTensorInfo->data_layout() == DataLayout::NCHW)
            {
                int* dim = input_tensor0->dims;
                assert(input_tensor0->dim_num == 4);

                pClTensorInfo->set_tensor_shape(TensorShape(dim[1], dim[3], dim[2], dim[0]));
                pClTensorInfo->set_data_layout(DataLayout::NHWC);
            }
            else
            {
                assert(pClTensorInfo->data_layout() == DataLayout::NHWC);
            }
        }
    }
    else
    {
        // TLOG_ERR("can't find node [%s]tensor named :%s\n", node->name, name);
        return false;
    }
    struct tensor* input_tensor1 = get_ir_graph_tensor(graph, node->input_tensors[1]);
    name = input_tensor1->name;
    CLTensor* itensor1 = nullptr;
    if (tensors_map_.count(name))
    {
        itensor1 = tensors_map_[name];
        if (bForcedNHWCMode_ == true)    //
        {
            TensorInfo* pClTensorInfo = itensor1->info();
            if (pClTensorInfo->data_layout() == DataLayout::NCHW)
            {
                int* dim = input_tensor1->dims;
                assert(input_tensor1->dim_num == 4);

                pClTensorInfo->set_tensor_shape(TensorShape(dim[1], dim[3], dim[2], dim[0]));
                pClTensorInfo->set_data_layout(DataLayout::NHWC);
            }
            else
            {
                assert(pClTensorInfo->data_layout() == DataLayout::NHWC);
            }
        }
    }
    else
    {
        // TLOG_ERR("can't find node [%s]tensor named :%s\n", node->name, name);
        return false;
    }
    /*output */
    struct tensor* o_tensor = get_ir_graph_tensor(graph, node->output_tensors[0]);
    name = o_tensor->name;
    int* dim_o = o_tensor->dims;
    CLTensor* otensor = new CLTensor();
    int TengineDataLayOut = o_tensor->layout;

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

    struct eltwise_param* param = ( struct eltwise_param* )node->op.param_mem;
    if (ELT_SUM == param->type)
    {
        std::shared_ptr<CLArithmeticAddition> add = std::make_shared<CLArithmeticAddition>();

        add->configure(itensor0, itensor1, otensor, ConvertPolicy::WRAP);
        functions_map_.push_back(add);
    }
    else
    {
        // TLOG_ERR("eltwise only support ADD!~~\n");
        return false;
    }

    return true;
}
