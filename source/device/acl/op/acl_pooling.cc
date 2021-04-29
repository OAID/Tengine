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
#include "pooling_param.h"
}

bool CLGraph::AddPoolingLayer(struct node* node)
{
    TLOG_INFO("Tengine ACL: Support OP(%d) OP_POOl.\n", node->index);
    struct graph* graph = node->graph;
    struct pool_param* param = ( struct pool_param* )node->op.param_mem;
    int pad_x = param->pad_w0;
    int pad_y = param->pad_h0;
    int stride_x = param->stride_w;
    int stride_y = param->stride_h;
    int kernel_w = param->kernel_w;
    int kernel_h = param->kernel_h;
    int type = param->pool_method;
    int global = param->global;

    struct tensor* input_tensor = get_ir_graph_tensor(graph, node->input_tensors[0]);
    int channel = input_tensor->dims[1];
    std::string name = input_tensor->name;
    CLTensor* itensor = nullptr;
    if (tensors_map_.count(name))
    {
        itensor = tensors_map_[name];
        if (bForcedNHWCMode_ == true)    //
        {
            TensorInfo* pClTensorInfo = itensor->info();
            if (pClTensorInfo->data_layout() == DataLayout::NCHW)
            {
                int* dim = input_tensor->dims;
                assert(input_tensor->dim_num == 4);

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
        return false;
    }

    /* output */
    struct tensor* o_tensor = get_ir_graph_tensor(graph, node->output_tensors[0]);

    int TengineDataLayOut = o_tensor->layout;
    TensorInfo* info = itensor->info();

    int out_h = o_tensor->dims[2];
    int out_w = o_tensor->dims[3];

    name = o_tensor->name;
    int* dim_o = o_tensor->dims;
    CLTensor* otensor = new CLTensor();
    DataLayout data_layout;

    if (bForcedNHWCMode_ == true && TengineDataLayOut == TENGINE_LAYOUT_NCHW)
    {
        // need to re init datalayout to nhwc
        TensorInfo ClTensorInfo_o = TensorInfo(TensorShape(channel, out_w, out_h, 1), 1, data_type_);
        ClTensorInfo_o.set_data_layout(DataLayout::NHWC);
        otensor->allocator()->init(ClTensorInfo_o);
        data_layout = DataLayout::NHWC;
    }
    else
    {
        // keep  the same datalayout
        assert(TENGINE_LAYOUT_NCHW == TengineDataLayOut);
        // dim_o[3], dim_o[2], dim_o[1], dim_o[0]
        TensorInfo ClTensorInfo_o = TensorInfo(TensorShape(out_w, out_h, channel, 1), 1, data_type_);
        ClTensorInfo_o.set_data_layout(DataLayout::NCHW);
        otensor->allocator()->init(ClTensorInfo_o);
        data_layout = DataLayout::NCHW;
    }

    tensors_map_[name] = otensor;

    std::shared_ptr<CLPoolingLayer> pooling = std::make_shared<CLPoolingLayer>();
    PoolingLayerInfo pooling_info;

    if (global)
        pooling_info = PoolingLayerInfo(type ? PoolingType::AVG : PoolingType::MAX, data_layout);
    else
        pooling_info =
            PoolingLayerInfo(type ? PoolingType::AVG : PoolingType::MAX, Size2D(kernel_w, kernel_h), data_layout,
                             PadStrideInfo(stride_x, stride_y, pad_x, pad_y, DimensionRoundingType::FLOOR));

    pooling->configure(itensor, otensor, pooling_info);

    functions_map_.push_back(pooling);

    return true;
}
