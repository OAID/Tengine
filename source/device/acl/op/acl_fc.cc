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

bool CLGraph::AddFCLayer(struct node* node)
{
    struct graph* graph = node->graph;
    /* Input */
    struct tensor* input_tensor = get_ir_graph_tensor(graph, node->input_tensors[0]);
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
    if (!itensor)
    {
        // TLOG_ERR("can't find node [%s]tensor named :%s\n", node->name, name);
        return false;
    }
    /* weight */
    struct tensor* w_tensor = get_ir_graph_tensor(graph, node->input_tensors[1]);
    name = w_tensor->name;
    int M = w_tensor->dims[0];
    int K = w_tensor->dims[1];
    CLTensor* wtensor = new CLTensor();
    wtensor->allocator()->init(TensorInfo(TensorShape(K, M), 1, data_type_));
    tensors_map_[name] = wtensor;
    /* bias */
    struct tensor* b_tensor = get_ir_graph_tensor(graph, node->input_tensors[2]);
    CLTensor* btensor = nullptr;

    if (b_tensor)
    {
        name = b_tensor->name;
        btensor = new CLTensor();
        btensor->allocator()->init(TensorInfo(TensorShape(M), 1, data_type_));
        tensors_map_[name] = btensor;
    }

    /*output */
    struct tensor* o_tensor = get_ir_graph_tensor(graph, node->output_tensors[0]);
    name = o_tensor->name;
    int* dim_w = o_tensor->dims;
    CLTensor* otensor = new CLTensor();
    otensor->allocator()->init(TensorInfo(TensorShape(dim_w[1]), 1, data_type_));
    tensors_map_[name] = otensor;

    /* FC Layer */
    bool transpose_w = (dim_w[1] == M) ? true : false;

    std::shared_ptr<CLFullyConnectedLayer> fc = std::make_shared<CLFullyConnectedLayer>();
    FullyConnectedLayerInfo fc_info;
    fc_info.set_transpose_weights(transpose_w);
    fc_info.set_weights_trained_layout(DataLayout::NCHW);    // lay out
    fc->configure(itensor, wtensor, btensor, otensor, fc_info);
    functions_map_.push_back(fc);
    wtensor->allocator()->allocate();
    wtensor->map();
    void* data = w_tensor->data;
    void* acl_data = wtensor->buffer();
    int size = w_tensor->elem_size * w_tensor->elem_num;
    copy_buffer(acl_data, data, size, data_type_, DataType::F32);
    wtensor->unmap();
    if (btensor)
    {
        btensor->allocator()->allocate();
        btensor->map();
        data = b_tensor->data;
        acl_data = btensor->buffer();
        int size = b_tensor->elem_size * b_tensor->elem_num;
        copy_buffer(acl_data, data, size, data_type_, DataType::F32);
        btensor->unmap();
    }
    return true;
}
