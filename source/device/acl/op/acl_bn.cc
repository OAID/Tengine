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


#include "acl_executor.hpp"

extern "C"
{
#include "operator/op.h"
#include "batchnorm_param.h"
}

bool CLGraph::AddBNLayer(struct node* node, struct node* node_scale)
{
    struct graph* graph = node->graph;
    struct graph* scale_graph = node_scale->graph;
    struct batchnorm_param* param = ( struct batchnorm_param* )node->op.param_mem;
    float eps = param->eps;

    /* input */
    struct tensor* input_tensor = get_ir_graph_tensor(graph, node->input_tensors[0]);
    char* name = input_tensor->name;
    int channel = input_tensor->dims[1];
    CLTensor* itensor = nullptr;
    if (tensors_map_.count(name))
    {
        itensor = tensors_map_[name];
        if (bForcedNHWCMode_ == true)
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
        // TLOG_ERR("can't find node [%s]tensor named :%s\n", node->name, name);
        return false;
    }

    /* gamma */
    struct tensor* gamma_tensor = get_ir_graph_tensor(scale_graph, node_scale->input_tensors[1]);
    CLTensor* gtensor = nullptr;
    if (gamma_tensor)
    {
        name = gamma_tensor->name;
        gtensor = new CLTensor();
        gtensor->allocator()->init(TensorInfo(TensorShape(channel), 1, data_type_));
        tensors_map_[name] = gtensor;
    }
    /* beta */
    struct tensor* beta_tensor = get_ir_graph_tensor(scale_graph, node_scale->input_tensors[2]);
    CLTensor* btensor = nullptr;
    if (beta_tensor)
    {
        name = beta_tensor->name;
        btensor = new CLTensor();
        btensor->allocator()->init(TensorInfo(TensorShape(channel), 1, data_type_));
        tensors_map_[name] = btensor;
    }

    /* means */
    struct tensor* means_tensor = get_ir_graph_tensor(graph, node_scale->input_tensors[3]);
    name = means_tensor->name;
    CLTensor* mtensor = new CLTensor();
    mtensor->allocator()->init(TensorInfo(TensorShape(channel), 1, data_type_));
    tensors_map_[name] = mtensor;

    /* var */
    struct tensor* var_tensor = get_ir_graph_tensor(graph, node_scale->input_tensors[4]);
    CLTensor* vtensor = nullptr;
    if (var_tensor)
    {
        name = var_tensor->name;
        vtensor = new CLTensor();
        vtensor->allocator()->init(TensorInfo(TensorShape(channel), 1, data_type_));
        tensors_map_[name] = vtensor;
    }
    /* output */
    struct tensor* out_tensor = get_ir_graph_tensor(graph, node_scale->output_tensors[0]);
    int* dim_o = out_tensor->dims;
    name = out_tensor->name;
    CLTensor* otensor = new CLTensor();

    int TengineDataLayOut = out_tensor->layout;

    if (bForcedNHWCMode_ == true && TengineDataLayOut == TENGINE_LAYOUT_NCHW)
    {
        // need to re init datalayout to nhwc
        TensorInfo ClTensorInfo_o = TensorInfo(TensorShape(dim_o[1], dim_o[2], dim_o[3], dim_o[0]), 1, data_type_);
        ClTensorInfo_o.set_data_layout(DataLayout::NHWC);
        otensor->allocator()->init(ClTensorInfo_o);
    }
    else
    {
        // keep  the same datalayout
        assert(TENGINE_LAYOUT_NCHW == TengineDataLayOut);
        // dim_o[3], dim_o[2], dim_o[1], dim_o[0]
        TensorInfo ClTensorInfo_o = TensorInfo(TensorShape(dim_o[3], dim_o[2], dim_o[1], dim_o[0]), 1, data_type_);
        ClTensorInfo_o.set_data_layout(DataLayout::NCHW);
        otensor->allocator()->init(ClTensorInfo_o);
    }

    tensors_map_[name] = otensor;

    std::shared_ptr<CLBatchNormalizationLayer> bn = std::make_shared<CLBatchNormalizationLayer>();
    bn->configure(itensor, otensor, mtensor, vtensor, btensor, gtensor, eps);

    functions_map_.push_back(bn);

    mtensor->allocator()->allocate();
    vtensor->allocator()->allocate();
    mtensor->map();
    vtensor->map();
    void* means_data = mtensor->buffer();
    void* vars_data = vtensor->buffer();
    void* means = means_tensor->data;
    void* vars = var_tensor->data;

    copy_buffer(means_data, means, channel * 4, data_type_, DataType::F32);
    copy_buffer(vars_data, vars, channel * 4, data_type_, DataType::F32);

    mtensor->unmap();
    vtensor->unmap();

    if (btensor)
    {
        btensor->allocator()->allocate();
        btensor->map();
        void* beta_data = btensor->buffer();
        void* beta = beta_tensor->data;
        copy_buffer(beta_data, beta, channel * 4, data_type_, DataType::F32);
        btensor->unmap();
    }
    if (gtensor)
    {
        gtensor->allocator()->allocate();
        gtensor->map();
        void* gamma_data = gtensor->buffer();
        void* gamma = gamma_tensor->data;
        copy_buffer(gamma_data, gamma, channel * 4, data_type_, DataType::F32);
        gtensor->unmap();
    }

    return true;
}
