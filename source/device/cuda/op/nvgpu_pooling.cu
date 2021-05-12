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
 * Author: hhchen@openailab.com
 */


#include "cuda_executor.hpp"

extern "C"
{
#include "pooling_param.h"

#include "graph/tensor.h"
#include "operator/op.h"
#include "utility/log.h"
}

void pooling_gpu_kernel(cudnnHandle_t& handle, struct graph* ir_graph, struct node* ir_node, dict_uint2voidx  gpu_addr_map)
{
    struct tensor* pool_input_data = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct tensor* pool_output_data = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    struct pool_param* pool_param = ( struct pool_param* )ir_node->op.param_mem;

    // input
    cudnnTensorDescriptor_t input_descriptor;
    cudnnCreateTensorDescriptor(&input_descriptor);
    auto ret_pool_in = cudnnSetTensor4dDescriptor(input_descriptor,
                                                  CUDNN_TENSOR_NCHW,
                                                  CUDNN_DATA_FLOAT,
                                                  pool_input_data->dims[0], pool_input_data->dims[1], pool_input_data->dims[2], pool_input_data->dims[3]);

    // output
    cudnnTensorDescriptor_t output_descriptor;
    cudnnCreateTensorDescriptor(&output_descriptor);
    auto ret_pool_out = cudnnSetTensor4dDescriptor(output_descriptor,
                                                   CUDNN_TENSOR_NCHW,
                                                   CUDNN_DATA_FLOAT,
                                                   pool_output_data->dims[0], pool_output_data->dims[1], pool_output_data->dims[2], pool_output_data->dims[3]);

    // pooling
    cudnnPoolingMode_t poolmode;
    switch (pool_param->pool_method)
    {
        case 0:
            poolmode = CUDNN_POOLING_MAX;
            break;
        case 1:
            poolmode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
            break;
        default:
            fprintf(stderr,"don't support this method pooling\n");
    }
    cudnnPoolingDescriptor_t pool_descriptor;
    cudnnCreatePoolingDescriptor(&pool_descriptor);
    auto ret_pool_des = cudnnSetPooling2dDescriptor(pool_descriptor,
                                                    poolmode,
                                                    CUDNN_NOT_PROPAGATE_NAN,
                                                    pool_param->kernel_h,
                                                    pool_param->kernel_w,
                                                    pool_param->pad_h0,
                                                    pool_param->pad_w0,
                                                    pool_param->stride_h,
                                                    pool_param->stride_w
    );

    /* pooling forward run */
    auto alpha = 1.0f, beta = 0.0f;
    auto ret = cudnnPoolingForward(handle, pool_descriptor,
                                   &alpha, input_descriptor, gpu_addr_map[pool_input_data->index],
                                   &beta, output_descriptor, gpu_addr_map[pool_output_data->index]
    );
    if (CUDNN_STATUS_SUCCESS != ret)
    {
        fprintf(stderr,"GPU: Fail to forward pooling!\n");
        switch(ret)
        {
            case CUDNN_STATUS_BAD_PARAM:
                fprintf(stderr,"CUDNN_STATUS_BAD_PARAM!\n");
                break;
            case CUDNN_STATUS_NOT_SUPPORTED:
                fprintf(stderr,"CUDNN_STATUS_NOT_SUPPORTED!\n");
                break;
            case CUDNN_STATUS_EXECUTION_FAILED:
                fprintf(stderr,"CUDNN_STATUS_EXECUTION_FAILED!\n");
                break;
            default:
                break;
        }
    }
}

void CUDAEngine::AddPoolingNode(struct graph* ir_graph, struct node* ir_node)
{
    TLOG_INFO("Tengine GPU: Support OP(%d) OP_POOL.\n", ir_node->index);
    cudnnCreate(&this->cudnn_handle);
    pooling_gpu_kernel(this->cudnn_handle, ir_graph, ir_node, this->gpu_addr_map);
    this->ops.push_back(std::bind(&pooling_gpu_kernel, this->cudnn_handle, ir_graph, ir_node, this->gpu_addr_map));
}
