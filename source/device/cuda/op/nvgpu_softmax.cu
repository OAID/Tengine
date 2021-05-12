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
#include "softmax_param.h"

#include "graph/tensor.h"
#include "operator/op.h"
#include "utility/log.h"
}

__global__ void softmax_max(float* k, float* result_block)
{
    __shared__ float sdata[1024];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = k[i];
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s /= 2)
    {
        if (tid < s)
        {
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) result_block[blockIdx.x] = sdata[0];
}

__global__ void softmax_k_upload(float* k, float* x, int N, int elem_perchannel_match, int elem_perchannel)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < N)
    {
        int idx_mul = i / elem_perchannel_match;
        int idx_new = i % elem_perchannel_match;
        if (elem_perchannel > idx_new )
            k[i] = x[idx_mul * elem_perchannel + idx_new];
        else
            k[i] = -9999.9f;
    }
}

__global__ void softmax_k_download(float* k, float* x, int N, int elem_perchannel_match, int elem_perchannel)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < N)
    {
        int idx_mul = idx / elem_perchannel_match;
        int idx_new = idx % elem_perchannel_match;
        if (elem_perchannel > idx_new )
            x[idx_mul * elem_perchannel + idx_new] = k[idx];
    }
}

__global__ void softmax_exp_sum(float* k, float* result_block, int elem_perchannel_match, int elem_perchannel)
{
    __shared__ float sdata[1024];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (elem_perchannel <= (i % elem_perchannel_match) )
        k[i] = 0;
    else
        k[i] = exp(k[i] - result_block[blockIdx.x]);
    __syncthreads();

    int tid = threadIdx.x;
    if (elem_perchannel <= (i % elem_perchannel_match) )
        sdata[tid] = 0;
    else
        sdata[tid] = k[i];
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s /= 2)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
        result_block[blockIdx.x] = sdata[0];
    __syncthreads();
}
__global__ void softmax_exp_div(float* k, float* result_block, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        k[i] = float(k[i] / result_block[blockIdx.x]);
}

void softmax_gpu_kernel(cudnnHandle_t& handle, struct graph* ir_graph, struct node* ir_node, dict_uint2voidx  gpu_addr_map, int ag)
{
    struct tensor* soft_input_data = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct tensor* soft_output_data = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    struct softmax_param* param = ( struct softmax_param* )ir_node->op.param_mem;
    fprintf(stderr,"### softmax axis %d\n",param->axis);

    // input
    cudnnTensorDescriptor_t input_descriptor;
    cudnnCreateTensorDescriptor(&input_descriptor);
    auto ret_pool_in = cudnnSetTensor4dDescriptor(input_descriptor,
                                                  CUDNN_TENSOR_NCHW,
                                                  CUDNN_DATA_FLOAT,
                                                  soft_input_data->dims[0], soft_input_data->dims[1], soft_input_data->dims[2], soft_input_data->dims[3]);

    // output
    cudnnTensorDescriptor_t output_descriptor;
    cudnnCreateTensorDescriptor(&output_descriptor);
    auto ret_pool_out = cudnnSetTensor4dDescriptor(output_descriptor,
                                                   CUDNN_TENSOR_NCHW,
                                                   CUDNN_DATA_FLOAT,
                                                   soft_output_data->dims[0], soft_output_data->dims[1], soft_output_data->dims[2], soft_output_data->dims[3]);

    cudnnSoftmaxAlgorithm_t algo = CUDNN_SOFTMAX_ACCURATE;

    cudnnSoftmaxMode_t mode;
    if (ag == 0)
        mode = CUDNN_SOFTMAX_MODE_INSTANCE;
    if (ag == 1)
        mode = CUDNN_SOFTMAX_MODE_CHANNEL;

    /* pooling forward run */
    auto alpha = 1.0f, beta = 0.0f;
    auto ret = cudnnSoftmaxForward(handle, algo, mode,
                                   &alpha, input_descriptor, gpu_addr_map[soft_input_data->index],
                                   &beta, output_descriptor, gpu_addr_map[soft_output_data->index]
    );
}
void softmax_gpu_kernel_2(struct graph* ir_graph, struct node* ir_node, dict_uint2voidx  gpu_addr_map)
{
//    double th = get_time();

    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    struct softmax_param* param = (struct softmax_param*)ir_node->op.param_mem;

    int channels = 1;
    for (int i = 0; i < param->axis; i++)
    {
        channels *= output_tensor->dims[i];
    }

    int elem_perchannel = output_tensor->elem_num / channels;
    int bs = (elem_perchannel / 32 + 1) * 32;
    if (bs > 1024)
        bs = 1024;
    int elem_perchannel_match = ((elem_perchannel - 1)/ bs + 1) * bs;
    int s = elem_perchannel_match * channels / bs;
    dim3 grid = dim3(s);

    float* k = NULL;
    cudaMalloc((void **)&k, elem_perchannel_match * channels * sizeof(float));
    softmax_k_upload<<<grid, bs>>>(k, (float*)gpu_addr_map[input_tensor->index], elem_perchannel_match * channels, elem_perchannel_match, elem_perchannel);

    float* result_block = NULL;
    cudaMalloc((void **)&result_block, s * sizeof(float));
    softmax_max<<<grid, bs>>>(k, result_block);

    if (s != channels)
    {
        float* result = (float*)malloc(channels * sizeof(float));
        float* result_tmp = (float*)malloc(s * sizeof(float));
        cudaMemcpy(result_tmp, result_block, s * sizeof(float),cudaMemcpyDeviceToHost);
        for (int i = 0; i < channels; i++)
        {
            result[i] = -9999.9f;
            for (int j = 0; j < s / channels; j++)
            {
                result[i] = max(result[i], result_tmp[i * s / channels + j]);
            }
            for (int j = 0; j < s / channels; j++)
            {
                result_tmp[i * s / channels + j] = result[i];
            }
        }
        cudaMemcpy(result_block, result_tmp, s * sizeof(float),cudaMemcpyHostToDevice);

        free(result);
        free(result_tmp);
    }

    softmax_exp_sum<<<grid, bs>>>(k, result_block, elem_perchannel_match, elem_perchannel);
    softmax_exp_div<<<grid, bs>>>(k, result_block, elem_perchannel_match * channels);

    softmax_k_download<<<grid, bs>>>( k, (float*)gpu_addr_map[output_tensor->index], elem_perchannel_match * channels, elem_perchannel_match, elem_perchannel );

    cudaFree(result_block);
    cudaFree(k);
}


void CUDAEngine::AddSoftmaxNode(struct graph* ir_graph, struct node* ir_node)
{
    TLOG_INFO("Tengine GPU: Support OP(%d) OP_SOFTMAX.\n", ir_node->index);
    struct softmax_param* param = ( struct softmax_param* )ir_node->op.param_mem;
    switch(param->axis)
    {
        case 0:
            cudnnCreate(&this->cudnn_handle);
            softmax_gpu_kernel(this->cudnn_handle, ir_graph, ir_node, this->gpu_addr_map, 0);
            this->ops.push_back(std::bind(&softmax_gpu_kernel, this->cudnn_handle, ir_graph, ir_node, this->gpu_addr_map, 0));
            break;
        case 1:
            cudnnCreate(&this->cudnn_handle);
            softmax_gpu_kernel(this->cudnn_handle, ir_graph, ir_node, this->gpu_addr_map, 1);
            this->ops.push_back(std::bind(&softmax_gpu_kernel, this->cudnn_handle, ir_graph, ir_node, this->gpu_addr_map, 1));
            break;
        case 2:
            softmax_gpu_kernel_2(ir_graph, ir_node, this->gpu_addr_map);
            this->ops.push_back(std::bind(&softmax_gpu_kernel_2, ir_graph, ir_node, this->gpu_addr_map));
            break;
        default:
            TLOG_INFO("Tengine GPU: Cannot support SOFTMAX axis(%d).\n", param->axis);
            break;
    }
}
