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

#include "cuda_executor.hpp"


CUDAEngine::CUDAEngine()
{

}

void CUDAEngine::CUDADataMalloc(struct ir_graph* ir_graph, int ir_tensor_idx)
{
    auto iter = this->gpu_addr_map.find(ir_tensor_idx);
    if (this->gpu_addr_map.end() == iter)
    {
        struct ir_tensor* ir_tensor = get_ir_graph_tensor(ir_graph, ir_tensor_idx);
        void* gpu_data = nullptr;
        if(cudaSuccess == cudaMalloc((void **)&gpu_data, ir_tensor->elem_num * ir_tensor->elem_size))
        {
            TLOG_INFO(" cuda malloc tensor(%d) name %s size %d addr %p\n",
                    ir_tensor->idx, ir_tensor->name, ir_tensor->elem_num * ir_tensor->elem_size, gpu_data);
        }
        if ( TENSOR_TYPE_CONST == ir_tensor->tensor_type || TENSOR_TYPE_DEP == ir_tensor->tensor_type )
        {
            TLOG_INFO(" cuda copy tensor(%d) name %s addr %p\n",ir_tensor->idx, ir_tensor->name, gpu_data);
            cudaMemcpy(gpu_data, ir_tensor->data, ir_tensor->elem_num * ir_tensor->elem_size, cudaMemcpyHostToDevice);
        }
        this->gpu_addr_map[ir_tensor_idx] = gpu_data;
    }
}

void CUDAEngine::DataUpload(struct ir_graph* ir_graph, int ir_tensor_idx)
{
    struct ir_tensor* ir_tensor = get_ir_graph_tensor(ir_graph, ir_tensor_idx);
    cudaMemcpy(this->gpu_addr_map[ir_tensor_idx], ir_tensor->data, ir_tensor->elem_num * ir_tensor->elem_size, cudaMemcpyHostToDevice);
}

void CUDAEngine::DataDownload(struct ir_graph* ir_graph, int ir_tensor_idx)
{
    struct ir_tensor* ir_tensor = get_ir_graph_tensor(ir_graph, ir_tensor_idx);
    cudaMemcpy(ir_tensor->data, this->gpu_addr_map[ir_tensor_idx], ir_tensor->elem_num * ir_tensor->elem_size, cudaMemcpyDeviceToHost);
}

int CUDAEngine::Build(struct subgraph* subgraph)
{
    struct ir_graph* ir_graph = subgraph->graph;

//    dump_graph(ir_graph);

    for (int i = 0; i < subgraph->node_num; i++)
    {
        uint16_t node_id = subgraph->node_list[i];
        struct ir_node* ir_node = get_ir_graph_node(ir_graph, node_id);
        auto op_type = ir_node->op.op_type;

        switch (op_type)
        {
            case OP_CLIP:
                this->AddClipNode(ir_graph, ir_node);
                break;
            case OP_CONCAT:
                this->AddConcatNode(ir_graph, ir_node);
                break;
            case OP_CONST:
                break;                
            case OP_CONV:
                this->AddConvolutionNode(ir_graph, ir_node);
                break;
            case OP_DROPOUT:
                this->AddDropoutNode(ir_graph, ir_node);
                break;
            case OP_ELTWISE:
                this->AddEltwiseNode(ir_graph, ir_node);
                break;
            case OP_INPUT:
                break;
            case OP_FC:
                this->AddFullyConnectionNode(ir_graph, ir_node);
                break;
            case OP_FLATTEN:
                this->AddFlattenNode(ir_graph, ir_node);
                break;
            case OP_PERMUTE:
                this->AddPermuteNode(ir_graph, ir_node);
                break;
            case OP_POOL:
                this->AddPoolingNode(ir_graph, ir_node);
                break;
            case OP_RELU:
                this->AddReluNode(ir_graph, ir_node);
                break;
            case OP_RESHAPE:
                this->AddReshapeNode(ir_graph, ir_node);
                break;
            case OP_SLICE:
                this->AddSliceNode(ir_graph, ir_node);
                break;
            case OP_SOFTMAX:
                this->AddSoftmaxNode(ir_graph, ir_node);
            default:
                TLOG_INFO("Tengine GPU: Cannot support OP(%d).\n", ir_node->idx);
                break;
        }
    }
    return 0;
}

int CUDAEngine::CUDAEnginePreRun(struct subgraph* subgraph)
{
    const auto cuda_status = cudaSetDevice(DEFAULT_DEVICE_ID);
    if (cuda_status != cudaSuccess)
    {
        fprintf(stderr, "Tengine: Cannot lock to socket %d.\n", DEFAULT_DEVICE_ID);
        return -1;
    }

    struct ir_graph* ir_graph = subgraph->graph;

    for (int i = 0; i < subgraph->node_num; i++)
    {
        uint16_t node_id = subgraph->node_list[i];
        struct ir_node* ir_node = get_ir_graph_node(ir_graph, node_id);
        for (int j = 0; j < ir_node->input_num; j++)
        {
            int ir_tensor_idx = ir_node->input_tensors[j];
            this->CUDADataMalloc(ir_graph, ir_tensor_idx);
        }
        for (int j = 0; j < ir_node->output_num; j++)
        {
            int ir_tensor_idx = ir_node->output_tensors[j];
            this->CUDADataMalloc(ir_graph, ir_tensor_idx);
        }
    }

    int val;
    cudaDeviceGetAttribute(&val, cudaDevAttrMaxThreadsPerBlock, 0);
    TLOG_INFO("cudaDevAttrMaxThreadsPerBlock %d\n",val);
    this->Build(subgraph);

    //handle
    cudnnCreate(&this->cudnn_handle);
    cublasCreate(&this->cublas_handle);

    for (int i = 0; i < subgraph->output_num; i++)
    {
        int ir_tensor_idx = subgraph->output_tensor_list[i];
        struct ir_tensor* graph_out_tensor = get_ir_graph_tensor(ir_graph, ir_tensor_idx);
        graph_out_tensor->data = (void*)malloc(graph_out_tensor->elem_num * graph_out_tensor->elem_size);
    }

    return 0;
};

int CUDAEngine::CUDAEngineRun(struct subgraph* subgraph)
{
    struct ir_graph* ir_graph = subgraph->graph;

    /* upload data */
    for (uint8_t i = 0; i < subgraph->input_num; i++)
    {
        int ir_tensor_idx = subgraph->input_tensor_list[i];
        this->DataUpload(ir_graph, ir_tensor_idx);
    }

    /* run */
    for (auto& func : this->ops)
    {
        func();
    }

    /* download data */
    for (uint8_t i = 0; i < subgraph->output_num; i++)
    {
        int ir_tensor_idx = subgraph->output_tensor_list[i];
        this->DataDownload(ir_graph, ir_tensor_idx);
    }

#ifdef DEBUG_DATA
    for (auto iter = this->gpu_addr_map.begin(); iter != this->gpu_addr_map.end(); iter++)
    {
        struct ir_tensor* ir_tensor = get_ir_graph_tensor(ir_graph, iter->first);
        cudaMemcpy(ir_tensor->data, iter->second, ir_tensor->elem_num * ir_tensor->elem_size, cudaMemcpyDeviceToHost);
    }
#endif

    return 0;
}

void CUDAEngine::CUDAEnginePostRun()
{
    for (auto iter = this->gpu_addr_map.begin(); iter != this->gpu_addr_map.end(); iter++)
    {
        cudaFree(iter->second);
    }
};

int cuda_dev_init(struct nn_device* dev)
{
    (void)dev;
    return 0;
}

int cuda_dev_prerun(struct nn_device* dev, struct subgraph* subgraph, int num_thread, int cpu_affinity, int mode)
{
    subgraph->exec_graph = new CUDAEngine;
    auto engine = (CUDAEngine*)subgraph->exec_graph;

    return engine->CUDAEnginePreRun(subgraph);
}

int cuda_dev_run(struct nn_device* dev, struct subgraph* subgraph)
{
    auto engine = (CUDAEngine*)subgraph->exec_graph;
    return engine->CUDAEngineRun(subgraph);
}

int cuda_dev_postrun(struct nn_device* dev, struct subgraph* subgraph)
{
    auto engine = (CUDAEngine*)subgraph->exec_graph;
    engine->CUDAEnginePostRun();
    delete engine;

    return 0;
}

int cuda_dev_release(struct nn_device* dev)
{
    (void)dev;
    return 0;
}
