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

#include "timvx_executor.hpp"
#include "timvx_define.h"

#ifdef TIMVX_MODEL_CACHE
#include "defines.h"
#include "cstdlib"
#endif

#ifdef TIMVX_MODEL_CACHE
#include "tim/vx/ops/nbg.h"
#include <fstream>
#endif


VXEngine::VXEngine()
{
    this->context = tim::vx::Context::Create();
    this->graph = context->CreateGraph();
};


int VXEngine::VXTensorMap(struct graph* ir_graph, int ir_tensor_idx, int spec_type)
{
    auto iter = this->vx_tensor_map.find(ir_tensor_idx);

    if (this->vx_tensor_map.end() == iter)
    {
        if (spec_type == SPEC_TYPE_INTERP)
        {
            this->vx_tensor_map[ir_tensor_idx] = nullptr;
            return 0;
        }
        struct tensor* ir_tensor = get_ir_graph_tensor(ir_graph, ir_tensor_idx);
        auto Dims = (unsigned int*)ir_tensor->dims;

        tim::vx::DataType datatype;
        switch(ir_tensor->data_type)
        {
            case (0):
                datatype = tim::vx::DataType::FLOAT32;
                break;
            case (1):
                datatype = tim::vx::DataType::FLOAT16;
                break;
            case (3):
                datatype = tim::vx::DataType::UINT8;
                break;
            case (4):
                datatype = tim::vx::DataType::INT32;
                break;
            default:
                TLOG_ERR("Tensor date type: Tensor_name(%s) tensor_index(%d) tensor_data_type(%d) .\n",ir_tensor->name, ir_tensor->index, ir_tensor->data_type);
                break;
        }

        tim::vx::ShapeType vx_shape;

        struct node* ir_node = get_ir_graph_node(ir_graph, ir_tensor->producer);
        if (ir_node->op.type == OP_FC && ir_node->output_tensors[0] == ir_tensor_idx)
        {
            for (int i = 1; i >= 0; i--)
            {
                vx_shape.push_back(Dims[i]);
            }
        }
        else if (spec_type == SPEC_TYPE_PRELU)
        {
            vx_shape.push_back(1);
            vx_shape.push_back(1);
            vx_shape.push_back(Dims[0]);
        }
        else
        {
            for (int i = ir_tensor->dim_num - 1; i >= 0; i--)
            {
                vx_shape.push_back(Dims[i]);
            }
        }

        /* set quant params */
        tim::vx::Quantization vx_quant(tim::vx::QuantType::ASYMMETRIC, ir_tensor->scale,
                                       ir_tensor->zero_point);

        /* create the vx tesnor */
        std::shared_ptr<tim::vx::Tensor> vx_tensor;

        TLOG_INFO("tensor name %s\n",ir_tensor->name);

        if (spec_type == SPEC_TYPE_OUTPUT)
        {
            tim::vx::TensorSpec vx_spec(datatype, vx_shape,
                                        tim::vx::TensorAttribute::OUTPUT, vx_quant);
            vx_tensor = this->graph->CreateTensor(vx_spec);
        }
        else if (ir_tensor->data_type == TENGINE_DT_FP32)
        {
            tim::vx::Quantization none_quant(tim::vx::QuantType::NONE, 1, 0);
            tim::vx::TensorSpec vx_spec(datatype, vx_shape,
                                        tim::vx::TensorAttribute::CONSTANT, none_quant);
            vx_tensor = this->graph->CreateTensor(vx_spec, ir_tensor->data);
        }
        else if (spec_type == SPEC_TYPE_DWCONV)
        {
            auto tmpvx = vx_shape[ir_tensor->dim_num - 2];
            vx_shape[ir_tensor->dim_num - 2] = vx_shape[ir_tensor->dim_num - 1];
            vx_shape[ir_tensor->dim_num - 1] = tmpvx;
            tim::vx::TensorSpec vx_spec(datatype, vx_shape,
                                        tim::vx::TensorAttribute::CONSTANT, vx_quant);
            vx_tensor = this->graph->CreateTensor(vx_spec, ir_tensor->data);
        }
        else if (spec_type == SPEC_TYPE_PRELU)
        {
            tim::vx::TensorSpec vx_spec(datatype, vx_shape,
                                        tim::vx::TensorAttribute::CONSTANT);
            vx_tensor = this->graph->CreateTensor(vx_spec, ir_tensor->data);
        }
        else if (ir_tensor->tensor_type == TENSOR_TYPE_INPUT )
        {
            tim::vx::TensorSpec vx_spec(datatype, vx_shape,
                                        tim::vx::TensorAttribute::INPUT, vx_quant);
            vx_tensor = this->graph->CreateTensor(vx_spec);
        }
        else if (ir_tensor->tensor_type == TENSOR_TYPE_VAR)
        {
            const char* env = getenv(TENGINE_DUMP_LAYER);
            if (env && env[0] == '1')
            {
                tim::vx::TensorSpec vx_spec(datatype, vx_shape,
                                            tim::vx::TensorAttribute::OUTPUT, vx_quant);
                vx_tensor = this->graph->CreateTensor(vx_spec);
            }
            else
            {
                tim::vx::TensorSpec vx_spec(datatype, vx_shape,
                                            tim::vx::TensorAttribute::TRANSIENT, vx_quant);
                vx_tensor = this->graph->CreateTensor(vx_spec);
            }
        }
        else if (ir_tensor->tensor_type == TENSOR_TYPE_CONST)
        {
            TLOG_INFO(" vx_shape %d %d %d %d\n", vx_shape[0], vx_shape[1], vx_shape[2], vx_shape[3]);
            tim::vx::TensorSpec vx_spec(datatype, vx_shape,
                                        tim::vx::TensorAttribute::CONSTANT, vx_quant);
            vx_tensor = this->graph->CreateTensor(vx_spec, ir_tensor->data);
        }
        this->vx_tensor_map[ir_tensor_idx] = vx_tensor;
    }

    return 0;
}

int VXEngine::Build(struct subgraph* subgraph)
{
//    dump_sub_graph(subgraph);
    struct graph* ir_graph = subgraph->graph;

    for (int i = 0; i < subgraph->node_num; i++)
    {
        uint16_t node_id = subgraph->node_list[i];
        struct node* ir_node = get_ir_graph_node(ir_graph, node_id);
        auto op_type = ir_node->op.type;

        switch (op_type)
        {
            case OP_BATCHNORM:
                this->AddBatchNormNode(ir_node);
                break;
            case OP_CLIP:
                this->AddClipNode(ir_node);
                break;
            case OP_CONCAT:
                this->AddConcatNode(ir_node);
                break;
            case OP_CONST:
            case OP_INPUT:
                continue;
            case OP_CONV:
                this->AddConvolutionNode(ir_node);
                break;
            case OP_DEPTHTOSPACE:
                this->AddDepthToSpaceNode(ir_node);
                break;
            case OP_DROPOUT:
                this->AddDropoutNode(ir_node);
                break;
            case OP_ELTWISE:
                this->AddEltwiseNode(ir_node);
                break;
            case OP_ELU:
                this->AddEluNode(ir_node);
                break;
            case OP_FC:
                this->AddFullyConnectionNode(ir_node);
                break;
            case OP_FLATTEN:
                this->AddFlattenNode(ir_node);
                break;
            case OP_GATHER:
                this->AddGatherNode(ir_node);
                break;
            case OP_HARDSWISH:
                this->AddHardSwishNode(ir_node);
                break;
            case OP_INTERP:
                this->AddInterpNode(ir_node);
                break;
            case OP_MISH:
                this->AddMishNode(ir_node);
                break;
            case OP_PERMUTE:
                this->AddPermuteNode(ir_node);
                break;
            case OP_POOL:
                this->AddPoolingNode(ir_node);
                break;
            case OP_PRELU:
                this->AddPReluNode(ir_node);
                break;
            case OP_RELU:
                this->AddReluNode(ir_node);
                break;
            case OP_RELU1:
                this->AddRelu1Node(ir_node);
                break;
            case OP_RESHAPE:
                this->AddReshapeNode(ir_node);
                break;
            case OP_RESIZE:
                this->AddResizeNode(ir_node);
                break;
            case OP_SCALE:
                this->AddScaleNode(ir_node);
                break;
            case OP_SIGMOID:
                this->AddSigmoidNode(ir_node);
                break;
            case OP_SLICE:
                this->AddSliceNode(ir_node);
                break;
            case OP_SOFTMAX:
                this->AddSoftmaxNode(ir_node);
                break;
            case OP_SPACETODEPTH:
                this->AddSpaceToDepthNode(ir_node);
                break;
            case OP_TANH:
                this->AddTanhNode(ir_node);
                break;
            case OP_TRANSPOSE:
                this->AddTransposeNode(ir_node);
                break;
            case OP_UPSAMPLE:
                this->AddUpsampleNode(ir_node);
                break;
            default:
                fprintf(stderr, "Tengine TIM-VX: Cannot support OP(%d).\n", ir_node->index);
                break;
        }
    }

    return 0;
}


int VXEngine::VXEnginePreRun(struct subgraph* subgraph)
{
    struct graph* ir_graph = subgraph->graph;

#ifdef TIMVX_MODEL_CACHE
    auto graph_node_count = subgraph->graph->node_num;
    auto graph_tensor_count = subgraph->graph->tensor_num;

    auto subgraph_node_count = subgraph->node_num;
    auto subgraph_tensor_count = 0;

    auto subgraph_input_count = subgraph->input_num;
    auto subgraph_output_count = subgraph->output_num;

    for (int i = 0; i < subgraph->node_num; i++)
    {
        uint16_t node_id = subgraph->node_list[i];
        struct node* ir_node = get_ir_graph_node(ir_graph, node_id);
        subgraph_tensor_count += ir_node->input_num;
        subgraph_tensor_count += ir_node->output_num;
    }

    std::string graph_name_field = std::to_string(graph_node_count) + "_" + std::to_string(graph_tensor_count);
    std::string subgraph_name_field = std::to_string(subgraph_node_count) + "_"
                                    + std::to_string(subgraph_input_count) + "_"
                                    + std::to_string(subgraph_output_count);

    std::string cache_file_name = "tm_" + graph_name_field + "_" + subgraph_name_field + ".tmcache";
    std::string full_cache_file_path;

    const char *env_cache_path = getenv(TE_MODEL_CACHE_PATH);
    if (nullptr != env_cache_path)
    {
        full_cache_file_path = std::string(env_cache_path) + "/" + full_cache_file_path;
    }
    else
    {
        full_cache_file_path = "./" + cache_file_name;
    }

    TLOG_INFO("Tengine: Model cache file for compiled is %s.", full_cache_file_path.c_str());

    bool cache_saved = false;
    std::ifstream read_stream;
    read_stream.open(full_cache_file_path, std::ios::in | std::ios::binary);
    if (read_stream.is_open())
    {
        cache_saved = true;
    }

    if (cache_saved)
    {
        /* Add TIM-VX Tensor */
        for (uint8_t i = 0; i < subgraph->output_num; i++)
        {
            int ir_tensor_idx = subgraph->output_tensor_list[i];
            this->VXTensorMap(ir_graph, ir_tensor_idx, SPEC_TYPE_OUTPUT);
        }

        /* Add TIM-VX Tensor */
        for (uint8_t i = 0; i < subgraph->input_num; i++)
        {
            int ir_tensor_idx = subgraph->input_tensor_list[i];
            this->VXTensorMap(ir_graph, ir_tensor_idx, 0);
        }

        read_stream.seekg(0, std::ifstream::beg);
        const auto start_length = read_stream.tellg();
        read_stream.seekg(0, std::ifstream::end);
        const auto end_length = read_stream.tellg();

        read_stream.seekg(0, std::ifstream::beg);

        auto file_size = end_length - start_length;

        nbg_buffer.reserve(file_size);
        nbg_buffer.insert(nbg_buffer.begin(), std::istreambuf_iterator<char>(read_stream), std::istreambuf_iterator<char>());
        read_stream.close();

        auto nbg_node = this->graph->CreateOperation<tim::vx::ops::NBG>(nbg_buffer.data(), subgraph_input_count, subgraph_output_count);

        std::vector<std::shared_ptr<tim::vx::Tensor>> inputs, outputs;
        for (uint8_t i = 0; i < subgraph->input_num; i++)
        {
            int ir_tensor_idx = subgraph->input_tensor_list[i];
            auto iter = this->vx_tensor_map[ir_tensor_idx];
            inputs.push_back(iter);
        }
        for (uint8_t i = 0; i < subgraph->output_num; i++)
        {
            int ir_tensor_idx = subgraph->output_tensor_list[i];
            auto iter = this->vx_tensor_map[ir_tensor_idx];
            outputs.push_back(iter);
        }
        (*nbg_node).BindInputs(inputs);
        (*nbg_node).BindOutputs(outputs);

        auto ret = this->graph->Compile();
        if (!ret)
        {
            TLOG_ERR("Tengine: Model compile from bin failed.");
            return -1;
        }
    }
    else
#endif
    {
        /* Add TIM-VX Tensor */
        for (uint8_t i = 0; i < subgraph->output_num; i++)
        {
            int ir_tensor_idx = subgraph->output_tensor_list[i];
            this->VXTensorMap(ir_graph, ir_tensor_idx, SPEC_TYPE_OUTPUT);
        }
        for (int i = 0; i < subgraph->node_num; i++)
        {
            uint16_t node_id = subgraph->node_list[i];
            struct node* ir_node = get_ir_graph_node(ir_graph, node_id);
            if (ir_node->op.type == OP_CONV)
            {
                auto conv_param = (struct conv_param*)ir_node->op.param_mem;
                if ((conv_param->group == conv_param->output_channel) && (conv_param->output_channel != 1))
                {
                    this->VXTensorMap(ir_graph, ir_node->input_tensors[1], SPEC_TYPE_DWCONV);
                }
            }
            else if (ir_node->op.type == OP_PRELU)
            {
                this->VXTensorMap(ir_graph, ir_node->input_tensors[1], SPEC_TYPE_PRELU);
            }
            else if (ir_node->op.type == OP_INTERP)
            {
                this->VXTensorMap(ir_graph, ir_node->input_tensors[1], SPEC_TYPE_INTERP);
                this->VXTensorMap(ir_graph, ir_node->input_tensors[2], SPEC_TYPE_INTERP);
            }
        }
        for (int i = 0; i < subgraph->node_num; i++)
        {
            uint16_t node_id = subgraph->node_list[i];
            struct node* ir_node = get_ir_graph_node(ir_graph, node_id);
            for (int j = 0; j < ir_node->input_num; j++)
            {
                int ir_tensor_idx = ir_node->input_tensors[j];
                this->VXTensorMap(ir_graph, ir_tensor_idx, 0);
            }
            for (int j = 0; j < ir_node->output_num; j++)
            {
                int ir_tensor_idx = ir_node->output_tensors[j];
                this->VXTensorMap(ir_graph, ir_tensor_idx, 0);
            }
        }

        /* Add TIM-VX Node */
        this->Build(subgraph);

#ifdef TIMVX_MODEL_CACHE
        size_t bin_size = -1;
        auto ret = graph->CompileToBinary(nullptr, &bin_size);
        if (-1 == bin_size || !ret)
        {
            TLOG_ERR("Tengine: Model compile to bin failed.");
            return -1;
        }

        this->nbg_buffer.resize(bin_size);
        ret = graph->CompileToBinary(nbg_buffer.data(), &bin_size);
        if (!ret)
        {
            TLOG_ERR("Tengine: Model compile to bin failed.");
            return -1;
        }

        std::ofstream nbg_stream;
        nbg_stream.open(full_cache_file_path, std::ios::out | std::ios::binary);
        if (nbg_stream.is_open())
        {
            TLOG_INFO("Tengine: Save compiled model to %s.", full_cache_file_path.c_str());
        }
        nbg_stream.write(this->nbg_buffer.data(), this->nbg_buffer.size());
        nbg_stream.close();
#else
        // fprintf(stderr,"subgraph->node_num %d\n",subgraph->node_num);
        if (subgraph->node_num > 0)
        {
            if (!this->graph->Compile()) {
                return -1;
            }
        }
#endif

    }

    return 0;
};

int VXEngine::VXEngineRun(struct subgraph* subgraph)
{
    struct graph* ir_graph = subgraph->graph;

    /* upload data */
//    fprintf(stderr,"subgraph->input_num %d\n",subgraph->input_num);
    if (subgraph->input_num > 0)
    {
        for (uint8_t i = 0; i < subgraph->input_num; i++)
        {
            int ir_tensor_idx = subgraph->input_tensor_list[i];
            struct tensor* ir_tensor = get_ir_graph_tensor(ir_graph, ir_tensor_idx);
            if (!this->vx_tensor_map[ir_tensor_idx]->CopyDataToTensor(ir_tensor->data, ir_tensor->elem_num * ir_tensor->elem_size)) {
                return -1;
            }
        }

        if (!this->graph->Run())
        {
            return -1;
        }

        /* download data */
        for (uint8_t i = 0; i < subgraph->output_num; i++)
        {
            int ir_tensor_idx = subgraph->output_tensor_list[i];
            struct tensor* ir_tensor = get_ir_graph_tensor(ir_graph, ir_tensor_idx);
            if (nullptr == ir_tensor->data)
            {
                auto u8data = (uint8_t*)malloc(ir_tensor->elem_size * ir_tensor->elem_num);
                ir_tensor->data = u8data;

                ir_tensor->free_host_mem = 1;
                ir_tensor->internal_allocated = 0;
            }

            if (!this->vx_tensor_map[ir_tensor_idx]->CopyDataFromTensor(ir_tensor->data)) 
            {
                TLOG_INFO("TIM-VX: Copy output data from VX tensor to CPU failed.\n");
                return -1;
            }
        }


        const char* env = getenv(TENGINE_DUMP_LAYER);
        if (env && env[0] == '1')
        {
            for (uint8_t i = 0; i < ir_graph->tensor_num; i++)
            {
                if (ir_graph->tensor_list[i]->tensor_type == TENSOR_TYPE_VAR)
                {
                    if (ir_graph->tensor_list[i]->data == nullptr)
                    {
                        TLOG_INFO("TIM-VX: Data pointer is nullptr.\n");
                        uint8_t* u8data = (uint8_t*)malloc(ir_graph->tensor_list[i]->elem_size * ir_graph->tensor_list[i]->elem_num);
                        ir_graph->tensor_list[i]->data = u8data;
                    }
                    if (!this->vx_tensor_map[i]->CopyDataFromTensor(ir_graph->tensor_list[i]->data))
                    {
                        TLOG_INFO("TIM-VX: Copy output data failed.\n");
                        return -1;
                    }
                }
            }

            for (uint8_t i = 0; i < ir_graph->tensor_num; i++)
            {
                TLOG_INFO("TIM-VX: Tensor type %d\n",ir_graph->tensor_list[i]->tensor_type);
                if (ir_graph->tensor_list[i]->tensor_type == TENSOR_TYPE_VAR)
                {
                    char dir_str[32] = { 0 };
                    sprintf(dir_str, "out[%d]", i);

                    if (NULL != ir_graph->tensor_list[i]->data)
                    {
                        extract_feature_from_tensor_timvx(dir_str, ir_graph->tensor_list[i]->name, ir_graph->tensor_list[i]);
                    }
                }
            }
        }// End TENGINE_DUMP_LAYER
    }

    return 0;
}

void VXEngine::VXEnginePostRun()
{

};
