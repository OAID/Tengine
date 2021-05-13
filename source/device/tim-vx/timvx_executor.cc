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


void dump_sub_graph(struct subgraph* sub_graph)
{
    TLOG_INFO("Sub graph[%d]: {%8s } has %d nodes, %d input tensors, %d output tensors.\n", sub_graph->index, sub_graph->device->name, sub_graph->node_num, sub_graph->input_num, sub_graph->output_num);
    TLOG_INFO("\tSub nodes: [ ");

    for (int j = 0; j < sub_graph->node_num - 1; j++)
    {
        int node_id = sub_graph->node_list[j];
        TLOG_INFO("%d, ", node_id);
    }
    TLOG_INFO("%d ].\n", sub_graph->node_list[sub_graph->node_num - 1]);

    TLOG_INFO("\tSub input tensors: [ ");
    for (int j = 0; j < sub_graph->input_num - 1; j++)
    {
        int tensor_id = sub_graph->input_tensor_list[j];
        TLOG_INFO("%d, ", tensor_id);
    }
    TLOG_INFO("%d ].\n", sub_graph->input_tensor_list[sub_graph->input_num - 1]);

    TLOG_INFO("\tSub output tensors: [ ");
    for (int j = 0; j < sub_graph->output_num - 1; j++)
    {
        int tensor_id = sub_graph->output_tensor_list[j];
        TLOG_INFO("%d, ", tensor_id);
    }
    TLOG_INFO("%d ].\n", sub_graph->output_tensor_list[sub_graph->output_num - 1]);
}

VXEngine::VXEngine()
{
    this->context = tim::vx::Context::Create();
    this->graph = context->CreateGraph();
};


void VXEngine::VXTensorMap(struct graph* ir_graph, int ir_tensor_idx, int spec_type)
{
    auto iter = this->vx_tensor_map.find(ir_tensor_idx);

    if (this->vx_tensor_map.end() == iter)
    {
        struct tensor* ir_tensor = get_ir_graph_tensor(ir_graph, ir_tensor_idx);
        auto Dims = (unsigned int*)ir_tensor->dims;

        tim::vx::DataType datatype;
        switch(ir_tensor->data_type)
        {
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
                TLOG_ERR("Tensor: Tensor_name(%s) tensor_index(%d) tensor_data_type(%d) .\n",ir_tensor->name, ir_tensor->index, ir_tensor->data_type);
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

        if (spec_type == SPEC_TYPE_OUTPUT)
        {
            tim::vx::TensorSpec vx_spec(datatype, vx_shape,
                                        tim::vx::TensorAttribute::OUTPUT, vx_quant);
            vx_tensor = this->graph->CreateTensor(vx_spec);
        }
        else if (spec_type == SPEC_TYPE_DWCONV)
        {
            vx_shape[ir_tensor->dim_num - 2] = vx_shape[ir_tensor->dim_num - 1];
            vx_shape[ir_tensor->dim_num - 1] = 1;
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
            tim::vx::TensorSpec vx_spec(datatype, vx_shape,
                                        tim::vx::TensorAttribute::TRANSIENT, vx_quant);
            vx_tensor = this->graph->CreateTensor(vx_spec);
        }
        else if (ir_tensor->tensor_type == TENSOR_TYPE_CONST)
        {
            tim::vx::TensorSpec vx_spec(datatype, vx_shape,
                                        tim::vx::TensorAttribute::CONSTANT, vx_quant);
            vx_tensor = this->graph->CreateTensor(vx_spec, ir_tensor->data);
        }
        this->vx_tensor_map[ir_tensor_idx] = vx_tensor;
    }
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
            if (conv_param->group == conv_param->output_channel)
            {
                this->VXTensorMap(ir_graph, ir_node->input_tensors[1], SPEC_TYPE_DWCONV);
            }       
        }
        else if (ir_node->op.type == OP_PRELU)
        {
            this->VXTensorMap(ir_graph, ir_node->input_tensors[1], SPEC_TYPE_PRELU);
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

    // fprintf(stderr,"subgraph->node_num %d\n",subgraph->node_num);
    if (subgraph->node_num > 0)
    {
        if (!this->graph->Compile()) {
            return -1;
        }
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
                TLOG_INFO("Tengine: Copy output data from VX tensor to CPU failed.\n");
                return -1;
            }
        }
    }

    return 0;
}

void VXEngine::VXEnginePostRun()
{

};
