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

#include "torch_executor.hpp"
#include "torch_define.h"


TORCHEngine::TORCHEngine()
{

};

int TORCHEngine::TORCHTensorMap(struct graph* ir_graph, int ir_tensor_idx, int spec_type = 0)
{
    auto iter = this->torch_tensor_map.find(ir_tensor_idx);

    if (this->torch_tensor_map.end() == iter)
    {
        struct tensor* ir_tensor = (struct tensor*)ir_graph->tensor_list[ir_tensor_idx];
        if (ir_tensor->tensor_type == TENSOR_TYPE_VAR || ir_tensor->tensor_type == TENSOR_TYPE_INPUT)
        {
            std::shared_ptr<torch::Tensor> torch_tensor = std::make_shared<torch::Tensor>();
            this->torch_tensor_map[ir_tensor_idx] = torch_tensor;
        }
        else
        {
            //To do
        }
    }

    return 0;
}

void dump_sub_graph_torch(struct subgraph* sub_graph)
{
    fprintf(stderr, "Sub graph[%d]: {%8s } has %d nodes, %d input tensors, %d output tensors.\n", sub_graph->index, sub_graph->device->name, sub_graph->node_num, sub_graph->input_num, sub_graph->output_num);
    fprintf(stderr, "\tSub nodes: [ ");

    for (int j = 0; j < sub_graph->node_num - 1; j++)
    {
        int node_id = sub_graph->node_list[j];
        fprintf(stderr, "%d, ", node_id);
    }
    fprintf(stderr, "%d ].\n", sub_graph->node_list[sub_graph->node_num - 1]);

    fprintf(stderr, "\tSub input tensors: [ ");
    for (int j = 0; j < sub_graph->input_num - 1; j++)
    {
        int tensor_id = sub_graph->input_tensor_list[j];
        fprintf(stderr, "%d, ", tensor_id);
    }
    fprintf(stderr, "%d ].\n", sub_graph->input_tensor_list[sub_graph->input_num - 1]);

    fprintf(stderr, "\tSub output tensors: [ ");
    for (int j = 0; j < sub_graph->output_num - 1; j++)
    {
        int tensor_id = sub_graph->output_tensor_list[j];
        fprintf(stderr, "%d, ", tensor_id);
    }
    fprintf(stderr, "%d ].\n", sub_graph->output_tensor_list[sub_graph->output_num - 1]);
}


int TORCHEngine::TORCHEnginePreRun(struct subgraph* subgraph)
{
    struct graph* ir_graph = subgraph->graph;

    /* Add TORCH Tensor */
    for (uint16_t i = 0; i < subgraph->node_num; i++)
    {
        uint16_t node_id = subgraph->node_list[i];
        struct node* ir_node = get_ir_graph_node(ir_graph, node_id);
        for (uint8_t j = 0; j < ir_node->input_num; j++)
        {
            int ir_tensor_idx = ir_node->input_tensors[j];
            this->TORCHTensorMap(ir_graph, ir_tensor_idx);
        }
        for (int j = 0; j < ir_node->output_num; j++)
        {
            int ir_tensor_idx = ir_node->output_tensors[j];
            this->TORCHTensorMap(ir_graph, ir_tensor_idx);
        }
    }

    /* Add TORCH OP */
    this->net = std::make_shared<Net>(subgraph, torch_tensor_map);

    torch::save(this->net, "net.pt");
//    torch::load(this->net, "net.pt");
    fprintf(stderr,"Torch prerun finish...\n");

    return 0;
};


int TORCHEngine::TORCHEngineRun(struct subgraph* subgraph)
{
    struct graph* ir_graph = subgraph->graph;

    std::vector<std::shared_ptr<torch::Tensor> > torch_input;
    std::vector<std::shared_ptr<torch::Tensor> > torch_out;
    for (uint8_t i = 0; i < subgraph->input_num; i++)
    {
        struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, subgraph->input_tensor_list[i]);
        int total_size = (int)(input_tensor->elem_num * input_tensor->elem_size);

        std::shared_ptr<torch::Tensor> torch_tensor = std::make_shared<torch::Tensor>();
        torch::Tensor t = torch::rand({input_tensor->dims[0], input_tensor->dims[1], input_tensor->dims[2], input_tensor->dims[3]});
        void* date_mem = t.data_ptr();
        memcpy(date_mem, input_tensor->data, total_size);
        *torch_tensor = t;

        torch_input.push_back(torch_tensor);
    }

    torch_out = this->net->forward(torch_input);

    for (uint8_t i = 0; i < subgraph->output_num; i++)
    {
        struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, subgraph->output_tensor_list[i]);
        int total_size = (int)(output_tensor->elem_num * output_tensor->elem_size);
        if (nullptr == output_tensor->data)
        {
            float* fp32data = (float*)malloc(total_size);
            output_tensor->data = fp32data;

            output_tensor->free_host_mem = 1;
            output_tensor->internal_allocated = 0;
        }

        void* date_mem = torch_out[i]->data_ptr();
        memcpy(output_tensor->data, date_mem, total_size);
    }

    return 0;
}

void TORCHEngine::TORCHEnginePostRun()
{

};
