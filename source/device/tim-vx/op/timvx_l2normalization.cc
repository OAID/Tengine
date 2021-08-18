/*
 * Copyright (c) 2021, Open AI Lab
 * Author: fhfang@openailab.com
 */
#include "timvx_executor.hpp"

extern "C"
{
#include "operator/op.h"
}

bool VXEngine::AddL2normalizationNode(struct node* ir_node)
{
    struct graph* ir_graph = ir_node->graph;

    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    auto l2normalization = graph->CreateOperation<tim::vx::ops::L2Normalization>(1);
    (*l2normalization)
        .BindInputs({ this->vx_tensor_map[input_tensor->index] })
        .BindOutputs({ this->vx_tensor_map[output_tensor->index] });

    return true;
}