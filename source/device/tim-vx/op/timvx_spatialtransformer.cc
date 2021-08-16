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
 * Author: fhfang@openailab.com
 */

#include "timvx_executor.hpp"

extern "C"
{
#include  <cstring>
#include "operator/op.h"
#include "utility/float.h"
#include "spatialtransformer_param.h"
}

bool VXEngine::AddSpatialtransformerNode(struct node* ir_node)
{
    struct graph* ir_graph = ir_node->graph;

    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct tensor* input_tensor1 = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    tim::vx::Quantization tmp_quant(tim::vx::QuantType::ASYMMETRIC,input_tensor1->scale, input_tensor1->zero_point);
    std::vector<std::shared_ptr<tim::vx::Tensor> > split_out_tensor(6);
    tim::vx::ShapeType vx_split_shape, vx_concat_shape, vx_dst_shape;

    int axis = 0;

    vx_split_shape.push_back(1);
    vx_split_shape.push_back(1);
    vx_concat_shape.push_back(6);
    vx_concat_shape.push_back(1);

    tim::vx::TensorSpec tmp_spec_0(tim::vx::DataType::UINT8, vx_split_shape, tim::vx::TensorAttribute::TRANSIENT, tmp_quant);
    split_out_tensor[0] = this->graph->CreateTensor(tmp_spec_0);
    tim::vx::TensorSpec tmp_spec_1(tim::vx::DataType::UINT8, vx_split_shape, tim::vx::TensorAttribute::TRANSIENT, tmp_quant);
    split_out_tensor[1] = this->graph->CreateTensor(tmp_spec_1);
    tim::vx::TensorSpec tmp_spec_2(tim::vx::DataType::UINT8, vx_split_shape, tim::vx::TensorAttribute::TRANSIENT, tmp_quant);
    split_out_tensor[2] = this->graph->CreateTensor(tmp_spec_2);
    tim::vx::TensorSpec tmp_spec_3(tim::vx::DataType::UINT8, vx_split_shape, tim::vx::TensorAttribute::TRANSIENT, tmp_quant);
    split_out_tensor[3] = this->graph->CreateTensor(tmp_spec_3);
    tim::vx::TensorSpec tmp_spec_4(tim::vx::DataType::UINT8, vx_split_shape, tim::vx::TensorAttribute::TRANSIENT, tmp_quant);
    split_out_tensor[4] = this->graph->CreateTensor(tmp_spec_4);
    tim::vx::TensorSpec tmp_spec_5(tim::vx::DataType::UINT8, vx_split_shape, tim::vx::TensorAttribute::TRANSIENT, tmp_quant);
    split_out_tensor[5] = this->graph->CreateTensor(tmp_spec_5);

    tim::vx::TensorSpec tmp_spec(tim::vx::DataType::UINT8, vx_concat_shape, tim::vx::TensorAttribute::TRANSIENT, tmp_quant);
    auto tmp_output = this->graph->CreateTensor(tmp_spec);
    auto reshape = graph->CreateOperation<tim::vx::ops::Reshape>(vx_concat_shape);
    vx_node_map[ir_node->index + ir_graph->node_num*1] = reshape;
    (*reshape)
            .BindInput({ this->vx_tensor_map[input_tensor1->index] })
            .BindOutputs({ tmp_output });
    
    std::vector<uint32_t> slices({1,1,1,1,1,1});
    auto split = graph->CreateOperation<tim::vx::ops::Split>(axis, slices);
    vx_node_map[ir_node->index + ir_graph->node_num*2] = split;
    (*split)
            .BindInput({ tmp_output })
            .BindOutputs(split_out_tensor);
    
    tim::vx::TensorSpec tmp_concat_spec(tim::vx::DataType::UINT8, vx_concat_shape, tim::vx::TensorAttribute::TRANSIENT, tmp_quant);
    auto tmp_concat = this->graph->CreateTensor(tmp_concat_spec);
    auto concat = graph->CreateOperation<tim::vx::ops::Concat>(axis, 6);
    vx_node_map[ir_node->index + ir_graph->node_num*3] = concat;
    (*concat)
        .BindInputs({split_out_tensor[4],split_out_tensor[3],split_out_tensor[5],split_out_tensor[1],split_out_tensor[0],split_out_tensor[2]})
        .BindOutputs({ tmp_concat });

    struct spatialtransformer_param* param = (struct spatialtransformer_param*)ir_node->op.param_mem;
    auto spatialtransformer = graph->CreateOperation<tim::vx::ops::SpatialTransformer>(param->target_shape[0],param->target_shape[1],
        false, false, false, false, false, false, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, true);

    (*spatialtransformer)
        .BindInputs({ this->vx_tensor_map[input_tensor->index], tmp_concat })
        .BindOutputs({ this->vx_tensor_map[output_tensor->index] });
    return true;
}
