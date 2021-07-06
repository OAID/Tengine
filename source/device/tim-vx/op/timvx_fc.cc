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

#include "timvx_executor.hpp"

extern "C"
{
#include "operator/op.h"
#include "convolution_param.h"
}


bool VXEngine::AddFullyConnectionNode(struct node* ir_node)
{
    struct graph* ir_graph = ir_node->graph;

    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct tensor* weight_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    int fc_dim_nim = 2;
    auto fc_shape_size = this->vx_tensor_map[input_tensor->index].get()->GetShape().size();
    if (fc_shape_size == 2)
    {
        fc_dim_nim = 0;
    }

    auto fc = graph->CreateOperation<tim::vx::ops::FullyConnected>(
            fc_dim_nim, weight_tensor->dims[0]);
    vx_node_map[ir_node->index] = fc;

    if (output_tensor->dim_num == 2)
    {
        if (ir_node->input_num > 2)
        {
            struct tensor* bias_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[2]);
            (*fc)
                    .BindInputs({this->vx_tensor_map[input_tensor->index], this->vx_tensor_map[weight_tensor->index], this->vx_tensor_map[bias_tensor->index]})
                    .BindOutputs({ this->vx_tensor_map[output_tensor->index] });
        }
        else
        {
            (*fc)
                    .BindInputs({ this->vx_tensor_map[input_tensor->index], this->vx_tensor_map[weight_tensor->index] })
                    .BindOutputs({ this->vx_tensor_map[output_tensor->index] });
        }
    }
    else if (output_tensor->dim_num == 4)
    {
        tim::vx::Quantization tmp_quant(tim::vx::QuantType::ASYMMETRIC,
                                        output_tensor->scale, output_tensor->zero_point);
        tim::vx::ShapeType vx_shape;
        std::vector<uint32_t> perm;
        for (int i = output_tensor->dim_num - 1; i >= 0; i--)
        {
            vx_shape.push_back(output_tensor->dims[i]);
            perm.push_back(output_tensor->dims[i]);
        }
        tim::vx::TensorSpec tmp_spec(tim::vx::DataType::UINT8, vx_shape, tim::vx::TensorAttribute::TRANSIENT, tmp_quant);
        auto tmp_output = this->graph->CreateTensor(tmp_spec);

        if (ir_node->input_num > 2)
        {
            struct tensor* bias_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[2]);
            (*fc)
                    .BindInputs({this->vx_tensor_map[input_tensor->index], this->vx_tensor_map[weight_tensor->index], this->vx_tensor_map[bias_tensor->index]})
                    .BindOutputs({ tmp_output });
        }
        else
        {
            (*fc)
                    .BindInputs({ this->vx_tensor_map[input_tensor->index], this->vx_tensor_map[weight_tensor->index] })
                    .BindOutputs({ tmp_output });
        }

        std::vector<uint32_t> perm_shape;
        for (int i = output_tensor->dim_num - 1; i >= 0; i--)
        {
            perm_shape.push_back(output_tensor->dims[i]);
        }

        auto reshape = graph->CreateOperation<tim::vx::ops::Reshape>(perm_shape);
        vx_node_map[ir_node->index + ir_graph->node_num] = reshape;

        (*reshape)
                .BindInputs({ tmp_output })
                .BindOutputs({ this->vx_tensor_map[output_tensor->index] });

    }

    return true;
}
