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
#include "deconv_param.h"
}


bool VXEngine::AddDeconvNode(struct node* ir_node)
{
    struct graph* ir_graph = ir_node->graph;

    struct deconv_param* param = (struct deconv_param*)ir_node->op.param_mem;

    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct tensor* weight_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    tim::vx::PadType padtype = tim::vx::PadType::SAME;

    int h = input_tensor->dims[2];
    int out_h = (h - 1) / param->stride_h + 1;
    int total_len_h = (out_h - 1) * param->stride_h + param->kernel_h;
    int pad_num_h = total_len_h - h;
    int pad_h0 = param->pad_h0;
    if (param->pad_h0 == pad_num_h / 2 && param->pad_h1 == pad_num_h - pad_num_h / 2 && param->pad_h0 != 0)
    {
        pad_h0 = -1;
    }

    int w = input_tensor->dims[3];
    int out_w = (w - 1) / param->stride_w + 1;
    int total_len_w = (out_w - 1) * param->stride_w + param->kernel_w;
    int pad_num_w = total_len_w - w;
    int pad_w0 = param->pad_w0;
    if (param->pad_w0 == pad_num_w / 2 && param->pad_w1 == pad_num_w - pad_num_w / 2 && param->pad_w0 != 0)
    {
        pad_w0 = -1;
    }

    if(pad_h0 == 0 && pad_w0 == 0)
    {
        padtype = tim::vx::PadType::VALID;
    }
    else
    {
        padtype = tim::vx::PadType::SAME;
    }

    auto deconv = this->graph->CreateOperation<tim::vx::ops::DeConv2d>(
            weight_tensor->dims[0], tim::vx::PadType::AUTO,
            std::array<uint32_t, 2>({ (unsigned int)param->kernel_w, (unsigned int)param->kernel_h }),
            std::array<uint32_t, 2>({ (unsigned int)param->stride_w, (unsigned int)param->stride_h }),
            std::array<uint32_t, 2>({ (unsigned int)param->output_pad_w0, (unsigned int)param->output_pad_h0 }),
            std::array<uint32_t, 4>({ (unsigned int)param->pad_w0, (unsigned int)param->pad_w1,
                                      (unsigned int)param->pad_h0, (unsigned int)param->pad_h1 }),
            (unsigned int)param->group  );
    
    vx_node_map[ir_node->index] = deconv;

    if (param->activation >= 0)
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
            (*deconv)
                    .BindInputs({ this->vx_tensor_map[input_tensor->index], this->vx_tensor_map[weight_tensor->index], this->vx_tensor_map[bias_tensor->index] })
                    .BindOutputs({ tmp_output });
        }
        else
        {
            (*deconv)
                    .BindInputs({ this->vx_tensor_map[input_tensor->index], this->vx_tensor_map[weight_tensor->index] })
                    .BindOutputs({ tmp_output });
        }
        this->vx_tensor_map[output_tensor->index + ir_graph->tensor_num] = tmp_output;
        if (param->activation == 0)
        {
            auto relu = this->graph->CreateOperation<tim::vx::ops::Relu>();
            (*relu).BindInput( tmp_output )
                    .BindOutput({ this->vx_tensor_map[output_tensor->index] });
        }
        else if (param->activation == 6)
        {
            auto relu = this->graph->CreateOperation<tim::vx::ops::Relu6>();
            (*relu).BindInput({ tmp_output })
                    .BindOutput({ this->vx_tensor_map[output_tensor->index] });
        }

    }
    else
    {
        if (ir_node->input_num > 2)
        {
            struct tensor* bias_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[2]);
            (*deconv)
                    .BindInputs({ this->vx_tensor_map[input_tensor->index], this->vx_tensor_map[weight_tensor->index], this->vx_tensor_map[bias_tensor->index] })
                    .BindOutputs({ this->vx_tensor_map[output_tensor->index] });
        }
        else
        {
            (*deconv)
                    .BindInputs({ this->vx_tensor_map[input_tensor->index], this->vx_tensor_map[weight_tensor->index] })
                    .BindOutputs({ this->vx_tensor_map[output_tensor->index] });
        }
    }

    return true;
}
