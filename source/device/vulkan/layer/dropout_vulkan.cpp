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
 * Parts of the following code in this file refs to
 * https://github.com/Tencent/ncnn/tree/master/src/layer/vulkan/
 * Tencent is pleased to support the open source community by making ncnn
 * available.
 *
 * Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
 *
 * Licensed under the BSD 3-Clause License (the "License"); you may not use this
 * file except in compliance with the License. You may obtain a copy of the
 * License at
 *
 * https://opensource.org/licenses/BSD-3-Clause
 */

/*
 * Copyright (c) 2020, Open AI Lab
 * Author: ddzhao@openailab.com
 */

#include "dropout_vulkan.hpp"
#include "../layer_shader_type.h"

namespace TEngine {

Dropout_vulkan::Dropout_vulkan()
{
    support_vulkan = true;
    support_image_storage = false;

    pipeline_dropout = 0;
    pipeline_dropout_pack4 = 0;
    pipeline_dropout_pack8 = 0;
}

Dropout_vulkan::Dropout_vulkan(ir_graph_t* ir_graph, ir_node_t* ir_node)
{
    support_vulkan = true;
    support_image_storage = false;

    pipeline_dropout = 0;
    pipeline_dropout_pack4 = 0;
    pipeline_dropout_pack8 = 0;

    graph = ir_graph;
    node = ir_node;

    struct tensor* input = get_ir_graph_tensor(graph, node->input_tensors[0]);
    std::string name = input->name;
    bottoms.push_back(name);

    struct tensor* output = get_ir_graph_tensor(graph, node->output_tensors[0]);
    name = output->name;
    tops.push_back(name);

    // params
    input_c = input->dims[1]; // param->input_channel;
    input_h = input->dims[2];
    input_w = input->dims[3];
    output_c = output->dims[1]; // param->output_channel;
    output_h = output->dims[2];
    output_w = output->dims[3];

    if (input->scale != 0)
        scale = input->scale;
    else
        scale = 1.0f;
}

int Dropout_vulkan::create_pipeline(const Option& opt)
{
    const Tensor& shape = Tensor(output_w, output_h, output_c, (void*)0); // top_shapes.empty() ? Tensor() : top_shapes[0];

    int elempack = 1;
    if (shape.dims == 1) elempack = opt.use_shader_pack8 && shape.w % 8 == 0 ? 8 : shape.w % 4 == 0 ? 4
                                                                                                    : 1;
    if (shape.dims == 2) elempack = opt.use_shader_pack8 && shape.h % 8 == 0 ? 8 : shape.h % 4 == 0 ? 4
                                                                                                    : 1;
    if (shape.dims == 3) elempack = opt.use_shader_pack8 && shape.c % 8 == 0 ? 8 : shape.c % 4 == 0 ? 4
                                                                                                    : 1;

    size_t elemsize;
    if (opt.use_fp16_storage)
    {
        elemsize = elempack * 2u;
    }
    else if (opt.use_fp16_packed)
    {
        elemsize = elempack == 1 ? 4u : elempack * 2u;
    }
    else
    {
        elemsize = elempack * 4u;
    }

    Tensor shape_packed;
    if (shape.dims == 1) shape_packed = Tensor(shape.w / elempack, (void*)0, elemsize, elempack);
    if (shape.dims == 2) shape_packed = Tensor(shape.w, shape.h / elempack, (void*)0, elemsize, elempack);
    if (shape.dims == 3) shape_packed = Tensor(shape.w, shape.h, shape.c / elempack, (void*)0, elemsize, elempack);

    std::vector<vk_specialization_type> specializations(1 + 5);
    specializations[0].f = scale;
    specializations[1 + 0].i = shape_packed.dims;
    specializations[1 + 1].i = shape_packed.w;
    specializations[1 + 2].i = shape_packed.h;
    specializations[1 + 3].i = shape_packed.c;
    specializations[1 + 4].i = shape_packed.cstep;

    Tensor local_size_xyz;
    if (shape_packed.dims == 1)
    {
        local_size_xyz.w = std::min(64, shape_packed.w);
        local_size_xyz.h = 1;
        local_size_xyz.c = 1;
    }
    if (shape_packed.dims == 2)
    {
        local_size_xyz.w = std::min(8, shape_packed.w);
        local_size_xyz.h = std::min(8, shape_packed.h);
        local_size_xyz.c = 1;
    }
    if (shape_packed.dims == 3)
    {
        local_size_xyz.w = std::min(4, shape_packed.w);
        local_size_xyz.h = std::min(4, shape_packed.h);
        local_size_xyz.c = std::min(4, shape_packed.c);
    }

    // pack1
    if (shape.dims == 0 || elempack == 1)
    {
        pipeline_dropout = new Pipeline(vkdev);
        pipeline_dropout->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_dropout->create(LayerShaderType::dropout, opt, specializations);
    }

    // pack4
    if (shape.dims == 0 || elempack == 4)
    {
        pipeline_dropout_pack4 = new Pipeline(vkdev);
        pipeline_dropout_pack4->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_dropout_pack4->create(LayerShaderType::dropout_pack4, opt, specializations);
    }

    // pack8
    if ((opt.use_shader_pack8 && shape.dims == 0) || elempack == 8)
    {
        pipeline_dropout_pack8 = new Pipeline(vkdev);
        pipeline_dropout_pack8->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_dropout_pack8->create(LayerShaderType::dropout_pack8, opt, specializations);
    }

    return 0;
}

int Dropout_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    delete pipeline_dropout;
    pipeline_dropout = 0;

    delete pipeline_dropout_pack4;
    pipeline_dropout_pack4 = 0;

    delete pipeline_dropout_pack8;
    pipeline_dropout_pack8 = 0;

    return 0;
}

int Dropout_vulkan::record_pipeline(VkTensor& bottom_top_blob, VkCompute& cmd, const Option& /*opt*/) const
{
    if (scale == 1.f)
    {
        return 0;
    }

    int elempack = bottom_top_blob.elempack;

    std::vector<VkTensor> bindings(1);
    bindings[0] = bottom_top_blob;

    std::vector<vk_constant_type> constants(5);
    constants[0].i = bottom_top_blob.dims;
    constants[1].i = bottom_top_blob.w;
    constants[2].i = bottom_top_blob.h;
    constants[3].i = bottom_top_blob.c;
    constants[4].i = bottom_top_blob.cstep;

    const Pipeline* pipeline = elempack == 8   ? pipeline_dropout_pack8
                               : elempack == 4 ? pipeline_dropout_pack4
                                               : pipeline_dropout;

    cmd.record_pipeline(pipeline, bindings, constants, bottom_top_blob);

    return 0;
}

} // namespace TEngine