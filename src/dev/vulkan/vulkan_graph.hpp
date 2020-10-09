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

#ifndef __VULKAN_GRAPH_HPP__
#define __VULKAN_GRAPH_HPP__

#include <array>
#include <random>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <vulkan/vulkan.h>

#include "vulkan_gpu.hpp"
#include "vulkan_pipeline.hpp"
#include "vulkan_command.hpp"
#include "vulkan_option.hpp"
#include "vulkan_layer.hpp"

#include "tengine_op.h"
#include "convolution_param.h"

namespace TEngine {

class VulkanDevice;

class VulkanGraph {

friend VulkanDevice;

public:
    const std::string& GetName(void) const {return name_;}

    VulkanGraph(const std::string& name);
    VulkanGraph(struct subgraph* graph);
    ~VulkanGraph();

    int record_convolution(VkCompute& cmd, ir_node* node);

    int UploadConvolutionWeight(VkTransfer& cmd, const Option& opt, ir_node* node);

    bool CreateConvolutionPipeline(ir_node* node);

    bool CreatePoolingPipeline(ir_node* node);

    std::unordered_map<std::string, ir_tensor*> tensor_map_;    // tengine lite cpu tensor list
    std::unordered_map<std::string, Tensor> tensor_map;         // vulkan cpu tensor list
    std::unordered_map<std::string, VkTensor> vktensor_map_;    // vulkan gpu tensor list

    bool OpSupported(const std::string& name);

    Option opt;
    Pipeline* pipeline_convolution;
    
    int record_graph_pipeline();

    int upload_model();

    int create_pipeline();

    int destory_pipeline();

protected:
    subgraph* sgraph;
    std::vector<Layer*> layers;

    const GPUDevice* vkdev;

    VkAllocator* weight_vkallocator;
    VkAllocator* weight_staging_vkallocator;
    
private:

    VkAllocator* local_blob_vkallocator;
    VkAllocator* local_staging_vkallocator;
    
    std::string name_;

    std::vector<void *> gpu_mem_vector_;
    std::vector<void *> mem_buf_vector_;

    std::map<std::string, ir_tensor*> iotensor_map_;
};

} //namespace TEngine

#endif
