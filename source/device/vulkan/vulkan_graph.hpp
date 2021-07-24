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
 * Copyright (c) 2021, OPEN AI LAB
 * Author: hhchen@openailab.com
 */

#pragma once

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

extern "C"
{
// #include "device/device.h"
// #include "graph/subgraph.h"

#include "api/c_api.h"
#include "device/device.h"
#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "graph/subgraph.h"
#include "executer/executer.h"
#include "optimizer/split.h"
#include "module/module.h"
#include "utility/vector.h"
#include "utility/log.h"


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

    int record_convolution(VkCompute& cmd, ir_node_t* node);

    int UploadConvolutionWeight(VkTransfer& cmd, const Option& opt, ir_node_t* node);

    bool CreateConvolutionPipeline(ir_node_t* node);

    bool CreatePoolingPipeline(ir_node_t* node);

    std::unordered_map<std::string, tensor*> tensor_map_;    // tengine lite cpu tensor list
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

    std::map<std::string, tensor*> iotensor_map_;
};

} //namespace TEngine


int vulkan_dev_init(struct device* dev);
int vulkan_dev_prerun(struct device* dev, struct subgraph* subgraph, void* options);
int vulkan_dev_run(struct device* dev, struct subgraph* subgraph);
int vulkan_dev_postrun(struct device* dev, struct subgraph* subgraph);
int vulkan_dev_release(struct device* dev);
}


/*




*/