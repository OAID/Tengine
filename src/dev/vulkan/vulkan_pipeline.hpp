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

#ifndef VULKAN_PIPELINE_HPP
#define VULKAN_PIPELINE_HPP

#include <vulkan/vulkan.h>
#include "vulkan_gpu.hpp"
#include "vulkan_tensor.hpp"
#include "vulkan_platform.hpp"
#include "vulkan_option.hpp"

namespace TEngine {

class Option;
class Pipeline
{
public:
    Pipeline(const GPUDevice* vkdev);
    virtual ~Pipeline();

public:
    void set_optimal_local_size_xyz(int w = 4, int h = 4, int c = 4);
    
    void set_optimal_local_size_xyz(const VkTensor& local_size_xyz);
    void set_optimal_local_size_xyz(const Tensor& local_size_xyz);
    void set_local_size_xyz(int w, int h, int c);

    int create(const uint32_t* spv_data, size_t spv_data_size, const std::vector<vk_specialization_type>& specializations);

    int create(int shader_type_index, const Option& opt, const std::vector<vk_specialization_type>& specializations);

    int create(VkShaderModule shader_module, const ShaderInfo& si, const std::vector<vk_specialization_type>& specializations);

    void destroy();

protected:
    int create_descriptorset_layout();
    int create_pipeline_layout();
    int create_pipeline(VkShaderModule shader_module, const std::vector<vk_specialization_type>& specializations);
    int create_descriptor_update_template();

public:
    const GPUDevice* vkdev;

    // local shader module
    VkShaderModule local_shader_module;

    VkDescriptorSetLayout descriptorset_layout;
    VkPipelineLayout pipeline_layout;

    // op forward TODO use pipeline cache ?
    VkPipeline pipeline;

    VkDescriptorUpdateTemplateKHR descriptor_update_template;

    ShaderInfo shader_info;

    uint32_t local_size_x;
    uint32_t local_size_y;
    uint32_t local_size_z;
};

#if __ANDROID_API__ >= 26
class VkCompute;
class ImportAndroidHardwareBufferPipeline : private Pipeline
{
public:
    ImportAndroidHardwareBufferPipeline(const GPUDevice* vkdev);
    ~ImportAndroidHardwareBufferPipeline();

    int create(VkAndroidHardwareBufferImageAllocator* ahb_im_allocator, int type_to, int rotate_from, const Option& opt);
    int create(VkAndroidHardwareBufferImageAllocator* ahb_im_allocator, int type_to, int rotate_from, int target_width, int target_height, const Option& opt);
    void destroy();

    friend class VkCompute;

protected:
    int create_sampler(VkAndroidHardwareBufferImageAllocator* ahb_im_allocator);
    int create_descriptorset_layout();
    int create_descriptor_update_template();

public:
    int type_to;
    int rotate_from;
    bool need_resize;

    VkSampler sampler;
};
#endif // __ANDROID_API__ >= 26

} // namespace TEngine

#endif // VULKAN_PIPELINE_HPP
