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
 * Copyright (c) 2020, OPEN AI LAB
 * Author: ddzhao@openailab.com
 */

#ifndef VULKAN_COMMAND_HPP
#define VULKAN_COMMAND_HPP

#include <vulkan/vulkan.h>
#include <vector>
#include "vulkan_allocator.hpp"
#include "vulkan_tensor.hpp"
#include "vulkan_pipeline.hpp"
#include "vulkan_option.hpp"
#include "vulkan_platform.hpp"
// #include "tengine_log.h"

namespace TEngine {

class Pipeline;
class VkCompute
{
public:
    VkCompute(const GPUDevice* vkdev);
    virtual ~VkCompute();

public:
    void record_upload(tensor* src, VkTensor& dst, const Option& opt);
    void record_upload(const Tensor& src, VkTensor& dst, const Option& opt);

    void record_download(const VkTensor& src, tensor* dst, const Option& opt);
    void record_download(const VkTensor& src, Tensor& dst, const Option& opt);

    void record_pipeline(const Pipeline* pipeline, const std::vector<VkTensor>& bindings, const std::vector<vk_constant_type>& constants, const VkTensor& dispatcher);
    void record_pipeline(const Pipeline* pipeline, const std::vector<VkImageTensor>& bindings, const std::vector<vk_constant_type>& constants, const VkImageTensor& dispatcher);
    void record_pipeline(const Pipeline* pipeline, const std::vector<VkTensor>& buffer_bindings, const std::vector<VkImageTensor>& image_bindings, const std::vector<vk_constant_type>& constants, const VkTensor& dispatcher);
    void record_pipeline(const Pipeline* pipeline, const std::vector<VkTensor>& buffer_bindings, const std::vector<VkImageTensor>& image_bindings, const std::vector<vk_constant_type>& constants, const VkImageTensor& dispatcher);
    void record_pipeline(const Pipeline* pipeline, const std::vector<VkTensor>& buffer_bindings, const std::vector<VkImageTensor>& image_bindings, const std::vector<vk_constant_type>& constants, int dispatcher_w, int dispatcher_h, int dispatcher_c);
    
    int submit_and_wait();

    int reset();

protected:
    int init();
    int begin_command_buffer();
    int end_command_buffer();

protected:
    const GPUDevice* vkdev;

    VkCommandPool compute_command_pool;
    VkCommandBuffer compute_command_buffer;
    VkFence compute_command_fence;

    std::vector<VkTensor> upload_staging_buffers;
    std::vector<VkTensor> download_post_buffers;
    std::vector<Tensor> download_post_tensors_fp16;
    std::vector<Tensor> download_post_tensors;
    std::vector<VkImageMemory*> image_blocks_to_destroy;

    // the good-old path for device without VK_KHR_push_descriptor
    std::vector<VkDescriptorPool> descriptor_pools;
    std::vector<VkDescriptorSet> descriptorsets;

    struct record
    {
        enum
        {
            TYPE_copy_buffer,
            TYPE_copy_image,
            TYPE_copy_buffer_to_image,
            TYPE_copy_image_to_buffer,
            TYPE_bind_pipeline,
            TYPE_bind_descriptorsets,
            TYPE_push_constants,
            TYPE_dispatch,
            TYPE_memory_barrers,
            TYPE_buffer_barrers,
            TYPE_image_barrers,
            TYPE_post_download,
            TYPE_post_cast_float16_to_float32,
        };

        int type;
        VkCommandBuffer command_buffer;

        union
        {
            struct { VkBuffer src; VkBuffer dst; uint32_t region_count; const VkBufferCopy* regions; } copy_buffer;
            struct { VkImage src; VkImageLayout src_layout; VkImage dst; VkImageLayout dst_layout; uint32_t region_count; const VkImageCopy* regions; } copy_image;
            struct { VkBuffer src; VkImage dst; VkImageLayout layout; uint32_t region_count; const VkBufferImageCopy* regions; } copy_buffer_to_image;
            struct { VkImage src; VkImageLayout layout; VkBuffer dst; uint32_t region_count; const VkBufferImageCopy* regions; } copy_image_to_buffer;

            struct { VkPipelineBindPoint bind_point; VkPipeline pipeline; } bind_pipeline;
            struct { VkPipelineBindPoint bind_point; VkPipelineLayout pipeline_layout; uint32_t descriptorset_count; uint32_t descriptorset_offset; } bind_descriptorsets;
            struct { VkPipelineLayout pipeline_layout; VkShaderStageFlags stage_flags; uint32_t size; const void* values; } push_constants;

            struct { uint32_t group_count_x; uint32_t group_count_y; uint32_t group_count_z; } dispatch;

            struct { VkPipelineStageFlags src_stage; VkPipelineStageFlags dst_stage; uint32_t barrier_count; const VkMemoryBarrier* barriers; } memory_barrers;
            struct { VkPipelineStageFlags src_stage; VkPipelineStageFlags dst_stage; uint32_t barrier_count; const VkBufferMemoryBarrier* barriers; } buffer_barrers;
            struct { VkPipelineStageFlags src_stage; VkPipelineStageFlags dst_stage; uint32_t barrier_count; const VkImageMemoryBarrier* barriers; } image_barrers;

            struct { uint32_t download_post_buffer_mat_offset; uint32_t download_post_mat_fp16_offset; } post_download;
            struct { uint32_t download_post_mat_fp16_offset; uint32_t download_post_mat_offset; } post_cast_float16_to_float32;
        };
    };

    std::vector<record> delayed_records;
};


class VkTransfer
{
public:
    VkTransfer(const GPUDevice* vkdev);
    ~VkTransfer();
public:
    void record_upload(const tensor* src, VkTensor& dst, const Option& opt);
    void record_upload(const Tensor& src, VkTensor& dst, const Option& opt);

    int submit_and_wait();

protected:
    int init();
    int begin_command_buffer();
    int end_command_buffer();

protected:
    const GPUDevice* vkdev;

    VkCommandPool compute_command_pool;
    VkCommandPool transfer_command_pool;

    VkCommandBuffer upload_command_buffer;
    VkCommandBuffer compute_command_buffer;

    VkSemaphore upload_compute_semaphore;

    VkFence upload_command_fence;
    VkFence compute_command_fence;

    std::vector<VkTensor> upload_staging_buffers;
};

} // namespace TEngine

#endif
