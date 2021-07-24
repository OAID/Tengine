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

#include "vulkan_command.hpp"

#include <algorithm>
#include "vulkan_option.hpp"
#include "vulkan_pipeline.hpp"
#include "vulkan_tensor.hpp"

namespace TEngine {

VkCompute::VkCompute(const GPUDevice* _vkdev) : vkdev(_vkdev)
{
    compute_command_pool = 0;
    compute_command_buffer = 0;
    compute_command_fence = 0;

    init();
}


VkCompute::~VkCompute()
{
    for (size_t i=0; i<image_blocks_to_destroy.size(); i++)
    {
        VkImageMemory* ptr = image_blocks_to_destroy[i];

        int old_command_refcount = TENGINE_XADD(&ptr->command_refcount, -1);
        if (ptr->refcount == 0 && old_command_refcount == 1)
        {
            // no userspace reference and we are the last command reference
            vkDestroyImageView(vkdev->vkdevice(), ptr->imageview, 0);
            vkDestroyImage(vkdev->vkdevice(), ptr->image, 0);

            delete ptr;
        }
        else
        {
            // reference exists in user code or other command
        }
    }
    image_blocks_to_destroy.clear();

    if (!vkdev->info.support_VK_KHR_push_descriptor)
    {
        for (size_t i=0; i<descriptorsets.size(); i++)
        {
            vkFreeDescriptorSets(vkdev->vkdevice(), descriptor_pools[i], 1, &descriptorsets[i]);
            vkDestroyDescriptorPool(vkdev->vkdevice(), descriptor_pools[i], 0);
        }
    }

    vkDestroyFence(vkdev->vkdevice(), compute_command_fence, 0);

    vkFreeCommandBuffers(vkdev->vkdevice(), compute_command_pool, 1, &compute_command_buffer);
    vkDestroyCommandPool(vkdev->vkdevice(), compute_command_pool, 0);
}

void VkCompute::record_upload(tensor* src, VkTensor& dst, const Option& opt)
{
    Tensor src_tensor = Tensor(src);
    record_upload(src_tensor, dst, opt);
//     // const ir_tensor* src_fp16;
//     // if (src.elemsize == src.elempack * 4u)
//     if(src->elem_size == opt.elempack * 4u)
//     {
//         // cpu cast to fp16 (discrete gpu)
//         if (vkdev->info.type == 0 && (opt.use_fp16_storage || (opt.use_fp16_packed && opt.elempack % 4 == 0)))
//         {
//             // ncnn::cast_float32_to_float16(src, src_fp16, opt);
//             printf("need to add cast_float32_to_float16 here, fix me!\n");
//         }
//         else
//         {
//             // src_fp16 = src;
//         }
//     }
//     else
//     {
//         // src_fp16 = src;
//     }

//     // upload
//     VkTensor dst_staging;
//     if (opt.blob_vkallocator->mappable)
//     {
//         // dst_staging.create_like(src_fp16, opt.blob_vkallocator);
//         dst_staging.create_like(src, opt.blob_vkallocator);
//     }
//     else
//     {
//         // dst_staging.create_like(src_fp16, opt.staging_vkallocator);
//         dst_staging.create_like(src, opt.staging_vkallocator);
//     }
//     if (dst_staging.empty())
//         return;

//     // stash staging
//     upload_staging_buffers.push_back(dst_staging);

// //     TLOG_INFO("upload_staging_buffer %p  ->   %p +%d ~%d", src_fp16.data, dst_staging.buffer(), dst_staging.buffer_offset(), dst_staging.buffer_capacity());

//     // memcpy src to device
//     // memcpy(dst_staging.mapped_ptr(), src_fp16->data, src_fp16->elem_size * src_fp16->elem_num);
//     memcpy(dst_staging.mapped_ptr(), src->data, src->elem_size * src->elem_num);
//     dst_staging.allocator->flush(dst_staging.data);

//     // mark device host-write @ null
//     dst_staging.data->access_flags = VK_ACCESS_HOST_WRITE_BIT;
//     dst_staging.data->stage_flags = VK_PIPELINE_STAGE_HOST_BIT;

//     // TODO
//     // not use pack for now------------------------
//     // // resolve dst_elempack
//     int dims = src->dim_num;
//     int elemcount = 0;
//     // src dims[0-3]  n c h w
//     // if (dims == 1) elemcount = opt.elempack * src_fp16.w;
//     // if (dims == 2) elemcount = opt.elempack * src_fp16.h;
//     // if (dims == 3) elemcount = opt.elempack * src_fp16.c;
//     if(dims == 4) 
//         elemcount = opt.elempack * src->dims[1];
//     else 
//         elemcount = opt.elempack * src->dims[0];

//     int dst_elempack = 1;
//     if (opt.use_shader_pack8)
//         dst_elempack = elemcount % 8 == 0 ? 8 : elemcount % 4 == 0 ? 4 : 1;
//     else
//         dst_elempack = elemcount % 4 == 0 ? 4 : 1;

//     vkdev->convert_packing(dst_staging, dst, dst_elempack, *this, opt);
}

void VkCompute::record_upload(const Tensor& src, VkTensor& dst, const Option& opt)
{
    //     TLOG_INFO("record_upload buffer");

    Tensor src_fp16;
    if (src.elemsize == src.elempack * 4u)
    {
        // cpu cast to fp16 (discrete gpu)
        if (vkdev->info.type == 0 && (opt.use_fp16_storage || (opt.use_fp16_packed && src.elempack % 4 == 0)))
        {
            // printf("do nothing for VkCompute record_upload cast_float32_to_float16, fix me\n");
            TEngine::cast_float32_to_float16(src, src_fp16, opt);
        }
        else
        {
            src_fp16 = src;
        }
    }
    else
    {
        src_fp16 = src;
    }

    // upload
    VkTensor dst_staging;
    if (opt.blob_vkallocator->mappable)
    {
        dst_staging.create_like(src_fp16, opt.blob_vkallocator);
    }
    else
    {
        dst_staging.create_like(src_fp16, opt.staging_vkallocator);
    }
    if (dst_staging.empty())
        return;

    // stash staging
    upload_staging_buffers.push_back(dst_staging);

//     TLOG_INFO("upload_staging_buffer %p  ->   %p +%d ~%d", src_fp16.data, dst_staging.buffer(), dst_staging.buffer_offset(), dst_staging.buffer_capacity());

    // memcpy src to device
    memcpy(dst_staging.mapped_ptr(), src_fp16.data, src_fp16.total() * src_fp16.elemsize);
    dst_staging.allocator->flush(dst_staging.data);

    // mark device host-write @ null
    dst_staging.data->access_flags = VK_ACCESS_HOST_WRITE_BIT;
    dst_staging.data->stage_flags = VK_PIPELINE_STAGE_HOST_BIT;

    // resolve dst_elempack
    int dims = src_fp16.dims;
    int elemcount = 0;
    if (dims == 1) elemcount = src_fp16.elempack * src_fp16.w;
    if (dims == 2) elemcount = src_fp16.elempack * src_fp16.h;
    if (dims == 3) elemcount = src_fp16.elempack * src_fp16.c;

    int dst_elempack = 1;
    if (opt.use_shader_pack8)
        dst_elempack = elemcount % 8 == 0 ? 8 : elemcount % 4 == 0 ? 4 : 1;
    else
        dst_elempack = elemcount % 4 == 0 ? 4 : 1;
    
    // gpu cast to fp16 on the fly (integrated gpu)
    vkdev->convert_packing(dst_staging, dst, dst_elempack, *this, opt);
}

void VkCompute::record_download(const VkTensor& src, tensor* dst, const Option& opt)
{
    Tensor dst_tensor;
    record_download(src, dst_tensor, opt);
    dst->data = dst_tensor.data;

    // Tensor feat;
    // if (opt.use_packing_layout)
    // {
    //     Tensor bottom_blob_unpacked;
    //     convert_packing(dst_tensor, bottom_blob_unpacked, 1, opt);
    //     feat = bottom_blob_unpacked;
    // }

    // if (opt.use_bf16_storage)
    // {
    //     if (feat.elemsize / feat.elempack == 2u)
    //     {
    //         Tensor feat_fp32;
    //         cast_bfloat16_to_float32(feat, feat_fp32, opt);
    //         feat = feat_fp32;
    //     }
    // }

    // dst->data = feat.data;
}

void VkCompute::record_download(const VkTensor& src, Tensor& dst, const Option& opt)
{
    int dims = src.dims;
    int elemcount = 0;
    if (dims == 1) elemcount = src.elempack * src.w;
    if (dims == 2) elemcount = src.elempack * src.h;
    if (dims == 3) elemcount = src.elempack * src.c;

    int dst_elempack = 1;
    if (opt.use_packing_layout)
        dst_elempack = elemcount % 4 == 0 ? 4 : 1;
    else
        dst_elempack = 1;

    // gpu cast to fp32 on the fly (integrated gpu)
    Option opt_staging = opt;
    if (vkdev->info.type != 0)
    {
        opt_staging.use_fp16_packed = false;
        opt_staging.use_fp16_storage = false;
    }

    VkTensor dst_staging;
    if (opt_staging.blob_vkallocator->mappable)
    {
        vkdev->convert_packing(src, dst_staging, dst_elempack, *this, opt);
    }
    else
    {
        opt_staging.blob_vkallocator = opt.staging_vkallocator;
        vkdev->convert_packing(src, dst_staging, dst_elempack, *this, opt_staging);
    }

    // barrier device any @ compute to host-read @ compute
    if (dst_staging.data->access_flags & VK_ACCESS_HOST_WRITE_BIT || dst_staging.data->stage_flags != VK_PIPELINE_STAGE_HOST_BIT)
    {
        VkBufferMemoryBarrier* barriers = new VkBufferMemoryBarrier[1];
        barriers[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barriers[0].pNext = 0;
        barriers[0].srcAccessMask = dst_staging.data->access_flags;
        barriers[0].dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].buffer = dst_staging.buffer();
        barriers[0].offset = dst_staging.buffer_offset();
        barriers[0].size = dst_staging.buffer_capacity();

        VkPipelineStageFlags src_stage = dst_staging.data->stage_flags;
        VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_HOST_BIT;

        if (vkdev->info.support_VK_KHR_push_descriptor)
        {
            vkCmdPipelineBarrier(compute_command_buffer, src_stage, dst_stage, 0, 0, 0, 1, barriers, 0, 0);
            delete[] barriers;
        }
        else
        {
            record r;
            r.type = record::TYPE_buffer_barrers;
            r.command_buffer = compute_command_buffer;
            r.buffer_barrers.src_stage = src_stage;
            r.buffer_barrers.dst_stage = dst_stage;
            r.buffer_barrers.barrier_count = 1;
            r.buffer_barrers.barriers = barriers;
            delayed_records.push_back(r);
        }

        // mark device host-read @ any
        dst_staging.data->access_flags = VK_ACCESS_HOST_READ_BIT;
        dst_staging.data->stage_flags = VK_PIPELINE_STAGE_HOST_BIT;
    }

    // create dst
    Tensor dst_fp16;
    dst_fp16.create_like(dst_staging, opt.blob_allocator);
    if (dst_fp16.empty())
        return;

    // download
    download_post_buffers.push_back(dst_staging);
    download_post_tensors_fp16.push_back(dst_fp16);

    // post memcpy device to dst
    {
        record r;
        r.type = record::TYPE_post_download;
        r.command_buffer = 0;
        r.post_download.download_post_buffer_mat_offset = download_post_buffers.size() - 1;
        r.post_download.download_post_mat_fp16_offset = download_post_tensors_fp16.size() - 1;
        delayed_records.push_back(r);
    }

    // cast to fp32 (discrete gpu)
    if (dst_fp16.elemsize == dst_fp16.elempack * 2u)
    {
        if (vkdev->info.type == 0 && (opt.use_fp16_storage || (opt.use_fp16_packed && dst_fp16.elempack % 4 == 0)))
        {
            int dims = dst_fp16.dims;
            if (dims == 1)
                dst.create(dst_fp16.w, (size_t)(dst_fp16.elempack * 4u), dst_fp16.elempack, opt.blob_allocator);
            if (dims == 2)
                dst.create(dst_fp16.w, dst_fp16.h, (size_t)(dst_fp16.elempack * 4u), dst_fp16.elempack, opt.blob_allocator);
            if (dims == 3)
                dst.create(dst_fp16.w, dst_fp16.h, dst_fp16.c, (size_t)(dst_fp16.elempack * 4u), dst_fp16.elempack, opt.blob_allocator);

            download_post_tensors_fp16.push_back(dst_fp16);
            download_post_tensors.push_back(dst);

            record r;
            r.type = record::TYPE_post_cast_float16_to_float32;
            r.command_buffer = 0;
            r.post_cast_float16_to_float32.download_post_mat_fp16_offset = download_post_tensors_fp16.size() - 1;
            r.post_cast_float16_to_float32.download_post_mat_offset = download_post_tensors.size() - 1;
            delayed_records.push_back(r);
        }
        else
        {
            dst = dst_fp16;
        }
    }
    else
    {
        dst = dst_fp16;
    }
}

int VkCompute::submit_and_wait()
{
    // printf("VkCompute submit_and_wait\n");
    if (!vkdev->info.support_VK_KHR_push_descriptor)
    {
        // printf("start to run begin command buffer\n");
        begin_command_buffer();
        const size_t record_count = delayed_records.size();
        // printf("delayed_records count:%d\n", record_count);

        // handle delayed records
        for (size_t i=0; i<record_count; i++)
        {
            const record& r = delayed_records[i];

            switch (r.type)
            {
                case record::TYPE_copy_buffer:
                {
                    // TODO
                    break;
                }
                case record::TYPE_copy_image:
                {
                    // TODO
                    break;
                }
                case record::TYPE_copy_buffer_to_image:
                {
                    // TODO
                    break;
                }
                case record::TYPE_copy_image_to_buffer:
                {
                    // TODO
                    break;
                }
                case record::TYPE_bind_pipeline:
                {
                    // TODO
                    break;
                }
                case record::TYPE_bind_descriptorsets:
                {
                    // TODO
                    break;
                }
                case record::TYPE_push_constants:
                {
                    // TODO
                    break;
                }
                case record::TYPE_dispatch:
                {
                    // TODO
                    break;
                }
                case record::TYPE_memory_barrers:
                {
                    // TODO
                    break;
                }
                case record::TYPE_buffer_barrers:
                {
                    // TODO
                    break;
                }
                case record::TYPE_image_barrers:
                {
                    // TODO
                    break;
                }
                case record::TYPE_post_download:
                case record::TYPE_post_cast_float16_to_float32:
                default:
                    break;	
            }
        }
    }

    // end command buffer
    {
        end_command_buffer();
    }

    VkQueue compute_queue = vkdev->acquire_queue(vkdev->info.compute_queue_family_index);
    if (compute_queue == 0)
    {
        printf("out of compute queue\n");
        return -1;
    }

    // submit compute
    {
        VkSubmitInfo submitInfo;
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.pNext = 0;
        submitInfo.waitSemaphoreCount = 0;
        submitInfo.pWaitSemaphores = 0;
        submitInfo.pWaitDstStageMask = 0;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &compute_command_buffer;
        submitInfo.signalSemaphoreCount = 0;
        submitInfo.pSignalSemaphores = 0;

        VkResult ret = vkQueueSubmit(compute_queue, 1, &submitInfo, compute_command_fence);
        if (ret != VK_SUCCESS)
        {
            printf("vkQueueSubmit failed %d", ret);
            vkdev->reclaim_queue(vkdev->info.compute_queue_family_index, compute_queue);
            return -1;
        }
    }

    vkdev->reclaim_queue(vkdev->info.compute_queue_family_index, compute_queue);

    // wait
    {
        VkResult ret = vkWaitForFences(vkdev->vkdevice(), 1, &compute_command_fence, VK_TRUE, UINT64_MAX);
        if (ret != VK_SUCCESS)
        {
            printf("vkWaitForFences failed %d", ret);
            return -1;
        }
    }

    // handle delayed post records
    for (size_t i=0; i<delayed_records.size(); i++)
    {
        const record& r = delayed_records[i];

        switch (r.type)
        {
            case record::TYPE_post_download:
            {
                const VkTensor& src = download_post_buffers[r.post_download.download_post_buffer_mat_offset];
                Tensor dst = download_post_tensors_fp16[r.post_download.download_post_mat_fp16_offset];

    //             TLOG_INFO("post_download  %p +%d ~%d  -> %p", src.buffer(), src.buffer_offset(), src.buffer_capacity(), dst.data);

                src.allocator->invalidate(src.data);
                // memcpy(dst.data, src.mapped_ptr(), dst.elem_size * dst.elem_num);
                memcpy(dst.data, src.mapped_ptr(), dst.total() * dst.elemsize);
                break;
            }
            case record::TYPE_post_cast_float16_to_float32:
            {
                // TODO
                printf("submit delayed_records TYPE_post_cast_float16_to_float32, Do nothing, fix me\n");
                break;
            }
            default:
                break;
        }
    }

    delayed_records.clear();

    return 0;
}


int VkCompute::init()
{
    // compute_command_pool
    {
        VkCommandPoolCreateInfo commandPoolCreateInfo;
        commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        commandPoolCreateInfo.pNext = 0;
        commandPoolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        commandPoolCreateInfo.queueFamilyIndex = vkdev->info.compute_queue_family_index;
        VkResult ret = vkCreateCommandPool(vkdev->vkdevice(), &commandPoolCreateInfo, 0, &compute_command_pool);
        if (ret != VK_SUCCESS)
        {
            printf("vkCreateCommandPool failed %d", ret);
            return -1;
        }
    }
    // compute_command_buffer
    {
        VkCommandBufferAllocateInfo commandBufferAllocateInfo;
        commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        commandBufferAllocateInfo.pNext = 0;
        commandBufferAllocateInfo.commandPool = compute_command_pool;
        commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        commandBufferAllocateInfo.commandBufferCount = 1;

        VkResult ret = vkAllocateCommandBuffers(vkdev->vkdevice(), &commandBufferAllocateInfo, &compute_command_buffer);
        if (ret != VK_SUCCESS)
        {
            printf("vkAllocateCommandBuffers failed %d", ret);
            return -1;
        }
    }

    // compute_command_fence
    {
        VkFenceCreateInfo fenceCreateInfo;
        fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceCreateInfo.pNext = 0;
        fenceCreateInfo.flags = 0;

        VkResult ret = vkCreateFence(vkdev->vkdevice(), &fenceCreateInfo, 0, &compute_command_fence);
        if (ret != VK_SUCCESS)
        {
            printf("vkCreateFence failed %d", ret);
            return -1;
        }
    }

    if (vkdev->info.support_VK_KHR_push_descriptor)
    {
        begin_command_buffer();
    }

    return 0;
}

int VkCompute::begin_command_buffer()
{
    VkCommandBufferBeginInfo commandBufferBeginInfo;
    commandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    commandBufferBeginInfo.pNext = 0;
    commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    commandBufferBeginInfo.pInheritanceInfo = 0;

    VkResult ret = vkBeginCommandBuffer(compute_command_buffer, &commandBufferBeginInfo);
    if (ret != VK_SUCCESS)
    {
        printf("vkBeginCommandBuffer failed %d", ret);
        return -1;
    }
    return 0;
}

int VkCompute::end_command_buffer()
{
    VkResult ret = vkEndCommandBuffer(compute_command_buffer);
    if (ret != VK_SUCCESS)
    {
        printf("vkEndCommandBuffer failed %d", ret);
        return -1;
    }

    return 0;
}

void VkCompute::record_pipeline(const Pipeline* pipeline, const std::vector<VkTensor>& bindings, const std::vector<vk_constant_type>& constants, const VkTensor& dispatcher)
{
    record_pipeline(pipeline, bindings, std::vector<VkImageTensor>(), constants, dispatcher);
}

void VkCompute::record_pipeline(const Pipeline* pipeline, const std::vector<VkImageTensor>& bindings, const std::vector<vk_constant_type>& constants, const VkImageTensor& dispatcher)
{
    record_pipeline(pipeline, std::vector<VkTensor>(), bindings, constants, dispatcher);
}

void VkCompute::record_pipeline(const Pipeline* pipeline, const std::vector<VkTensor>& buffer_bindings, const std::vector<VkImageTensor>& image_bindings, const std::vector<vk_constant_type>& constants, const VkTensor& dispatcher)
{
    // Mat dispatcher_mat(dispatcher.w, dispatcher.h, dispatcher.c, (void*)0);

    record_pipeline(pipeline, buffer_bindings, image_bindings, constants, dispatcher.w, dispatcher.h, dispatcher.c);
}

void VkCompute::record_pipeline(const Pipeline* pipeline, const std::vector<VkTensor>& buffer_bindings, const std::vector<VkImageTensor>& image_bindings, const std::vector<vk_constant_type>& constants, const VkImageTensor& dispatcher)
{
    // VkTensor dispatcher_VkTensor(dispatcher.w, dispatcher.h, dispatcher.c, (void*)0);

    record_pipeline(pipeline, buffer_bindings, image_bindings, constants, dispatcher.w, dispatcher.h, dispatcher.c);
}

void VkCompute::record_pipeline(const Pipeline* pipeline, const std::vector<VkTensor>& buffer_bindings, const std::vector<VkImageTensor>& image_bindings, const std::vector<vk_constant_type>& constants, int dispatcher_w, int dispatcher_h, int dispatcher_c)
{
    const int buffer_binding_count = (int)buffer_bindings.size();
    const int image_binding_count = (int)image_bindings.size();
    const int constant_count = (int)constants.size();

    const int binding_count = buffer_binding_count + image_binding_count;

    if (binding_count != pipeline->shader_info.binding_count)
    {
        printf("binding_count not match, expect %d but got %d + %d", pipeline->shader_info.binding_count, buffer_binding_count, image_binding_count);
    }

    if (constant_count != pipeline->shader_info.push_constant_count)
    {
        printf("push_constant_count not match, expect %d but got %d", pipeline->shader_info.push_constant_count, constant_count);
    }

    int buffer_index = 0;
    int image_index = 0;
    for (int i=0; i<binding_count; i++)
    {
        int binding_type = pipeline->shader_info.binding_types[i];

        if (binding_type == 1)
        {
            const VkTensor& binding = buffer_bindings[buffer_index].empty() ? vkdev->get_dummy_buffer() : buffer_bindings[buffer_index];
            buffer_index++;

//             TLOG_INFO("binding #%d buffer = %d %d %d %d @ %lu %d = %p +%ld ~%ld", i, binding.dims, binding.w, binding.h, binding.c, binding.elemsize, binding.elempack, binding.buffer(), binding.buffer_offset(), binding.buffer_capacity());

            if (binding.data->access_flags & VK_ACCESS_SHADER_WRITE_BIT || binding.data->stage_flags != VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT)
            {
                // barrier device any @ compute/null to shader-readwrite @ compute
                VkBufferMemoryBarrier* barriers = new VkBufferMemoryBarrier[1];
                barriers[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
                barriers[0].pNext = 0;
                barriers[0].srcAccessMask = binding.data->access_flags;
                barriers[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
                barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barriers[0].buffer = binding.buffer();
                barriers[0].offset = binding.buffer_offset();
                barriers[0].size = binding.buffer_capacity();

                VkPipelineStageFlags src_stage = binding.data->stage_flags;
                VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

                if (vkdev->info.support_VK_KHR_push_descriptor)
                {
                    vkCmdPipelineBarrier(compute_command_buffer, src_stage, dst_stage, 0, 0, 0, 1, barriers, 0, 0);
                    delete[] barriers;
                }
                else
                {
                    record r;
                    r.type = record::TYPE_buffer_barrers;
                    r.command_buffer = compute_command_buffer;
                    r.buffer_barrers.src_stage = src_stage;
                    r.buffer_barrers.dst_stage = dst_stage;
                    r.buffer_barrers.barrier_count = 1;
                    r.buffer_barrers.barriers = barriers;
                    delayed_records.push_back(r);
                }

                // mark device shader-readwrite @ compute
                binding.data->access_flags = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
                binding.data->stage_flags = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
            }
        }
        else if (binding_type == 2)
        {
            const VkImageTensor& binding = image_bindings[image_index].empty() ? vkdev->get_dummy_image() : image_bindings[image_index];
            image_index++;

//             TLOG_INFO("binding #%d image = %d %d %d %d @ %lu %d = %p +%ld ~%ld %p", i, binding.dims, binding.w, binding.h, binding.c, binding.elemsize, binding.elempack, binding.image(), binding.data->bind_offset, binding.data->bind_capacity, binding.imageview());

            if (binding.data->access_flags & VK_ACCESS_SHADER_WRITE_BIT || binding.data->image_layout != VK_IMAGE_LAYOUT_GENERAL || binding.data->stage_flags != VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT)
            {
                // image layout transform any @ any to shader-write @ compute
                VkImageMemoryBarrier* barriers = new VkImageMemoryBarrier[1];
                barriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                barriers[0].pNext = 0;
                barriers[0].srcAccessMask = binding.data->access_flags;
                barriers[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
                barriers[0].oldLayout = binding.data->image_layout;
                barriers[0].newLayout = VK_IMAGE_LAYOUT_GENERAL;
                barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barriers[0].image = binding.image();
                barriers[0].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                barriers[0].subresourceRange.baseMipLevel = 0;
                barriers[0].subresourceRange.levelCount = 1;
                barriers[0].subresourceRange.baseArrayLayer = 0;
                barriers[0].subresourceRange.layerCount = 1;

                VkPipelineStageFlags src_stage = binding.data->stage_flags;
                VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

                if (vkdev->info.support_VK_KHR_push_descriptor)
                {
                    vkCmdPipelineBarrier(compute_command_buffer, src_stage, dst_stage, 0, 0, 0, 0, 0, 1, barriers);
                    delete[] barriers;
                }
                else
                {
                    record r;
                    r.type = record::TYPE_image_barrers;
                    r.command_buffer = compute_command_buffer;
                    r.image_barrers.src_stage = src_stage;
                    r.image_barrers.dst_stage = dst_stage;
                    r.image_barrers.barrier_count = 1;
                    r.image_barrers.barriers = barriers;
                    delayed_records.push_back(r);
                }

                // mark image shader-write @ compute
                binding.data->access_flags = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
                binding.data->image_layout = VK_IMAGE_LAYOUT_GENERAL;
                binding.data->stage_flags = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
            }

            // image and imageview can not be destroyed until command execution ends
            TENGINE_XADD(&binding.data->command_refcount, 1);
            image_blocks_to_destroy.push_back(binding.data);
        }
        else // if (binding_type == 3)
        {
            const VkImageTensor& binding = image_bindings[image_index].empty() ? vkdev->get_dummy_image() : image_bindings[image_index];
            image_index++;

//             TLOG_INFO("binding #%d sampler = %d %d %d %d @ %lu %d = %p +%ld ~%ld %p", i, binding.dims, binding.w, binding.h, binding.c, binding.elemsize, binding.elempack, binding.image(), binding.data->bind_offset, binding.data->bind_capacity, binding.imageview());

            // if the same image used for both storage image and combined image sampler
            // only apply image layout transition to general
            for (int j=0; j<image_binding_count; j++)
            {
                if (pipeline->shader_info.binding_types[j] == 2 && binding.data == image_bindings[j].data)
                {
                    // the same image is used as storage image, skip it
                    continue;
                }
            }

            if (binding.data->access_flags & VK_ACCESS_SHADER_WRITE_BIT || binding.data->image_layout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL || binding.data->stage_flags != VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT)
            {
                // image layout transform any @ any to shader-readonly-optimal @ compute
                VkImageMemoryBarrier* barriers = new VkImageMemoryBarrier[1];
                barriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                barriers[0].pNext = 0;
                barriers[0].srcAccessMask = binding.data->access_flags;
                barriers[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                barriers[0].oldLayout = binding.data->image_layout;
                barriers[0].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barriers[0].image = binding.image();
                barriers[0].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                barriers[0].subresourceRange.baseMipLevel = 0;
                barriers[0].subresourceRange.levelCount = 1;
                barriers[0].subresourceRange.baseArrayLayer = 0;
                barriers[0].subresourceRange.layerCount = 1;

                VkPipelineStageFlags src_stage = binding.data->stage_flags;
                VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

                if (vkdev->info.support_VK_KHR_push_descriptor)
                {
                    vkCmdPipelineBarrier(compute_command_buffer, src_stage, dst_stage, 0, 0, 0, 0, 0, 1, barriers);
                    delete[] barriers;
                }
                else
                {
                    record r;
                    r.type = record::TYPE_image_barrers;
                    r.command_buffer = compute_command_buffer;
                    r.image_barrers.src_stage = src_stage;
                    r.image_barrers.dst_stage = dst_stage;
                    r.image_barrers.barrier_count = 1;
                    r.image_barrers.barriers = barriers;
                    delayed_records.push_back(r);
                }

                // mark image shader-readonly-optimal @ compute
                binding.data->access_flags = VK_ACCESS_SHADER_READ_BIT;
                binding.data->image_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                binding.data->stage_flags = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
            }

            // image and imageview can not be destroyed until command execution ends
            TENGINE_XADD(&binding.data->command_refcount, 1);
            image_blocks_to_destroy.push_back(binding.data);
        }
    }
    // record bind pipeline
    {
        if (vkdev->info.support_VK_KHR_push_descriptor)
        {
            vkCmdBindPipeline(compute_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline);
        }
        else
        {
            record r;
            r.type = record::TYPE_bind_pipeline;
            r.command_buffer = compute_command_buffer;
            r.bind_pipeline.bind_point = VK_PIPELINE_BIND_POINT_COMPUTE;
            r.bind_pipeline.pipeline = pipeline->pipeline;
            delayed_records.push_back(r);
        }
    }

    // record update bindings
    if (binding_count > 0)
    {
        std::vector<unsigned char> descriptorInfos;
        {
            descriptorInfos.resize(sizeof(VkDescriptorBufferInfo) * buffer_binding_count + sizeof(VkDescriptorImageInfo) * image_binding_count);

            unsigned char* p_descriptorInfos = descriptorInfos.data();
            int descriptorBufferInfo_index = 0;
            int descriptorImageInfo_index = 0;
            for (int i=0; i<binding_count; i++)
            {
                int binding_type = pipeline->shader_info.binding_types[i];

                if (binding_type == 1)
                {
                    const VkTensor& binding = buffer_bindings[descriptorBufferInfo_index].empty() ? vkdev->get_dummy_buffer() : buffer_bindings[descriptorBufferInfo_index];
                    descriptorBufferInfo_index++;

                    VkDescriptorBufferInfo descriptorBufferInfo;
                    descriptorBufferInfo.buffer = binding.buffer();
                    descriptorBufferInfo.offset = binding.buffer_offset();
                    descriptorBufferInfo.range = binding.total() * binding.elemsize;

                    memcpy(p_descriptorInfos, &descriptorBufferInfo, sizeof(VkDescriptorBufferInfo));
                    p_descriptorInfos += sizeof(VkDescriptorBufferInfo);
                }
                else //if (binding_type == 2 || binding_type == 3)
                {
                    const VkImageTensor& binding = image_bindings[descriptorImageInfo_index].empty() ? vkdev->get_dummy_image() : image_bindings[descriptorImageInfo_index];
                    descriptorImageInfo_index++;

                    // we always use immutable nearest sampler set in descroptor layout during pipeline creation
                    VkDescriptorImageInfo descriptorImageInfo;
                    descriptorImageInfo.sampler = 0;
                    descriptorImageInfo.imageView = binding.imageview();
                    descriptorImageInfo.imageLayout = binding.data->image_layout;

                    memcpy(p_descriptorInfos, &descriptorImageInfo, sizeof(VkDescriptorImageInfo));
                    p_descriptorInfos += sizeof(VkDescriptorImageInfo);
                }
            }
        }

        if (vkdev->info.support_VK_KHR_push_descriptor)
        {
            vkdev->vkCmdPushDescriptorSetWithTemplateKHR(compute_command_buffer, pipeline->descriptor_update_template, pipeline->pipeline_layout, 0, descriptorInfos.data());
        }
        else
        {
            // create new descriptor_pool and descriptorset
            VkDescriptorPool descriptor_pool;
            {
                int image_binding_count = 0;
                int sampler_binding_count = 0;
                for (int i=0; i<binding_count; i++)
                {
                    int binding_type = pipeline->shader_info.binding_types[i];

                    if (binding_type == 2)
                        image_binding_count++;
                    else // if (binding_type == 3)
                        sampler_binding_count++;
                }

                VkDescriptorPoolSize poolSizes[3];
                poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                poolSizes[0].descriptorCount = buffer_binding_count;
                poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                poolSizes[1].descriptorCount = image_binding_count;
                poolSizes[2].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                poolSizes[2].descriptorCount = sampler_binding_count;

                VkDescriptorPoolCreateInfo descriptorPoolCreateInfo;
                descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
                descriptorPoolCreateInfo.pNext = 0;
                descriptorPoolCreateInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
                descriptorPoolCreateInfo.maxSets = 1;
                descriptorPoolCreateInfo.poolSizeCount = 3;
                descriptorPoolCreateInfo.pPoolSizes = poolSizes;

                VkResult ret = vkCreateDescriptorPool(vkdev->vkdevice(), &descriptorPoolCreateInfo, 0, &descriptor_pool);
                if (ret != VK_SUCCESS)
                {
                    printf("vkCreateDescriptorPool failed %d", ret);
                    return;
                }
            }
            descriptor_pools.push_back(descriptor_pool);

            VkDescriptorSet descriptorset;
            {
                VkDescriptorSetAllocateInfo descriptorSetAllocateInfo;
                descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                descriptorSetAllocateInfo.pNext = 0;
                descriptorSetAllocateInfo.descriptorPool = descriptor_pool;
                descriptorSetAllocateInfo.descriptorSetCount = 1;
                descriptorSetAllocateInfo.pSetLayouts = &pipeline->descriptorset_layout;

                VkResult ret = vkAllocateDescriptorSets(vkdev->vkdevice(), &descriptorSetAllocateInfo, &descriptorset);
                if (ret != VK_SUCCESS)
                {
                    printf("vkAllocateDescriptorSets failed %d", ret);
                    return;
                }
            }
            descriptorsets.push_back(descriptorset);

            if (vkdev->info.support_VK_KHR_descriptor_update_template)
            {
                vkdev->vkUpdateDescriptorSetWithTemplateKHR(vkdev->vkdevice(), descriptorset, pipeline->descriptor_update_template, descriptorInfos.data());
            }
            else
            {
                std::vector<VkWriteDescriptorSet> writeDescriptorSets(binding_count);
                {
                    const unsigned char* p_descriptorInfos = descriptorInfos.data();
                    for (int i=0; i<binding_count; i++)
                    {
                        int binding_type = pipeline->shader_info.binding_types[i];

                        writeDescriptorSets[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                        writeDescriptorSets[i].pNext = 0;
                        writeDescriptorSets[i].dstSet = descriptorset;
                        writeDescriptorSets[i].dstBinding = i;
                        writeDescriptorSets[i].dstArrayElement = 0;
                        writeDescriptorSets[i].descriptorCount = 1;
                        writeDescriptorSets[i].pTexelBufferView = 0;

                        if (binding_type == 1)
                        {
                            writeDescriptorSets[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                            writeDescriptorSets[i].pImageInfo = 0;
                            writeDescriptorSets[i].pBufferInfo = (const VkDescriptorBufferInfo*)p_descriptorInfos;

                            p_descriptorInfos += sizeof(VkDescriptorBufferInfo);
                        }
                        else if (binding_type == 2)
                        {
                            writeDescriptorSets[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                            writeDescriptorSets[i].pImageInfo = (const VkDescriptorImageInfo*)p_descriptorInfos;
                            writeDescriptorSets[i].pBufferInfo = 0;

                            p_descriptorInfos += sizeof(VkDescriptorImageInfo);
                        }
                        else // if (binding_type == 3)
                        {
                            writeDescriptorSets[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                            writeDescriptorSets[i].pImageInfo = (const VkDescriptorImageInfo*)p_descriptorInfos;
                            writeDescriptorSets[i].pBufferInfo = 0;

                            p_descriptorInfos += sizeof(VkDescriptorImageInfo);
                        }
                    }
                }

                vkUpdateDescriptorSets(vkdev->vkdevice(), binding_count, writeDescriptorSets.data(), 0, 0);
            }

            record r;
            r.type = record::TYPE_bind_descriptorsets;
            r.command_buffer = compute_command_buffer;
            r.bind_descriptorsets.bind_point = VK_PIPELINE_BIND_POINT_COMPUTE;
            r.bind_descriptorsets.pipeline_layout = pipeline->pipeline_layout;
            r.bind_descriptorsets.descriptorset_count = 1;
            r.bind_descriptorsets.descriptorset_offset = descriptorsets.size() - 1;
            delayed_records.push_back(r);
        }
    }

    // record push constants
    if (constant_count > 0)
    {
        if (vkdev->info.support_VK_KHR_push_descriptor)
        {
            vkCmdPushConstants(compute_command_buffer, pipeline->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, constant_count * sizeof(vk_constant_type), constants.data());
        }
        else
        {
            uint32_t size = constant_count * sizeof(vk_constant_type);
            unsigned char* constant_values = new unsigned char[size];
            memcpy(constant_values, constants.data(), size);

            record r;
            r.type = record::TYPE_push_constants;
            r.command_buffer = compute_command_buffer;
            r.push_constants.pipeline_layout = pipeline->pipeline_layout;
            r.push_constants.stage_flags = VK_SHADER_STAGE_COMPUTE_BIT;
            r.push_constants.size = size;
            r.push_constants.values = constant_values;
            delayed_records.push_back(r);
        }
    }

    // record dispatch
    {
        uint32_t group_count_x = (dispatcher_w + pipeline->local_size_x - 1) / pipeline->local_size_x;
        uint32_t group_count_y = (dispatcher_h + pipeline->local_size_y - 1) / pipeline->local_size_y;
        uint32_t group_count_z = (dispatcher_c + pipeline->local_size_z - 1) / pipeline->local_size_z;

        if (vkdev->info.support_VK_KHR_push_descriptor)
        {
            vkCmdDispatch(compute_command_buffer, group_count_x, group_count_y, group_count_z);
        }
        else
        {
            record r;
            r.type = record::TYPE_dispatch;
            r.command_buffer = compute_command_buffer;
            r.dispatch.group_count_x = group_count_x;
            r.dispatch.group_count_y = group_count_y;
            r.dispatch.group_count_z = group_count_z;
            delayed_records.push_back(r);
        }
    }
}

VkTransfer::VkTransfer(const GPUDevice* _vkdev) : vkdev(_vkdev)
{
    compute_command_pool = 0;
    transfer_command_pool = 0;

    upload_command_buffer = 0;
    compute_command_buffer = 0;

    upload_compute_semaphore = 0;

    upload_command_fence = 0;
    compute_command_fence = 0;

    init();
}

VkTransfer::~VkTransfer()
{
    vkDestroyFence(vkdev->vkdevice(), compute_command_fence, 0);

    vkFreeCommandBuffers(vkdev->vkdevice(), compute_command_pool, 1, &compute_command_buffer);
    vkDestroyCommandPool(vkdev->vkdevice(), compute_command_pool, 0);

    if (!vkdev->info.unified_compute_transfer_queue)
    {
        vkDestroyFence(vkdev->vkdevice(), upload_command_fence, 0);

        vkDestroySemaphore(vkdev->vkdevice(), upload_compute_semaphore, 0);

        vkFreeCommandBuffers(vkdev->vkdevice(), transfer_command_pool, 1, &upload_command_buffer);
        vkDestroyCommandPool(vkdev->vkdevice(), transfer_command_pool, 0);
    }
}

int VkTransfer::init()
{
    // compute_command_pool
    {
        VkCommandPoolCreateInfo commandPoolCreateInfo;
        commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        commandPoolCreateInfo.pNext = 0;
        commandPoolCreateInfo.flags = 0;
        commandPoolCreateInfo.queueFamilyIndex = vkdev->info.compute_queue_family_index;

        VkResult ret = vkCreateCommandPool(vkdev->vkdevice(), &commandPoolCreateInfo, 0, &compute_command_pool);
        if (ret != VK_SUCCESS)
        {
            printf("vkCreateCommandPool failed %d", ret);
            return -1;
        }
    }

    // compute_command_buffer
    {
        VkCommandBufferAllocateInfo commandBufferAllocateInfo;
        commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        commandBufferAllocateInfo.pNext = 0;
        commandBufferAllocateInfo.commandPool = compute_command_pool;
        commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        commandBufferAllocateInfo.commandBufferCount = 1;

        VkResult ret = vkAllocateCommandBuffers(vkdev->vkdevice(), &commandBufferAllocateInfo, &compute_command_buffer);
        if (ret != VK_SUCCESS)
        {
            printf("vkAllocateCommandBuffers failed %d", ret);
            return -1;
        }
    }

    // compute_command_fence
    {
        VkFenceCreateInfo fenceCreateInfo;
        fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceCreateInfo.pNext = 0;
        fenceCreateInfo.flags = 0;

        VkResult ret = vkCreateFence(vkdev->vkdevice(), &fenceCreateInfo, 0, &compute_command_fence);
        if (ret != VK_SUCCESS)
        {
            printf("vkCreateFence failed %d", ret);
            return -1;
        } 
    }

    if (!vkdev->info.unified_compute_transfer_queue)
    {
        // transfer_command_pool
        {
            VkCommandPoolCreateInfo commandPoolCreateInfo;
            commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            commandPoolCreateInfo.pNext = 0;
            commandPoolCreateInfo.flags = 0;
            commandPoolCreateInfo.queueFamilyIndex = vkdev->info.transfer_queue_family_index;

            VkResult ret = vkCreateCommandPool(vkdev->vkdevice(), &commandPoolCreateInfo, 0, &transfer_command_pool);
            if (ret != VK_SUCCESS)
            {
                printf("vkCreateCommandPool failed %d", ret);
                return -1;
            }
        }

    // upload_command_buffer
    {
            VkCommandBufferAllocateInfo commandBufferAllocateInfo;
            commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            commandBufferAllocateInfo.pNext = 0;
            commandBufferAllocateInfo.commandPool = transfer_command_pool;
            commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            commandBufferAllocateInfo.commandBufferCount = 1;

            VkResult ret = vkAllocateCommandBuffers(vkdev->vkdevice(), &commandBufferAllocateInfo, &upload_command_buffer);
            if (ret != VK_SUCCESS)
            {
                printf("vkAllocateCommandBuffers failed %d", ret);
                return -1;
            }
    }

    // upload_compute_semaphore
    {
            VkSemaphoreCreateInfo semaphoreCreateInfo;
            semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
            semaphoreCreateInfo.pNext = 0;
            semaphoreCreateInfo.flags = 0;

            VkResult ret = vkCreateSemaphore(vkdev->vkdevice(), &semaphoreCreateInfo, 0, &upload_compute_semaphore);

        if (ret != VK_SUCCESS)
        {
                printf("vkCreateSemaphore failed %d", ret);
        return -1;
        }
    }

    // upload_command_fence
    {
            VkFenceCreateInfo fenceCreateInfo;
            fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            fenceCreateInfo.pNext = 0;
            fenceCreateInfo.flags = 0;

            VkResult ret = vkCreateFence(vkdev->vkdevice(), &fenceCreateInfo, 0, &upload_command_fence);

        if (ret != VK_SUCCESS)
            {
                printf("vkCreateFence failed %d", ret);
                return -1;
        }
    }
    }

    begin_command_buffer();

    return 0;
}

int VkTransfer::begin_command_buffer()
{
    {
        VkCommandBufferBeginInfo commandBufferBeginInfo;
        commandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        commandBufferBeginInfo.pNext = 0;
        commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        commandBufferBeginInfo.pInheritanceInfo = 0;

        VkResult ret = vkBeginCommandBuffer(compute_command_buffer, &commandBufferBeginInfo);
        if (ret != VK_SUCCESS)
        {
            printf("vkBeginCommandBuffer failed %d", ret);
            return -1;
        }
    }

    if (!vkdev->info.unified_compute_transfer_queue)
    {
        {
            VkCommandBufferBeginInfo commandBufferBeginInfo;
            commandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            commandBufferBeginInfo.pNext = 0;
            commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            commandBufferBeginInfo.pInheritanceInfo = 0;

            VkResult ret = vkBeginCommandBuffer(upload_command_buffer, &commandBufferBeginInfo);
            if (ret != VK_SUCCESS)
            {
                printf("vkBeginCommandBuffer failed %d", ret);
                return -1;
            }
        }
    }
    return 0;
}


int VkTransfer::end_command_buffer()
{
    {
        VkResult ret = vkEndCommandBuffer(compute_command_buffer);
        if (ret != VK_SUCCESS)
        {
            printf("vkEndCommandBuffer failed %d", ret);
            return -1;
        }
    }

    if (!vkdev->info.unified_compute_transfer_queue)
    {
        {
            VkResult ret = vkEndCommandBuffer(upload_command_buffer);
            if (ret != VK_SUCCESS)
            {
                printf("vkEndCommandBuffer failed %d", ret);
                return -1;
            }
        }
    }
    return 0;
}

int VkTransfer::submit_and_wait()
{
    // end command buffer
    {
        end_command_buffer();
    }

    VkQueue compute_queue = vkdev->acquire_queue(vkdev->info.compute_queue_family_index);
    if (compute_queue == 0)
    {
        printf("out of compute queue");
        return -1;
    }

    if (vkdev->info.unified_compute_transfer_queue)
    {
        // submit compute
        {
            VkSubmitInfo submitInfo;
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.pNext = 0;
            submitInfo.waitSemaphoreCount = 0;
            submitInfo.pWaitSemaphores = 0;
            submitInfo.pWaitDstStageMask = 0;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &compute_command_buffer;
            submitInfo.signalSemaphoreCount = 0;
            submitInfo.pSignalSemaphores = 0;

            VkResult ret = vkQueueSubmit(compute_queue, 1, &submitInfo, compute_command_fence);
            if (ret != VK_SUCCESS)
            {
                printf("vkQueueSubmit failed %d", ret);
                vkdev->reclaim_queue(vkdev->info.compute_queue_family_index, compute_queue);
                return -1;
            }
        }
    }
    else
    {
        VkQueue transfer_queue = vkdev->acquire_queue(vkdev->info.transfer_queue_family_index);
        if (transfer_queue == 0)
        {
            printf("out of transfer queue");
            vkdev->reclaim_queue(vkdev->info.compute_queue_family_index, compute_queue);
            return -1;
        }

        // submit upload compute
        {
            VkSubmitInfo submitInfo;
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.pNext = 0;
            submitInfo.waitSemaphoreCount = 0;
            submitInfo.pWaitSemaphores = 0;
            submitInfo.pWaitDstStageMask = 0;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &upload_command_buffer;
            submitInfo.signalSemaphoreCount = 1;
            submitInfo.pSignalSemaphores = &upload_compute_semaphore;

            VkResult ret = vkQueueSubmit(transfer_queue, 1, &submitInfo, upload_command_fence);
            if (ret != VK_SUCCESS)
            {
                printf("vkQueueSubmit failed %d", ret);
                vkdev->reclaim_queue(vkdev->info.transfer_queue_family_index, transfer_queue);
                vkdev->reclaim_queue(vkdev->info.compute_queue_family_index, compute_queue);
                return -1;
            }
        }
        
        {
            VkPipelineStageFlags wait_dst_stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;// FIXME
            VkSubmitInfo submitInfo;
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.pNext = 0;
            submitInfo.waitSemaphoreCount = 1;
            submitInfo.pWaitSemaphores = &upload_compute_semaphore;
            submitInfo.pWaitDstStageMask = &wait_dst_stage;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &compute_command_buffer;
            submitInfo.signalSemaphoreCount = 0;
            submitInfo.pSignalSemaphores = 0;

            VkResult ret = vkQueueSubmit(compute_queue, 1, &submitInfo, compute_command_fence);

            if (ret != VK_SUCCESS)
            {
                printf("vkQueueSubmit failed %d", ret);
                vkdev->reclaim_queue(vkdev->info.transfer_queue_family_index, transfer_queue);
                vkdev->reclaim_queue(vkdev->info.compute_queue_family_index, compute_queue);
                return -1;
            }
        }
        
        vkdev->reclaim_queue(vkdev->info.transfer_queue_family_index, transfer_queue);
    }
    vkdev->reclaim_queue(vkdev->info.compute_queue_family_index, compute_queue);
    
    // wait
    if (vkdev->info.unified_compute_transfer_queue)
    {
        VkResult ret = vkWaitForFences(vkdev->vkdevice(), 1, &compute_command_fence, VK_TRUE, UINT64_MAX);
        if (ret != VK_SUCCESS)
        {
            printf("vkWaitForFences failed %d", ret);
            return -1;
        }
    }
    else
    {
        VkFence fences[2] = { upload_command_fence, compute_command_fence };

        VkResult ret = vkWaitForFences(vkdev->vkdevice(), 2, fences, VK_TRUE, UINT64_MAX);
        if (ret != VK_SUCCESS)
        {
            printf("vkWaitForFences failed %d", ret);
            return -1;
        }
    }
    return 0;
}

void VkTransfer::record_upload(const Tensor& src, VkTensor& dst, const Option& opt)
{
//     TLOG_INFO("record_upload src = %d | %d %d %d @ %d", src.dims, src.w, src.h, src.c, src.elempack);

    // NOTE keep the hack here ?
    if (src.elemsize == src.elempack * 4u)
    {
        if (opt.use_fp16_storage || (opt.use_fp16_packed && src.elempack % 4 == 0))
        {
            // printf("VkTransfer record_upload, cast fp32 to fp16, need to be done, fix me\n");
            Tensor src_fp16;
            TEngine::cast_float32_to_float16(src, src_fp16);
            record_upload(src_fp16, dst, opt);

            return;
        }
    }

    Tensor src_flattened = src.reshape(src.w * src.h * src.c);

    // create dst
    dst.create_like(src_flattened, opt.blob_vkallocator);

    if (dst.empty())
    {
        return;
    }

    if (dst.allocator->mappable)
    {
        // memcpy src_flattened to device
        memcpy(dst.mapped_ptr(), src_flattened.data, src_flattened.total() * src_flattened.elemsize);
        dst.allocator->flush(dst.data);

        // barrier device host-write @ null to shader-read @ compute
        {
            VkBufferMemoryBarrier barrier;
            barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            barrier.pNext = 0;
            barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.buffer = dst.buffer();
            barrier.offset = dst.buffer_offset();
            barrier.size = dst.buffer_capacity();

            VkPipelineStageFlags src_stage = VK_PIPELINE_STAGE_HOST_BIT;
            VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

            vkCmdPipelineBarrier(compute_command_buffer, src_stage, dst_stage, 0, 0, 0, 1, &barrier, 0, 0);
        }

        // mark device shader-readwrite @ compute
        dst.data->access_flags = VK_ACCESS_SHADER_READ_BIT;
        dst.data->stage_flags = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

        return;
    }

    // create staging
    VkTensor dst_staging;
    dst_staging.create_like(src_flattened, opt.staging_vkallocator);

    // memcpy src_flattened to staging
    memcpy(dst_staging.mapped_ptr(), src_flattened.data, src_flattened.total() * src_flattened.elemsize);
    dst_staging.allocator->flush(dst_staging.data);

    VkCommandBuffer command_buffer;
    if (vkdev->info.unified_compute_transfer_queue)
    {
        command_buffer = compute_command_buffer;
    }
    else
    {
        command_buffer = upload_command_buffer;
    }

    // barrier staging host-write @ null to transfer-read @ queue
    {
        VkBufferMemoryBarrier barrier;
        barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barrier.pNext = 0;
        barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.buffer = dst_staging.buffer();
        barrier.offset = dst_staging.buffer_offset();
        barrier.size = dst_staging.buffer_capacity();

        VkPipelineStageFlags src_stage = VK_PIPELINE_STAGE_HOST_BIT;
        VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;

        vkCmdPipelineBarrier(command_buffer, src_stage, dst_stage, 0, 0, 0, 1, &barrier, 0, 0);
    }

    // record staging to device
    {
        VkBufferCopy region;
        region.srcOffset = dst_staging.buffer_offset();
        region.dstOffset = dst.buffer_offset();
        region.size = std::min(dst_staging.buffer_capacity(), dst.buffer_capacity());

        vkCmdCopyBuffer(command_buffer, dst_staging.buffer(), dst.buffer(), 1, &region);
    }

    if (vkdev->info.unified_compute_transfer_queue)
    {
        // barrier device transfer-write @ compute to shader-read @ compute
        {
            VkBufferMemoryBarrier barrier;
            barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            barrier.pNext = 0;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.buffer = dst.buffer();
            barrier.offset = dst.buffer_offset();
            barrier.size = dst.buffer_capacity();

            VkPipelineStageFlags src_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

            vkCmdPipelineBarrier(command_buffer, src_stage, dst_stage, 0, 0, 0, 1, &barrier, 0, 0);
        }
    }
    else
    {
        // queue ownership transfer transfer-write @ transfer to shader-read @ compute

        // release
        {
            VkBufferMemoryBarrier barrier;
            barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            barrier.pNext = 0;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = 0;
            barrier.srcQueueFamilyIndex = vkdev->info.transfer_queue_family_index;
            barrier.dstQueueFamilyIndex = vkdev->info.compute_queue_family_index;
            barrier.buffer = dst.buffer();
            barrier.offset = dst.buffer_offset();
            barrier.size = dst.buffer_capacity();

            VkPipelineStageFlags src_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;

            vkCmdPipelineBarrier(upload_command_buffer, src_stage, dst_stage, 0, 0, 0, 1, &barrier, 0, 0);
        }

        // acquire
        {
            VkBufferMemoryBarrier barrier;
            barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            barrier.pNext = 0;
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            barrier.srcQueueFamilyIndex = vkdev->info.transfer_queue_family_index;
            barrier.dstQueueFamilyIndex = vkdev->info.compute_queue_family_index;
            barrier.buffer = dst.buffer();
            barrier.offset = dst.buffer_offset();
            barrier.size = dst.buffer_capacity();

            VkPipelineStageFlags src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

            vkCmdPipelineBarrier(compute_command_buffer, src_stage, dst_stage, 0, 0, 0, 1, &barrier, 0, 0);
        }
    }

    // mark device shader-readwrite @ compute
    dst.data->access_flags = VK_ACCESS_SHADER_READ_BIT;
    dst.data->stage_flags = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

    // stash staging
    upload_staging_buffers.push_back(dst_staging);
}

void VkTransfer::record_upload(const tensor* src, VkTensor& dst, const Option& opt)
{
//     TLOG_INFO("record_upload src = %d | %d %d %d @ %d", src.dims, src.w, src.h, src.c, src.elempack);

    // NOTE keep the hack here ?
    // printf("elem size: %d, elempack:%d\n", src.elemsize, src.elempack);
    if (src->elem_size == opt.elempack * 4u)
    {
        if (opt.use_fp16_storage || (opt.use_fp16_packed && opt.elempack % 4 == 0))
        {
            printf("VkTransfer record_upload, cast fp32 to fp16, need to be done, fix me\n");
            // Mat src_fp16;
            // cast_float32_to_float16(src, src_fp16);

            // record_upload(src_fp16, dst, opt);

            return;
        }
    }

    // Mat src_flattened = src.reshape(src.w * src.h * src.c);

    // create dst
    // dst.create_like(src_flattened, opt.blob_vkallocator);
    // int elemnum = src->elem_num;    //  src->GetTotalSize()/sizeof(float);
    dst.create(src->elem_num, src->elem_size, opt.blob_vkallocator);

    if (dst.empty())
    {
        return;
    }

    if (dst.allocator->mappable)
    {
        // memcpy src_flattened to device
        memcpy(dst.mapped_ptr(), src->data, src->elem_num * src->elem_size);
        dst.allocator->flush(dst.data);

        // barrier device host-write @ null to shader-read @ compute
        {
            VkBufferMemoryBarrier barrier;
            barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            barrier.pNext = 0;
            barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.buffer = dst.buffer();
            barrier.offset = dst.buffer_offset();
            barrier.size = dst.buffer_capacity();

            VkPipelineStageFlags src_stage = VK_PIPELINE_STAGE_HOST_BIT;
            VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

            vkCmdPipelineBarrier(compute_command_buffer, src_stage, dst_stage, 0, 0, 0, 1, &barrier, 0, 0);
        }

        // mark device shader-readwrite @ compute
        dst.data->access_flags = VK_ACCESS_SHADER_READ_BIT;
        dst.data->stage_flags = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

        return;
    }

    printf("run create staging\n");
    // create staging
    VkTensor dst_staging;
    dst_staging.create(src->elem_num, src->elem_size, opt.staging_vkallocator);
    // dst_staging.create_like(src_flattened, opt.staging_vkallocator);

    // memcpy src_flattened to staging
    memcpy(dst_staging.mapped_ptr(), src->data, src->elem_num * src->elem_size);
    dst_staging.allocator->flush(dst_staging.data);

    VkCommandBuffer command_buffer;
    if (vkdev->info.unified_compute_transfer_queue)
    {
        command_buffer = compute_command_buffer;
    }
    else
    {
        command_buffer = upload_command_buffer;
    }

    // barrier staging host-write @ null to transfer-read @ queue
    {
        VkBufferMemoryBarrier barrier;
        barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barrier.pNext = 0;
        barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.buffer = dst_staging.buffer();
        barrier.offset = dst_staging.buffer_offset();
        barrier.size = dst_staging.buffer_capacity();

        VkPipelineStageFlags src_stage = VK_PIPELINE_STAGE_HOST_BIT;
        VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;

        vkCmdPipelineBarrier(command_buffer, src_stage, dst_stage, 0, 0, 0, 1, &barrier, 0, 0);
    }

    // record staging to device
    {
        VkBufferCopy region;
        region.srcOffset = dst_staging.buffer_offset();
        region.dstOffset = dst.buffer_offset();
        region.size = std::min(dst_staging.buffer_capacity(), dst.buffer_capacity());

        vkCmdCopyBuffer(command_buffer, dst_staging.buffer(), dst.buffer(), 1, &region);
    }

    if (vkdev->info.unified_compute_transfer_queue)
    {
        // barrier device transfer-write @ compute to shader-read @ compute
        {
            VkBufferMemoryBarrier barrier;
            barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            barrier.pNext = 0;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.buffer = dst.buffer();
            barrier.offset = dst.buffer_offset();
            barrier.size = dst.buffer_capacity();

            VkPipelineStageFlags src_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

            vkCmdPipelineBarrier(command_buffer, src_stage, dst_stage, 0, 0, 0, 1, &barrier, 0, 0);
        }
    }
    else
    {
        // queue ownership transfer transfer-write @ transfer to shader-read @ compute

        // release
        {
            VkBufferMemoryBarrier barrier;
            barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            barrier.pNext = 0;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = 0;
            barrier.srcQueueFamilyIndex = vkdev->info.transfer_queue_family_index;
            barrier.dstQueueFamilyIndex = vkdev->info.compute_queue_family_index;
            barrier.buffer = dst.buffer();
            barrier.offset = dst.buffer_offset();
            barrier.size = dst.buffer_capacity();

            VkPipelineStageFlags src_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;

            vkCmdPipelineBarrier(upload_command_buffer, src_stage, dst_stage, 0, 0, 0, 1, &barrier, 0, 0);
        }

        // acquire
        {
            VkBufferMemoryBarrier barrier;
            barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            barrier.pNext = 0;
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            barrier.srcQueueFamilyIndex = vkdev->info.transfer_queue_family_index;
            barrier.dstQueueFamilyIndex = vkdev->info.compute_queue_family_index;
            barrier.buffer = dst.buffer();
            barrier.offset = dst.buffer_offset();
            barrier.size = dst.buffer_capacity();

            VkPipelineStageFlags src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

            vkCmdPipelineBarrier(compute_command_buffer, src_stage, dst_stage, 0, 0, 0, 1, &barrier, 0, 0);
        }
    }

    // mark device shader-readwrite @ compute
    dst.data->access_flags = VK_ACCESS_SHADER_READ_BIT;
    dst.data->stage_flags = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

    // stash staging
    upload_staging_buffers.push_back(dst_staging);
}

} // namespace TEngine
