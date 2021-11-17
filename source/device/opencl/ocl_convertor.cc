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
 * Author: hbshi@openailab.com
 */

#include "ocl_cpp_helper.hpp"
#include "ocl_executor.hpp"
#include "ocl_convertor.hpp"
extern "C" {
#include "operator/op.h"
#include "convolution_param.h"
}

bool ocl_convertor::nchw_buffer_to_image(struct tensor* ir_tensor, cl::Buffer* input, cl::Image* output, bool needWait)
{
    int N, C, H, W;
    N = ir_tensor->dims[0];
    C = ir_tensor->dims[1];
    H = ir_tensor->dims[2];
    W = ir_tensor->dims[3];
    size_t image_width = UP_DIV(C, 4) * W;
    size_t image_height = N * H;
    uint32_t output_global_work_size[2] = {(uint32_t)image_width, (uint32_t)image_height};
    if (nchw_buffer_image_kernel.get() == nullptr)
    {
        std::set<std::string> build_options;
        build_options.emplace("-DBUFFER_IMAGE_IO_TRANS");
        nchw_buffer_image_kernel = engine->build_kernel("buffer_to_image", "nchw_buffer_to_image", build_options);
    }
    uint32_t idx = 0;
    int err = nchw_buffer_image_kernel.setArg(idx++, output_global_work_size[0]);
    err = nchw_buffer_image_kernel.setArg(idx++, output_global_work_size[1]);
    err = nchw_buffer_image_kernel.setArg(idx++, *input);
    err = nchw_buffer_image_kernel.setArg(idx++, H);
    err = nchw_buffer_image_kernel.setArg(idx++, W);
    err = nchw_buffer_image_kernel.setArg(idx++, C);
    err = nchw_buffer_image_kernel.setArg(idx++, *output);
    if (err != CL_SUCCESS)
    {
        TLOG_ERR("nchw_buffer_to_image err: %d \n", err);
    }

    auto max_work_group_size = engine->get_max_work_group_size(nchw_buffer_image_kernel);
    std::vector<uint32_t> lws = {16, std::max((uint32_t)1, (uint32_t)max_work_group_size / 16)};
    std::vector<uint32_t> round_up_group_work_size(lws.size());
    for (size_t i = 0; i < lws.size(); ++i)
    {
        round_up_group_work_size[i] = ROUND_UP(output_global_work_size[i], lws[i]);
    }
    cl::Event event;
    engine->get_command_queue().enqueueNDRangeKernel(nchw_buffer_image_kernel, cl::NullRange,
                                                     cl::NDRange(round_up_group_work_size[0], round_up_group_work_size[1]),
                                                     cl::NDRange(lws[0], lws[1]), nullptr, &event);

#if 0
    printf("----------- after upload-------- %lld\n", output);
    std::vector<float> debugData;
    debugData.resize(image_width * image_height * 4);
    engine->get_command_queue().enqueueReadImage(*output,
                                                 CL_TRUE, {0, 0, 0}, {image_width, image_height, 1}, image_width * sizeof(float) * 4,
                                                 0, debugData.data());
    int debugIndex = 0;
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < image_width; ++j)
        {
            for (int k = 0; k < 4; ++k)
            {
                printf("%.4f,", debugData[debugIndex]);
                debugIndex++;
            }
            printf("  ");
        }
        printf("\n");
    }
#endif

    return true;
}

bool ocl_convertor::buffer_to_image(cl::Buffer* input, cl::Image* output, int w, int h)
{
    if (buffer_to_image_kernel.get() == nullptr)
    {
        std::set<std::string> buildOptions;
        buildOptions.emplace("-DBUFFER_INP_FP32");
        buffer_to_image_kernel = engine->build_kernel("copy_buffer_to_image2d", "copy_buffer_to_image2d", buildOptions);
    }
    //    cl::Kernel bias_kernel;
    //    std::set<std::string> buildOptions;
    //    buildOptions.emplace("-DBUFFER_INP_FP32");
    //    buffer_to_image_kernel = engine->build_kernel("copy_buffer_to_image2d", "copy_buffer_to_image2d", buildOptions);
    auto status = buffer_to_image_kernel.setArg(0, *input);
    if (status != CL_SUCCESS)
    {
        TLOG_ERR("copyBufferToImage error: %d", status);
    }
    status = buffer_to_image_kernel.setArg(1, *output);
    if (status != CL_SUCCESS)
    {
        TLOG_ERR("copyBufferToImage error: %d", status);
    }
    status = buffer_to_image_kernel.setArg(2, w);
    if (status != CL_SUCCESS)
    {
        TLOG_ERR("copyBufferToImage error: %d", status);
    }
    status = buffer_to_image_kernel.setArg(3, h);
    if (status != CL_SUCCESS)
    {
        TLOG_ERR("copyBufferToImage error: %d", status);
    }
    engine->get_command_queue().enqueueNDRangeKernel(buffer_to_image_kernel, cl::NullRange, cl::NDRange(w, h), cl::NDRange(1, 1));

#if 0
    printf("----------- after upload-------- \n");
    std::vector<float> debugData;
    debugData.resize(w * h * 4);
    engine->get_command_queue().enqueueReadImage(*output,
                                                 CL_TRUE, {0, 0, 0}, {(uint32_t)w, (uint32_t)h, 1}, w * sizeof(float) * 4,
                                                 0, debugData.data());
    int debugIndex = 0;
    for (int i = 0; i < 1; ++i)
    {
        for (int j = 0; j < w; ++j)
        {
            for (int k = 0; k < 4; ++k)
            {
                printf("%.4f,", debugData[debugIndex]);
                debugIndex++;
            }
            printf("  ");
        }
        printf("\n");
    }
#endif
    return true;
}

bool ocl_convertor::conv2d_buffer_to_image(struct conv_param* conv_param, cl::Buffer* input, cl::Image* output)
{
    int global_width = conv_param->input_channel;
    int global_height = conv_param->kernel_w * conv_param->kernel_h * UP_DIV(conv_param->output_channel, 4);
    uint32_t global_work_size[2] = {(uint32_t)global_width, (uint32_t)global_height};
    uint32_t kernel_shape[2] = {(uint32_t)conv_param->kernel_h, (uint32_t)conv_param->kernel_w};
    const int c_w_h = conv_param->input_channel * conv_param->kernel_h * conv_param->kernel_w;
    const int w_h = conv_param->kernel_w * conv_param->kernel_h;
    if (conv2d_filter_buffer_to_image.get() == nullptr)
    {
        std::set<std::string> build_options;
        build_options.emplace("-DBUFFER_INP_FP32");
        conv2d_filter_buffer_to_image = engine->build_kernel("buffer_to_image", "conv2d_filter_buffer_to_image", build_options);
    }
    int idx = 0;
    conv2d_filter_buffer_to_image.setArg(idx++, global_work_size[0]);
    conv2d_filter_buffer_to_image.setArg(idx++, global_work_size[1]);
    conv2d_filter_buffer_to_image.setArg(idx++, *input);
    conv2d_filter_buffer_to_image.setArg(idx++, conv_param->output_channel);
    conv2d_filter_buffer_to_image.setArg(idx++, sizeof(kernel_shape), kernel_shape);
    conv2d_filter_buffer_to_image.setArg(idx++, c_w_h);
    conv2d_filter_buffer_to_image.setArg(idx++, w_h);
    conv2d_filter_buffer_to_image.setArg(idx++, *output);

    const uint32_t maxWorkGroupSize = engine->get_max_work_group_size(conv2d_filter_buffer_to_image);
    const std::vector<uint32_t> local_work_size = {16, std::max((uint32_t)1, maxWorkGroupSize / 16)};

    cl::Event event;
    cl_int res;

    std::vector<uint32_t> round_up_global_size(local_work_size.size());
    for (int i = 0; i < local_work_size.size(); ++i)
    {
        round_up_global_size[i] = ROUND_UP(global_work_size[i], local_work_size[i]);
    }

    res = engine->get_command_queue().enqueueNDRangeKernel(conv2d_filter_buffer_to_image,
                                                           cl::NullRange,
                                                           cl::NDRange(round_up_global_size[0], round_up_global_size[1]),
                                                           cl::NDRange(local_work_size[0], local_work_size[1]),
                                                           nullptr, &event);

#if 0
    std::vector<float> debugData;
    debugData.resize(3 * 9 * 4);
    engine->get_command_queue().enqueueReadImage(*output,
                                                 CL_TRUE, {0, 0, 0}, {3, 9, 1}, 3 * sizeof(float) * 4,
                                                 0, debugData.data());
    int debugIndex = 0;
    printf("weight:: -- \n");
    for (int i = 0; i < 9; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            for (int k = 0; k < 4; ++k)
            {
                printf("%.4f,", debugData[debugIndex]);
                debugIndex++;
            }
            printf("  ");
        }
        printf("\n");
    }
#endif

    return false;
}
bool ocl_convertor::image_to_buffer(struct tensor* ir_tensor, cl::Image* input, cl::Buffer* output, int w, int h)
{
    //image_to_nchw_buffer
    if (image_to_nchw_buffer.get() == nullptr)
    {
        std::set<std::string> build_options;
        build_options.emplace("-DBUFFER_IMAGE_IO_TRANS");
        image_to_nchw_buffer = engine->build_kernel("buffer_to_image", "image_to_nchw_buffer", build_options);
    }
    uint32_t idx = 0;

    auto status = image_to_nchw_buffer.setArg(idx++, w);
    if (status != CL_SUCCESS)
    {
        TLOG_ERR("copyBufferToImage error: %d", status);
    }
    status = image_to_nchw_buffer.setArg(idx++, h);
    if (status != CL_SUCCESS)
    {
        TLOG_ERR("copyBufferToImage error: %d", status);
    }
    status = image_to_nchw_buffer.setArg(idx++, *output);
    if (status != CL_SUCCESS)
    {
        TLOG_ERR("copyBufferToImage error: %d", status);
    }
    status = image_to_nchw_buffer.setArg(idx++, ir_tensor->dims[2]);
    if (status != CL_SUCCESS)
    {
        TLOG_ERR("copyBufferToImage error: %d", status);
    }
    status = image_to_nchw_buffer.setArg(idx++, ir_tensor->dims[3]);
    if (status != CL_SUCCESS)
    {
        TLOG_ERR("copyBufferToImage error: %d", status);
    }
    status = image_to_nchw_buffer.setArg(idx++, ir_tensor->dims[1]);
    if (status != CL_SUCCESS)
    {
        TLOG_ERR("copyBufferToImage error: %d", status);
    }
    status = image_to_nchw_buffer.setArg(idx++, *input);
    if (status != CL_SUCCESS)
    {
        TLOG_ERR("copyBufferToImage error: %d", status);
    }

    const uint32_t max_group_work_size = engine->get_max_work_group_size(image_to_nchw_buffer);
    const std::vector<uint32_t> local_size = {16, std::max((uint32_t)1, max_group_work_size / 16)};
    cl::Event event;
    std::vector<uint32_t> round_group_size(local_size.size());
    round_group_size[0] = ROUND_UP(w, local_size[0]);
    round_group_size[1] = ROUND_UP(h, local_size[1]);
    engine->get_command_queue().enqueueNDRangeKernel(image_to_nchw_buffer, cl::NullRange, cl::NDRange(round_group_size[0], round_group_size[1]),
                                                     cl::NDRange(local_size[0], local_size[1]), nullptr, nullptr);
    return true;
}

bool ocl_convertor::dw_filter_buffer_to_image(struct conv_param* conv_param, cl::Buffer* input, cl::Image* output)
{
    int global_width = conv_param->kernel_h * conv_param->kernel_w;
    int global_height = UP_DIV(conv_param->output_channel, 4);
    if (dw_filter_to_image_kernel.get() == nullptr)
    {
        std::set<std::string> build_options;
        dw_filter_to_image_kernel = engine->build_kernel("buffer_to_image", "dw_filter_buffer_to_image", build_options);
    }
    uint32_t idx = 0;
    cl_int err;
    uint32_t shape[] = {1, (uint32_t)conv_param->output_channel, (uint32_t)conv_param->kernel_h, (uint32_t)conv_param->kernel_w};
    err = dw_filter_to_image_kernel.setArg(idx++, global_width);
    err |= dw_filter_to_image_kernel.setArg(idx++, global_height);
    err |= dw_filter_to_image_kernel.setArg(idx++, *input);
    err |= dw_filter_to_image_kernel.setArg(idx++, sizeof(shape), shape);
    err |= dw_filter_to_image_kernel.setArg(idx++, global_width);
    err |= dw_filter_to_image_kernel.setArg(idx++, *output);

    const uint32_t max_group_work_size = engine->get_max_work_group_size(image_to_nchw_buffer);
    const std::vector<uint32_t> local_size = {16, std::max((uint32_t)1, max_group_work_size / 16)};
    cl::Event event;
    std::vector<uint32_t> round_group_size(local_size.size());
    round_group_size[0] = ROUND_UP(global_width, local_size[0]);
    round_group_size[1] = ROUND_UP(global_height, local_size[1]);
    engine->get_command_queue().enqueueNDRangeKernel(dw_filter_to_image_kernel, cl::NullRange, cl::NDRange(round_group_size[0], round_group_size[1]),
                                                     cl::NDRange(local_size[0], local_size[1]), nullptr, nullptr);

    return true;
}

ocl_convertor::ocl_convertor(OCLEngine* _engine)
{
    this->engine = _engine;
    // upload
    if (nchw_buffer_image_kernel.get() == nullptr)
    {
        std::set<std::string> build_options;
        build_options.emplace("-DBUFFER_IMAGE_IO_TRANS");
        nchw_buffer_image_kernel = engine->build_kernel("buffer_to_image", "nchw_buffer_to_image", build_options);
    }

    // download
    if (image_to_nchw_buffer.get() == nullptr)
    {
        std::set<std::string> build_options;
        build_options.emplace("-DBUFFER_IMAGE_IO_TRANS");
        image_to_nchw_buffer = engine->build_kernel("buffer_to_image", "image_to_nchw_buffer", build_options);
    }
}
