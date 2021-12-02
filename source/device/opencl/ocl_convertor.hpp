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

#ifndef TENGINE_LITE_SOURCE_DEVICE_OPENCL_OCL_CONVERTOR_HPP_
#define TENGINE_LITE_SOURCE_DEVICE_OPENCL_OCL_CONVERTOR_HPP_

#include "ocl_cpp_helper.hpp"

class OCLEngine;
class ocl_convertor
{
public:
    ocl_convertor(OCLEngine* _engine);
    ~ocl_convertor() = default;
    bool nchw_buffer_to_image(struct tensor* ir_tensor, cl::Buffer* input, cl::Image* output, bool needWait);

    bool buffer_to_image(cl::Buffer* input, cl::Image* output, int w, int h);
    bool conv2d_buffer_to_image(struct conv_param* conv_param, cl::Buffer* input, cl::Image* output);
    bool image_to_buffer(struct tensor* ir_tensor, cl::Image* input, cl::Buffer* output, int w, int h);
    bool dw_filter_buffer_to_image(struct conv_param* conv_param, cl::Buffer* input, cl::Image* output);

private:
    OCLEngine* engine;
    cl::Kernel nchw_buffer_image_kernel;
    cl::Kernel buffer_to_image_kernel;
    cl::Kernel conv2d_filter_buffer_to_image;
    cl::Kernel image_to_nchw_buffer;
    cl::Kernel dw_filter_to_image_kernel;
};

#endif //TENGINE_LITE_SOURCE_DEVICE_OPENCL_OCL_CONVERTOR_HPP_
