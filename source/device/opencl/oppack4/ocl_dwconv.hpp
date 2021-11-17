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

#pragma once
#ifndef TENGINE_LITE_SOURCE_DEVICE_OPENCL_OPPACK4_OCL_DWCONV_H_
#define TENGINE_LITE_SOURCE_DEVICE_OPENCL_OPPACK4_OCL_DWCONV_H_

#include "ocl_node.hpp"

extern "C" {
#include "operator/op.h"
#include "convolution_param.h"
}

class ocl_dwconv : public ocl_node
{
public:
    ocl_dwconv(OCLEngine* engine, struct node* ir_node);

    void pre_run() override;
    void run(struct subgraph* subgraph) override;

private:
    std::shared_ptr<cl::Image> gpu_bias;
    std::shared_ptr<cl::Image> gpu_weight;

    struct conv_param* conv2d_param;

    cl::Kernel conv2d_kernel;
    std::vector<int> strides;
    std::vector<int> dilations;
    std::vector<int> paddings;

    std::vector<uint32_t> global_work_size;
    std::vector<uint32_t> local_work_size;
    int max_work_group_size;
    void upload_bias_gpu(tensor* tensor);
};

#endif //TENGINE_LITE_SOURCE_DEVICE_OPENCL_OPPACK4_OCL_DWCONV_H_
