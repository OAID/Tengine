//
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

#ifndef TENGINE_LITE_SOURCE_DEVICE_OPENCL_OPPACK4_OCL_WINOGRAD_HPP_
#define TENGINE_LITE_SOURCE_DEVICE_OPENCL_OPPACK4_OCL_WINOGRAD_HPP_

#include "ocl_node.hpp"
#include "ocl_cpp_helper.hpp"

extern "C" {
#include "operator/op.h"
#include "convolution_param.h"
}

class ocl_winograd : public ocl_node
{
public:
    ocl_winograd(OCLEngine* engine, struct node* ir_node);
    void pre_run() override;
    void run(struct subgraph* subgraph) override;

private:
    static void weight_transform(struct tensor* weight_tensor, float* weight_dst);
    static void trans_kernel(const float* src, float* dest);
    void upload_bias_gpu(const float* bias_data, int bias_size);

    struct conv_param* conv2d_param;
    std::vector<int> strides;
    std::vector<int> dilations;
    std::vector<int> paddings;

    std::shared_ptr<cl::Image> gpu_bias;
    std::shared_ptr<cl::Image> gpu_weight;
    std::shared_ptr<cl::Image> gpu_source;
    std::shared_ptr<cl::Image> gpu_dest;

    cl::Kernel source_transform;
    cl::Kernel dot_mul;
    cl::Kernel dest_transform;

    std::vector<uint32_t> global_work_size_source;
    std::vector<uint32_t> global_work_size_dot;
    std::vector<uint32_t> global_work_size_dest;

    std::vector<uint32_t> local_work_size_source;
    std::vector<uint32_t> local_work_size_dot;
    std::vector<uint32_t> local_work_size_dest;

    uint32_t max_work_group_size_source;
    uint32_t max_work_group_size_dot;
    uint32_t max_work_group_size_dest;
};

#endif //TENGINE_LITE_SOURCE_DEVICE_OPENCL_OPPACK4_OCL_WINOGRAD_HPP_
