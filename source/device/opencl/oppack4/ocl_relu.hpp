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

#ifndef TENGINE_LITE_SOURCE_DEVICE_OPENCL_OPPACK4_OCL_RELU_HPP_
#define TENGINE_LITE_SOURCE_DEVICE_OPENCL_OPPACK4_OCL_RELU_HPP_

#include "ocl_node.hpp"
#include "ocl_cpp_helper.hpp"

class ocl_relu : public ocl_node
{
public:
    void pre_run() override;
    void run(struct subgraph* subgraph) override;
    ocl_relu(OCLEngine* engine, struct node* ir_node);

private:
    cl::Kernel leaky_relu_kernel;
    uint32_t max_work_group_size;
    std::vector<uint32_t> global_work_size = {1, 1, 1};
    std::vector<uint32_t> local_work_size = {1, 1, 1};
};

#endif //TENGINE_LITE_SOURCE_DEVICE_OPENCL_OPPACK4_OCL_RELU_HPP_
