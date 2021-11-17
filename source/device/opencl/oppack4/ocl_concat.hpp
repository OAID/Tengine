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

#ifndef TENGINE_LITE_SOURCE_DEVICE_OPENCL_OPPACK4_OCL_CONCAT_HPP_
#define TENGINE_LITE_SOURCE_DEVICE_OPENCL_OPPACK4_OCL_CONCAT_HPP_

#include "ocl_node.hpp"

class ocl_concat : public ocl_node
{
public:
    ocl_concat(OCLEngine* engine, struct node* ir_node);
    void pre_run() override;
    void run(struct subgraph* subgraph) override;

private:
    cl::Kernel concat_kernel;
    std::vector<uint32_t> global_work_size = {1, 1, 1};
    std::vector<uint32_t> local_work_size = {1, 1, 1};
    uint32_t max_group_work_item = 1;

    std::vector<cl::Kernel> concat_multi_kernels;
    std::vector<uint32_t> concat_multi_max_group_size;
    std::vector<std::vector<uint32_t> > concat_multi_local_size;
    std::vector<std::vector<uint32_t> > concat_multi_global_size;

    std::vector<std::shared_ptr<cl::Buffer> > concat_input_buffers;

    // 0 image concat4  input_num = 2
    // 1 image concat123 input_num = 2
    // 2 image concat input_num > 2
    // 3 buffer concat input_num > 2
    int type_concat = 0;
    //
    bool use_image_anyway = true;

    void build_kernel();
    void pre_run_type_concat_0();
    void pre_run_type_concat_2();
    void run_type_concat_0();
    void run_type_concat_2();
};

#endif //TENGINE_LITE_SOURCE_DEVICE_OPENCL_OPPACK4_OCL_CONCAT_HPP_
