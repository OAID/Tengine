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

#ifndef TENGINE_LITE_SOURCE_DEVICE_OPENCL_OPPACK4_OCL_DROPOUT_HPP_
#define TENGINE_LITE_SOURCE_DEVICE_OPENCL_OPPACK4_OCL_DROPOUT_HPP_

#include "ocl_executor.hpp"
#include "ocl_node.hpp"

class ocl_dropout : public ocl_node
{
public:
    ocl_dropout(OCLEngine* engine, struct node* ir_node);
    void pre_run() override;
    void run(struct subgraph* subgraph) override;
};

#endif //TENGINE_LITE_SOURCE_DEVICE_OPENCL_OPPACK4_OCL_DROPOUT_HPP_
