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

#include "vulkan_option.hpp"

namespace TEngine {

Option::Option()
{
    lightmode = true;
    num_threads = 1;
    elempack = 1;
    blob_allocator = 0;
    workspace_allocator = 0;

    blob_vkallocator = 0;
    workspace_vkallocator = 0;
    staging_vkallocator = 0;

    use_winograd_convolution = true;
    use_sgemm_convolution = true;
    use_int8_inference = true;
    use_vulkan_compute = true;

    use_fp16_packed = true; 
    use_fp16_storage = true;
    use_fp16_arithmetic = false;
    use_int8_storage = false;
    use_int8_arithmetic = false;

    use_packing_layout = true;
    use_shader_pack8 = false;
    use_image_storage = false;
    use_bf16_storage = false;
}

} // namespace TEngine
