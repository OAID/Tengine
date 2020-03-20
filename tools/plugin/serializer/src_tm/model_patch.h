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
 * Copyright (c) 2018, Open AI Lab
 * Author: haitao@openailab.com
 */
#ifndef __MODEL_PATCH_HPP__
#define __MODEL_PATCH_HPP__

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_MODEL_NAME_LEN (64 - 1)

struct model_patch
{
    char model_name[MAX_MODEL_NAME_LEN + 1];
    uint32_t vendor_id;
    uint32_t nn_id;
    uint32_t total_size;
    uint32_t patch_off;
    uint32_t patch_size;
    void* addr;
};

#ifdef __cplusplus
};
#endif

#endif
