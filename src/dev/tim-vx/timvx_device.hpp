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
 * Copyright (c) 2021, Open AI Lab
 * Author: lswang@openailab.com
 */

#ifndef __TIMVX_DEVICE_H__
#define __TIMVX_DEVICE_H__

#define TIMVX_DEV_NAME "TIMVX"

extern "C"
{
#include "tengine_c_api.h"

struct timvx_device
{
    struct nn_device base;

    int (*load_graph)(struct timvx_device* dev);

    int (*load_ir_graph)(struct timvx_device* dev);

    int (*unload_graph)(struct timvx_device* dev);
};

DLLEXPORT int register_timvx_device(void);
}

#endif
