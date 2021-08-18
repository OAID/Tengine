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
 * Author: haitao@openailab.com
 * Revised: lswang@openailab.com
 */

#pragma once

#include <stdarg.h>

struct graph;

/*!
 * @struct serializer_t
 * @brief  Abstract serializer
 */
typedef struct serializer
{
    const char* (*get_name)(struct serializer*);

    //!< load model from file
    int (*load_model)(struct serializer*, struct graph*, const char* fname, va_list ap);

    //!< load model from memory
    int (*load_mem)(struct serializer*, struct graph*, const void* addr, int size, va_list ap);

    //!< unload model, free serializer and device related resource
    int (*unload_graph)(struct serializer*, struct graph*, void* s_priv, void* dev_priv);

    //!< interface exposed for register operator extension
    int (*register_op_loader)(struct serializer*, int op_type, int op_ver, void* op_load_func, void* op_type_map_func, void* op_ver_map_func);

    //!< interface exposed for register operator extension
    int (*unregister_op_loader)(struct serializer*, int op_type, int op_ver, void* op_load_func);

    //!< interface exposed for initialize serializer
    int (*init)(struct serializer*);

    //!< interface exposed for release serializer
    int (*release)(struct serializer*);
} serializer_t;

/*!
 * @brief Initialize serializer
 *
 * @param [in]  serializer: specific serializer.
 */
void init_serializer(struct serializer* serializer);
