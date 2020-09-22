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
 * Copyright (c) 2020, OPEN AI LAB
 * Author: haitao@openailab.com
 */

#ifndef __TENGINE_SERIALIZER_H__
#define __TENGINE_SERIALIZER_H__

struct ir_graph;

struct serializer
{
    const char* (*get_name)(struct serializer*);

    /* load graph from file */
    int (*load_model)(struct serializer*, struct ir_graph*, const char* fname, va_list ap);

    /* load graph from memory */
    int (*load_mem)(struct serializer*, struct ir_graph*, const void* addr, int size, va_list ap);

    /* unload graph, free serializer related and device releated  resource */
    int (*unload_graph)(struct serializer*, struct ir_graph*, void* s_priv, void* dev_priv);

    /* those interface exposed for operator extension */
    int (*register_op_loader)(struct serializer*, int op_type, int op_ver, void* op_load_func, void* op_type_map_func,
                              void* op_ver_map_func);
    int (*unregister_op_loader)(struct serializer*, int op_type, int op_ver, void* op_load_func);

    /* interface for regiser and unregister */
    int (*init)(struct serializer*);
    int (*release)(struct serializer*);
};

struct serializer* find_serializer(const char* name);

int register_serializer(struct serializer* serializer);
int unregister_serializer(struct serializer* serializer);

/* called by init_tengine/release_tengine */

int init_serializer_registry(void);
int release_serializer_registry(void);

#endif
