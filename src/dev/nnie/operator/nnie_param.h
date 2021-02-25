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
 * Author: bsun@openailab.com
 */

#ifndef __NNIE_PARAM_H__
#define __NNIE_PARAM_H__

#ifdef __cplusplus
extern "C" {
#endif
#include "parameter.h"
#include "tengine_op.h"

#ifdef STANDLONE_MODE
void init_nnie_ops(void);
#endif

#ifdef __cplusplus
}
#endif

#define NNIE_OP_FORWARD "NnieOpForward"
#define NNIE_OP_FORWARD_WITH_BBOX "NnieOpForwardWithBbox"
#define NNIE_OP_CPU_PROPOSAL "NnieOpCpuProrosal"

#define OP_VERSION 1

enum
{
    OP_NNIE_FORWARD = OP_BUILTIN_LAST + 128,
    OP_NNIE_FORWARD_WITHBBOX,
    OP_NNIE_CPU_PROPOSAL,
};

typedef struct nnie_param
{
    void* nnie_node;
    void* software_param;
} nnie_param_t;

#endif
