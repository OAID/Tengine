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

#ifndef __NNIE_SERIALIZER_H__
#define __NNIE_SERIALIZER_H__

#include <stdbool.h>
#include "tengine_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif
#include "tengine_ir.h"
#include "tengine_serializer.h"

#ifdef STANDLONE_MODE
void init_nnie_serializer(void);
#endif
#ifdef __cplusplus
}
#endif

#include "nnie_param.h"

#include "sample_comm_nnie.h"

typedef struct te_NNIE_CONTEXT_S
{
    SAMPLE_SVP_NNIE_CFG_S stNnieCfg;
    SAMPLE_SVP_NNIE_PARAM_S stCnnNnieParam;
    SAMPLE_SVP_NNIE_MODEL_S stCnnModel;
} TE_NNIE_CONTEXT_S;

struct nnie_serializer
{
    struct serializer base;
    int nnieContextCount;
    bool segNet;
};

const char * get_nnie_tensor_layout(SVP_NNIE_NODE_S* pstNode);
int get_nnie_tensor_size(SVP_NNIE_NODE_S *pstDstNode);

#endif
