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

#include <stdio.h>
#include <assert.h>

#include "sys_port.h"
#include "tengine_ir.h"
#include "tengine_errno.h"
#include "tengine_log.h"
#include "tengine_op.h"
#include "parameter.h"
#include "normalize_param.h"

DEFINE_PARM_PARSE_ENTRY(normalize_param, across_spatial, channel_shared);

static int init_op(struct ir_op* op)
{
    normalize_param_t* normalize_param = ( normalize_param_t* )sys_malloc(sizeof(normalize_param_t));

    if (normalize_param == NULL)
    {
        set_tengine_errno(ENOMEM);
        return -1;
    }

    normalize_param->across_spatial = 0;
    normalize_param->channel_shared = 0;

    op->param_mem = normalize_param;
    op->param_size = sizeof(normalize_param_t);
    op->same_shape = 1;
    op->infer_shape = NULL;

    return 0;
}

static void release_op(struct ir_op* op)
{
    sys_free(op->param_mem);
}

static int register_normalize_op(void* arg)
{
    struct op_method m;

    m.op_version = 1;
    m.init_op = init_op;
    m.release_op = release_op;
    m.access_param_entry = access_param_entry;

    return register_op(OP_NORMALIZE, OP_NORMALIZE_NAME, &m);
}

static int unregister_normalize_op(void* arg)
{
    sys_free(GET_PARAM_PARSE_MAP(normalize_param));
    return unregister_op(OP_NORMALIZE, 1);
}

AUTO_REGISTER_OP(register_normalize_op);
AUTO_UNREGISTER_OP(unregister_normalize_op);
