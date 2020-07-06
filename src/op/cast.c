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
#include "cast_param.h"

DEFINE_PARM_PARSE_ENTRY(cast_param, type_from, type_to);

static int init_op(struct ir_op* op)
{
    struct cast_param* cast_param = ( struct cast_param* )sys_malloc(sizeof(struct cast_param));

    if (cast_param == NULL)
    {
        set_tengine_errno(ENOMEM);
        return -1;
    }

    /*set the param default value */
    cast_param->type_from = 0;
    cast_param->type_to = 0;

    op->param_mem = cast_param;
    op->param_size = sizeof(struct cast_param);
    op->same_shape = 1;
    op->infer_shape = NULL;

    return 0;
}

static void release_op(struct ir_op* op)
{
    sys_free(op->param_mem);
}

static int register_cast_op(void* arg)
{
    struct op_method m;

    m.op_version = 1;
    m.init_op = init_op;
    m.release_op = release_op;
    m.access_param_entry = access_param_entry;

    return register_op(OP_CAST, OP_CAST_NAME, &m);
}

static int unregister_cast_op(void* arg)
{
    sys_free(GET_PARAM_PARSE_MAP(cast_param));
    return unregister_op(OP_CAST, 1);
}

AUTO_REGISTER_OP(register_cast_op);
AUTO_UNREGISTER_OP(unregister_cast_op);
