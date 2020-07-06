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
 * Author: zpluo@openailab.com
 */

#include <stdio.h>
#include <assert.h>

#include "sys_port.h"
#include "tengine_ir.h"
#include "tengine_errno.h"
#include "tengine_log.h"
#include "tengine_op.h"
#include "parameter.h"
#include "mvn_param.h"

DEFINE_PARM_PARSE_ENTRY(mvn_param, normalize_variance, across_channels, eps);

static int init_op(struct ir_op* op)
{
    struct mvn_param* param = ( struct mvn_param* )sys_malloc(sizeof(struct mvn_param));

    if (param == NULL)
    {
        set_tengine_errno(ENOMEM);
        return -1;
    }

    /*set the param default value */
    op->param_mem = param;
    op->param_size = sizeof(struct mvn_param);
    op->same_shape = 1;
    op->infer_shape = NULL;

    return 0;
}

static void release_op(struct ir_op* op)
{
    sys_free(op->param_mem);
}

static int register_mvn_op(void* arg)
{
    struct op_method m;

    m.op_version = 1;
    m.init_op = init_op;
    m.release_op = release_op;
    m.access_param_entry = access_param_entry;

    return register_op(OP_MVN, OP_MVN_NAME, &m);
}

static int unregister_mvn_op(void* arg)
{
    return unregister_op(OP_MVN, 1);
}

AUTO_REGISTER_OP(register_mvn_op);
AUTO_UNREGISTER_OP(unregister_mvn_op);
