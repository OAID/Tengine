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
 * Author: xlchen@openailab.com
 */

#include <stdio.h>
#include <assert.h>
#include "sys_port.h"
#include "tengine_ir.h"
#include "tengine_errno.h"
#include "tengine_log.h"
#include "tengine_op.h"

static int init_op(struct ir_op* op)
{
    op->same_shape = 1;

    return 0;
}

static int register_add_n_op(void* arg)
{
    struct op_method m;
    m.op_version = 1;
    m.init_op = init_op;
    m.release_op = NULL;
    m.access_param_entry = NULL;

    return register_op(OP_ADD_N, OP_ADD_N_NAME, &m);
}

static int unregister_add_n_op(void* arg)
{
    return unregister_op(OP_ADD_N, 1);
}

AUTO_REGISTER_OP(register_add_n_op);
AUTO_UNREGISTER_OP(unregister_add_n_op);
