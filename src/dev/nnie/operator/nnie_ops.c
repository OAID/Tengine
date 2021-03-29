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

#include <stdio.h>
#include <assert.h>

#include "sys_port.h"
#include "tengine_ir.h"
#include "tengine_errno.h"
#include "tengine_log.h"
#include "compiler.h"

#include "nnie_param.h"
#include "tengine_nnie_plugin.h"

DEFINE_PARM_PARSE_ENTRY(nnie_param, nnie_node, software_param);

#define NNIE_OP_FORWARD_NAME "NnieOpForward"
#define NNIE_OP_FORWARD_WITH_BBOX_NAME "NnieOpForwardWithBbox"
#define NNIE_OP_CPU_PROPOSAL_NAME "NnieOpCpuProrosal"


static int infer_shape(struct ir_node* node)
{
    return 0;
}

static int init_op(struct ir_op* op)
{
    nnie_param_t* nnieparam = (nnie_param_t* )sys_malloc(sizeof(nnie_param_t));

    if (nnieparam == NULL)
    {
        set_tengine_errno(ENOMEM);
        return -1;
    }

    nnieparam->nnie_node = 0;
    nnieparam->software_param = 0;

    op->param_mem = nnieparam;
    op->param_size = sizeof(nnie_param_t);
    op->same_shape = 0;
    op->infer_shape = infer_shape;

    return 0;
}

static void release_op(struct ir_op* op)
{
    sys_free(op->param_mem);
}

/* nnieforward_op */
static int register_nnieforward_op(void)
{
    struct op_method m;

    m.op_version = OP_VERSION;
    m.init_op = init_op;
    m.release_op = release_op;
    m.access_param_entry = access_param_entry;

    return register_op(OP_NNIE_FORWARD, NNIE_OP_FORWARD_NAME, &m);
}

static int unregister_nnieforward_op(void* arg)
{
    return unregister_op(OP_NNIE_FORWARD, 1);
}

/* nnieforward_withbbox_op */
static int register_nnieforward_withbbox_op(void)
{
    struct op_method m;

    m.op_version = OP_VERSION;
    m.init_op = init_op;
    m.release_op = release_op;
    m.access_param_entry = access_param_entry;

    return register_op(OP_NNIE_FORWARD_WITHBBOX, NNIE_OP_FORWARD_WITH_BBOX_NAME, &m);
}

static int unregister_nnieforward_withbbox_op(void* arg)
{
    return unregister_op(OP_NNIE_FORWARD_WITHBBOX, 1);
}

/* cpu_proposal_op */
static int register_cpu_proposal_op(void)
{
    struct op_method m;

    m.op_version = OP_VERSION;
    m.init_op = init_op;
    m.release_op = release_op;
    m.access_param_entry = access_param_entry;

    return register_op(OP_NNIE_CPU_PROPOSAL, NNIE_OP_CPU_PROPOSAL_NAME, &m);
}

static int unregister_cpu_proposal_op(void* arg)
{
    return unregister_op(OP_NNIE_CPU_PROPOSAL, 1);
}

#ifndef STANDLONE_MODE
static void init_nnie_ops(void)
#else
void init_nnie_ops(void)
#endif
{
    register_nnieforward_op();
    register_module_exit(MOD_OP_LEVEL, "unregister_nnieforward_op", unregister_nnieforward_op, NULL);

    register_nnieforward_withbbox_op();
    register_module_exit(MOD_OP_LEVEL, "unregister_nnieforward_withbbox_op", unregister_nnieforward_withbbox_op, NULL);

    register_cpu_proposal_op();
    register_module_exit(MOD_OP_LEVEL, "unregister_cpu_proposal_op", unregister_cpu_proposal_op, NULL);
}

#ifndef STANDLONE_MODE
DECLARE_AUTO_INIT_FUNC(init_nnie_ops);
#endif
