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
#include "depthtospace_param.h"

DEFINE_PARM_PARSE_ENTRY(depthtospace_param, block_size)

static int infer_shape(struct ir_node* node)
{
    struct ir_graph* graph = node->graph;
    struct ir_tensor* input = get_ir_graph_tensor(graph, node->input_tensors[0]);
    struct ir_tensor* output = get_ir_graph_tensor(graph, node->output_tensors[0]);

    struct depthtospace_param* depthtospace_param = ( struct depthtospace_param* )(node->op.param_mem);

    /* todo reshape */
    int dims[4];
    int block_size = depthtospace_param->block_size;

    dims[0] = input->dims[0];    // batch
    dims[1] = input->dims[1] / (block_size * block_size);    // channel
    dims[2] = input->dims[2] * block_size;    // height
    dims[3] = input->dims[3] * block_size;    // width

    set_ir_tensor_shape(output, dims, 4);

    return 0;
}

static int init_op(struct ir_op* op)
{
    struct depthtospace_param* depthtospace_param =
        ( struct depthtospace_param* )sys_malloc(sizeof(struct depthtospace_param));

    if (depthtospace_param == NULL)
    {
        set_tengine_errno(ENOMEM);
        return -1;
    }

    /*set the param default value */
    depthtospace_param->block_size = 1;

    op->param_mem = depthtospace_param;
    op->param_size = sizeof(struct depthtospace_param);
    op->same_shape = 0;
    op->infer_shape = infer_shape;

    return 0;
}

static void release_op(struct ir_op* op)
{
    sys_free(op->param_mem);
}

static int register_depthtospace_op(void* arg)
{
    struct op_method m;

    m.op_version = 1;
    m.init_op = init_op;
    m.release_op = release_op;
    m.access_param_entry = access_param_entry;

    return register_op(OP_DEPTHTOSPACE, OP_DEPTHTOSPACE_NAME, &m);
}

static int unregister_depthtospace_op(void* arg)
{
    return unregister_op(OP_DEPTHTOSPACE, 1);
}

AUTO_REGISTER_OP(register_depthtospace_op);
AUTO_UNREGISTER_OP(unregister_depthtospace_op);
