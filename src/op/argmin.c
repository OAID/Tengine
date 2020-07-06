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
#include "parameter.h"
#include "argmin_param.h"

DEFINE_PARM_PARSE_ENTRY(argmin_param, axis, keepdims);

static int infer_shape(struct ir_node* node)
{
    struct ir_graph* graph = node->graph;
    struct ir_tensor* input = get_ir_graph_tensor(graph, node->input_tensors[0]);
    struct ir_tensor* output = get_ir_graph_tensor(graph, node->output_tensors[0]);

    struct argmin_param* argmin_param = ( struct argmin_param* )(node->op.param_mem);

    int axis = argmin_param->axis;

    if (axis >= input->dim_num)
    {
        set_tengine_errno(ENOENT);
        return -1;
    }

    int outdims[input->dim_num];

    // Change HWC to CHW
    int tmp = input->dims[2];
    input->dims[2] = input->dims[1];
    input->dims[1] = input->dims[0];
    input->dims[0] = tmp;
    input->dims[3] = 1;

    if (input->dims[0] != 1)    // input 3 keepdimss
    {
        for (int i = 0, j = 0; i < 3; i++)
        {
            if (i != axis)
                outdims[j++] = input->dims[i];
        }
    }
    else    // input 2 keepdimss
    {
        for (int i = 0, j = 0; i < 4; i++)
            outdims[j++] = input->dims[i];
        outdims[axis + 1] = outdims[axis + 2];
    }
    outdims[2] = outdims[3] = 1;

    if (argmin_param->keepdims == 2)
    {
        // Change CHW to HWC
        tmp = input->dims[0];
        input->dims[0] = input->dims[1];
        input->dims[1] = input->dims[2];
        input->dims[2] = tmp;
    }

    set_ir_tensor_shape(output, outdims, input->dim_num);

    return 0;
}

static int init_op(struct ir_op* op)
{
    struct argmin_param* argmin_param = ( struct argmin_param* )sys_malloc(sizeof(struct argmin_param));

    if (argmin_param == NULL)
    {
        set_tengine_errno(ENOMEM);
        return -1;
    }

    /*set the param default value */
    argmin_param->axis = 0;
    argmin_param->keepdims = 1;

    op->param_mem = argmin_param;
    op->param_size = sizeof(struct argmin_param);
    op->same_shape = 0;
    op->infer_shape = infer_shape;

    return 0;
}

static void release_op(struct ir_op* op)
{
    sys_free(op->param_mem);
}

static int register_argmin_op(void* arg)
{
    struct op_method m;
    m.op_version = 1;
    m.init_op = init_op;
    m.release_op = release_op;
    m.access_param_entry = access_param_entry;

    return register_op(OP_ARGMIN, OP_ARGMIN_NAME, &m);
}

static int unregister_argmin_op(void* arg)
{
    return unregister_op(OP_ARGMIN, 1);
}

AUTO_REGISTER_OP(register_argmin_op);
AUTO_UNREGISTER_OP(unregister_argmin_op);
