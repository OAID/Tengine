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
 * Author: qli@openailab.com
 */

#include <stdio.h>
#include <assert.h>
#include "sys_port.h"
#include "tengine_ir.h"
#include "tengine_errno.h"
#include "tengine_log.h"
#include "tengine_op.h"
#include "parameter.h"
#include "generic_param.h"

DEFINE_PARM_PARSE_ENTRY(generic_param, op_name, max_input_num, max_output_num)

static int infer_shape(struct ir_node* node)
{
    struct ir_graph* graph = node->graph;
    struct ir_tensor* input = get_ir_graph_tensor(graph, node->input_tensors[0]);
    struct ir_tensor* output = get_ir_graph_tensor(graph, node->output_tensors[0]);

    struct generic_param* generic_param = ( struct generic_param* )(node->op.param_mem);

    /* check input/output number */
    int input_num = input->elem_num;
    int output_num = output->elem_num;

    if (input_num > generic_param->max_input_num)
    {
        fprintf(stderr, "generic node: input number mismatch: max =%d , real = %d \n", generic_param->max_input_num, input_num);

        set_tengine_errno(EINVAL);
        return -1;
    }

    if (output_num > generic_param->max_output_num)
    {
        fprintf(stderr, "generic node: input number mismatch: max =%d , real = %d \n", generic_param->max_output_num,
               output_num);
        set_tengine_errno(EINVAL);
        return -1;
    }

    // TODO : need add customer kernel op windows.
    return -1;
}

static int init_op(struct ir_op* op)
{
    struct generic_param* generic_param = ( struct generic_param* )sys_malloc(sizeof(struct generic_param));

    if (generic_param == NULL)
    {
        set_tengine_errno(ENOMEM);
        return -1;
    }

    op->param_mem = generic_param;
    op->param_size = sizeof(struct generic_param);
    op->same_shape = 0;
    op->infer_shape = infer_shape;

    return 0;
}

static void release_op(struct ir_op* op)
{
    sys_free(op->param_mem);
}

static int register_generic_op(void* arg)
{
    struct op_method m;

    m.op_version = 1;
    m.init_op = init_op;
    m.release_op = release_op;

    return register_op(OP_GENERIC, OP_GENERIC_NAME, &m);
}

static int unregister_generic_op(void* arg)
{
    return unregister_op(OP_GENERIC, 1);
}

AUTO_REGISTER_OP(register_generic_op);
AUTO_UNREGISTER_OP(unregister_generic_op);
