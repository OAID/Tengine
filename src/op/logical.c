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
 * Author: qtang@openailab.com
 */

#include <stdio.h>
#include <assert.h>
#include "sys_port.h"
#include "tengine_ir.h"
#include "tengine_errno.h"
#include "tengine_log.h"
#include "tengine_op.h"
#include "parameter.h"
#include "logical_param.h"

DEFINE_PARM_PARSE_ENTRY(logical_param, type)

static int infer_shape(struct ir_node* node)
{
    if (node->input_num == 1)
    {
        struct ir_graph* graph = node->graph;
        struct ir_tensor* input = get_ir_graph_tensor(graph, node->input_tensors[0]);
        struct ir_tensor* output = get_ir_graph_tensor(graph, node->output_tensors[0]);

        set_ir_tensor_shape(output, input->dims, input->dim_num);

        return 0;
    }

    if (node->input_num == 2)
    {
        struct ir_graph* graph = node->graph;
        struct ir_tensor* input0 = get_ir_graph_tensor(graph, node->input_tensors[0]);
        struct ir_tensor* input1 = get_ir_graph_tensor(graph, node->input_tensors[1]);
        struct ir_tensor* output = get_ir_graph_tensor(graph, node->output_tensors[0]);

        if (input0->elem_num > input1->elem_num)
            set_ir_tensor_shape(output, input0->dims, input0->dim_num);
        else
            set_ir_tensor_shape(output, input1->dims, input1->dim_num);

        return 0;
    }

    return -1;
}

static int init_op(struct ir_op* op)
{
    struct logical_param* logical_param = ( struct logical_param* )sys_malloc(sizeof(struct logical_param));

    if (logical_param == NULL)
    {
        set_tengine_errno(ENOMEM);
        return -1;
    }

    /*set the param default value */
    logical_param->type = 0;

    op->param_mem = logical_param;
    op->param_size = sizeof(struct logical_param);
    op->same_shape = 0;
    op->infer_shape = infer_shape;

    return 0;
}

static void release_op(struct ir_op* op)
{
    sys_free(op->param_mem);
}

static int register_logical_op(void* arg)
{
    struct op_method m;

    m.op_version = 1;
    m.init_op = init_op;
    m.release_op = release_op;
    m.access_param_entry = access_param_entry;

    return register_op(OP_LOGICAL, OP_LOGICAL_NAME, &m);
}

static int unregister_logical_op(void* arg)
{
    sys_free(GET_PARAM_PARSE_MAP(logical_param));
    return unregister_op(OP_LOGICAL, 1);
}

AUTO_REGISTER_OP(register_logical_op);
AUTO_UNREGISTER_OP(unregister_logical_op);
