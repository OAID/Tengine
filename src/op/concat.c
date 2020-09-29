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
#include "concat_param.h"

DEFINE_PARM_PARSE_ENTRY(concat_param, axis);

static int infer_shape(struct ir_node* node)
{
    struct ir_graph* graph = node->graph;
    struct ir_tensor* output = get_ir_graph_tensor(graph, node->output_tensors[0]);

    struct concat_param* concat_param = ( struct concat_param* )(node->op.param_mem);

    int concat_shape = 0;
    int axis = concat_param->axis;

    /* transpose axis from nhwc to nchw */
    if (graph->model_layout == TENGINE_LAYOUT_NHWC)
    {
        struct ir_tensor* input_tensor = get_ir_graph_tensor(graph, node->input_tensors[0]);
        if (input_tensor->dim_num == 4)
        {
            if (axis == 1)
                axis = 2;
            else if (axis == 2)
                axis = 3;
            else if (axis == 3)
                axis = 1;
            else
            {
                fprintf(stderr, "concat infershape axis value error\n");
                return -1;
            }
            concat_param->axis = axis;
        }
        else if (input_tensor->dim_num == 3)
        {
            if (axis == 1)
                axis = 2;
            else if (axis == 2)
                axis = 1;
            else
            {
                fprintf(stderr, "concat infershape axis value error\n");
                return -1;
            }
            concat_param->axis = axis;
        }
    }

    for (int i = 0; i < node->input_num; i++)
    {
        struct ir_tensor* input_tensor = get_ir_graph_tensor(graph, node->input_tensors[0]);

        if (axis < 0)
        {
            axis = input_tensor->dim_num + axis;
            concat_param->axis = axis;
        }

        struct ir_tensor* input = get_ir_graph_tensor(graph, node->input_tensors[i]);
        concat_shape += input->dims[axis];
    }

    struct ir_tensor* input = get_ir_graph_tensor(graph, node->input_tensors[0]);
    int dims[4];
    for (int i = 0; i < input->dim_num; i++)
    {
        dims[i] = input->dims[i];
    }

    dims[axis] = concat_shape;
    output->layout = input->layout;
    set_ir_tensor_shape(output, dims, input->dim_num);

    return 0;
}

static int init_op(struct ir_op* op)
{
    struct concat_param* concat_param = ( struct concat_param* )sys_malloc(sizeof(struct concat_param));

    if (concat_param == NULL)
    {
        set_tengine_errno(ENOMEM);
        return -1;
    }

    /*set the param default value */
    concat_param->axis = 0;

    op->param_mem = concat_param;
    op->param_size = sizeof(struct concat_param);
    op->same_shape = 0;
    op->infer_shape = infer_shape;

    return 0;
}

static void release_op(struct ir_op* op)
{
    sys_free(op->param_mem);
}

static int register_concat_op(void* arg)
{
    struct op_method m;

    m.op_version = 1;
    m.init_op = init_op;
    m.release_op = release_op;
    m.access_param_entry = access_param_entry;

    return register_op(OP_CONCAT, OP_CONCAT_NAME, &m);
}

static int unregister_concat_op(void* arg)
{
    sys_free(GET_PARAM_PARSE_MAP(concat_param));
    return unregister_op(OP_CONCAT, 1);
}

AUTO_REGISTER_OP(register_concat_op);
AUTO_UNREGISTER_OP(unregister_concat_op);
