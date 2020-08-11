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
#include "reshape_param.h"

DEFINE_PARM_PARSE_ENTRY(reshape_param, re_shape, reverse, is_mxnet, is_onnx, dim_size);

static int infer_shape(struct ir_node* node)
{
    struct ir_graph* graph = node->graph;
    struct ir_tensor* input = get_ir_graph_tensor(graph, node->input_tensors[0]);
    struct ir_tensor* output = get_ir_graph_tensor(graph, node->output_tensors[0]);
    reshape_param_t* param = ( struct reshape_param* )(node->op.param_mem);

    const int size = input->elem_num;

    int new_size = 1;
    int new_shape_size = param->dim_size;
    int in_idx = 0;

    struct vector* new_shape = create_vector(sizeof(int), NULL);
    int input_dim_size = input->dim_num;

    for (int i = 0; i < new_shape_size; ++i)
    {
        if (0 == param->re_shape[i])
        {
            if (param->is_mxnet)
                push_vector_data(new_shape, ( void* )&input->dims[in_idx]);
            else
            {
                int temp = 1;
                if (i == 0)
                    push_vector_data(new_shape, ( void* )&temp);
            }

            in_idx++;
        }
        else if (-1 == param->re_shape[i])
        {
            int temp = -1;
            push_vector_data(new_shape, ( void* )&temp);
            in_idx++;
        }
        else if (-2 == param->re_shape[i])
        {
            for (; in_idx < input_dim_size; ++in_idx)
            {
                push_vector_data(new_shape, ( void* )&input->dims[in_idx]);
            }
        }
        else if (-3 == param->re_shape[i])
        {
            int temp = input->dims[in_idx] * input->dims[in_idx + 1];
            push_vector_data(new_shape, ( void* )&temp);
            in_idx = in_idx + 2;
        }
        else if (-4 == param->re_shape[i])
        {
            int muti_val = param->re_shape[i + 1];
            if (muti_val == -1)
                muti_val = 1;
            push_vector_data(new_shape, ( void* )&muti_val);
            push_vector_data(new_shape, ( void* )&param->re_shape[i + 2]);
            i = i + 2;
            in_idx++;
        }
        else
        {
            push_vector_data(new_shape, ( void* )&param->re_shape[i]);
        }
    }

    int idx = -1;
    int dim_size = get_vector_num(new_shape);
    for (int i = 0; i < dim_size; i++)
    {
        int temp = (( int* )get_vector_data(new_shape, i))[0];
        if (temp == -1)
            idx = i;
        else
            new_size *= temp;
    }

    if (idx >= 0)
    {
        int temp = size / new_size;
        set_vector_data(new_shape, idx, ( void* )&temp);
    }

    if ((( int* )get_vector_data(new_shape, 0))[0] == -1 && get_vector_num(new_shape) == 1)
    {
        set_vector_data(new_shape, 0, ( void* )&size);
    }

    if (param->reverse)
    {
        struct vector* tmp = create_vector(sizeof(int), NULL);

        for (int i = 0; i < get_vector_num(new_shape); i++)
        {
            set_vector_data(tmp, i, get_vector_data(new_shape, i));
        }

        int j = 0;
        for (int i = dim_size - 1; i >= 0; --i)
        {
            set_vector_data(new_shape, j++, get_vector_data(tmp, i));
        }
    }

    int* new_shape_temp = ( int* )sys_malloc(get_vector_num(new_shape) * sizeof(int));

    for (int i = 0; i < get_vector_num(new_shape); i++)
    {
        int* a = ( int* )get_vector_data(new_shape, i);
        new_shape_temp[i] = *a;
    }

    output->layout = input->layout;

    set_ir_tensor_shape(output, new_shape_temp, get_vector_num(new_shape));

    sys_free(new_shape_temp);
    release_vector(new_shape);

    return 0;
}

static int init_op(struct ir_op* op)
{
    struct reshape_param* reshape_param = ( struct reshape_param* )sys_malloc(sizeof(struct reshape_param));

    if (reshape_param == NULL)
    {
        set_tengine_errno(ENOMEM);
        return -1;
    }

    /*set the param default value */
    memset(reshape_param, 0, sizeof(struct reshape_param));
    op->param_mem = reshape_param;
    op->param_size = sizeof(struct reshape_param);
    op->same_shape = 0;
    op->infer_shape = infer_shape;

    return 0;
}

static void release_op(struct ir_op* op)
{
    struct reshape_param* reshape_param = ( struct reshape_param* )op->param_mem;

    if (reshape_param->re_shape)
        sys_free(reshape_param->re_shape);

    sys_free(op->param_mem);
}

static int register_reshape_op(void* arg)
{
    struct op_method m;

    m.op_version = 1;
    m.init_op = init_op;
    m.release_op = release_op;
    m.access_param_entry = NULL;

    return register_op(OP_RESHAPE, OP_RESHAPE_NAME, &m);
}

static int unregister_reshape_op(void* arg)
{
    sys_free(GET_PARAM_PARSE_MAP(reshape_param));
    return unregister_op(OP_RESHAPE, 1);
}

AUTO_REGISTER_OP(register_reshape_op);
AUTO_UNREGISTER_OP(unregister_reshape_op);
