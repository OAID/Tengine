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
#include "split_param.h"

DEFINE_PARM_PARSE_ENTRY(split_param, axis, split_dim, is_caffe, is_onnx, split_sizes_);

static int infer_shape(struct ir_node* node)
{
    struct ir_graph* graph = node->graph;
    struct ir_tensor* input = get_ir_graph_tensor(graph, node->input_tensors[0]);
    struct split_param* split_param = ( struct split_param* )(node->op.param_mem);

    int axis = split_param->axis;

    int input_dim[4];
    for (int i = 0; i < input->dim_num; i++)
        input_dim[i] = input->dims[i];

    if (split_param->is_caffe)
    {
        for (int i = 0; i < node->output_num; i++)
        {
            struct ir_tensor* output = get_ir_graph_tensor(graph, node->output_tensors[i]);
            set_ir_tensor_shape(output, input->dims, input->dim_num);
        }
    }
    else
    {
        if (get_vector_num(split_param->split_sizes_) != 0)
        {
            int sumcheck = 0;
            int input_slice_num = input_dim[axis];

            for (int i = 0; i < get_vector_num(split_param->split_sizes_); i++)
            {
                sumcheck += (( int* )get_vector_data(split_param->split_sizes_, i))[0];
            }

            if (sumcheck != input_slice_num)
            {
                fprintf(stderr, "sumcheck != input_slice_num, %d, %d\n", sumcheck, input_slice_num);
                return -1;
            }

            for (int i = 0; i < get_vector_num(split_param->split_sizes_); i++)
            {
                input_dim[axis] = (( int* )get_vector_data(split_param->split_sizes_, i))[0];
                struct ir_tensor* output = get_ir_graph_tensor(graph, node->output_tensors[i]);
                set_ir_tensor_shape(output, input_dim, input->dim_num);
            }
        }
        else
        {
            int split_dim = split_param->split_dim;
            int split_shape = 0;

            if (input_dim[axis] % split_dim != 0)
            {
                fprintf(stderr, "input_dim[axis] %% split_dim != 0\n");
                return -1;
            }

            split_shape = input_dim[axis] / split_dim;
            input_dim[axis] = split_shape;

            if (split_shape == 1)
            {
                int output_dim[4];
                for (int i = 0; i < input->dim_num - 1; i++)
                {
                    if (i >= axis)
                        output_dim[i] = input_dim[i + axis];
                    else
                        output_dim[i] = input_dim[i];
                }

                for (int i = 0; i < node->output_num; i++)
                {
                    struct ir_tensor* output = get_ir_graph_tensor(graph, node->output_tensors[i]);
                    set_ir_tensor_shape(output, input->dims, input->dim_num - 1);
                }
            }

            for (int i = 0; i < node->output_num; i++)
            {
                struct ir_tensor* output = get_ir_graph_tensor(graph, node->output_tensors[i]);
                set_ir_tensor_shape(output, input->dims, input->dim_num);
            }
        }
    }

    return 0;
}

static int init_op(struct ir_op* op)
{
    struct split_param* split_param = ( struct split_param* )sys_malloc(sizeof(struct split_param));

    if (split_param == NULL)
    {
        set_tengine_errno(ENOMEM);
        return -1;
    }

    /*set the param default value */
    split_param->axis = 0;
    split_param->split_dim = 1;
    split_param->is_caffe = 0;
    split_param->is_onnx = 0;
    split_param->split_sizes_ = NULL;

    op->param_mem = split_param;
    op->param_size = sizeof(struct split_param);
    op->same_shape = 0;
    op->infer_shape = infer_shape;

    return 0;
}

static void release_op(struct ir_op* op)
{
    struct split_param* split_param = ( struct split_param* )op->param_mem;

    if (split_param->split_sizes_)
        release_vector(split_param->split_sizes_);

    sys_free(op->param_mem);
}

static int register_split_op(void* arg)
{
    struct op_method m;

    m.op_version = 1;
    m.init_op = init_op;
    m.release_op = release_op;
    m.access_param_entry = access_param_entry;

    return register_op(OP_SPLIT, OP_SPLIT_NAME, &m);
}

static int unregister_split_op(void* arg)
{
    sys_free(GET_PARAM_PARSE_MAP(split_param));
    return unregister_op(OP_SPLIT, 1);
}

AUTO_REGISTER_OP(register_split_op);
AUTO_UNREGISTER_OP(unregister_split_op);
