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
 * Author: bhu@openailab.com
 */

#include <stdio.h>
#include <assert.h>
#include <stdbool.h>

#include "sys_port.h"
#include "tengine_ir.h"
#include "tengine_errno.h"
#include "tengine_log.h"
#include "tengine_op.h"
#include "parameter.h"
#include "reduction_param.h"
#include "vector.h"

DEFINE_PARM_PARSE_ENTRY(reduction_param, dim_0, dim_1, dim_2, dim_3, keepdim, type);

static int infer_shape(struct ir_node* node)
{
    struct ir_graph* ir_graph = node->graph;
    struct ir_tensor* input = get_ir_graph_tensor(ir_graph, node->input_tensors[0]);
    struct ir_tensor* output = get_ir_graph_tensor(ir_graph, node->output_tensors[0]);
    struct reduction_param* reduction_param = ( struct reduction_param* )node->op.param_mem;

    int kd = reduction_param->keepdim;

    int* in_dim = ( int* )sys_malloc(input->dim_num * sizeof(int));

    for (int i = 0; i < input->dim_num; i++)
    {
        in_dim[i] = input->dims[i];
    }
    // int new_shape[]={0};
    int count = 0;
    if (reduction_param->dim_0 != -2)
    {
        // new_shape[count]=reduction_param->dim_0;
        count++;
    }
    if (reduction_param->dim_1 != -2)
    {
        // new_shape[count]=reduction_param->dim_1;
        count++;
    }
    if (reduction_param->dim_2 != -2)
    {
        // new_shape[count]=reduction_param->dim_2;
        count++;
    }
    if (reduction_param->dim_3 != -2)
    {
        // new_shape[count]=reduction_param->dim_3;
        count++;
    }
    int* new_shape = ( int* )sys_malloc(count * sizeof(int));
    int size = 0;
    if (reduction_param->dim_0 != -2)
    {
        new_shape[size] = reduction_param->dim_0;
        size++;
    }
    if (reduction_param->dim_1 != -2)
    {
        new_shape[size] = reduction_param->dim_1;
        size++;
    }
    if (reduction_param->dim_2 != -2)
    {
        new_shape[size] = reduction_param->dim_2;
        size++;
    }
    if (reduction_param->dim_3 != -2)
    {
        new_shape[size] = reduction_param->dim_3;
        size++;
    }

    bool should_reduced[4] = {false, false, false, false};

    int reduceddim = 0;
    int real_shape[4] = {0, 1, 2, 3};
    int newshape_size = size;

    if (newshape_size)
    {
        for (int i = 0; i < newshape_size; i++)
        {
            if (new_shape[i] >= 0)
            {
                int idx = new_shape[i];
                if (input->layout == TENGINE_LAYOUT_NHWC)
                    idx = real_shape[idx];
                if (idx >= 0 && idx < 4)
                {
                    should_reduced[idx] = true;
                    ++reduceddim;
                }
            }
            else if (new_shape[i] < 0)
            {
                int current = input->dim_num + new_shape[i];
                if (input->layout == TENGINE_LAYOUT_NHWC)
                {
                    current = real_shape[current];
                }
                should_reduced[current] = true;
                ++reduceddim;
            }
        }
    }
    else
    {
        for (int idx = 0; idx < input->dim_num; ++idx)
        {
            should_reduced[idx] = true;
            ++reduceddim;
        }
    }
    if (input->dim_num - reduceddim == 0)
    {
        if (kd == 0)
        {
            int odim[1] = {1};
            set_ir_tensor_shape(output, odim, 1);
            return 0;
        }
        else
        {
            int* odim = ( int* )sys_malloc(input->dim_num * sizeof(int));
            for (int i_idx = 0, o_idx = 0; i_idx < input->dim_num; i_idx++)
            {
                odim[o_idx++] = 1;
            }
            // TShape shape;
            // shape.SetDim(odim);

            set_ir_tensor_shape(output, odim, input->dim_num);
            sys_free(odim);
            sys_free(in_dim);
            sys_free(new_shape);
            return 0;
        }

        return -1;
    }
    else
    {
        int o_size = 0;
        if (kd == 0)
        {
            o_size = input->dim_num - reduceddim;
        }
        else
        {
            o_size = input->dim_num;
        }
        // std::vector<int> odim(o_size);
        int* odim = ( int* )sys_malloc(o_size * sizeof(int));
        for (int i_idx = 0, o_idx = 0; i_idx < input->dim_num; i_idx++)
        {
            if (!should_reduced[i_idx])
            {
                odim[o_idx++] = in_dim[i_idx];
            }
            else if (should_reduced[i_idx] && kd == 1)
            {
                odim[o_idx++] = 1;
            }
        }
        // TShape shape;
        // shape.SetDim(odim);

        // shape.SetDataLayout(input.GetDataLayout());
        // oshape[0] = shape;
        set_ir_tensor_shape(output, odim, o_size);
        sys_free(odim);
        sys_free(in_dim);
        sys_free(new_shape);
        return 0;
    }

    return -1;
}

static int init_op(struct ir_op* op)
{
    struct reduction_param* reduction_param = ( struct reduction_param* )sys_malloc(sizeof(struct reduction_param));

    if (reduction_param == NULL)
    {
        set_tengine_errno(ENOMEM);
        return -1;
    }

    reduction_param->dim_0 = -2;
    reduction_param->dim_1 = -2;
    reduction_param->dim_2 = -2;
    reduction_param->dim_3 = -2;
    reduction_param->keepdim = 0;
    reduction_param->type = 0;

    op->param_mem = reduction_param;
    op->param_size = sizeof(struct reduction_param);
    op->same_shape = 0;
    op->infer_shape = infer_shape;

    return 0;
}

static void release_op(struct ir_op* op)
{
    sys_free(op->param_mem);
    sys_free(GET_PARAM_PARSE_MAP(reduction_param));
}

static int register_reduction_op(void* arg)
{
    struct op_method m;

    m.op_version = 1;
    m.init_op = init_op;
    m.release_op = release_op;
    m.access_param_entry = access_param_entry;

    return register_op(OP_REDUCTION, OP_REDUCTION_NAME, &m);
}

static int unregister_reduction_op(void* arg)
{
    return unregister_op(OP_REDUCTION, 1);
}

AUTO_REGISTER_OP(register_reduction_op);
AUTO_UNREGISTER_OP(unregister_reduction_op);
