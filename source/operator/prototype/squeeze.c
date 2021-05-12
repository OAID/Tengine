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
 * Copyright (c) 2021, OPEN AI LAB
 * Author: haitao@openailab.com
 */

#include "squeeze_param.h"

#include "api/c_api.h"
#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "module/module.h"
#include "utility/sys_port.h"
#include "utility/log.h"


static int infer_shape(struct node* node)
{
    struct graph* ir_graph = node->graph;
    struct tensor* input = get_ir_graph_tensor(ir_graph, node->input_tensors[0]);
    struct tensor* output = get_ir_graph_tensor(ir_graph, node->output_tensors[0]);
    struct squeeze_param* squeeze_param = ( struct squeeze_param* )node->op.param_mem;

    int in_size = input->dim_num;

    int new_shape[4];
    int dim_size = 0;
    if (squeeze_param->dim_0 != -2)
    {
        new_shape[dim_size] = squeeze_param->dim_0;
        dim_size++;
    }
    if (squeeze_param->dim_1 != -2)
    {
        new_shape[dim_size] = squeeze_param->dim_1;
        dim_size++;
    }
    if (squeeze_param->dim_2 != -2)
    {
        new_shape[dim_size] = squeeze_param->dim_2;
        dim_size++;
    }
    if (squeeze_param->dim_3 != -2)
    {
        new_shape[dim_size] = squeeze_param->dim_3;
        dim_size++;
    }

    int8_t should_squeeze[4] = { 0 };
    int squeezeddim = 0;
    int newshape_size = dim_size;
    int real_shape[4] = {0, 2, 3, 1};

    if (newshape_size)
    {
        for (int i = 0; i < newshape_size; i++)
        {
            if (new_shape[i] >= 0)
            {
                int idx = new_shape[i];
                if (ir_graph->graph_layout == TENGINE_LAYOUT_NCHW)
                    idx = real_shape[idx];
                if (input->dims[idx] == 1 && idx >= 0 && idx < 4)
                {
                    should_squeeze[idx] = 1;
                    ++squeezeddim;
                }
            }
            else if (new_shape[i] < 0)
            {
                int idx = new_shape[i];
                if (ir_graph->graph_layout == TENGINE_LAYOUT_NCHW)
                    idx = real_shape[idx];
                if (input->dims[idx] == 1 && idx > 0 && idx < 3)
                {
                    int current = input->dim_num + idx;
                    should_squeeze[current] = 1;
                    ++squeezeddim;
                }
            }
        }
    }
    else
    {
        for (int idx = 0; idx < in_size; ++idx)
        {
            if (input->dims[idx] == 1)
            {
                should_squeeze[idx] = 1;
                ++squeezeddim;
            }
        }
    }

    int* odim = ( int* )sys_malloc((in_size - squeezeddim) * sizeof(int));
    int o_idx = 0;
    for (int i_idx = 0; i_idx < in_size; i_idx++)
    {
        if (!should_squeeze[i_idx])
            odim[o_idx++] = input->dims[i_idx];
    }

    set_ir_tensor_shape(output, odim, o_idx);

    sys_free(odim);
    return 0;
}


static int init_op(struct op* op)
{
    struct squeeze_param* squeeze_param = ( struct squeeze_param* )sys_malloc(sizeof(struct squeeze_param));

    if (squeeze_param == NULL)
    {
        return -1;
    }

    squeeze_param->dim_0 = -2;
    squeeze_param->dim_1 = -2;
    squeeze_param->dim_2 = -2;
    squeeze_param->dim_3 = -2;

    op->param_mem = squeeze_param;
    op->param_size = sizeof(struct squeeze_param);
    op->same_shape = 0;
    op->infer_shape = infer_shape;

    return 0;
}


static void release_op(struct op* op)
{
    sys_free(op->param_mem);
}


int register_squeeze_op()
{
    struct method m;

    m.version = 1;
    m.init = init_op;
    m.release = release_op;

    return register_op(OP_SQUEEZE, OP_SQUEEZE_NAME, &m);
}


int unregister_squeeze_op()
{
    return unregister_op(OP_SQUEEZE, 1);
}
