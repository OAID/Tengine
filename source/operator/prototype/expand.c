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
 * Author: bzhang@openailab.com
 */

#include "expand_param.h"

#include "api/c_api.h"
#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "module/module.h"
#include "utility/sys_port.h"

#include <math.h>


static int infer_shape(struct node* node)
{
    struct graph* graph = node->graph;
    struct tensor* input = get_ir_graph_tensor(graph, node->input_tensors[0]);
    struct tensor* output = get_ir_graph_tensor(graph, node->output_tensors[0]);

    struct expand_param* expand_param = (struct expand_param*)(node->op.param_mem);

    // is MAX_SHAPE_DIM_NUM enough?
    int dims[MAX_SHAPE_DIM_NUM];

    if (input->dim_num == expand_param->shape_size)
    {
        for (int i = 0; i < expand_param->shape_size; i++)
        {
            dims[i] = input->dims[i] >= expand_param->shape[i] ? input->dims[i] : expand_param->shape[i];
        }
    }
    else
    {
        int diff = abs(input->dim_num - expand_param->shape_size);
        if (input->dim_num > expand_param->shape_size)
        {
            for (int i = 0; i < input->dim_num; i++)
            {
                dims[i] = input->dims[i];
            }
            for (int i = 0; i < input->dim_num - diff; i++)
            {
                dims[i + input->dim_num] = input->dims[i + diff] > expand_param->shape[i] ? input->dims[i + diff] : expand_param->shape[i];
            }
        }
        else
        {
            for (int i = 0; i < expand_param->shape_size; i++)
            {
                dims[i] = expand_param->shape[i];
            }
            for (int i = 0; i < expand_param->shape_size - diff; i++)
            {
                dims[i + expand_param->shape_size] = expand_param->shape[i + diff] > input->dim_num[i] ? expand_param->shape[i + diff] : expand_param->shape[i];
            }
        }
    }

    return 0;
}


static int init_op(struct op* op)
{
    struct expand_param* expand_param = (struct expand_param*)sys_malloc(sizeof(struct expand_param));

    if (expand_param == NULL)
    {
        return -1;
    }

    /*set the param default value */
    expand_param->shape_size = 0;
    expand_param->shape = NULL;

    op->param_mem = expand_param;
    op->param_size = sizeof(struct expand_param);
    op->same_shape = 0;
    op->infer_shape = infer_shape;

    return 0;
}


static void release_op(struct op* op)
{
    sys_free(op->param_mem);
}


int register_expand_op()
{
    struct method m;

    m.version = 1;
    m.init = init_op;
    m.release = release_op;

    return register_op(OP_EXPAND, OP_EXPAND_NAME, &m);
}


int unregister_expand_op()
{
    return unregister_op(OP_EXPAND, 1);
}
