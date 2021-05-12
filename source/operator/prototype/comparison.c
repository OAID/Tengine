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
 * Author: ddzhao@openailab.com
 */

#include "comparison_param.h"

#include "defines.h"
#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "module/module.h"
#include "utility/sys_port.h"

#include <string.h>


#define CALC_TENSOR_SHAPE_SIZE(outval, IR_TENSOR)       \
    {                                                   \
        outval = 1;                                     \
        for (int ii = 0; ii < IR_TENSOR->dim_num; ++ii) \
        {                                               \
            outval *= IR_TENSOR->dims[ii];              \
        }                                               \
    }


static int infer_shape(struct node* node)
{
    struct graph* graph = node->graph;
    struct tensor* input = get_ir_graph_tensor(graph, node->input_tensors[0]);
    struct tensor* output = get_ir_graph_tensor(graph, node->output_tensors[0]);

    if (node->input_num == 1)
    {
        set_ir_tensor_shape(output, input->dims, 1);
        return 0;
    }

    if (node->input_num != 2)
    {
        return 1;
    }

    struct tensor* input_1 = get_ir_graph_tensor(graph, node->input_tensors[1]);
    int i0_size = 1;
    int i1_size = 1;

    CALC_TENSOR_SHAPE_SIZE(i0_size, input);
    CALC_TENSOR_SHAPE_SIZE(i1_size, input_1);

    int dims[2] = {0};
    dims[0] = 1;
    for (int ii = 0; ii < input->dim_num; ++ii)
    {
        dims[0] *= input->dims[ii];
    }

    if (i0_size >= i1_size)
    {
        output->dims[0] = input->dims[0];
    }
    else if (i0_size < i1_size)
    {
        output->dims[0] = input_1->dims[0];
    }

    return 0;
}


static int init_op(struct op* op)
{
    struct comparison_param* param = ( struct comparison_param* )sys_malloc(sizeof(struct comparison_param));

    if (param == NULL)
    {
        return -1;
    }

    /*set the param default value */
    memset(param, 0, sizeof(struct comparison_param));
    op->param_mem = param;
    op->param_size = sizeof(struct comparison_param);
    op->same_shape = 0;
    op->infer_shape = infer_shape;

    return 0;
}


static void release_op(struct op* op)
{
    sys_free(op->param_mem);
}


int register_comparison_op()
{
    struct method m;

    m.version = 1;
    m.init = init_op;
    m.release = release_op;


    return register_op(OP_COMPARISON, OP_COMPARISON_NAME, &m);
}


int unregister_comparison_op()
{
    return unregister_op(OP_COMPARISON, 1);
}
