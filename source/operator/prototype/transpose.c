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
 * Author: bhu@openailab.com
 */

#include "transpose_param.h"

#include "api/c_api.h"
#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "module/module.h"
#include "utility/sys_port.h"
#include "utility/log.h"


static int infer_shape(struct node* node)
{
    struct graph* graph = node->graph;
    struct tensor* input = get_ir_graph_tensor(graph, node->input_tensors[0]);
    struct tensor* output = get_ir_graph_tensor(graph, node->output_tensors[0]);
    struct transpose_param* param = ( struct transpose_param* )(node->op.param_mem);

    int new_shape_size = param->tr_shape_size;
    int* out_dims = ( int* )sys_malloc(new_shape_size * sizeof(int));

    for (int i = 0; i < new_shape_size; i++)
    {
        out_dims[i] = input->dims[param->tr_shape[i]];
    }

    set_ir_tensor_shape(output, out_dims, new_shape_size);
    sys_free(out_dims);

    return 0;
}


static int init_op(struct op* op)
{
    struct transpose_param* transpose_param = ( struct transpose_param* )sys_malloc(sizeof(struct transpose_param));

    if (transpose_param == NULL)
    {
        return -1;
    }

    /*set the param default value */
    transpose_param->dim_0 = -2;
    transpose_param->dim_1 = -2;
    transpose_param->dim_2 = -2;
    transpose_param->dim_3 = -2;

    op->param_mem = transpose_param;
    op->param_size = sizeof(struct transpose_param);
    op->same_shape = 0;
    op->infer_shape = infer_shape;

    return 0;
}


static void release_op(struct op* op)
{
    struct transpose_param* transpose_param = ( struct transpose_param* )op->param_mem;

    if (transpose_param->tr_shape)
        sys_free(transpose_param->tr_shape);

    sys_free(op->param_mem);
}


int register_transpose_op()
{
    struct method m;

    m.version = 1;
    m.init = init_op;
    m.release = release_op;


    return register_op(OP_TRANSPOSE, OP_TRANSPOSE_NAME, &m);
}


int unregister_transpose_op()
{
    return unregister_op(OP_TRANSPOSE, 1);
}
