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
 * Author: zpluo@openailab.com
 */

#include "hardsigmoid_param.h"

#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "module/module.h"
#include "utility/sys_port.h"

#include <string.h>


static int infer_shape(struct node* node)
{
    struct graph* graph = node->graph;
    struct tensor* input = get_ir_graph_tensor(graph, node->input_tensors[0]);
    struct tensor* output = get_ir_graph_tensor(graph, node->output_tensors[0]);

    set_ir_tensor_shape(output, input->dims, input->dim_num);

    return 0;
}


static int init_op(struct op* op)
{
    struct hard_sigmoid_param* param = ( struct hard_sigmoid_param* )sys_malloc(sizeof(struct hard_sigmoid_param));

    if (param == NULL)
    {
        return -1;
    }

    /*set the param default value */
    memset(param, 0, sizeof(struct hard_sigmoid_param));
    op->param_mem = param;
    op->param_size = sizeof(struct hard_sigmoid_param);
    op->same_shape = 0;
    op->infer_shape = infer_shape;

    return 0;
}


static void release_op(struct op* op)
{
    sys_free(op->param_mem);
}


int register_hardsigmoid_op()
{
    struct method m;

    m.version = 1;
    m.init = init_op;
    m.release = release_op;

    return register_op(OP_HARDSIGMOID, OP_HARDSIGMOID_NAME, &m);
}


int unregister_hardsigmoid_op()
{
    return unregister_op(OP_HARDSIGMOID, 1);
}
