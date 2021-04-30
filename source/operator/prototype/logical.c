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
 * Author: qtang@openailab.com
 */

#include "logical_param.h"

#include "api/c_api.h"
#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "module/module.h"
#include "utility/sys_port.h"
#include "utility/log.h"


static int infer_shape(struct node* node)
{
    if (node->input_num == 1)
    {
        struct graph* graph = node->graph;
        struct tensor* input = get_ir_graph_tensor(graph, node->input_tensors[0]);
        struct tensor* output = get_ir_graph_tensor(graph, node->output_tensors[0]);

        set_ir_tensor_shape(output, input->dims, input->dim_num);

        return 0;
    }

    if (node->input_num == 2)
    {
        struct graph* graph = node->graph;
        struct tensor* input0 = get_ir_graph_tensor(graph, node->input_tensors[0]);
        struct tensor* input1 = get_ir_graph_tensor(graph, node->input_tensors[1]);
        struct tensor* output = get_ir_graph_tensor(graph, node->output_tensors[0]);

        if (input0->elem_num > input1->elem_num)
            set_ir_tensor_shape(output, input0->dims, input0->dim_num);
        else
            set_ir_tensor_shape(output, input1->dims, input1->dim_num);

        return 0;
    }

    return -1;
}


static int init_op(struct op* op)
{
    struct logical_param* logical_param = ( struct logical_param* )sys_malloc(sizeof(struct logical_param));

    if (logical_param == NULL)
    {
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


static void release_op(struct op* op)
{
    sys_free(op->param_mem);
}


int register_logical_op()
{
    struct method m;

    m.version = 1;
    m.init = init_op;
    m.release = release_op;

    return register_op(OP_LOGICAL, OP_LOGICAL_NAME, &m);
}


int unregister_logical_op()
{
    return unregister_op(OP_LOGICAL, 1);
}
