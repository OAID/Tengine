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

#include "api/c_api.h"
#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "module/module.h"
#include "utility/log.h"


static int infer_shape(struct node* node)
{
    struct graph* graph = node->graph;
    struct tensor* input0 = get_ir_graph_tensor(graph, node->input_tensors[0]);
    struct tensor* input1 = get_ir_graph_tensor(graph, node->input_tensors[1]);
    struct tensor* output = get_ir_graph_tensor(graph, node->output_tensors[0]);

    if (input1->dim_num != input0->dim_num)
    {
        TLOG_ERR("dim's size of inputs must be qual for operator matmul\n");
        return -1;
    }

    if (input0->dim_num == 2)
    {
        int dims[2];
        dims[0] = input0->dims[0];
        dims[1] = input1->dims[1];
        set_ir_tensor_shape(output, dims, 2);

        return 0;
    }
    else if (input0->dim_num == 3)
    {
        int dims[3];
        dims[0] = input0->dims[0];
        dims[1] = input0->dims[1];
        dims[2] = input1->dims[2];
        set_ir_tensor_shape(output, dims, 3);

        return 0;
    }
    else if (input0->dim_num == 4)
    {
        int dims[4];
        dims[0] = input0->dims[0];
        dims[1] = input0->dims[1];
        dims[2] = input0->dims[2];
        dims[3] = input1->dims[3];
        set_ir_tensor_shape(output, dims, 4);

        return 0;
    }        

    return -1;
}


static int init_op(struct op* op)
{
    op->same_shape = 0;
    op->infer_shape = infer_shape;
    return 0;
}


int register_matmul_op()
{
    struct method m;

    m.version = 1;
    m.init = init_op;
    m.release = NULL;


    return register_op(OP_MATMUL, OP_MATMUL_NAME, &m);
}


int unregister_matmul_op()
{
    return unregister_op(OP_MATMUL, 1);
}
