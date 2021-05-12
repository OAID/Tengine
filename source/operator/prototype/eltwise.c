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

#include "eltwise_param.h"

#include "api/c_api.h"
#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "module/module.h"
#include "utility/sys_port.h"
#include "utility/log.h"

#include <string.h>


static int infer_shape(struct node* node)
{
    struct graph* graph = node->graph;
    struct tensor* input0 = get_ir_graph_tensor(graph, node->input_tensors[0]);
    struct tensor* output = get_ir_graph_tensor(graph, node->output_tensors[0]);

    struct eltwise_param* eltwise_param = ( struct eltwise_param* )(node->op.param_mem);

    if (node->input_num == 1)
    {
        set_ir_tensor_shape(output, input0->dims, input0->dim_num);
        return 0;
    }

    if (node->input_num != 2)
    {
        TLOG_ERR("eltwise infer shape error : input tensor number : %d \n", node->input_num);
        return -1;
    }

    struct tensor* input1 = get_ir_graph_tensor(graph, node->input_tensors[1]);

    int i0_size = input0->elem_num;
    int i1_size = input1->elem_num;
    int dim_num = 0;

    if (i0_size >= i1_size)
    {
        memcpy(output->dims, input0->dims, input0->dim_num * sizeof(int));
        dim_num = input0->dim_num;
    }
    else
    {
        memcpy(output->dims, input1->dims, input1->dim_num * sizeof(int));
        dim_num = input1->dim_num;
    }

    set_ir_tensor_shape(output, output->dims, dim_num);

    return 0;
}


static int init_op(struct op* op)
{
    struct eltwise_param* eltwise_param = ( struct eltwise_param* )sys_malloc(sizeof(struct eltwise_param));

    if (eltwise_param == NULL)
    {
        return -1;
    }

    /*set the param default value */
    eltwise_param->type = 0;

    op->param_mem = eltwise_param;
    op->param_size = sizeof(struct eltwise_param);
    op->same_shape = 0;
    op->infer_shape = infer_shape;

    return 0;
}


static void release_op(struct op* op)
{
    sys_free(op->param_mem);
}


int register_eltwise_op()
{
    struct method m;

    m.version = 1;
    m.init = init_op;
    m.release = release_op;


    return register_op(OP_ELTWISE, OP_ELTWISE_NAME, &m);
}


int unregister_eltwise_op()
{
    return unregister_op(OP_ELTWISE, 1);
}
