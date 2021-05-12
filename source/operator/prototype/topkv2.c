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

#include "topkv2_param.h"

#include "api/c_api.h"
#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "module/module.h"
#include "utility/sys_port.h"


static int infer_shape(struct node* node)
{
    struct topkv2_param* topkv2_param = (struct topkv2_param*)node->op.param_mem;

    struct graph* ir_graph = node->graph;
    struct tensor* input = get_ir_graph_tensor(ir_graph, node->input_tensors[0]);
    struct tensor* output = get_ir_graph_tensor(ir_graph, node->output_tensors[0]);
    struct tensor* output1 = get_ir_graph_tensor(ir_graph, node->output_tensors[1]);

    int in_size = input->dim_num;
    int* in_dim = ( int* )sys_malloc((in_size) * sizeof(int));

    if (topkv2_param->k > input->dims[in_size - 1])
    {
        return false;
    }
    for (int i = 0; i < in_size - 1; i++)
    {
        in_dim[i] = input->dims[i];
    }
    in_dim[in_size - 1] = topkv2_param->k;
    set_ir_tensor_shape(output, in_dim, in_size);

    set_ir_tensor_shape(output1, in_dim, in_size);

    sys_free(in_dim);
    return 0;
}


static int init_op(struct op* op)
{
    struct topkv2_param* topkv2_param = ( struct topkv2_param* )sys_malloc(sizeof(struct topkv2_param));

    if (topkv2_param == NULL)
    {
        return -1;
    }

    topkv2_param->k = 1;
    topkv2_param->sorted = false;

    op->param_mem = topkv2_param;
    op->param_size = sizeof(struct topkv2_param);
    op->same_shape = 0;
    op->infer_shape = infer_shape;

    return 0;
}


static void release_op(struct op* op)
{
    sys_free(op->param_mem);
}


int register_topkv2_op()
{
    struct method m;

    m.version = 1;
    m.init = init_op;
    m.release = release_op;


    return register_op(OP_TOPKV2, OP_TOPKV2_NAME, &m);
}


int unregister_topkv2_op()
{
    return unregister_op(OP_TOPKV2, 1);
}
