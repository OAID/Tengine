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
 * Author: qli@openailab.com
 */

#include "gemm_param.h"

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
    struct tensor* weight = get_ir_graph_tensor(graph, node->input_tensors[1]);

    struct gemm_param* gemm_param = ( struct gemm_param* )(node->op.param_mem);

    int dims[2];
    if (gemm_param->transA)
        dims[0] = input->dims[1];
    else
        dims[0] = input->dims[0];

    if (gemm_param->transB)
        dims[1] = weight->dims[0];
    else
        dims[1] = weight->dims[1];

    set_ir_tensor_shape(output, dims, 2);

    return 0;
}


static int init_op(struct op* op)
{
    struct gemm_param* gemm_param = ( struct gemm_param* )sys_malloc(sizeof(struct gemm_param));

    if (gemm_param == NULL)
    {
        return -1;
    }
    op->param_mem = gemm_param;
    op->param_size = sizeof(struct gemm_param);
    op->same_shape = 0;
    op->infer_shape = infer_shape;

    return 0;
}


static void release_op(struct op* op)
{
    sys_free(op->param_mem);
}


int register_gemm_op()
{
    struct method m;

    m.version = 1;
    m.init = init_op;
    m.release = release_op;

    return register_op(OP_GEMM, OP_GEMM_NAME, &m);
}


int unregister_gemm_op()
{
    return unregister_op(OP_GEMM, 1);
}
