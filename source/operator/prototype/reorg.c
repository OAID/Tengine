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

#include "reorg_param.h"

#include "api/c_api.h"
#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "module/module.h"
#include "utility/sys_port.h"

static int infer_shape(struct node* node)
{
    struct graph* graph = node->graph;
    struct tensor* input = get_ir_graph_tensor(graph, node->input_tensors[0]);
    struct tensor* output = get_ir_graph_tensor(graph, node->output_tensors[0]);
    struct reorg_param* reorg_param = (struct reorg_param*)(node->op.param_mem);

    int stride = reorg_param->stride;

    int in_n = input->dims[0];
    int in_c = input->dims[1];
    int in_h = input->dims[2];
    int in_w = input->dims[3];

    int dims[4];

    dims[0] = in_n;
    dims[1] = in_c * stride * stride;
    dims[2] = in_h / stride;
    dims[3] = in_w / stride;

    set_ir_tensor_shape(output, dims, 4);

    return 0;
}

static int init_op(struct op* op)
{
    struct reorg_param* reorg_param = (struct reorg_param*)sys_malloc(sizeof(struct reorg_param));

    if (reorg_param == NULL)
    {
        return -1;
    }

    /*set the param default value */
    reorg_param->stride = 1;

    op->param_mem = reorg_param;
    op->param_size = sizeof(struct reorg_param);
    op->same_shape = 0;
    op->infer_shape = infer_shape;

    return 0;
}

static void release_op(struct op* op)
{
    sys_free(op->param_mem);
}

int register_reorg_op()
{
    struct method m;

    m.version = 1;
    m.init = init_op;
    m.release = release_op;

    return register_op(OP_REORG, OP_REORG_NAME, &m);
}

int unregister_reorg_op()
{
    return unregister_op(OP_REORG, 1);
}
