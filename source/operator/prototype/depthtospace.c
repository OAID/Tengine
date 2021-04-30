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

#include "depthtospace_param.h"

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

    struct depthtospace_param* depthtospace_param = ( struct depthtospace_param* )(node->op.param_mem);

    /* todo reshape */
    int dims[4];
    int block_size = depthtospace_param->block_size;

    dims[0] = input->dims[0];    // batch
    dims[1] = input->dims[1] / (block_size * block_size);    // channel
    dims[2] = input->dims[2] * block_size;    // height
    dims[3] = input->dims[3] * block_size;    // width

    set_ir_tensor_shape(output, dims, 4);

    return 0;
}


static int init_op(struct op* op)
{
    struct depthtospace_param* depthtospace_param =
        ( struct depthtospace_param* )sys_malloc(sizeof(struct depthtospace_param));

    if (depthtospace_param == NULL)
    {
        return -1;
    }

    /*set the param default value */
    depthtospace_param->block_size = 1;

    op->param_mem = depthtospace_param;
    op->param_size = sizeof(struct depthtospace_param);
    op->same_shape = 0;
    op->infer_shape = infer_shape;

    return 0;
}


static void release_op(struct op* op)
{
    sys_free(op->param_mem);
}


int register_depthtospace_op()
{
    struct method m;

    m.version = 1;
    m.init = init_op;
    m.release = release_op;


    return register_op(OP_DEPTHTOSPACE, OP_DEPTHTOSPACE_NAME, &m);
}


int unregister_depthtospace_op()
{
    return unregister_op(OP_DEPTHTOSPACE, 1);
}
