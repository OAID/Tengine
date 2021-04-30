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
 * Author: chh@openailab.com
 */

#include "roialign_param.h"

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

    struct roialign_param* roialign_param = ( struct roialign_param* )(node->op.param_mem);

    int out_dim[4];

    out_dim[0] = input->dims[0];
    out_dim[1] = input->dims[1];
    out_dim[2] = roialign_param->pooled_height;
    out_dim[3] = roialign_param->pooled_width;

    set_ir_tensor_shape(output, out_dim, 4);

    return 0;
}


static int init_op(struct op* op)
{
    struct roialign_param* roialign_param = ( struct roialign_param* )sys_malloc(sizeof(struct roialign_param));

    if (roialign_param == NULL)
    {
        return -1;
    }

    /*set the param default value */
    roialign_param->pooled_width = 0;
    roialign_param->pooled_height = 0;
    roialign_param->spatial_scale = 0.f;

    op->param_mem = roialign_param;
    op->param_size = sizeof(struct roialign_param);
    op->same_shape = 0;
    op->infer_shape = infer_shape;

    return 0;
}


static void release_op(struct op* op)
{
    sys_free(op->param_mem);
}


int register_roialign_op()
{
    struct method m;

    m.version = 1;
    m.init = init_op;
    m.release = release_op;


    return register_op(OP_ROIALIGN, OP_ROIALIGN_NAME, &m);
}


int unregister_roialign_op()
{
    return unregister_op(OP_ROIALIGN, 1);
}
