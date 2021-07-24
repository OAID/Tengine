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

#include "psroipooling_param.h"

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

    struct psroipooling_param* psroipooling_param = ( struct psroipooling_param* )(node->op.param_mem);

    int output_n = input->dims[0];
    int output_c = psroipooling_param->output_dim;
    int output_h = psroipooling_param->pooled_h;
    int output_w = psroipooling_param->pooled_w;

    int out_dim[4];

    out_dim[0] = output_n;
    out_dim[1] = output_c;
    out_dim[2] = output_h;
    out_dim[3] = output_w;

    set_ir_tensor_shape(output, out_dim, 4);

    return 0;
}


static int init_op(struct op* op)
{
    struct psroipooling_param* psroipooling_param = (struct psroipooling_param*)sys_malloc(sizeof(struct psroipooling_param));

    if (psroipooling_param == NULL)
    {
        return -1;
    }

    /*set the param default value */
    psroipooling_param->pooled_w = 0;
    psroipooling_param->pooled_h = 0;
    psroipooling_param->spatial_scale = 0.f;
    psroipooling_param->output_dim = 0;

    op->param_mem = psroipooling_param;
    op->param_size = sizeof(struct psroipooling_param);
    op->same_shape = 0;
    op->infer_shape = infer_shape;

    return 0;
}


static void release_op(struct op* op)
{
    sys_free(op->param_mem);
}


int register_psroipooling_op()
{
    struct method m;

    m.version = 1;
    m.init = init_op;
    m.release = release_op;

    return register_op(OP_PSROIPOOLING, OP_PSROIPOOLING_NAME, &m);
}


int unregister_psroipooling_op()
{
    return unregister_op(OP_PSROIPOOLING, 1);
}
